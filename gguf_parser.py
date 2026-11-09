"""
GGUF & Multi-Format Model Parser and Metadata Inspector
Extracts architecture, quantization, parameter counts, context length, tensor counts,
and key-value attributes directly from GGUF binary headers (v1, v2, v3), SafeTensors
headers, and PyTorch / ONNX / binary model files without loading weights into RAM.
"""

import os
import re
import json
import struct
from typing import Dict, Any, List

# GGUF Constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

SUPPORTED_MODEL_EXTENSIONS = (
    ".gguf",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
)


class GGUFReader:
    """Reads and parses GGUF and other model headers without loading weights."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.ext = os.path.splitext(file_path)[1].lower()
        self.is_gguf = False
        self.version = 0
        self.tensor_count = 0
        self.metadata_kv_count = 0
        self.computed_params = 0
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        self._parse()

    def _read_string(self, f) -> str:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            return ""
        length = struct.unpack("<Q", length_bytes)[0]
        data = f.read(length)
        return data.decode("utf-8", errors="replace")

    def _read_value(self, f, val_type: int) -> Any:
        if val_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif val_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif val_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif val_type == GGUF_TYPE_INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif val_type == GGUF_TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif val_type == GGUF_TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif val_type == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif val_type == GGUF_TYPE_BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif val_type == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif val_type == GGUF_TYPE_ARRAY:
            sub_type, count = struct.unpack("<IQ", f.read(12))
            # Fast-skip large fixed-size scalar arrays when over 100 items
            scalar_sizes = {
                GGUF_TYPE_UINT8: 1,
                GGUF_TYPE_INT8: 1,
                GGUF_TYPE_BOOL: 1,
                GGUF_TYPE_UINT16: 2,
                GGUF_TYPE_INT16: 2,
                GGUF_TYPE_UINT32: 4,
                GGUF_TYPE_INT32: 4,
                GGUF_TYPE_FLOAT32: 4,
                GGUF_TYPE_UINT64: 8,
                GGUF_TYPE_INT64: 8,
                GGUF_TYPE_FLOAT64: 8,
            }
            if count > 100 and sub_type in scalar_sizes:
                items = [self._read_value(f, sub_type) for _ in range(100)]
                f.seek((count - 100) * scalar_sizes[sub_type], os.SEEK_CUR)
                items.append(f"... ({count} total items)")
                return items
            items = []
            for _ in range(count):
                val = self._read_value(f, sub_type)
                if len(items) < 100:
                    items.append(val)
                elif len(items) == 100:
                    items.append(f"... ({count} total items)")
            return items
        elif val_type == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif val_type == GGUF_TYPE_INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif val_type == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        return None

    def _parse(self):
        if not os.path.isfile(self.file_path):
            return

        with open(self.file_path, "rb") as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                return
            magic = struct.unpack("<I", magic_bytes)[0]
            if magic == GGUF_MAGIC:
                self.is_gguf = True
                self._parse_gguf(f)
            elif self.ext == ".safetensors":
                f.seek(0)
                self._parse_safetensors(f)
            else:
                self._parse_generic_model()

    def _parse_gguf(self, f):
        self.version = struct.unpack("<I", f.read(4))[0]
        self.tensor_count = struct.unpack("<Q", f.read(8))[0]
        self.metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        for _ in range(self.metadata_kv_count):
            key = self._read_string(f)
            val_type = struct.unpack("<I", f.read(4))[0]
            val = self._read_value(f, val_type)
            self.metadata[key] = val

        # Read tensor info headers to compute exact parameter count
        total_elements = 0
        try:
            for _ in range(self.tensor_count):
                t_name = self._read_string(f)
                if not t_name:
                    break
                n_dims_bytes = f.read(4)
                if len(n_dims_bytes) < 4:
                    break
                n_dims = struct.unpack("<I", n_dims_bytes)[0]
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                t_type, t_offset = struct.unpack("<IQ", f.read(12))
                numel = 1
                for d in dims:
                    numel *= d
                total_elements += numel
            self.computed_params = total_elements
        except Exception:
            pass

    def _parse_safetensors(self, f):
        try:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if 0 < header_len < 100 * 1024 * 1024:
                header_json = json.loads(f.read(header_len).decode("utf-8", errors="replace"))
                meta = header_json.get("__metadata__", {})
                self.metadata["general.architecture"] = meta.get("modelspec.architecture", "safetensors")
                dtypes = set()
                total_elements = 0
                t_count = 0
                for k, v in header_json.items():
                    if k == "__metadata__" or not isinstance(v, dict):
                        continue
                    t_count += 1
                    if "dtype" in v:
                        dtypes.add(v["dtype"])
                    shape = v.get("shape", [])
                    if shape:
                        numel = 1
                        for d in shape:
                            numel *= int(d)
                        total_elements += numel
                self.tensor_count = t_count
                self.computed_params = total_elements
                self.metadata["safetensors.dtypes"] = ", ".join(sorted(dtypes)) if dtypes else "F16/BF16"
        except Exception:
            pass

    def _parse_generic_model(self):
        format_names = {
            ".pt": "PyTorch (.pt)",
            ".pth": "PyTorch (.pth)",
            ".bin": "Binary Weights (.bin)",
            ".onnx": "ONNX Graph (.onnx)",
        }
        self.metadata["general.architecture"] = format_names.get(self.ext, f"Model ({self.ext})")

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "Unknown"))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("general.name", os.path.basename(self.file_path)))

    @property
    def file_type_desc(self) -> str:
        if not self.is_gguf:
            if self.ext == ".safetensors":
                return f"SafeTensors ({self.metadata.get('safetensors.dtypes', 'Tensor')})"
            return self.ext.upper().lstrip(".") or "RAW"

        ftype = self.metadata.get("general.file_type")
        types_map = {
            0: "ALL_F32",
            1: "MOSTLY_F16",
            2: "MOSTLY_Q4_0",
            3: "MOSTLY_Q4_1",
            7: "MOSTLY_Q8_0",
            8: "MOSTLY_Q5_0",
            9: "MOSTLY_Q5_1",
            10: "MOSTLY_Q2_K",
            11: "MOSTLY_Q3_K_S",
            12: "MOSTLY_Q3_K_M",
            13: "MOSTLY_Q3_K_L",
            14: "MOSTLY_Q4_K_S",
            15: "MOSTLY_Q4_K_M",
            16: "MOSTLY_Q5_K_S",
            17: "MOSTLY_Q5_K_M",
            18: "MOSTLY_Q6_K",
            19: "MOSTLY_IQ2_XXS",
            20: "MOSTLY_IQ2_XS",
            21: "MOSTLY_Q2_K_S",
            22: "MOSTLY_IQ3_XS",
            23: "MOSTLY_IQ3_XXS",
            24: "MOSTLY_IQ1_S",
            25: "MOSTLY_IQ4_NL",
            26: "MOSTLY_IQ3_S",
            27: "MOSTLY_IQ3_M",
            28: "MOSTLY_IQ2_S",
            29: "MOSTLY_IQ2_M",
            30: "MOSTLY_IQ4_XS",
        }
        return types_map.get(ftype, f"Type {ftype}" if ftype is not None else "Unknown")

    @property
    def context_length(self) -> int:
        arch = self.architecture
        keys = [
            f"{arch}.context_length",
            "context_length",
            "llama.context_length",
            "qwen2.context_length",
            "qwen2vl.context_length",
            "phi3.context_length",
        ]
        for k in keys:
            if k in self.metadata:
                return int(self.metadata[k])
        return 4096 if self.is_gguf else 0

    @property
    def parameter_count(self) -> str:
        params = self.metadata.get("general.parameter_count") or self.computed_params
        if params:
            count = float(params)
            if count >= 1e9:
                return f"{count / 1e9:.2f}B"
            elif count >= 1e6:
                return f"{count / 1e6:.1f}M"
            return str(int(count))
        size_label = self.metadata.get("general.size_label")
        if size_label:
            return str(size_label)
        # Fallback regex check on filename (e.g. 7B, 8B, 14B)
        m = re.search(r"(\d+(?:\.\d+)?[bBmM])\b", os.path.basename(self.file_path))
        if m:
            return m.group(1).upper()
        return "Unknown"

    @property
    def layer_count(self) -> int:
        arch = self.architecture
        keys = [
            f"{arch}.block_count",
            "llama.block_count",
            "qwen2.block_count",
            "qwen2vl.block_count",
            "block_count",
        ]
        for k in keys:
            if k in self.metadata:
                return int(self.metadata[k])
        return 0

    def get_summary(self) -> Dict[str, Any]:
        return {
            "Filename": os.path.basename(self.file_path),
            "Size": f"{self.file_size / (1024**3):.2f} GB",
            "Architecture": self.architecture,
            "Parameters": self.parameter_count,
            "Quantization": self.file_type_desc,
            "Context Length": self.context_length,
            "Layers (Blocks)": self.layer_count,
            "Tensors Count": self.tensor_count,
            "GGUF Version": f"v{self.version}" if self.is_gguf else self.ext.upper().lstrip("."),
        }
