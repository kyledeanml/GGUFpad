"""
GGUF File Parser and Metadata Inspector
Extracts architecture, quantization, context length, tensor counts, and key-value attributes
directly from GGUF binary headers (GGUF v1, v2, and v3).
"""

import os
import struct
from typing import Dict, Any, List, Tuple

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


class GGUFReader:
    """Reads and parses GGUF model headers without loading weights."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        self.version = 0
        self.tensor_count = 0
        self.metadata_kv_count = 0
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
            items = []
            for _ in range(count):
                val = self._read_value(f, sub_type)
                # Cap extremely large arrays (e.g. tokenizer tokens) in metadata display
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
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a valid GGUF file. Magic: {hex(magic)}")

            self.version = struct.unpack("<I", f.read(4))[0]
            self.tensor_count = struct.unpack("<Q", f.read(8))[0]
            self.metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(self.metadata_kv_count):
                key = self._read_string(f)
                val_type = struct.unpack("<I", f.read(4))[0]
                val = self._read_value(f, val_type)
                self.metadata[key] = val

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "Unknown"))

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("general.name", os.path.basename(self.file_path)))

    @property
    def file_type_desc(self) -> str:
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
            "phi3.context_length",
        ]
        for k in keys:
            if k in self.metadata:
                return int(self.metadata[k])
        return 4096

    @property
    def parameter_count(self) -> str:
        params = self.metadata.get("general.parameter_count")
        if params:
            count = float(params)
            if count >= 1e9:
                return f"{count / 1e9:.1f}B"
            elif count >= 1e6:
                return f"{count / 1e6:.1f}M"
            return str(params)
        return "Unknown"

    @property
    def layer_count(self) -> int:
        arch = self.architecture
        keys = [f"{arch}.block_count", "llama.block_count", "block_count"]
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
            "GGUF Version": f"v{self.version}",
        }
