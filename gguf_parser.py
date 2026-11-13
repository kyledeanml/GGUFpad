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


GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
}

IMAGE_ARCHITECTURES = {
    "flux",
    "flux1",
    "flux2",
    "flux3",
    "z_image",
    "z-image",
    "lumina2",
    "nextdit",
    "qwen_image",
    "qwen-image",
    "qwen_image_2_1",
    "sd3",
    "sd3.5",
    "sdxl",
    "sd1",
    "sd2",
    "sd",
    "stable-diffusion",
    "chroma",
    "wan",
    "hunyuan_video",
}


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
        self.tensor_names: List[str] = []
        self.tensor_type_counts: Dict[int, int] = {}
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

        # Read tensor info headers to compute exact parameter count and detect architecture/quantization
        total_elements = 0
        try:
            for idx in range(self.tensor_count):
                t_name = self._read_string(f)
                if not t_name:
                    break
                if idx < 800:
                    self.tensor_names.append(t_name)
                n_dims_bytes = f.read(4)
                if len(n_dims_bytes) < 4:
                    break
                n_dims = struct.unpack("<I", n_dims_bytes)[0]
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                t_type, t_offset = struct.unpack("<IQ", f.read(12))
                self.tensor_type_counts[t_type] = self.tensor_type_counts.get(t_type, 0) + 1
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
                arch = meta.get("modelspec.architecture") or meta.get("ss_base_model_version") or "safetensors"
                self.metadata["general.architecture"] = arch
                dtypes = set()
                total_elements = 0
                t_count = 0
                for k, v in header_json.items():
                    if k == "__metadata__" or not isinstance(v, dict):
                        continue
                    t_count += 1
                    if len(self.tensor_names) < 800:
                        self.tensor_names.append(k)
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

    def _infer_image_profile_from_tensors_and_name(self) -> str:
        """Detects text-to-image diffusion family from GGUF metadata, tensor names, or filename."""
        raw_arch = str(self.metadata.get("general.architecture", "")).lower().strip()
        fname = os.path.basename(self.file_path).lower()
        t_joined = " ".join(self.tensor_names[:300])

        # Exclude standalone text encoders & VLMs even if named similarly (e.g. Qwen3-4B, Qwen3VL-8B, T5, CLIP)
        if any(k in fname for k in ("t5", "clip_l", "clip_g", "mmproj", "instruct", "qwen2.5-vl", "qwen3-4b", "qwen3vl")):
            if "diffusion" not in fname and "image" not in fname and "flux" not in fname:
                return ""
        if raw_arch in ("llama", "qwen2", "qwen2vl", "qwen3", "qwen3vl", "mistral", "phi3", "gemma2", "t5", "t5encoder"):
            return ""
        if any(p in t_joined for p in ("blk.0.attn_q.weight", "encoder.block.0.layer", "text_model.encoder.layers")):
            if "double_blocks." not in t_joined and "joint_blocks." not in t_joined and "x_embedder." not in t_joined:
                return ""

        # 1. Explicit architecture tag
        if raw_arch in ("z_image", "z-image", "lumina2", "nextdit"):
            return "z-image-turbo" if "turbo" in fname else "z-image"
        if raw_arch in ("qwen_image", "qwen-image", "qwen_image_2_1"):
            if "2.1" in fname or "2_1" in fname:
                return "qwen-image-2.1-turbo" if "turbo" in fname else "qwen-image-2.1"
            return "qwen-image"
        if raw_arch.startswith("flux"):
            if "flux3" in fname or "flux-3" in fname or raw_arch == "flux3":
                return "flux3-klein"
            if "flux2" in fname or "flux-2" in fname or raw_arch == "flux2":
                return "flux2-klein" if "klein" in fname else "flux2-dev"
            if "schnell" in fname:
                return "flux1-schnell"
            if "dev" in fname or raw_arch in ("flux", "flux1"):
                return "flux1-dev"
        if raw_arch in ("sd3", "sd3.5"):
            return "sd3.5"
        if "sdxl" in raw_arch:
            return "sdxl"
        if raw_arch in ("sd", "sd1", "sd2", "stable-diffusion"):
            return "sd1.5"

        # 2. Tensor-name structural signatures (handles GGUFs lacking general.architecture)
        has_flux_blocks = (
            ("double_blocks." in t_joined and "single_blocks." in t_joined)
            or ("double_stream_blocks." in t_joined and "single_stream_blocks." in t_joined)
        )
        if has_flux_blocks:
            if "flux3" in fname or "flux-3" in fname:
                return "flux3-klein"
            if "flux2" in fname or "flux-2" in fname:
                return "flux2-klein" if "klein" in fname else "flux2-dev"
            return "flux1-schnell" if "schnell" in fname else "flux1-dev"

        if "cap_embedder." in t_joined or "noise_refiner." in t_joined or "context_refiner." in t_joined:
            return "z-image-turbo" if "turbo" in fname else "z-image"

        if "transformer_blocks." in t_joined and ("img_mod." in t_joined or "txt_mod." in t_joined or "img_mlp." in t_joined):
            if "2.1" in fname or "2_1" in fname:
                return "qwen-image-2.1-turbo" if "turbo" in fname else "qwen-image-2.1"
            return "qwen-image"

        if "joint_blocks." in t_joined and ("context_embedder." in t_joined or "x_embedder." in t_joined):
            return "sd3.5"

        if "model.diffusion_model.input_blocks." in t_joined or "conditioner.embedders." in t_joined:
            if "conditioner.embedders." in t_joined or "label_emb." in t_joined or "sdxl" in fname or "sd_xl" in fname:
                return "sdxl"
            return "sd1.5"

        # 3. Filename fallback for diffusion GGUFs/safetensors
        if "z_image" in fname or "z-image" in fname:
            return "z-image-turbo" if "turbo" in fname else "z-image"
        if "qwen_image" in fname or "qwen-image" in fname:
            if "2.1" in fname or "2_1" in fname:
                return "qwen-image-2.1-turbo" if "turbo" in fname else "qwen-image-2.1"
            return "qwen-image"
        if "flux3" in fname or "flux-3" in fname:
            return "flux3-klein"
        if "flux2" in fname or "flux-2" in fname:
            return "flux2-klein" if "klein" in fname else "flux2-dev"
        if "flux" in fname and ("dev" in fname or "schnell" in fname or "kontext" in fname or "klein" in fname):
            return "flux1-schnell" if "schnell" in fname else "flux1-dev"
        if "sd3.5" in fname or "sd3_" in fname or "sd3-" in fname:
            return "sd3.5"
        if "sdxl" in fname or "sd_xl" in fname:
            return "sdxl"

        return ""

    @property
    def image_profile(self) -> str:
        return self._infer_image_profile_from_tensors_and_name()

    @property
    def is_image_model(self) -> bool:
        return bool(self.image_profile)

    @property
    def modality(self) -> str:
        return "image" if self.is_image_model else "llm"

    @property
    def architecture(self) -> str:
        if "general.architecture" in self.metadata and self.metadata["general.architecture"] not in ("", "safetensors"):
            return str(self.metadata["general.architecture"])
        prof = self.image_profile
        if prof:
            return prof
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
            31: "MOSTLY_IQ1_M",
            32: "MOSTLY_BF16",
        }
        if ftype in types_map:
            return types_map[ftype]
        if ftype is not None:
            return f"Type {ftype}"
        # Fallback for diffusion GGUFs without general.file_type metadata: inspect quantized tensor types
        if self.tensor_type_counts:
            quant_counts = {k: v for k, v in self.tensor_type_counts.items() if k not in (0, 1, 30)}
            pick_map = quant_counts if quant_counts else self.tensor_type_counts
            dom_type = max(pick_map.items(), key=lambda kv: kv[1])[0]
            if dom_type in GGML_TYPE_NAMES:
                return f"MOSTLY_{GGML_TYPE_NAMES[dom_type]}"
        return "Unknown"

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
        if self.is_image_model:
            return 0
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
        # Infer block count from diffusion tensor names if present
        if self.tensor_names and self.is_image_model:
            block_indices = set()
            for tn in self.tensor_names:
                m = re.search(r"(?:double_blocks|single_blocks|joint_blocks|transformer_blocks|layers)\.(\d+)\.", tn)
                if m:
                    block_indices.add(int(m.group(1)))
            if block_indices:
                return max(block_indices) + 1
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
            "Modality": self.modality,
            "Image Profile": self.image_profile,
        }
