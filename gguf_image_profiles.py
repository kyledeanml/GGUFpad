"""
GGUFpanel Diffusion / Text-to-Image Model Profiles, Hardware Recommendations,
and Companion Component Auto-Discovery (VAE, LLM Encoder, CLIP-L, CLIP-G, T5-XXL).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ImageModelProfile:
    """Configuration profile for a stable-diffusion.cpp compatible image architecture."""
    profile_id: str
    display_name: str
    model_flag: str  # "--diffusion-model" or "-m"
    required_components: Tuple[str, ...]  # subset of ("vae", "llm", "clip_l", "clip_g", "t5xxl")
    optional_components: Tuple[str, ...] = ()
    default_steps: int = 20
    default_cfg: float = 1.0
    default_sampler: str = "euler"
    default_width: int = 1024
    default_height: int = 1024
    default_clip_on_cpu: bool = False
    default_offload_to_cpu: bool = False
    default_vae_tiling: bool = False
    default_flash_attn: bool = True
    default_sigmas: str = ""
    default_flow_shift: Optional[float] = None
    supports_negative_prompt: bool = False
    supports_reference_edit: bool = False
    vram_note: str = ""


IMAGE_PROFILES: Dict[str, ImageModelProfile] = {
    "z-image-turbo": ImageModelProfile(
        profile_id="z-image-turbo",
        display_name="Z-Image-Turbo (6B Distilled)",
        model_flag="--diffusion-model",
        required_components=("vae", "llm"),
        default_steps=8,
        default_cfg=1.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_clip_on_cpu=False,
        default_offload_to_cpu=False,
        default_vae_tiling=False,
        default_flash_attn=True,
        supports_negative_prompt=False,
        vram_note="Fits fully in 12GB VRAM (~6.9 GB total with Q5_0 + Qwen3-4B Q4_K_M + ae.safetensors).",
    ),
    "z-image": ImageModelProfile(
        profile_id="z-image",
        display_name="Z-Image Base (6B)",
        model_flag="--diffusion-model",
        required_components=("vae", "llm"),
        default_steps=25,
        default_cfg=5.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_clip_on_cpu=False,
        default_offload_to_cpu=False,
        default_flash_attn=True,
        supports_negative_prompt=True,
        vram_note="Fits in 12GB VRAM with Q4_K_M / Q5_0 + Qwen3-4B Q4_K_M.",
    ),
    "flux1-dev": ImageModelProfile(
        profile_id="flux1-dev",
        display_name="FLUX.1-dev (12B DiT)",
        model_flag="--diffusion-model",
        required_components=("vae", "clip_l", "t5xxl"),
        default_steps=20,
        default_cfg=1.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_clip_on_cpu=True,
        default_offload_to_cpu=False,
        default_vae_tiling=False,
        default_flash_attn=True,
        supports_negative_prompt=False,
        vram_note="Q4_K_S (6.3 GB) fits in 12GB VRAM; T5-XXL offloads cleanly to 32GB RAM via --clip-on-cpu.",
    ),
    "flux1-schnell": ImageModelProfile(
        profile_id="flux1-schnell",
        display_name="FLUX.1-schnell (12B 4-Step)",
        model_flag="--diffusion-model",
        required_components=("vae", "clip_l", "t5xxl"),
        default_steps=4,
        default_cfg=1.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_clip_on_cpu=True,
        default_offload_to_cpu=False,
        default_flash_attn=True,
        supports_negative_prompt=False,
        vram_note="Fast 4-step FLUX; Q4_K (6.4 GB) fits in 12GB VRAM with --clip-on-cpu.",
    ),
    "qwen-image-2.1": ImageModelProfile(
        profile_id="qwen-image-2.1",
        display_name="Qwen-Image-2.1 (Typography & Editing)",
        model_flag="--diffusion-model",
        required_components=("vae", "llm"),
        optional_components=("llm_vision",),
        default_steps=24,
        default_cfg=6.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_clip_on_cpu=False,
        default_offload_to_cpu=True,
        default_vae_tiling=True,
        default_flash_attn=True,
        supports_negative_prompt=True,
        supports_reference_edit=True,
        vram_note="Q4_K (3.9 GB) + Qwen3VL-8B Q4_K_M (4.7 GB) runs smoothly with --offload-to-cpu.",
    ),
    "qwen-image-2.1-turbo": ImageModelProfile(
        profile_id="qwen-image-2.1-turbo",
        display_name="Qwen-Image-2.1 Turbo (8-Step)",
        model_flag="--diffusion-model",
        required_components=("vae", "llm"),
        optional_components=("llm_vision",),
        default_steps=8,
        default_cfg=1.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_offload_to_cpu=True,
        default_vae_tiling=True,
        default_flash_attn=True,
        default_sigmas="qwen-image-2.1-turbo",
        supports_negative_prompt=False,
        supports_reference_edit=True,
        vram_note="8-step Turbo schedule (--sigmas qwen-image-2.1-turbo) with --offload-to-cpu.",
    ),
    "qwen-image": ImageModelProfile(
        profile_id="qwen-image",
        display_name="Qwen-Image (v1)",
        model_flag="--diffusion-model",
        required_components=("vae", "llm"),
        default_steps=24,
        default_cfg=2.5,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_offload_to_cpu=True,
        default_vae_tiling=True,
        default_flash_attn=True,
        default_flow_shift=3.0,
        supports_negative_prompt=True,
        vram_note="Runs on 12GB VRAM with --offload-to-cpu and --flow-shift 3.",
    ),
    "flux2-klein": ImageModelProfile(
        profile_id="flux2-klein",
        display_name="FLUX.2-klein (4B/9B)",
        model_flag="--diffusion-model",
        required_components=("vae", "llm"),
        default_steps=4,
        default_cfg=1.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_offload_to_cpu=True,
        default_flash_attn=True,
        supports_reference_edit=True,
        vram_note="Uses Qwen3-4B/8B + flux2 ae.safetensors.",
    ),
    "flux3-klein": ImageModelProfile(
        profile_id="flux3-klein",
        display_name="FLUX.3 Image Klein",
        model_flag="--diffusion-model",
        required_components=("vae", "llm"),
        default_steps=16,
        default_cfg=1.0,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_offload_to_cpu=True,
        default_flash_attn=True,
        supports_reference_edit=True,
        vram_note="Uses Qwen3-VL-4B GGUF + flux2 ae.safetensors.",
    ),
    "sd3.5": ImageModelProfile(
        profile_id="sd3.5",
        display_name="Stable Diffusion 3.5 (Large/Medium)",
        model_flag="-m",
        required_components=("clip_l", "clip_g", "t5xxl"),
        default_steps=28,
        default_cfg=4.5,
        default_sampler="euler",
        default_width=1024,
        default_height=1024,
        default_clip_on_cpu=True,
        default_flash_attn=True,
        supports_negative_prompt=True,
        vram_note="SD3.5 Large Q4_0/Q5_0 (4.4-5.4 GB) fits comfortably in 12GB VRAM with --clip-on-cpu.",
    ),
    "sdxl": ImageModelProfile(
        profile_id="sdxl",
        display_name="Stable Diffusion XL (SDXL)",
        model_flag="-m",
        required_components=(),
        optional_components=("vae",),
        default_steps=25,
        default_cfg=7.0,
        default_sampler="euler_a",
        default_width=1024,
        default_height=1024,
        default_vae_tiling=True,
        default_flash_attn=True,
        supports_negative_prompt=True,
        vram_note="Single-file checkpoint; fits comfortably inside 12GB VRAM.",
    ),
    "sd1.5": ImageModelProfile(
        profile_id="sd1.5",
        display_name="Stable Diffusion 1.5 / 2.x",
        model_flag="-m",
        required_components=(),
        optional_components=("vae",),
        default_steps=20,
        default_cfg=7.0,
        default_sampler="euler_a",
        default_width=512,
        default_height=512,
        default_flash_attn=True,
        supports_negative_prompt=True,
        vram_note="Lightweight (~2 GB VRAM).",
    ),
}


# Top 3 recommended GGUF text-to-image models tailored for:
# NVIDIA RTX 3060 (12GB VRAM) + 32GB System RAM + Ryzen 7 3700X
RECOMMENDED_T2I_MODELS: List[Dict[str, object]] = [
    {
        "rank": 1,
        "profile_id": "z-image-turbo",
        "name": "Z-Image-Turbo (Q5_0 GGUF)",
        "tagline": "Fastest 8-Step Photorealism — Fits Entirely in 12GB VRAM",
        "summary": (
            "6B distilled Diffusion Transformer that generates crisp 1024x1024 images in just 8 steps. "
            "With Q5_0 (4.23 GB) + Qwen3-4B-Instruct-Q4_K_M (2.33 GB) + ae.safetensors (0.31 GB), "
            "the entire pipeline (~6.9 GB) stays inside your RTX 3060 12GB VRAM with zero CPU offload penalty."
        ),
        "vram_estimate": "~7.5–9.0 GB VRAM (All-GPU)",
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "files": [
            {
                "role": "diffusion",
                "subdir": "diffusion_models",
                "filename": "z_image_turbo-Q5_0.gguf",
                "size_label": "4.23 GB",
                "url": "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q5_0.gguf",
            },
            {
                "role": "vae",
                "subdir": "vae",
                "filename": "ae.safetensors",
                "size_label": "0.31 GB",
                "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            },
            {
                "role": "llm",
                "subdir": "text_encoders",
                "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                "size_label": "2.33 GB",
                "url": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            },
        ],
    },
    {
        "rank": 2,
        "profile_id": "flux1-dev",
        "name": "FLUX.1-dev (Q4_K_S GGUF)",
        "tagline": "Flagship 12B Quality & Community Ecosystem",
        "summary": (
            "Black Forest Labs' 12B flagship DiT for photorealism, anatomy, lighting, and LoRA support. "
            "At Q4_K_S (6.34 GB), the diffusion model sits inside your 12GB VRAM while --clip-on-cpu "
            "places CLIP-L + T5-XXL (Q4_K_M, 2.70 GB) smoothly in your 32GB system RAM."
        ),
        "vram_estimate": "~7.5–9.5 GB VRAM + ~3.5 GB RAM (--clip-on-cpu)",
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "files": [
            {
                "role": "diffusion",
                "subdir": "diffusion_models",
                "filename": "flux1-dev-Q4_K_S.gguf",
                "size_label": "6.34 GB",
                "url": "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf",
            },
            {
                "role": "vae",
                "subdir": "vae",
                "filename": "ae.safetensors",
                "size_label": "0.31 GB",
                "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            },
            {
                "role": "clip_l",
                "subdir": "text_encoders",
                "filename": "clip_l.safetensors",
                "size_label": "0.23 GB",
                "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
            },
            {
                "role": "t5xxl",
                "subdir": "text_encoders",
                "filename": "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                "size_label": "2.70 GB",
                "url": "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q4_K_M.gguf",
            },
        ],
    },
    {
        "rank": 3,
        "profile_id": "qwen-image-2.1",
        "name": "Qwen-Image-2.1 (Q4_K GGUF)",
        "tagline": "Best-in-Class Typography, Complex Layouts & RGBA Alpha",
        "summary": (
            "Alibaba's modern multimodal DiT powered by Qwen3-VL-8B. Excels at dense text rendering, "
            "multi-subject prompt adherence, and transparent RGBA outputs. Q4_K (3.91 GB) + Qwen3VL-8B "
            "(4.68 GB) runs reliably on RTX 3060 12GB + 32GB RAM with --offload-to-cpu."
        ),
        "vram_estimate": "~6.5–10.5 GB VRAM + RAM (--offload-to-cpu)",
        "steps": 24,
        "cfg": 6.0,
        "sampler": "euler",
        "files": [
            {
                "role": "diffusion",
                "subdir": "diffusion_models",
                "filename": "qwen_image_2.1-Q4_K.gguf",
                "size_label": "3.91 GB",
                "url": "https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q4_K.gguf",
            },
            {
                "role": "vae",
                "subdir": "vae",
                "filename": "qwen_image_2.1_vae_bf16.safetensors",
                "size_label": "0.63 GB",
                "url": "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
            },
            {
                "role": "llm",
                "subdir": "text_encoders",
                "filename": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                "size_label": "4.68 GB",
                "url": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            },
        ],
    },
]


COMPONENT_LABELS = {
    "vae": "VAE (--vae)",
    "llm": "LLM Encoder (--llm)",
    "llm_vision": "Vision Proj (--llm_vision)",
    "clip_l": "CLIP-L (--clip_l)",
    "clip_g": "CLIP-G (--clip_g)",
    "t5xxl": "T5-XXL (--t5xxl)",
}


def auto_discover_components(
    model_path: str,
    profile_id: str,
    extra_dirs: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Scans the model's directory tree and extra model directories for companion
    VAE and text encoder files matching the active diffusion profile.
    """
    found: Dict[str, str] = {}
    search_roots = []
    if model_path:
        model_dir = os.path.dirname(os.path.abspath(model_path))
        search_roots.append(model_dir)
        parent_dir = os.path.dirname(model_dir)
        if parent_dir and parent_dir != model_dir:
            search_roots.append(parent_dir)
    if extra_dirs:
        for d in extra_dirs:
            if d and os.path.isdir(d) and d not in search_roots:
                search_roots.append(d)

    candidate_files: List[Tuple[str, str]] = []
    seen = set()
    for root_dir in search_roots:
        if not os.path.isdir(root_dir):
            continue
        for cur_root, _dirs, files in os.walk(root_dir):
            # Avoid descending endlessly into unrelated deep trees
            rel = os.path.relpath(cur_root, root_dir)
            if rel != "." and rel.count(os.sep) > 2:
                continue
            for fn in sorted(files):
                low = fn.lower()
                if not low.endswith((".gguf", ".safetensors", ".sft", ".pt", ".ckpt")):
                    continue
                full = os.path.join(cur_root, fn)
                if full not in seen:
                    seen.add(full)
                    candidate_files.append((low, full))

    def pick(match_fn) -> str:
        for low, full in candidate_files:
            if match_fn(low):
                return full
        return ""

    # VAE matching by profile family
    if profile_id.startswith("qwen-image-2.1"):
        vae = pick(lambda f: "qwen_image_2.1" in f and "vae" in f)
    elif profile_id.startswith("qwen-image"):
        vae = pick(lambda f: "qwen_image" in f and "vae" in f and "2.1" not in f)
    elif profile_id.startswith("flux2") or profile_id.startswith("flux3"):
        vae = pick(lambda f: ("flux2" in f and "ae" in f) or f in ("ae.safetensors", "ae.sft"))
    elif profile_id == "sdxl":
        vae = pick(lambda f: "sdxl" in f and "vae" in f)
    else:
        vae = pick(lambda f: f in ("ae.safetensors", "ae.sft") or ("flux" in f and "ae" in f and "flux2" not in f))
        if not vae:
            vae = pick(lambda f: "vae" in f and "qwen" not in f and "wan" not in f)
    if vae:
        found["vae"] = vae

    # LLM text encoder matching
    if profile_id.startswith("z-image") or profile_id == "flux2-klein":
        llm = pick(lambda f: ("qwen3" in f or "qwen_3" in f) and "4b" in f and "vl" not in f and "mmproj" not in f)
        if not llm:
            llm = pick(lambda f: "qwen" in f and ("instruct" in f or "text" in f or "4b" in f) and "mmproj" not in f and "image" not in f)
    elif profile_id == "flux3-klein":
        llm = pick(lambda f: "qwen3" in f and "vl" in f and "4b" in f and "mmproj" not in f)
    elif profile_id.startswith("qwen-image-2.1"):
        llm = pick(lambda f: ("qwen3vl" in f or "qwen3-vl" in f or "qwen_3_vl" in f) and "8b" in f and "mmproj" not in f)
        llm_vis = pick(lambda f: "mmproj" in f and "qwen3" in f and "8b" in f)
        if llm_vis:
            found["llm_vision"] = llm_vis
    elif profile_id == "qwen-image":
        llm = pick(lambda f: "qwen2.5-vl" in f or "qwen_2.5_vl" in f)
    else:
        llm = ""
    if llm:
        found["llm"] = llm

    # CLIP-L / CLIP-G / T5-XXL matching
    clip_l = pick(lambda f: "clip_l" in f or "clip-l" in f)
    if clip_l:
        found["clip_l"] = clip_l

    clip_g = pick(lambda f: "clip_g" in f or "clip-g" in f)
    if clip_g:
        found["clip_g"] = clip_g

    t5 = pick(lambda f: ("t5xxl" in f or "t5-v1_1-xxl" in f or "t5_xxl" in f) and f.endswith((".gguf", ".safetensors")))
    if t5:
        found["t5xxl"] = t5

    return found
