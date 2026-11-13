--------------------------![GGUFpanel](ggufpanel.svg)


# GGUFpanel

A hardware-accelerated GGUF model inspector, parameter tuner, and local LLM studio designed for Linux Mint Cinnamon.

Styled after **PythonPanel** with a sleek dark neon theme (`#0b0f19`), GGUFpanel enables instant model inspection, GPU layer offloading controls, and real-time streaming chat.


## Features

- **Instant GGUF Binary Parser**: Inspects GGUF headers (v1, v2, v3) without loading full weights into memory:
  - Architecture detection (`llama`, `qwen2`, `phi3`, `mistral`, `gemma2`, plus diffusion families `z-image-turbo`, `flux1-dev`, `qwen-image-2.1`, `sd3.5`, `sdxl`, etc.)
  - Quantization formats (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ3_XS`, `IQ4_NL`, `F16`, etc.)
  - Total parameter count, context window size, block/layer counts, and tensor specifications.
- **GGUF Text-to-Image Studio (`stable-diffusion.cpp` / `sd-cli`)**:
  - Auto-detects GGUF diffusion models (**Z-Image-Turbo**, **FLUX.1-dev/schnell**, **Qwen-Image-2.1**, **SD 3.5**, **SDXL**) and companion files (`--vae`, `--llm`, `--clip_l`, `--clip_g`, `--t5xxl`).
  - Built-in **⬇ T2I Models** setup & downloader tailored for RTX 3060 12GB VRAM + 32GB RAM.
  - Inline image preview, progress tracking, resolution presets, samplers, and RTX 3060 CPU/VRAM offload toggles.
- **Hardware & GPU Offload Controls**:
  - Auto-detects NVIDIA GPUs (e.g. GeForce RTX 3060 12GB VRAM) via PyTorch/CUDA.
  - Interactive slider for `-ngl` (number of GPU layers offloaded).
  - Temperature, context length, and sampling configuration.
- **PythonPanel Aesthetic**:
  - Dark neon `#0b0f19` theme with cyan borders and highlights.
  - Linked HeaderBar controls (`Open`, `Clear Session`, `Inspector Toggle`).
  - Real-time status bar showing active model format, GPU readiness, and token counts.
- **Preset System Roles**: Quick templates for General Assistant, Python & CUDA Engineer, 3D & Vision Specialist, and Creative Writer.

## Installation

```bash
git clone https://github.com/kyledeanml/GGUFpanel.git
cd GGUFpanel
./install.sh
```

The installer automatically:
- Installs the vector icon (`ggufpanel.svg`) and multi-size panel icon set (`16x16` through `512x512`) into `~/.local/share/icons/hicolor/`.
- Installs the desktop launcher (`~/.local/share/applications/ggufpanel.desktop`) with `StartupWMClass=ggufpanel`.
- Creates the CLI executable symlink at `~/.local/bin/ggufpanel`.
- Adds GGUFpanel to your Cinnamon Menu favorites.

## Running GGUFpanel

```bash
ggufpanel
```

Or pass a GGUF model directly:

```bash
ggufpanel /path/to/model.gguf
```

## License

MIT License
