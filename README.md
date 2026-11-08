#![GGUFpad](ggufpad.svg)


# GGUFpad

A hardware-accelerated GGUF model inspector, parameter tuner, and local LLM studio designed for Linux Mint Cinnamon.

Styled after **PythonPad** with a sleek dark neon theme (`#0b0f19`), GGUFpad enables instant model inspection, GPU layer offloading controls, and real-time streaming chat.


## Features

- **Instant GGUF Binary Parser**: Inspects GGUF headers (v1, v2, v3) without loading full weights into memory:
  - Architecture detection (`llama`, `qwen2`, `phi3`, `mistral`, `gemma2`, etc.)
  - Quantization formats (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ3_XS`, `IQ4_NL`, `F16`, etc.)
  - Total parameter count, context window size, block/layer counts, and tensor specifications.
- **Hardware & GPU Offload Controls**:
  - Auto-detects NVIDIA GPUs (e.g. GeForce RTX 3060 12GB VRAM) via PyTorch/CUDA.
  - Interactive slider for `-ngl` (number of GPU layers offloaded).
  - Temperature, context length, and sampling configuration.
- **PythonPad Aesthetic**:
  - Dark neon `#0b0f19` theme with cyan borders and highlights.
  - Linked HeaderBar controls (`Open`, `Clear Session`, `Inspector Toggle`).
  - Real-time status bar showing active model format, GPU readiness, and token counts.
- **Preset System Roles**: Quick templates for General Assistant, Python & CUDA Engineer, 3D & Vision Specialist, and Creative Writer.

## Installation

```bash
git clone https://github.com/kyledeanml/GGUFpad.git
cd GGUFpad
chmod +x run.sh ggufpad.py

# Install desktop shortcut & icon
mkdir -p ~/.local/share/applications ~/.local/share/icons/hicolor/scalable/apps
cp ggufpad.svg ~/.local/share/icons/hicolor/scalable/apps/
cp ggufpad.desktop ~/.local/share/applications/
update-desktop-database ~/.local/share/applications/ 2>/dev/null || true
```

## Running GGUFpad

```bash
./run.sh
```

Or pass a GGUF file directly:

```bash
./run.sh /path/to/model.gguf
```

## License

MIT License
