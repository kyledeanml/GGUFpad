#!/usr/bin/env python3
"""
Integrity & Unit Test Suite for GGUFpad
Validates:
1. GGUF v3 binary header + tensor table parsing & exact parameter calculation
2. Multi-format model inspection (.safetensors, .pt, .bin, .onnx)
3. GGUFEngine GPU/CUDA detection & llama-server binary lookup
4. SVG icon & watermark loading integrity
5. UI code structure (temperature slider removal & Model Files browser presence)
"""

import os
import sys
import json
import struct
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf

from gguf_parser import (
    GGUFReader,
    GGUF_MAGIC,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    SUPPORTED_MODEL_EXTENSIONS,
)
from gguf_engine import GGUFEngine


class TestGGUFpadIntegrity(unittest.TestCase):
    def test_gguf_binary_and_tensor_parsing(self):
        """Verify GGUF header parsing and exact tensor dimension parameter counting."""
        with tempfile.NamedTemporaryFile(suffix="-8B.gguf", delete=False) as tf:
            temp_path = tf.name
            # Header: magic, version=3, tensor_count=2, kv_count=3
            tf.write(struct.pack("<I", GGUF_MAGIC))
            tf.write(struct.pack("<I", 3))
            tf.write(struct.pack("<Q", 2))
            tf.write(struct.pack("<Q", 3))

            def write_str(s: str):
                b = s.encode("utf-8")
                tf.write(struct.pack("<Q", len(b)) + b)

            # KV 1: general.architecture = llama
            write_str("general.architecture")
            tf.write(struct.pack("<I", GGUF_TYPE_STRING))
            write_str("llama")

            # KV 2: llama.block_count = 32
            write_str("llama.block_count")
            tf.write(struct.pack("<I", GGUF_TYPE_UINT32))
            tf.write(struct.pack("<I", 32))

            # KV 3: general.file_type = 17 (MOSTLY_Q5_K_M)
            write_str("general.file_type")
            tf.write(struct.pack("<I", GGUF_TYPE_UINT32))
            tf.write(struct.pack("<I", 17))

            # Tensor 1: 1000 x 2000 = 2,000,000 params
            write_str("blk.0.attn_q.weight")
            tf.write(struct.pack("<I", 2))
            tf.write(struct.pack("<2Q", 1000, 2000))
            tf.write(struct.pack("<IQ", 0, 0))

            # Tensor 2: 500 x 1000 = 500,000 params
            write_str("blk.0.attn_k.weight")
            tf.write(struct.pack("<I", 2))
            tf.write(struct.pack("<2Q", 500, 1000))
            tf.write(struct.pack("<IQ", 0, 0))

        try:
            reader = GGUFReader(temp_path)
            summary = reader.get_summary()
            self.assertTrue(reader.is_gguf)
            self.assertEqual(summary["Architecture"], "llama")
            self.assertEqual(summary["Quantization"], "MOSTLY_Q5_K_M")
            self.assertEqual(summary["Layers (Blocks)"], 32)
            self.assertEqual(summary["Tensors Count"], 2)
            self.assertEqual(summary["Parameters"], "2.5M")
        finally:
            os.remove(temp_path)

    def test_safetensors_and_other_formats(self):
        """Verify multi-format model inspection (.safetensors, .pt, .onnx)."""
        self.assertIn(".gguf", SUPPORTED_MODEL_EXTENSIONS)
        self.assertIn(".safetensors", SUPPORTED_MODEL_EXTENSIONS)
        self.assertIn(".pt", SUPPORTED_MODEL_EXTENSIONS)
        self.assertIn(".bin", SUPPORTED_MODEL_EXTENSIONS)
        self.assertIn(".onnx", SUPPORTED_MODEL_EXTENSIONS)

        header_dict = {
            "__metadata__": {"modelspec.architecture": "flux"},
            "model.diffusion_model.weight": {
                "dtype": "BF16",
                "shape": [1000, 4000],
                "data_offsets": [0, 8000000],
            },
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tf:
            st_path = tf.name
            tf.write(struct.pack("<Q", len(header_bytes)))
            tf.write(header_bytes)

        try:
            reader = GGUFReader(st_path)
            summary = reader.get_summary()
            self.assertFalse(reader.is_gguf)
            self.assertEqual(summary["Architecture"], "flux")
            self.assertIn("BF16", summary["Quantization"])
            self.assertEqual(summary["Parameters"], "4.0M")
            self.assertEqual(summary["Tensors Count"], 1)
        finally:
            os.remove(st_path)

    def test_engine_and_gpu_detection(self):
        """Verify GGUFEngine initializes and detects CUDA/llama-server."""
        engine = GGUFEngine()
        info = engine.get_gpu_info()
        self.assertIn("cuda_available", info)
        self.assertIn("device_name", info)
        server_bin = engine.find_llama_server()
        self.assertIsNotNone(server_bin)
        self.assertTrue(os.path.exists(server_bin))

    def test_svg_and_watermark_integrity(self):
        """Verify ggufpad.svg loads cleanly via GdkPixbuf and contains no GeForce branding."""
        svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufpad.svg")
        self.assertTrue(os.path.isfile(svg_path))
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        self.assertNotIn("GEFORCE", svg_content.upper())
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(svg_path, 360, 360, True)
        self.assertIsNotNone(pixbuf)
        self.assertEqual(pixbuf.get_width(), 360)

    def test_ui_temperature_removed_and_model_browser_present(self):
        """Verify temperature slider is removed and model file browser controls exist."""
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufpad.py")
        with open(ui_path, "r", encoding="utf-8") as f:
            ui_code = f.read()
        self.assertNotIn("self.temp_scale", ui_code)
        self.assertNotIn("self.temp_label", ui_code)
        self.assertIn("MODEL FILES BROWSER", ui_code)
        self.assertIn("on_browse_models_dialog", ui_code)
        self.assertIn("refresh_model_browser", ui_code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
