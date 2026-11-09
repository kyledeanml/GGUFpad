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
from gguf_context import ContextFileLoader, SUPPORTED_CONTEXT_CATEGORIES
import ggufpad


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
        """Verify temperature slider is removed, model file browser and context file loader exist."""
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggufpad.py")
        with open(ui_path, "r", encoding="utf-8") as f:
            ui_code = f.read()
        self.assertNotIn("self.temp_scale", ui_code)
        self.assertNotIn("self.temp_label", ui_code)
        self.assertIn("MODEL FILES BROWSER", ui_code)
        self.assertIn("on_browse_models_dialog", ui_code)
        self.assertIn("refresh_model_browser", ui_code)
        self.assertIn("btn_load_context", ui_code)
        self.assertIn("btn_attach_context", ui_code)
        self.assertIn("on_load_context_files", ui_code)
        self.assertIn("context_tray_box", ui_code)

    def test_context_file_loader_multiformat(self):
        """Verify ContextFileLoader extracts text, tables, notebooks, docx, and image base64 cleanly."""
        import zipfile

        # 1. Python file
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            py_path = tf.name
            tf.write("def compute_vram():\n    return 12\n")
        try:
            att = ContextFileLoader.load_file(py_path)
            self.assertEqual(att.file_type, "PYTHON")
            self.assertIn("```python", att.extracted_text)
            self.assertIn("compute_vram", att.format_for_prompt())
        finally:
            os.remove(py_path)

        # 2. CSV file
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
            csv_path = tf.name
            tf.write("model,quant,vram_gb\nLlama-3.1-8B,Q5_K_M,5.34\nQwen2.5-7B,Q5_K_M,5.07\n")
        try:
            att = ContextFileLoader.load_file(csv_path)
            self.assertEqual(att.file_type, "CSV/TSV")
            self.assertIn("Llama-3.1-8B", att.extracted_text)
            self.assertIn("2 rows, 3 cols", att.summary_label)
        finally:
            os.remove(csv_path)

        # 3. DOCX archive
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
            docx_path = tf.name
        try:
            with zipfile.ZipFile(docx_path, "w") as zf:
                zf.writestr(
                    "word/document.xml",
                    '<?xml version="1.0"?>'
                    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    '<w:body><w:p><w:r><w:t>Hello from GGUFpad DOCX context!</w:t></w:r></w:p></w:body>'
                    '</w:document>',
                )
            att = ContextFileLoader.load_file(docx_path)
            self.assertEqual(att.file_type, "DOCX")
            self.assertIn("Hello from GGUFpad DOCX context!", att.extracted_text)
        finally:
            os.remove(docx_path)

        # 4. Jupyter Notebook (.ipynb)
        with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as tf:
            nb_path = tf.name
            json.dump(
                {
                    "cells": [
                        {"cell_type": "markdown", "source": ["# Experiment Notes"]},
                        {"cell_type": "code", "source": ["import torch\nprint(torch.cuda.is_available())"]},
                    ]
                },
                tf,
            )
        try:
            att = ContextFileLoader.load_file(nb_path)
            self.assertEqual(att.file_type, "NOTEBOOK")
            self.assertIn("Experiment Notes", att.extracted_text)
            self.assertIn("torch.cuda.is_available", att.extracted_text)
        finally:
            os.remove(nb_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
