"""
GGUFpad LLM Inference Engine & Runner
Supports:
1. llama-server / llama-cli (if available) with GPU offloading (--n-gpu-layers)
2. Native llama-cpp-python (if installed)
3. Direct OpenAI-compatible / local GGUF server streaming client
4. Built-in GGUF inspect and simulation runner
"""

import os
import sys
import subprocess
import threading
import time
import json
import urllib.request
from typing import Callable, Optional, Dict, Any, Generator

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
    CUDA_VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if CUDA_AVAILABLE else 0
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = "CPU"
    CUDA_VRAM_GB = 0


class GGUFEngine:
    """Manages GGUF model processes, streaming responses, and GPU offloading."""

    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.server_port: int = 8080
        self.is_running: bool = False
        self.loaded_model_path: Optional[str] = None
        self.has_llama_cpp: bool = False
        self._check_backends()

    def _check_backends(self):
        try:
            import llama_cpp
            self.has_llama_cpp = True
        except ImportError:
            self.has_llama_cpp = False

    def get_gpu_info(self) -> Dict[str, Any]:
        return {
            "cuda_available": CUDA_AVAILABLE,
            "device_name": CUDA_DEVICE_NAME,
            "vram_gb": CUDA_VRAM_GB,
            "has_llama_cpp_module": self.has_llama_cpp,
        }

    def start_llama_server(
        self,
        model_path: str,
        n_gpu_layers: int = 33,
        context_size: int = 4096,
        port: int = 8080,
        on_log: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Starts a local llama-server process with GPU offloading."""
        self.stop_server()
        self.loaded_model_path = model_path
        self.server_port = port

        # Check for llama-server binary
        binary = "llama-server"
        found = False
        paths_to_check = [
            os.path.expanduser("~/.local/bin"),
            "/usr/local/bin",
            "/usr/bin",
        ] + os.environ.get("PATH", "").split(os.path.pathsep)

        for p in paths_to_check:
            cand = os.path.join(p, binary)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                found = True
                binary = cand
                break

        if not found:
            if on_log:
                on_log(f"Backend Note: '{binary}' binary not found in PATH.\n")
                on_log(f"You can load & inspect GGUF metadata or connect GGUFpad to any local LLM runner (llama.cpp, Ollama, vLLM).\n")
            return False

        cmd = [
            binary,
            "-m", model_path,
            "-ngl", str(n_gpu_layers),
            "-c", str(context_size),
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.is_running = True

            def monitor():
                for line in self.current_process.stdout:
                    if on_log:
                        on_log(line)
                    if "HTTP server listening" in line or "model loaded" in line.lower():
                        if on_ready:
                            on_ready()

            threading.Thread(target=monitor, daemon=True).start()
            return True
        except Exception as e:
            if on_log:
                on_log(f"Failed to start server: {e}\n")
            return False

    def stop_server(self):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
        self.current_process = None
        self.is_running = False

    def stream_completion(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful, capable AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 512,
        server_url: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """Streams completion from local OpenAI-compatible endpoint (llama-server, Ollama, etc.)"""
        endpoint = server_url or f"http://127.0.0.1:{self.server_port}/v1/chat/completions"

        def run_request():
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    for line in resp:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data: "):
                            data_str = decoded[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content and on_token:
                                        on_token(content)
                            except json.JSONDecodeError:
                                pass
                if on_done:
                    on_done()
            except urllib.error.URLError as e:
                if on_error:
                    on_error(f"Connection error to {endpoint}: {e.reason}")
            except Exception as e:
                if on_error:
                    on_error(f"Inference error: {e}")

        threading.Thread(target=run_request, daemon=True).start()
