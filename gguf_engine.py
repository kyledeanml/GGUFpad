"""
GGUFpanel LLM Inference Engine & Runner
Supports:
1. Automatic llama-server launch with GPU offloading (--n-gpu-layers) & health-check polling
2. Native llama-cpp-python (if installed)
3. Direct OpenAI-compatible / local GGUF server streaming client with multi-turn history
"""

import os
import sys
import subprocess
import threading
import time
import json
import urllib.request
import urllib.error
from typing import Callable, Optional, Dict, Any, List

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
    """Manages GGUF model processes, readiness polling, streaming responses, and GPU offloading."""

    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.server_port: int = 8080
        self.is_running: bool = False
        self.is_ready: bool = False
        self.loaded_model_path: Optional[str] = None
        self.loaded_ngl: int = -1
        self.has_llama_cpp: bool = False
        self._lock = threading.Lock()
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

    def find_llama_server(self) -> Optional[str]:
        binary = "llama-server"
        paths_to_check = [
            os.path.expanduser("~/.local/bin"),
            "/usr/local/bin",
            "/usr/bin",
        ] + os.environ.get("PATH", "").split(os.path.pathsep)

        for p in paths_to_check:
            if not p:
                continue
            cand = os.path.join(p, binary)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        return None

    def start_llama_server(
        self,
        model_path: str,
        n_gpu_layers: int = 33,
        context_size: int = 8192,
        port: int = 8080,
        on_log: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Starts a local llama-server process with GPU offloading."""
        with self._lock:
            if (
                self.is_running
                and self.current_process
                and self.current_process.poll() is None
                and self.loaded_model_path == model_path
                and self.loaded_ngl == n_gpu_layers
            ):
                return True

            self.stop_server()
            self.loaded_model_path = model_path
            self.loaded_ngl = n_gpu_layers
            self.server_port = port
            self.is_ready = False

            binary = self.find_llama_server()
            if not binary:
                if on_log:
                    on_log("Backend Note: 'llama-server' binary not found in PATH.\n")
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

                proc = self.current_process

                def monitor():
                    if not proc.stdout:
                        return
                    for line in proc.stdout:
                        if on_log:
                            on_log(line)
                        low = line.lower()
                        if "http server listening" in low or "server is listening" in low or "all slots are idle" in low:
                            self.is_ready = True
                            if on_ready:
                                on_ready()
                    if proc.poll() is not None and self.current_process == proc:
                        self.is_running = False
                        self.is_ready = False

                threading.Thread(target=monitor, daemon=True).start()
                return True
            except Exception as e:
                if on_log:
                    on_log(f"Failed to start server: {e}\n")
                return False

    def wait_until_ready(self, timeout: float = 45.0) -> bool:
        """Polls /health on the local llama-server until model finishes loading into GPU VRAM."""
        health_url = f"http://127.0.0.1:{self.server_port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.current_process and self.current_process.poll() is not None:
                return False
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        self.is_ready = True
                        return True
            except urllib.error.HTTPError as e:
                # 503 means llama-server is alive and currently loading model weights into VRAM
                if e.code == 503:
                    time.sleep(0.4)
                    continue
            except Exception:
                pass
            time.sleep(0.4)
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
        self.is_ready = False

    def stream_completion(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful, capable AI assistant.",
        history: Optional[List[Dict[str, Any]]] = None,
        context_block: str = "",
        image_data_urls: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model_path: Optional[str] = None,
        n_gpu_layers: int = 33,
        server_url: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """Streams completion from local OpenAI-compatible endpoint, auto-starting and waiting for readiness."""
        endpoint = server_url or f"http://127.0.0.1:{self.server_port}/v1/chat/completions"

        def run_request():
            if model_path and model_path.lower().endswith(".gguf"):
                need_start = (
                    not self.is_running
                    or not self.current_process
                    or self.current_process.poll() is not None
                    or self.loaded_model_path != model_path
                    or self.loaded_ngl != n_gpu_layers
                )
                if need_start:
                    if on_status:
                        on_status("GPU: Loading model weights into RTX 3060 VRAM...")
                    started = self.start_llama_server(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,
                        port=self.server_port,
                    )
                    if not started:
                        if on_error:
                            on_error("Unable to start local llama-server binary.")
                        return

                if not self.is_ready:
                    if on_status:
                        on_status("GPU: Loading model weights into RTX 3060 VRAM...")
                    if not self.wait_until_ready(timeout=60.0):
                        if on_error:
                            on_error("Timed out waiting for llama-server to finish loading model weights.")
                        return

                if on_status:
                    on_status(f"GPU: Active ({CUDA_DEVICE_NAME})")

            effective_user_text = prompt
            if context_block:
                effective_user_text = (
                    f"Use the following loaded context files to answer the user's request:\n\n"
                    f"{context_block}\n\n"
                    f"User Request:\n{prompt}"
                )

            messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history)

            if image_data_urls:
                content_parts: List[Dict[str, Any]] = [{"type": "text", "text": effective_user_text}]
                for url in image_data_urls:
                    content_parts.append({"type": "image_url", "image_url": {"url": url}})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": "user", "content": effective_user_text})

            payload = {
                "messages": messages,
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

            full_reply = []
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for line in resp:
                        decoded = line.decode("utf-8", errors="replace").strip()
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
                                    if content:
                                        full_reply.append(content)
                                        if on_token:
                                            on_token(content)
                            except json.JSONDecodeError:
                                pass
                if on_done:
                    on_done("".join(full_reply))
            except urllib.error.URLError as e:
                if on_error:
                    on_error(f"Connection error to {endpoint}: {e.reason}")
            except Exception as e:
                if on_error:
                    on_error(f"Inference error: {e}")

        threading.Thread(target=run_request, daemon=True).start()
