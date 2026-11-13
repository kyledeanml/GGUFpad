"""
GGUFpanel LLM & Diffusion Image Inference Engine
Supports:
1. Automatic llama-server launch with GPU offloading (--n-gpu-layers) & health-check polling
2. Native llama-cpp-python (if installed)
3. Direct OpenAI-compatible / local GGUF server streaming client with multi-turn history
4. Text-to-image / image-editing generation via stable-diffusion.cpp (sd-cli) with CUDA
"""

import os
import re
import sys
import subprocess
import threading
import time
import json
import urllib.request
import urllib.error
from typing import Callable, Optional, Dict, Any, List

from gguf_image_profiles import IMAGE_PROFILES

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
    """Manages GGUF LLM and Diffusion (stable-diffusion.cpp) processes and GPU offloading."""

    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.sd_process: Optional[subprocess.Popen] = None
        self.server_port: int = 8080
        self.is_running: bool = False
        self.is_ready: bool = False
        self.is_generating_image: bool = False
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
        sd_cli = self.find_sd_cli()
        return {
            "cuda_available": CUDA_AVAILABLE,
            "device_name": CUDA_DEVICE_NAME,
            "vram_gb": CUDA_VRAM_GB,
            "has_llama_cpp_module": self.has_llama_cpp,
            "has_sd_cli": bool(sd_cli),
            "sd_cli_path": sd_cli,
        }

    def _find_binary(self, names: List[str]) -> Optional[str]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        paths_to_check = [
            os.path.join(base_dir, "bin"),
            os.path.expanduser("~/.local/bin"),
            os.path.expanduser("~/.local/share/stable-diffusion.cpp/build/bin"),
            "/usr/local/bin",
            "/usr/bin",
        ] + os.environ.get("PATH", "").split(os.path.pathsep)

        for p in paths_to_check:
            if not p:
                continue
            for binary in names:
                cand = os.path.join(p, binary)
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    return cand
        return None

    def find_llama_server(self) -> Optional[str]:
        return self._find_binary(["llama-server"])

    def find_sd_cli(self) -> Optional[str]:
        return self._find_binary(["sd-cli", "sd"])

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

    def stop_image_generation(self):
        if self.sd_process and self.sd_process.poll() is None:
            self.sd_process.terminate()
            try:
                self.sd_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.sd_process.kill()
        self.sd_process = None
        self.is_generating_image = False

    def stop_all(self):
        self.stop_server()
        self.stop_image_generation()

    def build_sd_command(
        self,
        sd_cli_path: str,
        model_path: str,
        profile_id: str,
        prompt: str,
        output_path: str,
        components: Optional[Dict[str, str]] = None,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 1.0,
        sampler: str = "euler",
        seed: int = -1,
        clip_on_cpu: bool = False,
        offload_to_cpu: bool = False,
        vae_tiling: bool = False,
        flash_attn: bool = True,
        sigmas: str = "",
        flow_shift: Optional[float] = None,
        ref_images: Optional[List[str]] = None,
        init_image: str = "",
        strength: float = 0.75,
    ) -> List[str]:
        """Constructs an sd-cli command line tailored to the diffusion model profile."""
        profile = IMAGE_PROFILES.get(profile_id)
        model_flag = profile.model_flag if profile else "--diffusion-model"
        comps = components or {}

        cmd = [sd_cli_path, model_flag, model_path]

        comp_flags = [
            ("vae", "--vae"),
            ("llm", "--llm"),
            ("llm_vision", "--llm_vision"),
            ("clip_l", "--clip_l"),
            ("clip_g", "--clip_g"),
            ("t5xxl", "--t5xxl"),
        ]
        for key, flag in comp_flags:
            val = (comps.get(key) or "").strip()
            if val:
                cmd.extend([flag, val])

        cmd.extend(["-p", prompt])
        if negative_prompt and negative_prompt.strip():
            cmd.extend(["-n", negative_prompt.strip()])

        cmd.extend([
            "-W", str(int(width)),
            "-H", str(int(height)),
            "--steps", str(int(steps)),
            "--cfg-scale", f"{float(cfg_scale):.2f}",
            "--sampling-method", sampler or "euler",
            "-s", str(int(seed)),
            "-o", output_path,
        ])

        if sigmas:
            cmd.extend(["--sigmas", sigmas])
        if flow_shift is not None:
            cmd.extend(["--flow-shift", str(flow_shift)])

        if clip_on_cpu:
            cmd.extend(["--backend", "te=cpu"])
        if offload_to_cpu:
            cmd.append("--offload-to-cpu")
        if vae_tiling:
            cmd.append("--vae-tiling")
        if flash_attn:
            cmd.extend(["--flash-attn", "on"])

        if ref_images and profile and profile.supports_reference_edit:
            for rpath in ref_images:
                if rpath and os.path.isfile(rpath):
                    cmd.extend(["-r", rpath])
        elif init_image and os.path.isfile(init_image):
            cmd.extend(["-i", init_image, "--strength", f"{float(strength):.2f}"])

        cmd.append("-v")
        return cmd

    def generate_image(
        self,
        model_path: str,
        profile_id: str,
        prompt: str,
        output_path: str,
        components: Optional[Dict[str, str]] = None,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 1.0,
        sampler: str = "euler",
        seed: int = -1,
        clip_on_cpu: bool = False,
        offload_to_cpu: bool = False,
        vae_tiling: bool = False,
        flash_attn: bool = True,
        sigmas: str = "",
        flow_shift: Optional[float] = None,
        ref_images: Optional[List[str]] = None,
        init_image: str = "",
        strength: float = 0.75,
        on_status: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str, float, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """Runs stable-diffusion.cpp (sd-cli) in a background thread and streams progress."""

        def run_sd():
            sd_cli = self.find_sd_cli()
            if not sd_cli:
                if on_error:
                    on_error(
                        "The 'sd-cli' binary (stable-diffusion.cpp) was not found in PATH or ~/.local/bin."
                    )
                return

            # Free GPU VRAM held by llama-server before running diffusion inference
            if self.is_running:
                if on_status:
                    on_status("GPU: Releasing LLM VRAM for diffusion...")
                self.stop_server()

            self.stop_image_generation()

            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            cmd = self.build_sd_command(
                sd_cli_path=sd_cli,
                model_path=model_path,
                profile_id=profile_id,
                prompt=prompt,
                output_path=output_path,
                components=components,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                seed=seed,
                clip_on_cpu=clip_on_cpu,
                offload_to_cpu=offload_to_cpu,
                vae_tiling=vae_tiling,
                flash_attn=flash_attn,
                sigmas=sigmas,
                flow_shift=flow_shift,
                ref_images=ref_images,
                init_image=init_image,
                strength=strength,
            )

            t0 = time.time()
            logs: List[str] = []
            step_re = re.compile(r"(\d+)\s*/\s*(\d+)")
            if on_status:
                on_status(f"GPU: Generating image ({profile_id})...")

            try:
                self.is_generating_image = True
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.sd_process = proc

                if proc.stdout:
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip("\r\n")
                        if line:
                            logs.append(line)
                            if on_log:
                                on_log(line)
                            m = step_re.search(line)
                            if m:
                                cur_s, tot_s = int(m.group(1)), int(m.group(2))
                                if tot_s > 0 and cur_s <= tot_s:
                                    if on_progress:
                                        on_progress(cur_s, tot_s, line)
                                    if on_status:
                                        on_status(f"GPU: Sampling step {cur_s}/{tot_s}...")

                rc = proc.wait()
                elapsed = time.time() - t0
                self.is_generating_image = False
                if self.sd_process == proc:
                    self.sd_process = None

                if rc == 0 and os.path.isfile(output_path):
                    if on_status:
                        on_status(f"GPU: Generated in {elapsed:.1f}s ({CUDA_DEVICE_NAME})")
                    if on_done:
                        on_done(output_path, elapsed, "\n".join(logs[-40:]))
                else:
                    tail = "\n".join(logs[-15:]) if logs else "No output from sd-cli."
                    if on_error:
                        on_error(f"sd-cli exited with code {rc}.\n\n{tail}")
            except Exception as e:
                self.is_generating_image = False
                if on_error:
                    on_error(f"Image generation error: {e}")

        threading.Thread(target=run_sd, daemon=True).start()
