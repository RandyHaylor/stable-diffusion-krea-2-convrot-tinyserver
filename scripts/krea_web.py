#!/usr/bin/env python3
"""Authenticated Krea web UI with a controllable single-worker queue."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, Response
from io import BytesIO
from PIL import Image
import uvicorn

from image_metadata import cached_civitai_hash, embed_generation_metadata
from krea2_edit_request import krea2_edit_native_args_fields, krea2_edit_payload_fields


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
INDEX_FILE = ROOT / "web" / "index.html"
SERVER_LOG_FILE = ROOT / ".runtime" / "krea-server.log"
LORA_DIR = ROOT / "models" / "loras"
OUTPUT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_SOURCE_IMAGE_BYTES = 128 * 1024 * 1024

MAX_SERVER_LOG_LINES = 400
SERVER_LOG_TAIL_BYTES = 262_144
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[A-Za-z@-_]|\[K")


def read_last_log_lines(log_path: Path, max_lines: int) -> list[str]:
    """Return the newest log lines, treating carriage-return progress updates as lines.

    The runtime redraws its progress bar in place with carriage returns and ANSI
    erase codes, so a whole sampling run occupies one physical line. Splitting on
    both separators keeps the most recent progress step visible as the last line.
    """
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - SERVER_LOG_TAIL_BYTES))
            tail_text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []

    lines = [ANSI_ESCAPE_PATTERN.sub("", segment).rstrip()
             for segment in re.split(r"[\r\n]", tail_text)]
    return [line for line in lines if line.strip()][-max_lines:]


def output_image_paths() -> list[Path]:
    if not OUTPUT_DIR.is_dir():
        return []
    return sorted(
        (path for path in OUTPUT_DIR.iterdir()
         if path.is_file() and path.suffix.lower() in OUTPUT_IMAGE_SUFFIXES),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def save_source_image(image_bytes: bytes, requested_name: str) -> dict:
    if not image_bytes:
        raise HTTPException(400, "Source image is empty")
    if len(image_bytes) > MAX_SOURCE_IMAGE_BYTES:
        raise HTTPException(413, "Source image exceeds the 128 MiB limit")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            image_format = image.format
    except OSError as exc:
        raise HTTPException(422, "Source must be a valid PNG or JPEG image") from exc
    if image_format not in {"PNG", "JPEG"}:
        raise HTTPException(415, "Only PNG and JPEG source images are supported")

    requested_name = Path(requested_name or "source-image").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(requested_name).stem).strip(".-")
    stem = stem[:100] or "source-image"
    requested_suffix = Path(requested_name).suffix.lower()
    suffix = requested_suffix if (image_format == "PNG" and requested_suffix == ".png") or (
        image_format == "JPEG" and requested_suffix in {".jpg", ".jpeg"}
    ) else (".png" if image_format == "PNG" else ".jpg")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{stem}{suffix}"
    if destination.exists():
        destination = OUTPUT_DIR / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"
    destination.write_bytes(image_bytes)
    return {"name": destination.name, "width": width, "height": height}


def read_generation_metadata(image: Image.Image) -> tuple[dict, str, dict]:
    """Read Krea generation settings, including legacy img2img outputs."""
    metadata: dict = {}
    prompt_json = image.info.get("prompt", "")
    if prompt_json:
        prompt_metadata = json.loads(prompt_json)
        metadata = dict(prompt_metadata.get("ui_params", prompt_metadata))
    if not metadata.get("extra_sample_args"):
        metadata["extra_sample_args"] = image.info.get("krea_extra_sample_args", "")
    if not metadata.get("prompt") and image.info.get("krea_generation"):
        metadata = json.loads(image.info["krea_generation"])

    custom_img2img: dict = {}
    if image.info.get("krea_img2img"):
        parsed_img2img = json.loads(image.info["krea_img2img"])
        if isinstance(parsed_img2img, dict):
            custom_img2img = parsed_img2img
    source_image = str(metadata.get("source_image") or custom_img2img.get("source_image") or "")
    generation_type = str(image.info.get("krea_generation_type") or
                          ("img2img" if source_image or custom_img2img else "txt2img"))
    if generation_type == "img2img":
        metadata["source_image"] = source_image
        if metadata.get("img2img_denoise") is None and custom_img2img.get("denoising_strength") is not None:
            metadata["img2img_denoise"] = custom_img2img["denoising_strength"]
        custom_img2img = {
            "source_image": source_image,
            "denoising_strength": metadata.get(
                "img2img_denoise", custom_img2img.get("denoising_strength", 0.0)
            ),
            "noise_add": metadata.get("img2img_noise_add", custom_img2img.get("noise_add", 0.0)),
        }
    metadata["generation_type"] = generation_type
    return metadata, generation_type, custom_img2img


def actual_seeds_from_backend_result(result: dict, image_count: int, fallback_seed: int) -> list[int]:
    """Return the concrete seeds reported by sd-server for generated images."""
    info = result.get("info", {})
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            info = {}
    seeds = info.get("all_seeds", []) if isinstance(info, dict) else []
    if not isinstance(seeds, list):
        seeds = []
    if not seeds and isinstance(info, dict) and info.get("seed") is not None:
        seeds = [info["seed"]]
    if not seeds and isinstance(result.get("parameters"), dict):
        seeds = [result["parameters"].get("seed", fallback_seed)]

    concrete: list[int] = []
    for index in range(image_count):
        candidate = seeds[index] if index < len(seeds) else fallback_seed
        try:
            parsed = int(candidate)
            concrete.append(int(fallback_seed) if parsed == -1 and fallback_seed != -1 else parsed)
        except (TypeError, ValueError):
            concrete.append(int(fallback_seed))
    return concrete


def load_output_image_as_base64(filename: str) -> str:
    """Read an image from the output folder, refusing paths that escape it."""
    image_path = (OUTPUT_DIR / Path(filename).name).resolve()
    if image_path.parent != OUTPUT_DIR.resolve() or not image_path.is_file():
        raise RuntimeError(f"image not found in outputs: {filename}")
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def extra_sample_args_including_scheduler_settings(p: dict) -> str:
    """The user's own extra sample args, plus the settings the chosen scheduler accepts.

    Only the beta scheduler reads alpha and beta, so sending them with any other
    scheduler would be inert noise in the request and in the saved metadata.
    """
    user_supplied_args = str(p.get("extra_sample_args", "")).strip()
    if p.get("scheduler") != "beta":
        return user_supplied_args
    beta_schedule_args = (f"alpha={float(p.get('beta_schedule_alpha', 0.5)):g}"
                          f",beta={float(p.get('beta_schedule_beta', 0.7)):g}")
    return f"{user_supplied_args},{beta_schedule_args}" if user_supplied_args else beta_schedule_args


class SessionTokenStore:
    """Holds login tokens in memory only.

    Nothing is written to a cookie or to browser storage, so reloading the page
    discards the token and the password must be entered again. Restarting the
    server also invalidates every outstanding token.
    """

    def __init__(self, token_lifetime_seconds: int = 12 * 60 * 60):
        self.token_lifetime_seconds = token_lifetime_seconds
        self.issued_tokens: dict[str, float] = {}
        self.lock = threading.Lock()

    def issue_token(self) -> str:
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.discard_expired_tokens()
            self.issued_tokens[token] = time.time()
        return token

    def is_valid(self, token: str | None) -> bool:
        if not token:
            return False
        with self.lock:
            self.discard_expired_tokens()
            return token in self.issued_tokens

    def revoke(self, token: str | None) -> None:
        with self.lock:
            self.issued_tokens.pop(token, None)

    def discard_expired_tokens(self) -> None:
        cutoff = time.time() - self.token_lifetime_seconds
        for token, issued_at in list(self.issued_tokens.items()):
            if issued_at < cutoff:
                del self.issued_tokens[token]


class QueueManager:
    def __init__(self, backend: str):
        self.backend = backend.rstrip("/")
        self.jobs: dict[str, dict] = {}
        self.pending: deque[str] = deque()
        self.current: str | None = None
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.loaded_checkpoint = Path(os.environ.get(
            "KREA_DIT", ROOT / "models" / "chkpt" / "krea2RawBaseInt8Row_v10.safetensors"
        )).name
        threading.Thread(target=self.run_queue_worker_loop, daemon=True, name="krea-web-worker").start()

    def add(self, params: dict, front: bool) -> dict:
        job_id = uuid.uuid4().hex[:10]
        job = {"id": job_id, "status": "queued", "created": time.time(),
               "params": params, "outputs": [], "error": None, "cancel_requested": False}
        with self.cv:
            self.jobs[job_id] = job
            (self.pending.appendleft if front else self.pending.append)(job_id)
            self.cv.notify()
        return job

    def snapshot(self) -> dict:
        with self.lock:
            current = self.job_state_for_client(self.jobs[self.current]) if self.current else None
            queued = [self.job_state_for_client(self.jobs[j]) for j in self.pending if j in self.jobs]
            history = sorted(
                (self.job_state_for_client(j) for j in self.jobs.values() if j["status"] not in {"queued", "running"}),
                key=lambda j: j["created"], reverse=True)[:24]
            return {"current": current, "queued": queued, "history": history,
                    "queue_count": len(queued)}

    @staticmethod
    def job_state_for_client(job: dict) -> dict:
        return {k: job.get(k) for k in
                ("id", "status", "created", "started", "completed", "params", "outputs", "error")}

    def cancel_queued(self, job_id: str) -> bool:
        with self.lock:
            if job_id not in self.pending:
                return False
            self.pending.remove(job_id)
            job = self.jobs[job_id]
            job.update(status="cancelled", completed=time.time(), error="Removed from queue")
            return True

    def clear_pending(self) -> int:
        with self.lock:
            ids = list(self.pending)
            self.pending.clear()
            for job_id in ids:
                self.jobs[job_id].update(status="cancelled", completed=time.time(), error="Queue cleared")
            return len(ids)

    def cancel_current(self) -> bool:
        with self.lock:
            if not self.current:
                return False
            self.jobs[self.current]["cancel_requested"] = True
        try:
            self.post_json_to_backend("/sdcpp/v1/cancel", {}, timeout=10)
        except Exception:
            pass
        return True

    def run_queue_worker_loop(self) -> None:
        while True:
            with self.cv:
                self.cv.wait_for(lambda: bool(self.pending))
                job_id = self.pending.popleft()
                self.current = job_id
                job = self.jobs[job_id]
                job.update(status="running", started=time.time())
            try:
                outputs = self.generate_job_outputs(job)
                with self.lock:
                    if job["cancel_requested"]:
                        job.update(status="cancelled", error="Cancelled by user")
                    else:
                        job.update(status="completed", outputs=outputs)
            except Exception as exc:
                with self.lock:
                    state = "cancelled" if job["cancel_requested"] else "failed"
                    job.update(status=state, error=str(exc))
            finally:
                with self.lock:
                    job["completed"] = time.time()
                    self.current = None

    def generate_job_outputs(self, job: dict) -> list[str]:
        p = job["params"]
        if int(p.get("seed", -1)) == -1:
            p["seed"] = secrets.randbelow(2**32)
            print(f"[web] random seed resolved to {p['seed']}", flush=True)
        checkpoint = str(p.get("checkpoint", ""))
        # The server may have been restarted outside this queue manager. Always
        # restart for an explicitly selected job so its checkpoint is definitive.
        if checkpoint:
            self.switch_checkpoint(checkpoint)
        outputs: list[str] = []
        if p.get("hires") and p.get("save_lowres"):
            outputs += self.run_single_backend_generation(p, hires=False, prefix=f"krea-web-lowres-{job['id']}")
            if job["cancel_requested"]:
                return outputs
        stage = "krea-web-highres" if p.get("hires") else "krea-web"
        outputs += self.run_single_backend_generation(p, hires=bool(p.get("hires")), prefix=f"{stage}-{job['id']}")
        return outputs

    def run_single_backend_generation(self, p: dict, hires: bool, prefix: str) -> list[str]:
        try:
            pag_layers = list(dict.fromkeys(
                int(layer.strip()) for layer in str(p.get("pag_layers", "")).split(",") if layer.strip()
            ))
        except ValueError as exc:
            raise RuntimeError("PAG layers must be comma-separated integers") from exc
        if any(layer < 0 for layer in pag_layers):
            raise RuntimeError("PAG layers must be non-negative")
        pag_start = float(p.get("pag_start", 0.0))
        pag_end = float(p.get("pag_end", 1.0))
        if not 0.0 <= pag_start <= pag_end <= 1.0:
            raise RuntimeError("PAG start/end must satisfy 0 <= start <= end <= 1")
        sample = {
            "sample_method": p["sampler"], "scheduler": p["scheduler"],
            "sample_steps": int(p["steps"]), "flow_shift": float(p["flow_shift"]),
            "extra_sample_args": extra_sample_args_including_scheduler_settings(p),
            "guidance": {"txt_cfg": float(p["cfg"])},
            "pag": {
                "enabled": bool(p.get("pag_enabled", False)),
                "scale": float(p.get("pag_scale", 1.0)),
                "layers": pag_layers,
                "start": pag_start,
                "end": pag_end,
            },
        }
        native = {
            "sample_params": sample,
            "vae_tiling_params": {"enabled": True, "tile_size_x": int(p["vae_tile_size"]),
                                  "tile_size_y": int(p["vae_tile_size"]), "target_overlap": 0.5},
            **krea2_edit_native_args_fields(p),
        }
        if hires:
            native["hires"] = {
                "enabled": True, "upscaler": p.get("hires_upscaler", "Latent"),
                "target_width": int(p["hires_width"]), "target_height": int(p["hires_height"]),
                "steps": int(p["hires_steps"]), "denoising_strength": float(p["hires_denoise"]),
            }
        prompt = p["prompt"] + " <sd_cpp_extra_args>" + json.dumps(native, separators=(",", ":")) + "</sd_cpp_extra_args>"
        payload = {
            "prompt": prompt, "negative_prompt": p.get("negative_prompt", ""),
            "width": int(p["width"]), "height": int(p["height"]), "steps": int(p["steps"]),
            "cfg_scale": float(p["cfg"]), "seed": int(p["seed"]), "batch_size": 1,
            "sampler_name": p["sampler"], "scheduler": p["scheduler"],
        }
        requested_loras = []
        for extra_lora in p.get("extra_loras", []):
            raw_name = str(extra_lora.get("filename", ""))
            lora_path = (LORA_DIR / Path(raw_name).name).resolve()
            if lora_path.parent != LORA_DIR.resolve() or not lora_path.is_file():
                raise RuntimeError(f"selected LoRA is not available in models/loras: {raw_name}")
            requested_loras.append({"path": lora_path.name,
                                    "multiplier": float(extra_lora.get("strength", 1.0)),
                                    "is_high_noise": False})
        if requested_loras:
            payload["lora"] = requested_loras
        print(f"[web] generation LoRAs: {[item['path'] for item in requested_loras]}", flush=True)
        model_name = str(p.get("checkpoint") or p.get("model_name", ""))
        if model_name:
            payload["model_name"] = model_name
        model_path = next((candidate for candidate in (ROOT / "models").rglob(model_name)
                           if candidate.is_file()), None) if model_name else None
        if model_path:
            payload["model_hash"] = cached_civitai_hash(model_path)
        payload["lora_hashes"] = [
            f"{Path(item['path']).stem}:{cached_civitai_hash(ROOT / 'models' / 'loras' / item['path'])}"
            for item in requested_loras
            if (ROOT / "models" / "loras" / item["path"]).is_file()
        ]
        endpoint = "/sdapi/v1/txt2img"
        krea2_edit_fields = krea2_edit_payload_fields(p, load_output_image_as_base64)
        source_name = "" if krea2_edit_fields else str(p.get("source_image", ""))
        payload.update(krea2_edit_fields)
        if source_name:
            payload["init_images"] = [load_output_image_as_base64(source_name)]
            payload["denoising_strength"] = float(p.get("img2img_denoise", 0.0))
            noise_add = max(0.0, min(1.0, float(p.get("img2img_noise_add", 0.0))))
            extra_args = str(payload.get("extra_sample_args", "")).strip()
            if extra_args:
                extra_args += ","
            payload["extra_sample_args"] = f"{extra_args}img2img_noise_add={noise_add:g}"
            endpoint = "/sdapi/v1/img2img"
        result = self.post_json_to_backend(endpoint, payload, timeout=7200)
        images = result.get("images", [])
        if not images:
            raise RuntimeError(f"backend returned no images: {result}")
        actual_seeds = actual_seeds_from_backend_result(result, len(images), int(p["seed"]))
        if actual_seeds:
            p["seed"] = actual_seeds[0]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        names = []
        for i, encoded in enumerate(images):
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            name = f"{prefix}-{stamp}-{i}.png"
            image_bytes = base64.b64decode(encoded)
            metadata_payload = dict(payload)
            metadata_payload["seed"] = actual_seeds[i]
            metadata_payload["ui_params"] = dict(p)
            metadata_payload["ui_params"]["seed"] = actual_seeds[i]
            (OUTPUT_DIR / name).write_bytes(embed_generation_metadata(image_bytes, metadata_payload))
            names.append(name)
        return names

    def fetch_backend_capabilities(self, timeout: int = 10) -> dict:
        with urllib.request.urlopen(self.backend + "/sdcpp/v1/capabilities", timeout=timeout) as response:
            return json.load(response)

    def available_checkpoints(self) -> list[str]:
        return sorted(path.name for path in (ROOT / "models" / "chkpt").glob("*.safetensors"))

    def switch_checkpoint(self, filename: str) -> str:
        checkpoint = (ROOT / "models" / "chkpt" / Path(filename).name).resolve()
        if checkpoint.parent != (ROOT / "models" / "chkpt").resolve() or not checkpoint.is_file():
            raise ValueError("Unknown checkpoint")
        script = ROOT / "krea-server.sh"
        print(f"Switching checkpoint for queued job: {checkpoint.name}", flush=True)
        subprocess.run([str(script), "stop"], check=True, timeout=45)
        env = os.environ.copy()
        env["KREA_LORA_DIR"] = str(LORA_DIR)
        subprocess.run([str(script), "start", str(checkpoint)], check=True, timeout=45, env=env)
        capabilities = self.wait_for_backend_after_switch(checkpoint.name)
        backend_loras = {str(item.get("path", "")) for item in capabilities.get("loras", [])}
        print(f"Checkpoint ready: {checkpoint.name}; LoRA cache: {len(backend_loras)} files", flush=True)
        self.loaded_checkpoint = checkpoint.name
        return checkpoint.name

    def wait_for_backend_after_switch(self, checkpoint_name: str, attempts: int = 120) -> dict:
        """Wait for the replacement process and initialize its model-file caches.

        sd-server creates an empty LoRA cache on every process start. Its
        capabilities handler refreshes that cache, while the SDAPI generation
        handler only reads it. Calling capabilities once before the first queued
        generation therefore prevents valid LoRAs from being rejected after a
        checkpoint restart. Checking the reported model also prevents an older
        process on the port from being mistaken for the replacement server.
        """
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                capabilities = self.fetch_backend_capabilities(timeout=10)
                live_model = Path(str(capabilities.get("model", {}).get("name", ""))).name
                if live_model != checkpoint_name:
                    raise RuntimeError(
                        f"replacement backend reported {live_model or 'no model'}; expected {checkpoint_name}"
                    )
                return capabilities
            except (OSError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f"checkpoint backend did not become ready: {checkpoint_name}: {last_error}"
                    ) from exc
                time.sleep(1)
        raise RuntimeError(f"checkpoint backend did not become ready: {checkpoint_name}")

    def post_json_to_backend(self, path: str, payload: dict, timeout: int) -> dict:
        req = urllib.request.Request(self.backend + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        for attempt in range(120):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(exc.read().decode(errors="replace")) from exc
            except urllib.error.URLError:
                if attempt == 119:
                    raise
                if attempt == 0:
                    print(f"Backend is starting; waiting to submit {path}", flush=True)
                time.sleep(1)


def build_app(user: str, password: str, backend: str) -> FastAPI:
    app = FastAPI(title="Krea Tinyserver Web")
    manager = QueueManager(backend)
    session_tokens = SessionTokenStore()

    def session_token_from_request(request: Request) -> str | None:
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            return header[len("Bearer "):]
        return request.query_params.get("token")

    def authenticate(request: Request) -> str:
        token = session_token_from_request(request)
        if not session_tokens.is_valid(token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Log in required")
        return token

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_FILE.read_text()

    @app.post("/api/login")
    async def login(request: Request):
        body = await request.json()
        submitted_user = str(body.get("username", ""))
        submitted_password = str(body.get("password", ""))
        credentials_match = (secrets.compare_digest(submitted_user.encode(), user.encode())
                             and secrets.compare_digest(submitted_password.encode(), password.encode()))
        if not credentials_match:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return {"token": session_tokens.issue_token()}

    @app.post("/api/logout")
    def logout(token: str = Depends(authenticate)):
        session_tokens.revoke(token)
        return {"status": "logged out"}

    @app.get("/api/loras")
    def available_loras(_: str = Depends(authenticate)):
        entries = []
        for lora_path in sorted(LORA_DIR.glob("*.safetensors")):
            entries.append({"filename": lora_path.name,
                            "name": lora_path.stem,
                            "size_mb": round(lora_path.stat().st_size / 1e6)})
        return {"loras": entries}

    @app.get("/api/server-log")
    def server_log(lines: int = 3, _: str = Depends(authenticate)):
        requested_lines = max(1, min(int(lines), MAX_SERVER_LOG_LINES))
        return {"lines": read_last_log_lines(SERVER_LOG_FILE, requested_lines)}

    @app.get("/api/backend-options")
    def backend_options(_: str = Depends(authenticate)):
        capabilities = manager.fetch_backend_capabilities()
        return {
            "samplers": capabilities.get("samplers", []),
            "schedulers": capabilities.get("schedulers", []),
            "upscalers": [u["name"] for u in capabilities.get("upscalers", [])],
            "loras": capabilities.get("loras", []),
            "limits": capabilities.get("limits", {}),
            "model": capabilities.get("model", {}).get("name", ""),
            "checkpoints": manager.available_checkpoints(),
        }

    @app.post("/api/checkpoint")
    async def switch_checkpoint(request: Request, _: str = Depends(authenticate)):
        body = await request.json()
        try:
            selected = manager.switch_checkpoint(str(body.get("filename", "")))
        except (RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"model": selected}

    @app.get("/api/state")
    def state(_: str = Depends(authenticate)):
        return manager.snapshot()

    @app.post("/api/jobs")
    async def add_job(request: Request, _: str = Depends(authenticate)):
        body = await request.json()
        params = body.get("params", {})
        if not str(params.get("prompt", "")).strip():
            raise HTTPException(400, "Prompt is required")
        checkpoint = str(params.get("checkpoint", ""))
        checkpoint_path = ROOT / "models" / "chkpt" / Path(checkpoint).name
        if checkpoint and checkpoint_path.is_file():
            params["model_hash"] = cached_civitai_hash(checkpoint_path)
        params["lora_hashes"] = [
            f"{Path(item['filename']).stem}:{cached_civitai_hash(ROOT / 'models' / 'loras' / item['filename'])}"
            for item in params.get("extra_loras", [])
            if (ROOT / "models" / "loras" / item.get("filename", "")).is_file()
        ]
        return manager.job_state_for_client(manager.add(params, bool(body.get("front"))))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, _: str = Depends(authenticate)):
        if not manager.cancel_queued(job_id):
            raise HTTPException(409, "Job is not queued")
        return {"status": "cancelled"}

    @app.post("/api/clear")
    def clear(_: str = Depends(authenticate)):
        return {"cleared": manager.clear_pending()}

    @app.post("/api/cancel-current")
    def cancel_current(_: str = Depends(authenticate)):
        return {"cancelling": manager.cancel_current()}

    @app.post("/api/kill-all")
    def kill_all(_: str = Depends(authenticate)):
        cleared = manager.clear_pending()
        return {"cleared": cleared, "cancelling": manager.cancel_current()}

    @app.post("/api/source-image")
    async def upload_source_image(request: Request, _: str = Depends(authenticate)):
        return save_source_image(
            await request.body(), request.headers.get("x-filename", "source-image")
        )

    @app.get("/api/output/{name}")
    def output(name: str, _: str = Depends(authenticate)):
        safe = Path(name).name
        path = OUTPUT_DIR / safe
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/output-thumb/{name}")
    def output_thumbnail(name: str, _: str = Depends(authenticate)):
        safe = Path(name).name
        path = OUTPUT_DIR / safe
        if not path.is_file():
            raise HTTPException(404)
        try:
            with Image.open(path) as image:
                image.thumbnail((240, 240))
                output = BytesIO()
                image.convert("RGB").save(output, format="JPEG", quality=82)
        except OSError as exc:
            raise HTTPException(422, "Invalid output image") from exc
        return Response(content=output.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/output-info/{name}")
    def output_info(name: str, _: str = Depends(authenticate)):
        safe = Path(name).name
        path = OUTPUT_DIR / safe
        if not path.is_file():
            raise HTTPException(404)
        try:
            with Image.open(path) as image:
                width, height = image.size
                metadata, generation_type, img2img = read_generation_metadata(image)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(422, "Invalid output image") from exc
        return {"name": safe, "width": width, "height": height,
                "generation_type": generation_type, "img2img": img2img,
                "params": metadata}

    @app.get("/api/outputs")
    def outputs(_: str = Depends(authenticate)):
        entries = []
        for path in output_image_paths():
            metadata = {}
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    metadata, generation_type, img2img = read_generation_metadata(image)
            except (OSError, ValueError, json.JSONDecodeError):
                width, height = 0, 0
                generation_type, img2img = "unknown", {}
            entries.append({"name": path.name, "created": path.stat().st_mtime,
                            "width": width, "height": height,
                            "generation_type": generation_type, "img2img": img2img,
                            "params": metadata})
        return {"outputs": entries}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--backend", default="http://127.0.0.1:1234")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()
    uvicorn.run(build_app(args.user, args.password, args.backend), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
