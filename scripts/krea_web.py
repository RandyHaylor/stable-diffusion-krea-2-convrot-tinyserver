#!/usr/bin/env python3
"""Authenticated Krea web UI with a controllable single-worker queue."""

from __future__ import annotations

import argparse
import base64
import json
import math
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

import wd14_tagging
from hires_staging import (
    hires_settings_vary_from_main,
    renders_hires_from_existing_source,
    select_loras_for_stage,
)
from executed_steps import scheduled_steps_for_executed_steps
from source_image_sizing import (
    DEFAULT_SOURCE_PIXEL_BUDGET_EDGE,
    DEFAULT_SOURCE_SIZE_INCREMENT,
    cover_crop_box_for_resolution,
    resolution_for_source_within_pixel_budget,
)
from tiled_diffusion import (
    largest_canvas_the_request_renders,
    tiled_diffusion_sample_args,
)
from image_metadata import cached_civitai_hash, embed_generation_metadata
from krea2_edit_request import (
    krea2_edit_native_args_fields,
    krea2_edit_payload_fields,
    krea2_edit_references,
    positive_weight_or_neutral,
)
from prompt_composition import (
    compose_hires_prompt,
    compose_prompt_with_tag_groups,
    describe_missing_prompt,
    DEFAULT_HIRES_VISION_SOURCE,
    DEFAULT_STAGE_ONE_TAG_MODE,
    NEUTRAL_TAG_PROMPT_WEIGHT,
    apply_stage_one_tags,
    hires_stage_uses_krea2_edit_references,
    hires_vision_source_needs_the_first_stage_image,
    stage_one_tags_need_the_first_stage_image,
    wrap_tag_group_in_attention_weight,
)


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


def without_non_finite_floats(value):
    """Replace NaN and Infinity with None, recursively.

    Python's json module reads NaN and Infinity happily, but responses are
    serialized with allow_nan=False. An image exported by another tool can carry
    such a value in its embedded metadata, and one of them anywhere would
    otherwise fail the entire output listing rather than that single entry.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: without_non_finite_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_non_finite_floats(item) for item in value]
    return value


def tagged_images_with_prompt_weights_for_stage(p: dict, stage: str) -> list[tuple[str, float]]:
    """Output filenames routed to one stage's prompt, each with its tag weight.

    Each image says which stages receive its tags, so references and the img2img
    source can both contribute, and either can feed one stage without the other.
    The weight is per image and not per stage, so an image feeding both stages
    pulls equally hard in each.
    """
    routing_key = f"tags_to_{stage}"
    tagged_images = [(str(reference.get("filename", "")), reference["tag_prompt_weight"])
                     for reference in krea2_edit_references(p)
                     if reference.get(routing_key)]
    source_image = str(p.get("source_image", "")).strip()
    if source_image and p.get(f"img2img_source_{routing_key}"):
        tagged_images.append((source_image, source_tag_prompt_weight(p)))
    return tagged_images


def source_tag_prompt_weight(p: dict) -> float:
    """How hard the img2img source's tags pull, 1.0 for untouched."""
    return positive_weight_or_neutral(
        p.get("img2img_source_tag_prompt_weight", NEUTRAL_TAG_PROMPT_WEIGHT),
        NEUTRAL_TAG_PROMPT_WEIGHT)


def images_giving_vision_to_stage(p: dict, stage: str) -> list[str]:
    """Output filenames whose vision tokens one stage should receive.

    Krea2 Edit references reach the vision tower through the reference preset
    instead, so they are not listed here.
    """
    source_image = str(p.get("source_image", "")).strip()
    if not source_image:
        return []
    if stage == "stage_one":
        return [source_image] if p.get("img2img_source_vision_to_stage_one") else []
    return [source_image] if hires_vision_source(p) == "img2img_source" else []


def backend_loras_for_stage(p: dict, stage: str) -> list[dict]:
    """One stage's LoRA selection, shaped for the backend and checked on disk.

    A path is resolved inside the LoRA directory and refused if it escapes it or
    is missing, so a selection that cannot be honoured fails the job rather than
    being quietly dropped.
    """
    backend_loras = []
    for extra_lora in select_loras_for_stage(p.get("extra_loras", []), stage):
        lora_path = (LORA_DIR / Path(extra_lora["path"]).name).resolve()
        if lora_path.parent != LORA_DIR.resolve() or not lora_path.is_file():
            raise RuntimeError(
                f"selected LoRA is not available in models/loras: {extra_lora['path']}")
        backend_loras.append({"path": lora_path.name,
                              "multiplier": extra_lora["multiplier"],
                              "is_high_noise": False})
    return backend_loras


def hires_vision_source(p: dict) -> str:
    """Which single input the hires stage is conditioned on besides its prompt."""
    return str(p.get("hires_vision_source", DEFAULT_HIRES_VISION_SOURCE))


def tag_groups_for_images(tagged_images: list[tuple[str, float]],
                          tag_group_cache: dict[str, str] | None = None) -> list[str]:
    """One comma-separated tag string per image, skipping any that yield nothing.

    A missing tagger model is reported once and treated as "no tags" rather than
    failing the generation, since the user asked for an image, not for tagging.

    The cache holds each image's unweighted tags, so an image routed to both
    stages is read once per job rather than once per stage; the weight is applied
    to the cached text afterwards.
    """
    tag_groups = []
    for image_name, tag_prompt_weight in tagged_images:
        if tag_group_cache is not None and image_name in tag_group_cache:
            tag_group = tag_group_cache[image_name]
        else:
            try:
                tags = wd14_tagging.tag_image_file(OUTPUT_DIR / Path(image_name).name)
            except wd14_tagging.TaggerUnavailable as exc:
                print(f"[web] WD14 tagging skipped: {exc}", flush=True)
                return []
            tag_group = wd14_tagging.format_danbooru_tags_for_prompt(tags)
            if tag_group_cache is not None:
                tag_group_cache[image_name] = tag_group
            if tag_group:
                print(f"[web] WD14 tagged {image_name}: {tag_group}", flush=True)
        weighted_tag_group = wrap_tag_group_in_attention_weight(tag_group, tag_prompt_weight)
        if weighted_tag_group:
            tag_groups.append(weighted_tag_group)
    return tag_groups


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
            "noise_multiplier": metadata.get("img2img_noise_multiplier",
                                             custom_img2img.get("noise_multiplier", 1.0)),
        }
    metadata["generation_type"] = generation_type
    return (without_non_finite_floats(metadata), generation_type,
            without_non_finite_floats(custom_img2img))


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


def fit_source_image_for_hires_stage(p: dict, source_name: str) -> tuple[str, int, int]:
    """The source encoded at its own aspect inside the chosen budget, and that size.

    The main pass resolution is not consulted: a source sent straight to the hires
    stage decides its own, so a portrait image is never squeezed into a square.
    """
    image_path = (OUTPUT_DIR / Path(source_name).name).resolve()
    if image_path.parent != OUTPUT_DIR.resolve() or not image_path.is_file():
        raise RuntimeError(f"image not found in outputs: {source_name}")
    source = Image.open(image_path).convert("RGB")
    width, height = resolution_for_source_within_pixel_budget(
        source.width, source.height,
        int(p.get("source_pixel_budget_edge", DEFAULT_SOURCE_PIXEL_BUDGET_EDGE)),
        int(p.get("source_size_increment", DEFAULT_SOURCE_SIZE_INCREMENT)))
    cropped = source.crop(cover_crop_box_for_resolution(
        source.width, source.height, width, height))
    fitted = cropped.resize((width, height), Image.LANCZOS)
    encoded = BytesIO()
    fitted.save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode("ascii"), width, height


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


def hires_reference_encode_size(p: dict) -> int:
    """Edge length the hires stage encodes Krea2 Edit references at, 0 for auto.

    Anything but auto has to run the hires stage as its own request, since one
    request carries a single ref_image_args for both stages.
    """
    try:
        return max(0, int(p.get("hires_reference_encode_size", 0) or 0))
    except (TypeError, ValueError):
        return 0


def img2img_noise_multiplier_sample_arg(p: dict,
                                        renders_hires_upscale: bool,
                                        has_img2img_source: bool) -> str:
    """The sample arg scaling the noise added to an img2img start latent.

    The denoising strength already picks the schedule's starting sigma; this
    multiplies the noise applied at that sigma, so 0 leaves the encoded source
    untouched and values above 1 push past the schedule's own starting point.
    A hires stage running as its own request is upscaling an already-denoised
    image, so it keeps that latent as it is.
    """
    if renders_hires_upscale:
        return "img2img_noise_multiplier=0"
    if not has_img2img_source:
        return ""
    multiplier = max(0.0, float(p.get("img2img_noise_multiplier", 1.0)))
    return f"img2img_noise_multiplier={multiplier:g}"


def join_sample_args(*sample_args: str) -> str:
    """Comma-join the sample arg fragments that are actually present."""
    return ",".join(fragment for fragment in sample_args if fragment)


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
        hires_enabled = bool(p.get("hires"))
        stage_one_tag_mode = str(p.get("stage_one_wd14_tags", DEFAULT_STAGE_ONE_TAG_MODE))
        # One cache per job, so an image routed to both stages is read once.
        tag_group_cache: dict[str, str] = {}
        main_tag_groups = tag_groups_for_images(
            tagged_images_with_prompt_weights_for_stage(p, "stage_one"), tag_group_cache)
        hires_image_tag_groups = (tag_groups_for_images(
            tagged_images_with_prompt_weights_for_stage(p, "hires"), tag_group_cache)
            if hires_enabled else [])
        stage_one_vision_images = images_giving_vision_to_stage(p, "stage_one")
        hires_vision_images = (images_giving_vision_to_stage(p, "hires")
                               if hires_enabled else [])

        settings_vary = hires_settings_vary_from_main(
            hires_enabled, p.get("extra_loras", []),
            str(p.get("hires_prompt", "")), str(p.get("hires_negative_prompt", "")),
            main_tag_groups=main_tag_groups,
            hires_tag_groups=hires_image_tag_groups,
            main_vision_images=stage_one_vision_images,
            hires_vision_images=hires_vision_images,
            hires_vision_source=hires_vision_source(p),
            stage_one_tag_mode=stage_one_tag_mode,
            hires_reference_encode_size=hires_reference_encode_size(p))

        wants_stage_one_vision = bool(
            hires_enabled and hires_vision_source_needs_the_first_stage_image(hires_vision_source(p)))
        wants_stage_one_tags = hires_enabled and stage_one_tags_need_the_first_stage_image(
            stage_one_tag_mode)
        # The first stage is exported for saving, tagging and vision while its
        # latent stays in the runtime, so the hires pass never restarts from a
        # decoded image and one generation covers both stages.
        pauses_between_stages = hires_enabled and (
            wants_stage_one_vision or wants_stage_one_tags or settings_vary)

        def supply_hires_input(first_stage_image_b64: str) -> dict:
            stage_one_name = f"krea-web-lowres-{job['id']}-{datetime.now():%Y%m%d-%H%M%S}-0.png"
            (OUTPUT_DIR / stage_one_name).write_bytes(base64.b64decode(first_stage_image_b64))
            # The generated image has no panel of its own, so its tags are unweighted.
            stage_one_tag_groups = (
                tag_groups_for_images([(stage_one_name, NEUTRAL_TAG_PROMPT_WEIGHT)])
                if wants_stage_one_tags else [])
            hires_prompt = apply_stage_one_tags(
                compose_hires_prompt(p["prompt"], str(p.get("hires_prompt", "")),
                                     str(p.get("hires_prompt_mode", "append")),
                                     hires_image_tag_groups),
                stage_one_tag_groups, stage_one_tag_mode)
            hires_negative_prompt = compose_hires_prompt(
                str(p.get("negative_prompt", "")), str(p.get("hires_negative_prompt", "")),
                str(p.get("hires_negative_prompt_mode", "append")), [])
            hires_vision_names = ([stage_one_name] if wants_stage_one_vision
                                  else list(hires_vision_images))
            # The hires stage carries its own LoRA selection. The runtime applies
            # these between the passes, so a LoRA ticked for one stage only is
            # honoured without the latent ever leaving memory.
            return {
                "prompt": hires_prompt,
                "negative_prompt": hires_negative_prompt,
                "vlm_images": [load_output_image_as_base64(name) for name in hires_vision_names],
                "lora": backend_loras_for_stage(p, "hires"),
            }

        stage = "krea-web-highres" if hires_enabled else "krea-web"

        return self.run_single_backend_generation(
            p,
            hires=hires_enabled,
            prefix=f"{stage}-{job['id']}",
            tag_groups=main_tag_groups,
            vlm_image_names=stage_one_vision_images,
            supply_hires_input=supply_hires_input if pauses_between_stages else None,
            lowres_prefix=(f"krea-web-lowres-{job['id']}"
                           if hires_enabled and p.get("save_lowres") else ""))

    def run_single_backend_generation(self, p: dict, hires: bool, prefix: str,
                                      tag_groups: list[str] | None = None,
                                      lowres_prefix: str = "",
                                      vlm_image_names: list[str] | None = None,
                                      supply_hires_input=None) -> list[str]:
        # The job params are what both the output panel and the saved PNG report, so
        # the tags this request actually used are recorded there before it is sent.
        p["wd14_tags"] = list(tag_groups or [])
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
        krea2_edit_fields = krea2_edit_payload_fields(p, load_output_image_as_base64)
        # The source image feeds three independent things; only its use as the
        # starting latent conflicts with edit mode, whose target starts as pure noise.
        # Sending the source straight to the hires stage is the same mechanism with
        # the first stage's sampling turned off: the source is encoded as the latent
        # the hires pass continues, so nothing is generated from noise.
        skips_first_stage_sampling = hires and renders_hires_from_existing_source(p)
        source_name = ("" if (krea2_edit_fields
                              or not (p.get("img2img_source_as_starting_latent")
                                      or skips_first_stage_sampling))
                       else str(p.get("source_image", "")))
        # A source sent to the hires stage decides the first stage's size itself, so
        # this is settled before anything that depends on the resolution.
        fitted_source_base64 = ""
        if skips_first_stage_sampling:
            fitted_source_base64, first_stage_width, first_stage_height = (
                fit_source_image_for_hires_stage(p, source_name))
            # The job params are what the queue label and the saved PNG report, so
            # the size actually rendered is written back over the unused one.
            p["width"], p["height"] = first_stage_width, first_stage_height
        else:
            first_stage_width, first_stage_height = int(p["width"]), int(p["height"])

        tiling_canvas_width, tiling_canvas_height = largest_canvas_the_request_renders(
            {**p, "hires": hires,
             "width": first_stage_width, "height": first_stage_height})
        sample = {
            "sample_method": p["sampler"], "scheduler": p["scheduler"],
            "sample_steps": int(p["steps"]), "flow_shift": float(p["flow_shift"]),
            "extra_sample_args": join_sample_args(
                extra_sample_args_including_scheduler_settings(p),
                img2img_noise_multiplier_sample_arg(p, False, bool(source_name)),
                tiled_diffusion_sample_args(p, tiling_canvas_width, tiling_canvas_height)),
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
            **krea2_edit_native_args_fields(
                p, hires_reference_encode_size(p) if hires else 0,
                source_gives_vision=bool(vlm_image_names)),
        }
        if hires:
            native["hires"] = {
                "enabled": True, "upscaler": p.get("hires_upscaler", "Latent"),
                "target_width": int(p["hires_width"]), "target_height": int(p["hires_height"]),
                "steps": scheduled_steps_for_executed_steps(
                    int(p["hires_steps"]), float(p["hires_denoise"]),
                    sample["extra_sample_args"]),
                "denoising_strength": float(p["hires_denoise"]),
                "noise_multiplier": float(p.get("hires_noise_multiplier", 1.0)),
                "return_lowres_image": bool(lowres_prefix),
                "await_stage_input": supply_hires_input is not None,
                "drop_reference_latents": not hires_stage_uses_krea2_edit_references(
                    hires_vision_source(p)),
            }
        prompt_text = compose_prompt_with_tag_groups(p["prompt"], tag_groups or [])
        negative_prompt_text = str(p.get("negative_prompt", ""))
        width, height = first_stage_width, first_stage_height
        # Only an img2img start latent makes the runtime truncate the schedule, so
        # only then does the count need scaling to execute the steps that were asked for.
        steps = (scheduled_steps_for_executed_steps(
                     int(p["steps"]), float(p.get("img2img_denoise", 0.0)),
                     sample["extra_sample_args"])
                 if source_name else int(p["steps"]))

        payload = {
            **native,
            "prompt": prompt_text, "negative_prompt": negative_prompt_text,
            "width": width, "height": height,
            "seed": int(p["seed"]), "batch_count": 1,
        }
        payload["sample_params"]["sample_steps"] = steps
        requested_loras = backend_loras_for_stage(p, "main")
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
        if krea2_edit_fields.get("extra_images"):
            payload["ref_images"] = krea2_edit_fields["extra_images"]
        if source_name:
            payload["init_image"] = (fitted_source_base64 or
                                     load_output_image_as_base64(source_name))
            payload["strength"] = (0.0 if skips_first_stage_sampling
                                   else float(p.get("img2img_denoise", 0.0)))
        # vlm_images reach the vision tower and are never encoded into reference
        # latents, so these cost nothing in the diffusion model's attention sequence.
        for vision_image_name in (vlm_image_names or []):
            payload.setdefault("vlm_images", []).append(
                load_output_image_as_base64(vision_image_name))
        result = self.run_generation_job(payload, supply_hires_input)
        images = result.get("images", [])
        if not images:
            raise RuntimeError(f"backend returned no images: {result}")
        actual_seeds = actual_seeds_from_backend_result(result, len(images), int(p["seed"]))
        if actual_seeds:
            p["seed"] = actual_seeds[0]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        names = []
        # The runtime returns the pre-upscale images first when it was asked for them,
        # so the leading entries are the low-res pass and the rest are the hires one.
        lowres_image_count = 1 if lowres_prefix and len(images) > 1 else 0
        for i, encoded in enumerate(images):
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            stage_prefix = lowres_prefix if i < lowres_image_count else prefix
            name = f"{stage_prefix}-{stamp}-{i}.png"
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

    def unload_models_from_vram(self) -> None:
        """Stop the model server so the GPU gets its memory back.

        The next queued job restarts it, since a job always switches to its own
        checkpoint first. Forgetting the loaded checkpoint is what makes that
        switch actually reload rather than assume the weights are still there.
        """
        print("Unloading models: stopping the Krea server", flush=True)
        subprocess.run([str(ROOT / "krea-server.sh"), "stop"], check=True, timeout=45)
        self.loaded_checkpoint = ""

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

    def run_generation_job(self, payload: dict, supply_hires_input=None) -> dict:
        """Run one generation, pausing between passes when the caller asks to.

        The runtime holds the first stage's latent while it is parked, so the
        hires pass continues from that latent no matter what happens here. The
        callback receives the decoded first stage and returns what the hires pass
        should be conditioned on.
        """
        submitted = self.post_json_to_backend("/sdcpp/v1/img_gen", payload, timeout=120)
        job_id = submitted["id"]
        while True:
            job = self.get_json_from_backend(f"/sdcpp/v1/jobs/{job_id}", timeout=120)
            status = job.get("status")
            if status == "awaiting_hires_input":
                hires_input = ({} if supply_hires_input is None
                               else supply_hires_input(job["first_stage"]["b64_json"]))
                self.post_json_to_backend(f"/sdcpp/v1/jobs/{job_id}/hires-input",
                                          hires_input, timeout=120)
                continue
            if status == "completed":
                return {"images": [entry["b64_json"] for entry in job["result"]["images"]]}
            if status in ("failed", "cancelled"):
                raise RuntimeError((job.get("error") or {}).get("message", status))
            time.sleep(0.25)

    def get_json_from_backend(self, path: str, timeout: int) -> dict:
        with urllib.request.urlopen(self.backend + path, timeout=timeout) as response:
            return json.load(response)

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
    app.state.queue_manager = manager
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

    @app.post("/api/unload-models")
    def unload_models(_: str = Depends(authenticate)):
        if manager.current:
            raise HTTPException(409, "A generation is running; cancel it before unloading")
        try:
            manager.unload_models_from_vram()
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"model": manager.loaded_checkpoint}

    @app.get("/api/state")
    def state(_: str = Depends(authenticate)):
        return manager.snapshot()

    @app.post("/api/jobs")
    async def add_job(request: Request, _: str = Depends(authenticate)):
        body = await request.json()
        params = body.get("params", {})
        missing_prompt_reason = describe_missing_prompt(
            str(params.get("prompt", "")),
            bool(tagged_images_with_prompt_weights_for_stage(params, "stage_one")),
            bool(params.get("hires")),
            str(params.get("hires_prompt", "")),
            str(params.get("hires_prompt_mode", "append")),
            first_stage_samples=not renders_hires_from_existing_source(params),
            hires_tagging_enabled=bool(
                tagged_images_with_prompt_weights_for_stage(params, "hires")))
        if missing_prompt_reason:
            raise HTTPException(400, missing_prompt_reason)
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
