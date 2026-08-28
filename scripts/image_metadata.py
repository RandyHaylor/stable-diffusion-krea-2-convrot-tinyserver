"""Embed generation metadata in PNGs using common A1111 and ComfyUI fields."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin


def cached_sha256(path: str | Path) -> str:
    """Return the full file SHA-256, cached in ``<filename>.hash``."""
    source = Path(path)
    cache = Path(f"{source}.hash")
    try:
        cached = cache.read_text(encoding="ascii").strip().lower()
        if len(cached) == 64 and all(char in "0123456789abcdef" for char in cached):
            return cached
    except OSError:
        pass

    import hashlib

    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    try:
        cache.write_text(value + "\n", encoding="ascii")
    except OSError:
        pass
    return value


def cached_civitai_hash(path: str | Path) -> str:
    """Return Civitai/A1111 AutoV2: the first 10 chars of full SHA-256."""
    return cached_sha256(path)[:10]


def _a1111_parameters(payload: dict[str, Any]) -> str:
    """Format the familiar Stable Diffusion WebUI parameters text block."""
    prompt = str(payload.get("prompt", ""))
    negative = str(payload.get("negative_prompt", ""))
    settings = [
        f"Steps: {payload.get('steps', '')}",
        f"Sampler: {payload.get('sampler_name', '')}",
        f"Schedule type: {payload.get('scheduler', '')}",
        f"CFG scale: {payload.get('cfg_scale', '')}",
        f"Seed: {payload.get('seed', '')}",
        f"Size: {payload.get('width', '')}x{payload.get('height', '')}",
    ]
    if payload.get("lora"):
        settings.append(f"LoRA: {json.dumps(payload['lora'], separators=(',', ':'))}")
    if payload.get("model_name"):
        if payload.get("model_hash"):
            settings.append(f"Model hash: {payload['model_hash']}")
        settings.append(f"Model: {payload['model_name']}")
    if payload.get("lora_hashes"):
        lora_hashes = []
        for item in payload["lora_hashes"]:
            name, separator, value = str(item).partition(":")
            lora_hashes.append(f"{name}: {value.strip()}" if separator else str(item))
        settings.append(f'Lora hashes: "{", ".join(lora_hashes)}"')
    generation_type = str(payload.get("generation_type", "txt2img"))
    img2img = payload.get("img2img", {}) if isinstance(payload.get("img2img"), dict) else {}
    if generation_type == "img2img":
        settings.append("Generation type: img2img")
        settings.append(f"Denoising strength: {img2img.get('denoising_strength', '')}")
        if img2img.get("source_image"):
            settings.append(f"Img2img source: {img2img['source_image']}")
    text = prompt
    if negative:
        text += f"\nNegative prompt: {negative}"
    return text + "\n" + ", ".join(settings)


def embed_generation_metadata(image_bytes: bytes, payload: dict[str, Any]) -> bytes:
    """Return PNG bytes with A1111 and ComfyUI-compatible text chunks."""
    metadata_payload = dict(payload)
    had_init_images = bool(metadata_payload.get("init_images"))
    # The source filename is sufficient to reproduce a web img2img job. Keeping
    # init_images would duplicate the complete base64 input image inside every
    # output PNG and is not part of either common metadata convention.
    metadata_payload.pop("init_images", None)
    ui_params = dict(metadata_payload.get("ui_params", metadata_payload))
    source_image = str(ui_params.get("source_image", ""))
    generation_type = str(metadata_payload.get(
        "generation_type", "img2img" if source_image or had_init_images else "txt2img"
    ))
    metadata_payload["generation_type"] = generation_type
    if generation_type == "img2img":
        metadata_payload["img2img"] = {
            "source_image": source_image,
            "denoising_strength": metadata_payload.get(
                "denoising_strength", ui_params.get("img2img_denoise", 0.75)
            ),
        }

    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", _a1111_parameters(metadata_payload))

        # A1111 reads ``parameters``. ComfyUI stores its API prompt and graph
        # in JSON text chunks named ``prompt`` and ``workflow`` respectively.
        prompt_json = json.dumps(metadata_payload, ensure_ascii=False, separators=(",", ":"))
        workflow_json = json.dumps(
            {
                "version": 1,
                "source": "krea_convrot",
                "nodes": [],
                "generation": metadata_payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        pnginfo.add_text("prompt", prompt_json)
        pnginfo.add_text("workflow", workflow_json)
        pnginfo.add_text("krea_extra_sample_args", str(ui_params.get("extra_sample_args", "")))
        pnginfo.add_text("krea_generation", json.dumps(ui_params, ensure_ascii=False, separators=(",", ":")))
        pnginfo.add_text("krea_generation_type", generation_type)
        if generation_type == "img2img":
            pnginfo.add_text("krea_img2img", json.dumps(
                metadata_payload["img2img"], ensure_ascii=False, separators=(",", ":")
            ))

        output = BytesIO()
        image.save(output, format="PNG", pnginfo=pnginfo)
        return output.getvalue()
