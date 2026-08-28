#!/usr/bin/env python3
"""Repair checkpoint/LoRA hashes in every saved Krea PNG.

Resource filenames are taken from each image's embedded generation settings,
matched against models/chkpt and models/loras, and rewritten using Civitai
AutoV2 hashes. Rewrites are atomic and retain each file's original timestamps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

from image_metadata import _a1111_parameters, cached_civitai_hash


ROOT = Path(__file__).resolve().parent.parent
SEED_PATTERN = re.compile(r"(?:^|,\s)Seed:\s*(-?\d+)(?:,|$)")


def json_object(value: Any) -> dict:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def file_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in directory.glob("*.safetensors"):
        index[path.name.lower()] = path
        index[path.stem.lower()] = path
    return index


def resolve_file(index: dict[str, Path], filename: str) -> Path | None:
    name = Path(str(filename)).name
    return index.get(name.lower()) or index.get(Path(name).stem.lower())


def cached_download_hash_index(model_root: Path) -> dict[str, tuple[str, str]]:
    """Index hashes retained by the Hugging Face cache for removed model files."""
    index: dict[str, tuple[str, str]] = {}
    pattern = "**/.cache/huggingface/download/*.safetensors.metadata"
    for metadata_path in model_root.glob(pattern):
        filename = metadata_path.name.removesuffix(".metadata")
        try:
            lines = [line.strip().lower() for line in metadata_path.read_text().splitlines()]
        except OSError:
            continue
        full_hash = next((line for line in lines if re.fullmatch(r"[0-9a-f]{64}", line)), "")
        if not full_hash:
            continue
        record = (filename, full_hash[:10])
        index[filename.lower()] = record
        index[Path(filename).stem.lower()] = record
    return index


def resolve_resource(index: dict[str, Path], cached_hashes: dict[str, tuple[str, str]],
                     filename: str) -> tuple[str, str] | None:
    path = resolve_file(index, filename)
    if path:
        return path.name, cached_civitai_hash(path)
    name = Path(str(filename)).name
    return cached_hashes.get(name.lower()) or cached_hashes.get(Path(name).stem.lower())


def payload_from_parameters(parameters: str) -> tuple[dict, dict]:
    marker = "SDCPP: "
    marker_index = parameters.rfind(marker)
    if marker_index < 0:
        return {}, {}
    try:
        sdcpp, _ = json.JSONDecoder().raw_decode(parameters[marker_index + len(marker):])
    except (json.JSONDecodeError, TypeError):
        return {}, {}
    if not isinstance(sdcpp, dict):
        return {}, {}

    prompt_data = sdcpp.get("prompt", {}) if isinstance(sdcpp.get("prompt"), dict) else {}
    sampling = sdcpp.get("sampling", {}) if isinstance(sdcpp.get("sampling"), dict) else {}
    guidance = sampling.get("guidance", {}) if isinstance(sampling.get("guidance"), dict) else {}
    models = sdcpp.get("models", {}) if isinstance(sdcpp.get("models"), dict) else {}
    vae_tiling = sdcpp.get("vae_tiling", {}) if isinstance(sdcpp.get("vae_tiling"), dict) else {}
    raw_loras = sdcpp.get("loras", []) if isinstance(sdcpp.get("loras"), list) else []
    extra_loras = [
        {"filename": str(item.get("name", "")), "strength": item.get("multiplier", 1.0)}
        for item in raw_loras if isinstance(item, dict) and item.get("name")
    ]
    model_name = str(models.get("diffusion_model", ""))
    seed = int(sdcpp.get("seed", 0))
    payload = {
        "prompt": str(prompt_data.get("positive", parameters.split("\nSteps:", 1)[0].strip())),
        "negative_prompt": str(prompt_data.get("negative", "")),
        "steps": sampling.get("steps", ""),
        "sampler_name": sampling.get("method", ""),
        "scheduler": sampling.get("scheduler", ""),
        "cfg_scale": guidance.get("txt_cfg", ""),
        "seed": seed,
        "width": sdcpp.get("width", ""),
        "height": sdcpp.get("height", ""),
        "model_name": model_name,
        "lora": [
            {"path": item["filename"], "multiplier": item["strength"], "is_high_noise": False}
            for item in extra_loras
        ],
        "generation_type": "txt2img",
    }
    ui_params = {
        "prompt": payload["prompt"], "negative_prompt": payload["negative_prompt"],
        "steps": payload["steps"], "sampler": payload["sampler_name"],
        "scheduler": payload["scheduler"], "cfg": payload["cfg_scale"],
        "seed": seed, "width": payload["width"], "height": payload["height"],
        "checkpoint": model_name, "model_name": model_name,
        "extra_loras": extra_loras,
        "flow_shift": sampling.get("flow_shift", ""),
        "extra_sample_args": sampling.get("extra_sample_args", ""),
        "vae_tile_size": vae_tiling.get("tile_size_x", 32),
        "hires": bool(sdcpp.get("hires", {}).get("enabled", False))
                 if isinstance(sdcpp.get("hires"), dict) else False,
    }
    return payload, ui_params


def payload_from_image_info(info: dict) -> tuple[dict, dict, dict]:
    payload = json_object(info.get("prompt"))
    workflow = json_object(info.get("workflow"))
    ui_params = payload.get("ui_params") if isinstance(payload.get("ui_params"), dict) else {}
    if not ui_params:
        ui_params = json_object(info.get("krea_generation"))
    ui_params = dict(ui_params)
    payload = dict(payload or ui_params)
    if not payload and not ui_params:
        payload, ui_params = payload_from_parameters(str(info.get("parameters", "")))

    field_map = {
        "prompt": "prompt", "negative_prompt": "negative_prompt", "steps": "steps",
        "sampler": "sampler_name", "scheduler": "scheduler", "cfg": "cfg_scale",
        "seed": "seed", "width": "width", "height": "height",
    }
    for ui_name, payload_name in field_map.items():
        if payload.get(payload_name) is None and ui_params.get(ui_name) is not None:
            payload[payload_name] = ui_params[ui_name]
    return payload, ui_params, workflow


def repair_resource_hashes(payload: dict, ui_params: dict,
                           checkpoints: dict[str, Path], loras: dict[str, Path],
                           cached_checkpoints: dict[str, tuple[str, str]]) -> tuple[int, list[str]]:
    repaired = 0
    missing: list[str] = []

    model_name = str(ui_params.get("checkpoint") or payload.get("model_name") or
                     ui_params.get("model_name") or "")
    model_resource = resolve_resource(checkpoints, cached_checkpoints, model_name) if model_name else None
    if model_resource:
        model_filename, model_hash = model_resource
        payload["model_name"] = model_filename
        payload["model_hash"] = model_hash
        ui_params["checkpoint"] = model_filename
        ui_params["model_hash"] = model_hash
        repaired += 1
    elif model_name:
        missing.append(f"checkpoint:{model_name}")

    selected_loras = ui_params.get("extra_loras", [])
    if not isinstance(selected_loras, list) or not selected_loras:
        selected_loras = [
            {"filename": item.get("path", ""), "strength": item.get("multiplier", 1.0)}
            for item in payload.get("lora", []) if isinstance(item, dict)
        ]

    lora_hashes: list[str] = []
    hashes: dict[str, str] = {}
    resources: list[dict] = []
    if model_resource:
        hashes["model"] = payload["model_hash"]
        resources.append({"type": "model", "name": model_filename, "hash": payload["model_hash"]})
    normalized_loras: list[dict] = []
    backend_loras: list[dict] = []
    for selected in selected_loras:
        if not isinstance(selected, dict):
            continue
        filename = str(selected.get("filename") or selected.get("path") or "")
        lora_path = resolve_file(loras, filename)
        if not lora_path:
            if filename:
                missing.append(f"lora:{filename}")
            continue
        strength = float(selected.get("strength", selected.get("multiplier", 1.0)))
        lora_hash = cached_civitai_hash(lora_path)
        name = lora_path.stem
        lora_hashes.append(f"{name}:{lora_hash}")
        hashes[f"lora:{name}"] = lora_hash
        resources.append({"type": "lora", "name": name, "hash": lora_hash, "weight": strength})
        normalized_loras.append({"filename": lora_path.name, "strength": strength})
        backend_loras.append({"path": lora_path.name, "multiplier": strength, "is_high_noise": False})
        repaired += 1

    payload["lora_hashes"] = lora_hashes
    ui_params["lora_hashes"] = lora_hashes
    ui_params["extra_loras"] = normalized_loras
    if backend_loras:
        payload["lora"] = backend_loras
    else:
        payload.pop("lora", None)
    payload["hashes"] = hashes
    payload["resources"] = resources
    payload["ui_params"] = ui_params
    return repaired, missing


def recover_seed_from_parameters(payload: dict, ui_params: dict, parameters: str) -> bool:
    if int(payload.get("seed", ui_params.get("seed", -1))) != -1:
        return False
    match = SEED_PATTERN.search(parameters)
    if not match or int(match.group(1)) == -1:
        return False
    seed = int(match.group(1))
    payload["seed"] = seed
    ui_params["seed"] = seed
    return True


def update_png(path: Path, checkpoints: dict[str, Path], loras: dict[str, Path],
               cached_checkpoints: dict[str, tuple[str, str]],
               dry_run: bool) -> tuple[bool, int, bool, list[str]]:
    original_stat = path.stat()
    with Image.open(path) as image:
        image.load()
        info = dict(image.info)
        payload, ui_params, workflow = payload_from_image_info(info)
        if not payload and not ui_params:
            return False, 0, False, []
        recovered_seed = recover_seed_from_parameters(payload, ui_params, str(info.get("parameters", "")))
        repaired, missing = repair_resource_hashes(
            payload, ui_params, checkpoints, loras, cached_checkpoints
        )
        payload["ui_params"] = ui_params

        text_metadata = {key: value for key, value in info.items() if isinstance(value, str)}
        text_metadata["parameters"] = _a1111_parameters(payload)
        text_metadata["prompt"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        text_metadata["krea_generation"] = json.dumps(ui_params, ensure_ascii=False, separators=(",", ":"))
        text_metadata["krea_extra_sample_args"] = str(ui_params.get("extra_sample_args", ""))
        if not workflow:
            workflow = {"version": 1, "source": "krea_convrot", "nodes": []}
        workflow["generation"] = payload
        text_metadata["workflow"] = json.dumps(workflow, ensure_ascii=False, separators=(",", ":"))

        changed = any(text_metadata.get(key) != info.get(key) for key in text_metadata)
        if dry_run or not changed:
            return changed, repaired, recovered_seed, missing

        pnginfo = PngImagePlugin.PngInfo()
        for key, value in text_metadata.items():
            pnginfo.add_text(key, value)
        temporary = path.with_name(f".{path.name}.metadata-{os.getpid()}.tmp")
        save_options: dict[str, Any] = {"format": "PNG", "pnginfo": pnginfo}
        if info.get("icc_profile"):
            save_options["icc_profile"] = info["icc_profile"]
        if info.get("exif"):
            save_options["exif"] = info["exif"]
        try:
            image.save(temporary, **save_options)
            os.chmod(temporary, stat.S_IMODE(original_stat.st_mode))
            os.replace(temporary, path)
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        finally:
            temporary.unlink(missing_ok=True)
    return True, repaired, recovered_seed, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoints = file_index(ROOT / "models" / "chkpt")
    loras = file_index(ROOT / "models" / "loras")
    cached_checkpoints = cached_download_hash_index(ROOT / "models")
    images = sorted(args.output_dir.glob("*.png"))
    updated = skipped = repaired_resources = recovered_seeds = 0
    missing_resources: set[str] = set()
    for index, path in enumerate(images, 1):
        before = path.stat().st_mtime_ns
        changed, repaired, recovered_seed, missing = update_png(
            path, checkpoints, loras, cached_checkpoints, args.dry_run
        )
        after = path.stat().st_mtime_ns
        if changed:
            updated += 1
            repaired_resources += repaired
            recovered_seeds += int(recovered_seed)
        else:
            skipped += 1
        missing_resources.update(missing)
        print(f"[{index}/{len(images)}] {'would update' if args.dry_run and changed else 'updated' if changed else 'skipped'} {path.name}")
        if not args.dry_run and before != after:
            raise RuntimeError(f"timestamp changed unexpectedly: {path}")

    print(f"images={len(images)} updated={updated} skipped={skipped} "
          f"resource_hashes={repaired_resources} recovered_seeds={recovered_seeds}")
    if missing_resources:
        print("unmatched metadata filenames:")
        for item in sorted(missing_resources):
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
