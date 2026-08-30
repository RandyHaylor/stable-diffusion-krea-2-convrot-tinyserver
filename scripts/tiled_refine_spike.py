#!/usr/bin/env python3
"""Run the two-tile Krea 2 refine spike against an already-running sd-server.

Generates one 832x1216 base, refines it as two overlapping 832x672 bands, and
writes both a hard join and a blended recombination so the seam can be judged.

This talks to the model server directly rather than through the web queue, so the
spike stays independent of the queue, the tunnel and the output folder. It needs
sd-server already up and holding the weights; it never starts, stops or reloads
it, and it submits nothing that would evict the loaded checkpoint.

Usage, with the server already running in its own terminal:

    python3 scripts/tiled_refine_spike.py            # variant A
    python3 scripts/tiled_refine_spike.py --variant B
    python3 scripts/tiled_refine_spike.py --variant C

Variants differ only in what the refine passes are told and how hard they push:

    A  base prompt on both tiles, denoise 0.25
    B  empty prompt on both tiles, denoise 0.25
    C  empty prompt on both tiles, denoise 0.35

Each variant writes to its own folder, and every variant reuses the same base
seed, so the bases are identical and only the refine differs.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiled_refine import (  # noqa: E402
    BASE_HEIGHT,
    BASE_WIDTH,
    VERTICAL_OVERLAP_PIXELS,
    blend_vertically_overlapping_tiles,
    join_tiles_without_blending,
    vertical_tile_crop_boxes,
)

DEFAULT_BACKEND_URL = "http://127.0.0.1:1234"
BASE_SEED = 12345

REFINE_PROMPT = (
    "full-body person standing before a tall window, "
    "one continuous red coat extending from shoulders to knees, "
    "wooden railing crossing the image horizontally, detailed interior"
)
NEGATIVE_PROMPT = "duplicate, repeated object, malformed, distorted"

# Sampling settings match the web UI's own defaults, so the spike's base image is
# representative of what the app normally produces rather than of a private setup.
# Those defaults assume the turbo LoRA: the checkpoint is a raw base, which the
# server itself would otherwise want ~52 steps and cfg 3.5 for.
SAMPLER_NAME = "euler"
SCHEDULER_NAME = "discrete"
BASE_STEPS = 6
REFINE_STEPS = 6
CFG_SCALE = 1.0
FLOW_SHIFT = 1.15
VAE_TILE_SIZE = 32
TURBO_LORA = {"path": "krea2_raw_to_turbo_r256.safetensors",
              "multiplier": 1.0,
              "is_high_noise": False}

REFINE_VARIANTS = {
    "A": {"uses_base_prompt": True, "denoise": 0.25},
    "B": {"uses_base_prompt": False, "denoise": 0.25},
    "C": {"uses_base_prompt": False, "denoise": 0.35},
}


def sample_params_for_steps(step_count: int) -> dict:
    return {
        "sample_method": SAMPLER_NAME,
        "scheduler": SCHEDULER_NAME,
        "sample_steps": step_count,
        "flow_shift": FLOW_SHIFT,
        "guidance": {"txt_cfg": CFG_SCALE},
    }


def common_generation_fields(step_count: int) -> dict:
    """The settings every pass in this spike shares.

    Both passes carry the same LoRA so the refine cannot drift in style from the
    base, and so the whole spike runs on one loaded weight set.
    """
    return {
        "sample_params": sample_params_for_steps(step_count),
        "vae_tiling_params": {"enabled": True, "tile_size_x": VAE_TILE_SIZE,
                              "tile_size_y": VAE_TILE_SIZE, "target_overlap": 0.5},
        "lora": [TURBO_LORA],
        "negative_prompt": NEGATIVE_PROMPT,
        "batch_count": 1,
    }


def post_json_to_backend(backend_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(backend_url + path,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{path} rejected the request: {exc.read().decode(errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach sd-server at {backend_url}. Start it in its own "
            f"terminal with ./restart-krea.sh before running this spike.") from exc


def get_json_from_backend(backend_url: str, path: str) -> dict:
    with urllib.request.urlopen(backend_url + path, timeout=300) as response:
        return json.load(response)


def run_generation(backend_url: str, payload: dict, description: str) -> Image.Image:
    """Submit one generation and wait for its single image."""
    print(f"  {description} ...", flush=True)
    started_at = time.monotonic()
    job_id = post_json_to_backend(backend_url, "/sdcpp/v1/img_gen", payload)["id"]
    while True:
        job = get_json_from_backend(backend_url, f"/sdcpp/v1/jobs/{job_id}")
        status = job.get("status")
        if status == "completed":
            images = job["result"]["images"]
            if not images:
                raise RuntimeError(f"{description} returned no images")
            print(f"  {description} took {time.monotonic() - started_at:.1f}s", flush=True)
            return Image.open(BytesIO(base64.b64decode(images[0]["b64_json"]))).convert("RGB")
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"{description} {status}: "
                               f"{(job.get('error') or {}).get('message', 'no message')}")
        time.sleep(0.25)


def generate_base_image(backend_url: str, width: int = BASE_WIDTH,
                        height: int = BASE_HEIGHT) -> Image.Image:
    return run_generation(backend_url, {
        **common_generation_fields(BASE_STEPS),
        "prompt": REFINE_PROMPT,
        "width": width,
        "height": height,
        "seed": BASE_SEED,
    }, f"base {width}x{height}")


def refine_tile(backend_url: str, tile: Image.Image, prompt: str, seed: int,
                denoise: float, refine_steps: int, description: str) -> Image.Image:
    tile_buffer = BytesIO()
    tile.save(tile_buffer, format="PNG")
    return run_generation(backend_url, {
        **common_generation_fields(refine_steps),
        "prompt": prompt,
        "width": tile.width,
        "height": tile.height,
        "seed": seed,
        "init_image": base64.b64encode(tile_buffer.getvalue()).decode("ascii"),
        "strength": denoise,
    }, description)


def run_spike(backend_url: str, variant_name: str, output_root: Path,
              refine_steps: int, run_label: str) -> int:
    variant = REFINE_VARIANTS[variant_name]
    refine_prompt = REFINE_PROMPT if variant["uses_base_prompt"] else ""
    denoise = variant["denoise"]
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # The runtime spends sample_steps * strength steps on an init image, rounded
    # down, so a low denoise on a short schedule can leave a single step.
    effective_refine_steps = max(1, int(refine_steps * denoise))
    print(f"Variant {variant_name}: prompt={'base' if refine_prompt else 'empty'}, "
          f"denoise={denoise}, refine schedule {refine_steps} steps "
          f"({effective_refine_steps} actually spent), writing to {output_dir}", flush=True)

    base = generate_base_image(backend_url)
    if base.size != (BASE_WIDTH, BASE_HEIGHT):
        raise RuntimeError(f"base came back {base.size}, expected {(BASE_WIDTH, BASE_HEIGHT)}")
    base.save(output_dir / "00_base.png")

    top_box, bottom_box = vertical_tile_crop_boxes()
    top_source = base.crop(top_box)
    bottom_source = base.crop(bottom_box)
    top_source.save(output_dir / "01_top_source.png")
    bottom_source.save(output_dir / "02_bottom_source.png")

    top_result = refine_tile(backend_url, top_source, refine_prompt,
                             BASE_SEED, denoise, refine_steps, "top tile refine")
    bottom_result = refine_tile(backend_url, bottom_source, refine_prompt,
                                BASE_SEED + 1, denoise, refine_steps, "bottom tile refine")
    for tile, name in ((top_result, "top"), (bottom_result, "bottom")):
        if tile.size != top_source.size:
            raise RuntimeError(f"{name} tile came back {tile.size}, expected {top_source.size}")
    top_result.save(output_dir / "03_top_result.png")
    bottom_result.save(output_dir / "04_bottom_result.png")

    join_tiles_without_blending(top_result, bottom_result, VERTICAL_OVERLAP_PIXELS).save(
        output_dir / "05_hard_join.png")
    blend_vertically_overlapping_tiles(top_result, bottom_result, VERTICAL_OVERLAP_PIXELS).save(
        output_dir / "06_blended.png")

    print(f"Variant {variant_name} complete. Compare 05_hard_join.png against "
          f"06_blended.png, and both against 00_base.png.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(REFINE_VARIANTS), default="A",
                        help="which prompt and denoise combination to run (default: A)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL,
                        help=f"sd-server base URL (default: {DEFAULT_BACKEND_URL})")
    parser.add_argument("--output-dir", default="tile_spike",
                        help="folder for this run's images (default: tile_spike)")
    parser.add_argument("--refine-steps", type=int, default=REFINE_STEPS,
                        help="schedule length for the refine passes; the runtime "
                             f"spends this times the denoise (default: {REFINE_STEPS})")
    parser.add_argument("--label", default="",
                        help="subfolder name for this run (default: variant_<variant>)")
    arguments = parser.parse_args()
    run_label = arguments.label or f"variant_{arguments.variant}"
    return run_spike(arguments.backend, arguments.variant, Path(arguments.output_dir),
                     arguments.refine_steps, run_label)


if __name__ == "__main__":
    raise SystemExit(main())
