#!/usr/bin/env python3
"""Unit tests for the tiled hires render loop, with the backend call stubbed.

Exercises the multi-hop path without an HTTP server or a GPU: the generation call
is replaced by one that returns a flat image of whatever size was asked for, so
what is under test is the loop's own geometry, sequencing and assembly.
"""
from __future__ import annotations

import base64
import sys
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import krea_web  # noqa: E402

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def flat_png_base64(width: int, height: int, shade: int) -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (shade, 80, 120)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RecordingBackend:
    """Stands in for run_generation_job, recording every payload it is given."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, payload: dict, _supply_hires_input=None) -> dict:
        self.payloads.append(payload)
        shade = 20 + (len(self.payloads) * 7) % 200
        return {"images": [flat_png_base64(int(payload["width"]),
                                          int(payload["height"]), shade)]}


def job_params(**overrides) -> dict:
    params = {
        "prompt": "a test", "negative_prompt": "",
        "width": 64, "height": 64,
        "steps": 2, "cfg": 1.0, "seed": 7,
        "sampler": "euler", "scheduler": "discrete", "flow_shift": 1.0,
        "extra_sample_args": "", "vae_tile_size": 32,
        "beta_schedule_alpha": 0.5, "beta_schedule_beta": 0.7,
        "hires": True,
        "hires_width": 132, "hires_height": 132,
        "hires_steps": 2, "hires_denoise": 0.5,
        "hires_tiling": "on", "hires_tile_overlap": 8,
        "hires_tile_source": "anchored",
        "extra_loras": [],
    }
    params.update(overrides)
    return params


def render_with_stubbed_backend(params: dict) -> tuple[Image.Image, RecordingBackend]:
    """Run the tiled hires path against a recording stub, in a temp output dir."""
    backend = RecordingBackend()
    original_output_dir = krea_web.OUTPUT_DIR
    original_run_generation_job = krea_web.QueueManager.run_generation_job
    with tempfile.TemporaryDirectory() as disposable:
        krea_web.OUTPUT_DIR = Path(disposable)
        krea_web.QueueManager.run_generation_job = (
            lambda _self, payload, supply=None: backend(payload, supply))
        try:
            first_stage_name = "first-stage.png"
            Image.new("RGB", (int(params["width"]), int(params["height"])), (10, 10, 10)).save(
                krea_web.OUTPUT_DIR / first_stage_name)
            manager = krea_web.QueueManager("http://127.0.0.1:1")
            produced = manager.render_hires_by_tiling(
                params, first_stage_name, "test-tiled", "a test", "")
            image = Image.open(krea_web.OUTPUT_DIR / produced[0])
            image.load()
            return image, backend
        finally:
            krea_web.OUTPUT_DIR = original_output_dir
            krea_web.QueueManager.run_generation_job = original_run_generation_job


def main() -> int:
    from hires_tiling import hires_tiling_hop_sizes  # noqa: E402
    from tiled_refine import covering_grid_tile_boxes  # noqa: E402

    multi_hop_params = job_params()
    hop_sizes = hires_tiling_hop_sizes(multi_hop_params)
    check("the test target really does take more than one hop",
          len(hop_sizes) > 1,
          f"hops {hop_sizes}")

    image, backend = render_with_stubbed_backend(multi_hop_params)
    check("the multi-hop result is exactly the requested target",
          image.size == (132, 132),
          f"got {image.size}")

    expected_tiles_per_hop = [len(covering_grid_tile_boxes(size[0], size[1], 64, 64, 8))
                              for size in hop_sizes]
    check("one request is sent per tile, across every hop",
          len(backend.payloads) == sum(expected_tiles_per_hop),
          f"sent {len(backend.payloads)}, expected {expected_tiles_per_hop} "
          f"= {sum(expected_tiles_per_hop)}")
    check("every tile request is the tile size, never the hop size",
          all(payload["width"] == 64 and payload["height"] == 64
              for payload in backend.payloads),
          "a tile larger than the proven size defeats the point of tiling")
    check("every tile request refines a supplied image",
          all("init_image" in payload for payload in backend.payloads))
    check("no tile request asks the runtime for a hires pass of its own",
          all("hires" not in payload for payload in backend.payloads))

    single_hop_params = job_params(hires_tiling_hops="single")
    single_image, single_backend = render_with_stubbed_backend(single_hop_params)
    check("single hop mode still reaches the same target",
          single_image.size == (132, 132),
          f"got {single_image.size}")
    check("single hop mode sends fewer requests than doubling",
          len(single_backend.payloads) < len(backend.payloads),
          f"single {len(single_backend.payloads)}, doubling {len(backend.payloads)}")

    # A hop's tiles must be cut from the previous hop's output, which is only
    # observable through the shades the stub gives each returned tile.
    first_hop_tile_count = expected_tiles_per_hop[0]
    second_hop_sources = {payload["init_image"]
                          for payload in backend.payloads[first_hop_tile_count:]}
    check("the second hop's tiles are not all identical",
          len(second_hop_sources) > 1,
          "tiles cut from a resampled earlier hop differ from one another")

    check("tile vision is absent unless asked for",
          all("vlm_images" not in payload for payload in backend.payloads),
          "vision tokens cost sequence length in every tile")

    _vision_image, vision_backend = render_with_stubbed_backend(
        job_params(hires_tile_vision="on", hires_tile_vision_weight=0.5))
    check("every tile sends its own starting pixels to the vision tower",
          all(payload.get("vlm_images") == [payload["init_image"]]
              for payload in vision_backend.payloads),
          "a tile must be described by its own content, not another tile's")
    check("the tile vision weight travels as a ref image arg",
          all(payload.get("ref_image_args") == "vlm_image_token_weight=0.5"
              for payload in vision_backend.payloads),
          f"got {vision_backend.payloads[0].get('ref_image_args')!r}")

    _neutral_vision_image, neutral_vision_backend = render_with_stubbed_backend(
        job_params(hires_tile_vision="on"))
    check("a neutral tile vision weight still attaches the image but no args",
          all("vlm_images" in payload and "ref_image_args" not in payload
              for payload in neutral_vision_backend.payloads),
          "the image is what the tower reads; the weight is optional")

    unanchored_image, unanchored_backend = render_with_stubbed_backend(
        job_params(hires_tile_source="independent"))
    check("independent mode reaches the target too",
          unanchored_image.size == (132, 132))
    check("independent mode sends the same number of requests as anchored",
          len(unanchored_backend.payloads) == len(backend.payloads),
          "anchoring changes what a tile starts from, never how many run")

    print()
    if failures:
        print(f"{len(failures)} tiled hires render loop check(s) failed")
        return 1
    print("all tiled hires render loop checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
