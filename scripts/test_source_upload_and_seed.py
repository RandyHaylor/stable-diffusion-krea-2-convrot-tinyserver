#!/usr/bin/env python3
"""Focused checks for imported img2img sources and concrete random seeds."""

from __future__ import annotations

import sys
import tempfile
import base64
import json
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent))
import krea_web  # noqa: E402
from image_metadata import embed_generation_metadata  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (37, 23), (120, 40, 210)).save(output, format="JPEG")
    return output.getvalue()


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), (20, 80, 140)).save(output, format="PNG")
    return output.getvalue()


def main() -> int:
    original_output_dir = krea_web.OUTPUT_DIR
    with tempfile.TemporaryDirectory() as temporary:
        krea_web.OUTPUT_DIR = Path(temporary)
        try:
            uploaded = krea_web.save_source_image(jpeg_bytes(), "portrait.jpeg")
            check(uploaded["name"] == "portrait.jpeg", "JPEG source upload succeeds")
            check((uploaded["width"], uploaded["height"]) == (37, 23),
                  "upload reports the decoded source resolution")
            check((krea_web.OUTPUT_DIR / uploaded["name"]).is_file(),
                  "uploaded source remains in the output folder")

            names = [path.name for path in krea_web.output_image_paths()]
            check(uploaded["name"] in names, "Output tab lists JPEG source images")

            duplicate = krea_web.save_source_image(jpeg_bytes(), "portrait.jpeg")
            check(duplicate["name"] != uploaded["name"], "upload never overwrites an existing image")
            try:
                krea_web.save_source_image(b"not an image", "bad.jpeg")
            except HTTPException as exc:
                check(exc.status_code == 422, "invalid source data is rejected")
            else:
                raise AssertionError("invalid source data is rejected")
        finally:
            krea_web.OUTPUT_DIR = original_output_dir

    manager = object.__new__(krea_web.QueueManager)
    observed_seeds: list[int] = []
    manager.run_single_backend_generation = (
        lambda params, hires, prefix, reference_image_names=None:
            observed_seeds.append(params["seed"]) or []
    )
    params = {"seed": -1, "checkpoint": "", "hires": False}
    manager.generate_job_outputs({"id": "seed-test", "params": params, "cancel_requested": False})
    check(0 <= params["seed"] < 2**32 and observed_seeds == [params["seed"]],
          "-1 resolves once to a concrete seed before generation")
    check(krea_web.actual_seeds_from_backend_result(
        {"info": '{"all_seeds":[987654321]}'}, 1, params["seed"]
    ) == [987654321], "backend-reported concrete seed is retained")
    check(krea_web.actual_seeds_from_backend_result(
        {"info": '{"all_seeds":[-1]}'}, 1, params["seed"]
    ) == [params["seed"]], "backend -1 cannot replace the resolved seed")

    with tempfile.TemporaryDirectory() as temporary:
        krea_web.OUTPUT_DIR = Path(temporary)
        try:
            metadata_manager = object.__new__(krea_web.QueueManager)
            metadata_manager.post_json_to_backend = lambda *_args, **_kwargs: {
                "images": [base64.b64encode(png_bytes()).decode()],
                "info": '{"all_seeds":[-1]}',
            }
            metadata_params = {
                "prompt": "seed metadata test", "negative_prompt": "",
                "width": 8, "height": 8, "steps": 1, "cfg": 1.0, "seed": -1,
                "sampler": "er_sde", "scheduler": "discrete", "flow_shift": 1.15,
                "extra_sample_args": "", "vae_tile_size": 32, "extra_loras": [],
                "beta_schedule_alpha": 0.5, "beta_schedule_beta": 0.7,
                "checkpoint": "", "model_name": "", "source_image": "",
                "hires": False, "save_lowres": False,
            }
            names = metadata_manager.generate_job_outputs({
                "id": "metadata-seed", "params": metadata_params, "cancel_requested": False,
            })
            with Image.open(krea_web.OUTPUT_DIR / names[0]) as output_image:
                embedded = json.loads(output_image.info["prompt"])
                a1111 = output_image.info["parameters"]
            actual_seed = metadata_params["seed"]
            check(embedded["seed"] == actual_seed
                  and embedded["ui_params"]["seed"] == actual_seed
                  and f"Seed: {actual_seed}" in a1111
                  and "Seed: -1" not in a1111,
                  "saved PNG embeds the concrete seed in JSON and A1111 metadata")

            bulky_base64_image = "A" * 200_000
            embedded_without_image_payloads = json.loads(Image.open(BytesIO(
                embed_generation_metadata(
                    png_bytes(),
                    {"seed": 1, "prompt": "x",
                     "init_images": [bulky_base64_image],
                     "extra_images": [bulky_base64_image, bulky_base64_image],
                     "ui_params": {"seed": 1}},
                ))).info["prompt"])
            check("extra_images" not in embedded_without_image_payloads
                  and "init_images" not in embedded_without_image_payloads,
                  "base64 reference and source images are kept out of PNG metadata")
            check(len(embed_generation_metadata(
                      png_bytes(),
                      {"seed": 1, "prompt": "x",
                       "extra_images": [bulky_base64_image, bulky_base64_image],
                       "ui_params": {"seed": 1}})) < 100_000,
                  "a PNG with large references stays small instead of embedding them")

            foreign_workflow_image = Image.open(BytesIO(png_bytes()))
            foreign_metadata = PngImagePlugin.PngInfo()
            foreign_metadata.add_text("prompt", json.dumps(
                {"83": {"class_type": "DPRandomGenerator", "is_changed": [float("nan")]}}))
            foreign_workflow_path = krea_web.OUTPUT_DIR / "comfyui-export.png"
            foreign_workflow_image.save(foreign_workflow_path, pnginfo=foreign_metadata)
            with Image.open(foreign_workflow_path) as saved_foreign_image:
                foreign_params, _, _ = krea_web.read_generation_metadata(saved_foreign_image)
            check(json.dumps(foreign_params, allow_nan=False) is not None,
                  "metadata from a foreign workflow stays JSON-serializable")
        finally:
            krea_web.OUTPUT_DIR = original_output_dir
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
