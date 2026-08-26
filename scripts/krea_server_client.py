#!/usr/bin/env python3
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate with the persistent Krea server")
    parser.add_argument("prompt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="discrete")
    parser.add_argument("--extra-sample-args")
    parser.add_argument("--flow-shift", type=float)
    parser.add_argument(
        "--vae-tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tile VAE decoding to reduce peak VRAM (default: enabled)",
    )
    parser.add_argument(
        "--vae-tile-size",
        type=int,
        default=32,
        help="latent-space tile edge; 32 equals 256 output pixels for Krea's 8x VAE",
    )
    parser.add_argument("--hires-width", type=int, default=0)
    parser.add_argument("--hires-height", type=int, default=0)
    parser.add_argument("--hires-steps", type=int, default=3)
    parser.add_argument("--hires-denoise", type=float, default=0.5)
    parser.add_argument(
        "--hires-upscaler",
        default="Latent",
        help="Latent performs bilinear latent-space interpolation (default: Latent)",
    )
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--output-prefix", default="krea")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "negative_prompt": "",
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "batch_size": 1,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "lora": [],
    }
    sample_params = {
        "sample_method": args.sampler,
        "scheduler": args.scheduler,
        "sample_steps": args.steps,
        "guidance": {"txt_cfg": args.cfg},
    }
    if args.extra_sample_args:
        sample_params["extra_sample_args"] = args.extra_sample_args
    if args.flow_shift is not None:
        sample_params["flow_shift"] = args.flow_shift
    native = {
        "sample_params": sample_params,
        "vae_tiling_params": {
            "enabled": args.vae_tiling,
            "tile_size_x": args.vae_tile_size,
            "tile_size_y": args.vae_tile_size,
            "target_overlap": 0.5,
        },
    }
    if args.hires_width > 0 or args.hires_height > 0:
        if args.hires_width <= 0 or args.hires_height <= 0:
            parser.error("--hires-width and --hires-height must be supplied together")
        native["hires"] = {
            "enabled": True,
            "upscaler": args.hires_upscaler,
            "target_width": args.hires_width,
            "target_height": args.hires_height,
            "steps": args.hires_steps,
            "denoising_strength": args.hires_denoise,
        }
    payload["prompt"] += f" <sd_cpp_extra_args>{json.dumps(native, separators=(',', ':'))}</sd_cpp_extra_args>"
    if args.lora:
        payload["lora"] = [
            {"path": args.lora.name, "multiplier": args.lora_strength, "is_high_noise": False}
        ]

    request = urllib.request.Request(
        f"http://{args.host}:{args.port}/sdapi/v1/txt2img",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"generation request failed: {exc}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"generation request failed: {exc}", file=sys.stderr)
        return 1

    images = result.get("images", [])
    if not images:
        print(f"server returned no images: {result}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for index, encoded in enumerate(images):
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        output = args.output_dir / f"{args.output_prefix}-{stamp}-{index}.png"
        output.write_bytes(base64.b64decode(encoded))
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
