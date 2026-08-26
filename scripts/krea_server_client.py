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
    parser.add_argument("--flow-shift", type=float)
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
        "sampler_name": "Euler",
        "lora": [],
    }
    sample_params = {
        "sample_method": "euler",
        "sample_steps": args.steps,
        "guidance": {"txt_cfg": args.cfg},
    }
    if args.flow_shift is not None:
        sample_params["flow_shift"] = args.flow_shift
    native = {"sample_params": sample_params}
    payload["prompt"] += f" <sd_cpp_extra_args>{json.dumps(native, separators=(',', ':'))}</sd_cpp_extra_args>"
    if args.lora:
        payload["lora"] = [
            {"path": str(args.lora.resolve()), "multiplier": args.lora_strength, "is_high_noise": False}
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
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
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
