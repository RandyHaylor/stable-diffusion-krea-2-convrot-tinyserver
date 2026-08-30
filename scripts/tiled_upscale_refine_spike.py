#!/usr/bin/env python3
"""Refine an upscaled image as a grid of tiles, past what one pass can render.

The base is generated at a size the GPU handles, resampled up to a canvas it
cannot render in one pass, then cut into overlapping tiles that are each refined
at the proven tile size. Resampling supplies the pixels; the refine is what puts
real detail into them, so the comparison that matters is the blended result
against the plain resampled canvas, not against the base.

Unlike the two-tile spike, every tile here is the same size as the original base
generation, so the model never sees a shape it has not already handled.

Usage, with sd-server already running in its own terminal:

    python3 scripts/tiled_upscale_refine_spike.py
    python3 scripts/tiled_upscale_refine_spike.py --denoise 0.45 --label softer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiled_refine import (  # noqa: E402
    blend_tiles_into_canvas,
    grid_tile_crop_boxes,
)
from tiled_refine_spike import (  # noqa: E402
    BASE_SEED,
    DEFAULT_BACKEND_URL,
    REFINE_PROMPT,
    generate_base_image,
    refine_tile,
)

# The tile is exactly the base size, which the earlier spike proved this GPU
# renders comfortably. The canvas is whatever four such tiles cover once they
# overlap, rather than a round number the tiles would not divide.
GRID_TILE_WIDTH = 832
GRID_TILE_HEIGHT = 1216
GRID_OVERLAP_PIXELS = 128
CANVAS_WIDTH = GRID_TILE_WIDTH * 2 - GRID_OVERLAP_PIXELS
CANVAS_HEIGHT = GRID_TILE_HEIGHT * 2 - GRID_OVERLAP_PIXELS

DEFAULT_REFINE_DENOISE = 0.6
DEFAULT_REFINE_STEPS = 6


def run_tiled_upscale_refine(backend_url: str, output_root: Path, run_label: str,
                             denoise: float, refine_steps: int,
                             uses_base_prompt: bool) -> int:
    output_dir = output_root / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    refine_prompt = REFINE_PROMPT if uses_base_prompt else ""
    steps_actually_spent = max(1, int(refine_steps * denoise))

    print(f"Tiled upscale refine into {CANVAS_WIDTH}x{CANVAS_HEIGHT} "
          f"from {GRID_TILE_WIDTH}x{GRID_TILE_HEIGHT} tiles overlapping "
          f"{GRID_OVERLAP_PIXELS}px", flush=True)
    print(f"  denoise {denoise}, {refine_steps} step schedule "
          f"({steps_actually_spent} actually spent), "
          f"prompt={'base' if refine_prompt else 'empty'}", flush=True)
    print(f"  writing to {output_dir}", flush=True)

    base = generate_base_image(backend_url, GRID_TILE_WIDTH, GRID_TILE_HEIGHT)
    base.save(output_dir / "00_base.png")

    resampled_canvas = base.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.LANCZOS)
    resampled_canvas.save(output_dir / "01_resampled_canvas.png")

    boxes = grid_tile_crop_boxes(CANVAS_WIDTH, CANVAS_HEIGHT,
                                 GRID_TILE_WIDTH, GRID_TILE_HEIGHT, GRID_OVERLAP_PIXELS)
    refined_tiles = []
    for index, box in enumerate(boxes):
        tile_source = resampled_canvas.crop(box)
        tile_source.save(output_dir / f"02_tile_{index}_source.png")
        refined = refine_tile(backend_url, tile_source, refine_prompt,
                              BASE_SEED + index, denoise, refine_steps,
                              f"tile {index + 1} of {len(boxes)} at {box}")
        refined.save(output_dir / f"03_tile_{index}_refined.png")
        refined_tiles.append(refined)

    # Pasting in order lets each tile overwrite the previous one's overlap, so the
    # seams the blend has to remove are visible for comparison.
    hard_pasted = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT))
    for tile, box in zip(refined_tiles, boxes):
        hard_pasted.paste(tile, (box[0], box[1]))
    hard_pasted.save(output_dir / "04_hard_pasted.png")

    blended = blend_tiles_into_canvas(refined_tiles, boxes,
                                      CANVAS_WIDTH, CANVAS_HEIGHT, GRID_OVERLAP_PIXELS)
    blended.save(output_dir / "05_blended.png")

    print(f"Done. Compare 05_blended.png against 01_resampled_canvas.png for detail "
          f"gain, and against 04_hard_pasted.png for seam removal.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--output-dir", default="tile_upscale_spike")
    parser.add_argument("--label", default="run")
    parser.add_argument("--denoise", type=float, default=DEFAULT_REFINE_DENOISE)
    parser.add_argument("--refine-steps", type=int, default=DEFAULT_REFINE_STEPS)
    parser.add_argument("--empty-prompt", action="store_true",
                        help="refine with no prompt instead of the base prompt")
    arguments = parser.parse_args()
    return run_tiled_upscale_refine(arguments.backend, Path(arguments.output_dir),
                                    arguments.label, arguments.denoise,
                                    arguments.refine_steps, not arguments.empty_prompt)


if __name__ == "__main__":
    raise SystemExit(main())
