#!/usr/bin/env python3
"""Unit tests for the two-tile refine geometry and overlap blending.

Pure image maths only: no backend, no GPU, no files. These pin the properties the
spike depends on before any generation time is spent, since a seam artefact is far
cheaper to find here than in a rendered image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiled_refine import (  # noqa: E402
    BASE_HEIGHT,
    BASE_WIDTH,
    TILE_HEIGHT,
    VERTICAL_OVERLAP_PIXELS,
    blend_tiles_into_canvas,
    blend_vertically_overlapping_tiles,
    grid_tile_crop_boxes,
    join_tiles_without_blending,
    tiles_needed_to_span,
    vertical_tile_crop_boxes,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def solid_tile(value: int, width: int = BASE_WIDTH, height: int = TILE_HEIGHT) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8), mode="RGB")


def vertical_gradient_image(width: int, height: int) -> Image.Image:
    """An image whose every row differs, so any row misplacement is detectable."""
    rows = (np.arange(height, dtype=np.float32) * 255.0 / max(1, height - 1))
    pixels = np.repeat(rows[:, None], width, axis=1)
    return Image.fromarray(np.stack([pixels] * 3, axis=2).astype(np.uint8), mode="RGB")


def main() -> int:
    check("the two tiles exactly cover the base height",
          TILE_HEIGHT * 2 - VERTICAL_OVERLAP_PIXELS == BASE_HEIGHT,
          f"{TILE_HEIGHT}*2-{VERTICAL_OVERLAP_PIXELS} covers "
          f"{TILE_HEIGHT * 2 - VERTICAL_OVERLAP_PIXELS} of {BASE_HEIGHT}")

    top_box, bottom_box = vertical_tile_crop_boxes()
    check("the top tile starts at the top edge",
          top_box == (0, 0, BASE_WIDTH, TILE_HEIGHT),
          f"got {top_box}")
    check("the bottom tile ends at the bottom edge",
          bottom_box == (0, TILE_HEIGHT - VERTICAL_OVERLAP_PIXELS, BASE_WIDTH, BASE_HEIGHT),
          f"got {bottom_box}")
    check("both crop boxes are exactly one tile tall",
          top_box[3] - top_box[1] == TILE_HEIGHT
          and bottom_box[3] - bottom_box[1] == TILE_HEIGHT,
          f"top={top_box[3] - top_box[1]} bottom={bottom_box[3] - bottom_box[1]}")
    check("the crop boxes overlap by exactly the overlap height",
          top_box[3] - bottom_box[1] == VERTICAL_OVERLAP_PIXELS,
          f"got {top_box[3] - bottom_box[1]}")

    blended = blend_vertically_overlapping_tiles(solid_tile(10), solid_tile(200),
                                                 VERTICAL_OVERLAP_PIXELS)
    check("the blended result is exactly the base size",
          blended.size == (BASE_WIDTH, BASE_HEIGHT),
          f"got {blended.size}")

    blended_pixels = np.asarray(blended, dtype=np.int16)
    top_only_height = TILE_HEIGHT - VERTICAL_OVERLAP_PIXELS
    check("the top-only region carries the top tile untouched",
          bool(np.all(blended_pixels[:top_only_height] == 10)),
          "a blend must not disturb rows only one tile covers")
    check("the bottom-only region carries the bottom tile untouched",
          bool(np.all(blended_pixels[TILE_HEIGHT:] == 200)))
    check("the overlap starts at the top tile's value",
          int(blended_pixels[top_only_height, 0, 0]) == 10,
          f"got {int(blended_pixels[top_only_height, 0, 0])}")
    check("the overlap ends at the bottom tile's value",
          int(blended_pixels[TILE_HEIGHT - 1, 0, 0]) == 200,
          f"got {int(blended_pixels[TILE_HEIGHT - 1, 0, 0])}")
    check("the overlap is monotonic from the top value to the bottom value",
          bool(np.all(np.diff(blended_pixels[top_only_height:TILE_HEIGHT, 0, 0]) >= 0)),
          "a non-monotonic ramp would read as banding across the seam")

    # The decisive property: refining two tiles that come back unchanged must
    # reassemble the original exactly, so any seam the blend itself introduces
    # shows up here rather than in a rendered image.
    base = vertical_gradient_image(BASE_WIDTH, BASE_HEIGHT)
    untouched_top = base.crop(top_box)
    untouched_bottom = base.crop(bottom_box)
    reassembled = blend_vertically_overlapping_tiles(untouched_top, untouched_bottom,
                                                    VERTICAL_OVERLAP_PIXELS)
    largest_difference = int(np.max(np.abs(np.asarray(reassembled, dtype=np.int16)
                                          - np.asarray(base, dtype=np.int16))))
    check("blending unmodified crops reproduces the base within rounding",
          largest_difference <= 1,
          f"largest per-channel difference was {largest_difference}")

    hard_joined = join_tiles_without_blending(untouched_top, untouched_bottom,
                                             VERTICAL_OVERLAP_PIXELS)
    check("the diagnostic hard join is also exactly the base size",
          hard_joined.size == (BASE_WIDTH, BASE_HEIGHT),
          f"got {hard_joined.size}")
    check("a hard join of unmodified crops reproduces the base exactly",
          bool(np.array_equal(np.asarray(hard_joined), np.asarray(base))),
          "the hard join does no arithmetic, so it must be lossless here")

    mismatched_heights_rejected = False
    try:
        blend_vertically_overlapping_tiles(solid_tile(10),
                                          solid_tile(200, height=TILE_HEIGHT - 8),
                                          VERTICAL_OVERLAP_PIXELS)
    except ValueError:
        mismatched_heights_rejected = True
    check("tiles of differing heights are refused rather than silently cropped",
          mismatched_heights_rejected)

    mismatched_widths_rejected = False
    try:
        blend_vertically_overlapping_tiles(solid_tile(10),
                                          solid_tile(200, width=BASE_WIDTH - 8),
                                          VERTICAL_OVERLAP_PIXELS)
    except ValueError:
        mismatched_widths_rejected = True
    check("tiles of differing widths are refused",
          mismatched_widths_rejected)

    oversized_overlap_rejected = False
    try:
        blend_vertically_overlapping_tiles(solid_tile(10), solid_tile(200),
                                           TILE_HEIGHT + 1)
    except ValueError:
        oversized_overlap_rejected = True
    check("an overlap taller than the tiles is refused",
          oversized_overlap_rejected,
          "this would index outside the arrays rather than fail cleanly")

    # --- 2x2 grid path, used for refining beyond a single pass's reach ---
    grid_tile_width, grid_tile_height, grid_overlap = 832, 1216, 128
    grid_canvas_width = grid_tile_width * 2 - grid_overlap
    grid_canvas_height = grid_tile_height * 2 - grid_overlap

    check("two overlapping tiles are what span a double-minus-overlap length",
          tiles_needed_to_span(grid_canvas_width, grid_tile_width, grid_overlap) == 2
          and tiles_needed_to_span(grid_canvas_height, grid_tile_height, grid_overlap) == 2)

    uncoverable_length_rejected = False
    try:
        tiles_needed_to_span(grid_tile_width * 2, grid_tile_width, grid_overlap)
    except ValueError:
        uncoverable_length_rejected = True
    check("a length the tiles cannot cover exactly is refused with the reachable ones",
          uncoverable_length_rejected,
          "silently leaving a gap or a short tile would be worse than failing")

    grid_boxes = grid_tile_crop_boxes(grid_canvas_width, grid_canvas_height,
                                      grid_tile_width, grid_tile_height, grid_overlap)
    check("a 2x2 grid produces four boxes in row-major order",
          len(grid_boxes) == 4 and grid_boxes[0][:2] == (0, 0)
          and grid_boxes[1][1] == 0 and grid_boxes[2][0] == 0,
          f"got {grid_boxes}")
    check("every grid tile is exactly the requested size",
          all(box[2] - box[0] == grid_tile_width and box[3] - box[1] == grid_tile_height
              for box in grid_boxes))
    check("the grid tiles reach both far edges of the canvas",
          max(box[2] for box in grid_boxes) == grid_canvas_width
          and max(box[3] for box in grid_boxes) == grid_canvas_height)

    # The same decisive property as the two-tile case, now in two axes: tiles that
    # come back unchanged must reassemble the canvas they were cut from.
    grid_canvas = vertical_gradient_image(grid_canvas_width, grid_canvas_height)
    grid_canvas_pixels = np.asarray(grid_canvas, dtype=np.int16)
    unchanged_tiles = [grid_canvas.crop(box) for box in grid_boxes]
    reassembled_grid = blend_tiles_into_canvas(unchanged_tiles, grid_boxes,
                                               grid_canvas_width, grid_canvas_height,
                                               grid_overlap)
    check("the reassembled grid is exactly the canvas size",
          reassembled_grid.size == (grid_canvas_width, grid_canvas_height),
          f"got {reassembled_grid.size}")
    grid_difference = int(np.max(np.abs(np.asarray(reassembled_grid, dtype=np.int16)
                                       - grid_canvas_pixels)))
    check("blending unmodified grid tiles reproduces the canvas within rounding",
          grid_difference <= 1,
          f"largest per-channel difference was {grid_difference}")

    # A horizontal gradient catches a weight error in the other axis, which a
    # vertical-only test cannot see.
    horizontal_canvas = vertical_gradient_image(grid_canvas_height, grid_canvas_width).transpose(
        Image.ROTATE_90)
    horizontal_tiles = [horizontal_canvas.crop(box) for box in grid_boxes]
    reassembled_horizontal = blend_tiles_into_canvas(horizontal_tiles, grid_boxes,
                                                    grid_canvas_width, grid_canvas_height,
                                                    grid_overlap)
    horizontal_difference = int(np.max(np.abs(
        np.asarray(reassembled_horizontal, dtype=np.int16)
        - np.asarray(horizontal_canvas, dtype=np.int16))))
    check("blending is correct across the horizontal axis too",
          horizontal_difference <= 1,
          f"largest per-channel difference was {horizontal_difference}")

    mismatched_tile_rejected = False
    try:
        blend_tiles_into_canvas([solid_tile(10, width=8, height=8)] + unchanged_tiles[1:],
                                grid_boxes, grid_canvas_width, grid_canvas_height, grid_overlap)
    except ValueError:
        mismatched_tile_rejected = True
    check("a tile that does not match its box is refused",
          mismatched_tile_rejected)

    tile_count_mismatch_rejected = False
    try:
        blend_tiles_into_canvas(unchanged_tiles[:3], grid_boxes,
                                grid_canvas_width, grid_canvas_height, grid_overlap)
    except ValueError:
        tile_count_mismatch_rejected = True
    check("fewer tiles than boxes is refused rather than leaving a hole",
          tile_count_mismatch_rejected)

    print()
    if failures:
        print(f"{len(failures)} tiled refine check(s) failed")
        return 1
    print("all tiled refine checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
