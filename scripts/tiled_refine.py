"""Two-tile vertical refine geometry and overlap blending.

A tall image is refined as two overlapping horizontal bands rather than in one
pass, so each band is generated at a size the model handles well. The bands are
then recombined, and the only thing standing between that and a visible seam is
the weighting across the shared rows.

The geometry is deliberately fixed to one base size and one tile size. Nothing
here is generalized: the point of the spike is to learn whether a two-tile refine
holds together at all, and a general tiler built on an unproven premise would
just be more code to throw away.

Pure image maths: no backend calls, no filesystem, so every property below is
testable without spending generation time.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

BASE_WIDTH = 832
BASE_HEIGHT = 1216
TILE_HEIGHT = 672
VERTICAL_OVERLAP_PIXELS = 128


def vertical_tile_crop_boxes() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """The top and bottom crop boxes, as PIL (left, top, right, bottom).

    The two tiles are anchored to opposite edges and meet in an overlap, so
    together they cover every row of the base exactly once or twice and never
    leave a gap.
    """
    if TILE_HEIGHT * 2 - VERTICAL_OVERLAP_PIXELS != BASE_HEIGHT:
        raise ValueError(
            f"two {TILE_HEIGHT}px tiles overlapping by {VERTICAL_OVERLAP_PIXELS}px "
            f"cover {TILE_HEIGHT * 2 - VERTICAL_OVERLAP_PIXELS}px, not {BASE_HEIGHT}px")
    top_box = (0, 0, BASE_WIDTH, TILE_HEIGHT)
    bottom_box = (0, TILE_HEIGHT - VERTICAL_OVERLAP_PIXELS, BASE_WIDTH, BASE_HEIGHT)
    return top_box, bottom_box


def validated_tile_pair(top_tile: Image.Image,
                        bottom_tile: Image.Image,
                        overlap_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Both tiles as float arrays, refusing any pair that cannot be recombined.

    Shapes are compared against each other rather than against this module's
    constants, so the maths stays correct for whatever pair it is handed.
    """
    if top_tile.size != bottom_tile.size:
        raise ValueError(f"tiles differ in size: {top_tile.size} and {bottom_tile.size}")
    if overlap_pixels <= 0 or overlap_pixels > top_tile.height:
        raise ValueError(f"overlap of {overlap_pixels}px does not fit a {top_tile.height}px tile")
    return (np.asarray(top_tile.convert("RGB"), dtype=np.float32),
            np.asarray(bottom_tile.convert("RGB"), dtype=np.float32))


def blend_vertically_overlapping_tiles(top_tile: Image.Image,
                                       bottom_tile: Image.Image,
                                       overlap_pixels: int) -> Image.Image:
    """Recombine two overlapping tiles, cross-fading linearly over the shared rows.

    The ramp runs the full width of the overlap, reaching pure top at its first
    row and pure bottom at its last, so it meets each tile's exclusive region at
    that tile's own values and leaves no step at either boundary.
    """
    top_pixels, bottom_pixels = validated_tile_pair(top_tile, bottom_tile, overlap_pixels)
    tile_height, width = top_pixels.shape[0], top_pixels.shape[1]
    exclusive_top_height = tile_height - overlap_pixels

    output = np.empty((tile_height * 2 - overlap_pixels, width, 3), dtype=np.float32)
    output[:exclusive_top_height] = top_pixels[:exclusive_top_height]

    bottom_weight = np.linspace(0.0, 1.0, overlap_pixels, dtype=np.float32)[:, None, None]
    output[exclusive_top_height:tile_height] = (
        top_pixels[exclusive_top_height:] * (1.0 - bottom_weight)
        + bottom_pixels[:overlap_pixels] * bottom_weight)

    output[tile_height:] = bottom_pixels[overlap_pixels:]
    return Image.fromarray(np.clip(np.rint(output), 0, 255).astype(np.uint8), mode="RGB")


def grid_tile_crop_boxes(canvas_width: int, canvas_height: int,
                         tile_width: int, tile_height: int,
                         overlap_pixels: int) -> list[tuple[int, int, int, int]]:
    """Crop boxes for a row-major grid of equally sized overlapping tiles.

    Every tile is exactly `tile_width` by `tile_height`, because that is the size
    the model is known to handle; the canvas is therefore whatever those tiles
    cover, not an arbitrary target. Raises when the canvas is not exactly covered,
    since a partial tile would either need a different size or leave a gap.
    """
    columns = tiles_needed_to_span(canvas_width, tile_width, overlap_pixels)
    rows = tiles_needed_to_span(canvas_height, tile_height, overlap_pixels)
    return [(column * (tile_width - overlap_pixels),
             row * (tile_height - overlap_pixels),
             column * (tile_width - overlap_pixels) + tile_width,
             row * (tile_height - overlap_pixels) + tile_height)
            for row in range(rows) for column in range(columns)]


def tiles_needed_to_span(canvas_length: int, tile_length: int, overlap_pixels: int) -> int:
    """How many overlapping tiles cover a length exactly, or an error saying why not."""
    stride = tile_length - overlap_pixels
    if stride <= 0:
        raise ValueError(f"a {tile_length}px tile overlapping by {overlap_pixels}px never advances")
    if (canvas_length - tile_length) % stride != 0:
        exact_lengths = [tile_length + stride * count for count in range(6)]
        raise ValueError(
            f"{canvas_length}px is not exactly covered by {tile_length}px tiles "
            f"overlapping {overlap_pixels}px; reachable lengths are {exact_lengths}")
    return (canvas_length - tile_length) // stride + 1


def tile_feather_weights(tile_width: int, tile_height: int, overlap_pixels: int,
                         box: tuple[int, int, int, int],
                         canvas_width: int, canvas_height: int) -> np.ndarray:
    """A tile's per-pixel contribution, ramping down only on edges it shares.

    Edges lying on the canvas boundary keep full weight, since nothing overlaps
    them and fading there would darken the border. Interior edges ramp linearly
    across the overlap, so a tile and its neighbour sum to exactly one everywhere
    they meet, corners included, because the ramps are separable.
    """
    ramp = np.linspace(0.0, 1.0, overlap_pixels, dtype=np.float64)
    horizontal = np.ones(tile_width, dtype=np.float64)
    vertical = np.ones(tile_height, dtype=np.float64)
    if box[0] > 0:
        horizontal[:overlap_pixels] = ramp
    if box[2] < canvas_width:
        horizontal[-overlap_pixels:] = ramp[::-1]
    if box[1] > 0:
        vertical[:overlap_pixels] = ramp
    if box[3] < canvas_height:
        vertical[-overlap_pixels:] = ramp[::-1]
    return np.outer(vertical, horizontal)[:, :, None]


def blend_tiles_into_canvas(tiles: list[Image.Image],
                            boxes: list[tuple[int, int, int, int]],
                            canvas_width: int, canvas_height: int,
                            overlap_pixels: int) -> Image.Image:
    """Recombine a grid of overlapping tiles by weighted accumulation.

    Each tile is added under its feather weights and the total is divided by the
    accumulated weight, so the result is a true weighted average no matter how
    many tiles meet at a pixel. Dividing rather than trusting the weights to sum
    to one keeps a boundary tile correct even if its ramps were clipped.
    """
    if len(tiles) != len(boxes):
        raise ValueError(f"{len(tiles)} tiles for {len(boxes)} boxes")
    accumulated = np.zeros((canvas_height, canvas_width, 3), dtype=np.float64)
    accumulated_weight = np.zeros((canvas_height, canvas_width, 1), dtype=np.float64)

    for tile, box in zip(tiles, boxes):
        expected_size = (box[2] - box[0], box[3] - box[1])
        if tile.size != expected_size:
            raise ValueError(f"tile for box {box} is {tile.size}, expected {expected_size}")
        weights = tile_feather_weights(tile.width, tile.height, overlap_pixels, box,
                                       canvas_width, canvas_height)
        pixels = np.asarray(tile.convert("RGB"), dtype=np.float64)
        accumulated[box[1]:box[3], box[0]:box[2]] += pixels * weights
        accumulated_weight[box[1]:box[3], box[0]:box[2]] += weights

    if not np.all(accumulated_weight > 0):
        raise ValueError("the tiles leave part of the canvas uncovered")
    averaged = accumulated / accumulated_weight
    return Image.fromarray(np.clip(np.rint(averaged), 0, 255).astype(np.uint8), mode="RGB")


def join_tiles_without_blending(top_tile: Image.Image,
                                bottom_tile: Image.Image,
                                overlap_pixels: int) -> Image.Image:
    """Butt the two tiles together with no cross-fade, for diagnosis only.

    Every shared row comes from the top tile, so the seam is left as visible as
    the two passes actually made it. Comparing this against the blended result is
    how a seam caused by the tiles themselves is told apart from one caused by
    the blend.
    """
    top_pixels, bottom_pixels = validated_tile_pair(top_tile, bottom_tile, overlap_pixels)
    tile_height, width = top_pixels.shape[0], top_pixels.shape[1]

    output = np.empty((tile_height * 2 - overlap_pixels, width, 3), dtype=np.float32)
    output[:tile_height] = top_pixels
    output[tile_height:] = bottom_pixels[overlap_pixels:]
    return Image.fromarray(np.clip(np.rint(output), 0, 255).astype(np.uint8), mode="RGB")
