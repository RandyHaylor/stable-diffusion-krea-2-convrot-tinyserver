"""Turning the UI's tiled diffusion settings into runtime sample args.

The runtime splits the latent into overlapping tiles, denoises each with the
real model, and fuses them under a raised cosine weight every step. Because the
fusion happens per step rather than per finished tile, neighbouring tiles cannot
drift apart, which is the failure a pixel blend of independently refined tiles
cannot repair.

The GRID is what the UI asks for, not a tile size. Tile count is the variable the
ghosting guidance was measured against, and it is what decides how many model
evaluations a step costs, so it is the number worth choosing. The tile size is
then derived per axis from the canvas, which keeps a portrait target's tiles
portrait instead of forcing square tiles onto it and wasting most of their area
on redundant overlap.

One request renders the main pass and then the hires pass, and both read the same
sample args. The runtime denoises a pass whole when it fits inside a single tile,
so a small main pass under a large hires target tiles only where it has to. That
is why the args are decided by the LARGEST canvas the request will reach.

The UI works in pixels because the rest of the app does. The runtime works in the
latent, so every size crosses this module and is divided by LATENT_SCALE.
"""
from __future__ import annotations

LATENT_SCALE = 8

TILED_DIFFUSION_GRIDS = ["1x1", "1x2", "2x1", "2x2", "2x3", "3x2", "3x3"]

DEFAULT_TILED_DIFFUSION_MODE = "off"
DEFAULT_TILED_DIFFUSION_GRID = "2x2"
# Labelling each tile with its true canvas position measured slightly worse for
# coherence than letting every tile claim the origin.
DEFAULT_TILED_DIFFUSION_ROPE_OFFSET_MODE = "off"
DEFAULT_TILED_DIFFUSION_OVERLAP_PIXELS = 128

# Below one latent cell of overlap the fusion has nothing to blend across, so a
# tile boundary becomes a hard edge again.
MINIMUM_TILED_DIFFUSION_OVERLAP_PIXELS = LATENT_SCALE


def tile_columns_and_rows(p: dict) -> tuple[int, int]:
    """The grid this job asks for, as columns and rows."""
    requested = str(p.get("tiled_diffusion_grid", DEFAULT_TILED_DIFFUSION_GRID))
    columns, _, rows = requested.partition("x")
    try:
        return max(1, int(columns)), max(1, int(rows))
    except ValueError:
        return 1, 1


def _overlap_pixels(p: dict) -> int:
    try:
        requested = int(p.get("tiled_diffusion_overlap",
                              DEFAULT_TILED_DIFFUSION_OVERLAP_PIXELS) or 0)
    except (TypeError, ValueError):
        requested = DEFAULT_TILED_DIFFUSION_OVERLAP_PIXELS
    return max(MINIMUM_TILED_DIFFUSION_OVERLAP_PIXELS, requested)


def tile_size_covering_length(length: int, tile_count: int, overlap: int) -> int:
    """The tile edge that covers one axis with this many tiles and this overlap.

    `count` tiles overlapping by `overlap` span `count * size - (count - 1) *
    overlap`, so solving for the size that spans exactly `length` gives this.
    Rounded UP to a whole latent cell, since covering the canvas matters more
    than landing on the requested overlap exactly.
    """
    length, tile_count = max(1, int(length)), max(1, int(tile_count))
    if tile_count == 1:
        return length
    overlap = max(0, min(int(overlap), length // tile_count))
    exact = (length + overlap * (tile_count - 1)) / tile_count
    rounded = int(-(-exact // LATENT_SCALE)) * LATENT_SCALE
    return min(length, max(LATENT_SCALE, rounded))


def actual_overlap_between_tiles(length: int, tile_size: int, tile_count: int) -> int:
    """How far neighbours really share once the tiles are spread across the axis.

    The requested overlap is a target, not a guarantee: rounding the tile to a
    whole latent cell moves it, so this reports what the canvas ends up with.
    """
    if int(tile_count) < 2:
        return 0
    stride = (int(length) - int(tile_size)) / (int(tile_count) - 1)
    return max(0, int(round(int(tile_size) - stride)))


def largest_canvas_the_request_renders(p: dict) -> tuple[int, int]:
    """The biggest size this request will sample at, across both of its passes."""
    width, height = int(p.get("width", 0)), int(p.get("height", 0))
    if not p.get("hires"):
        return width, height
    return (max(width, int(p.get("hires_width", 0))),
            max(height, int(p.get("hires_height", 0))))


def renders_by_tiled_diffusion(p: dict, canvas_width: int, canvas_height: int) -> bool:
    """Whether this canvas will actually be split into tiles."""
    if str(p.get("tiled_diffusion", DEFAULT_TILED_DIFFUSION_MODE)) != "on":
        return False
    columns, rows = tile_columns_and_rows(p)
    return columns * rows > 1 and int(canvas_width) > 0 and int(canvas_height) > 0


def tile_start_positions_covering_length(length: int, tile_size: int, overlap: int) -> list[int]:
    """Where each tile begins along one axis, mirroring the runtime's spread.

    The count comes from the stride, but the positions are then spread evenly
    across the length instead of marching at that stride, so the overlap is
    uniform. An exact doubling therefore needs three tiles per axis: two would
    only abut, leaving the join nothing to blend across.
    """
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    count = (length - tile_size + stride - 1) // stride + 1
    return [0 if count == 1 else (index * (length - tile_size)) // (count - 1)
            for index in range(count)]


def tile_sizes_for_canvas(p: dict, canvas_width: int, canvas_height: int) -> tuple[int, int]:
    """The per-axis tile size this grid and overlap produce on this canvas."""
    columns, rows = tile_columns_and_rows(p)
    overlap = _overlap_pixels(p)
    return (tile_size_covering_length(canvas_width, columns, overlap),
            tile_size_covering_length(canvas_height, rows, overlap))


def tiled_diffusion_sample_args(p: dict, canvas_width: int, canvas_height: int) -> str:
    """The sample args carrying this job's tiling to the runtime, in latent units."""
    if not renders_by_tiled_diffusion(p, canvas_width, canvas_height):
        return ""
    tile_width, tile_height = tile_sizes_for_canvas(p, canvas_width, canvas_height)
    columns, rows = tile_columns_and_rows(p)
    overlap = min(actual_overlap_between_tiles(canvas_width, tile_width, columns),
                  actual_overlap_between_tiles(canvas_height, tile_height, rows))
    if overlap <= 0:
        overlap = _overlap_pixels(p)
    rope_offset = 1 if str(p.get("tiled_diffusion_rope_offset",
                                 DEFAULT_TILED_DIFFUSION_ROPE_OFFSET_MODE)) == "on" else 0
    return (f"tiled_diffusion_tile_width={tile_width // LATENT_SCALE}"
            f",tiled_diffusion_tile_height={tile_height // LATENT_SCALE}"
            f",tiled_diffusion_overlap={max(1, overlap // LATENT_SCALE)}"
            f",tiled_diffusion_rope_offset={rope_offset}")


def describe_tiled_diffusion_settings(p: dict, canvas_width: int, canvas_height: int) -> str:
    """The UI notice saying what the current settings will actually do."""
    if str(p.get("tiled_diffusion", DEFAULT_TILED_DIFFUSION_MODE)) != "on":
        return ""
    columns, rows = tile_columns_and_rows(p)
    if columns * rows <= 1:
        return (f"A 1x1 grid denoises {canvas_width}x{canvas_height} as a single tile, "
                f"so tiled diffusion changes nothing.")
    tile_width, tile_height = tile_sizes_for_canvas(p, canvas_width, canvas_height)
    overlap_across = actual_overlap_between_tiles(canvas_width, tile_width, columns)
    overlap_down = actual_overlap_between_tiles(canvas_height, tile_height, rows)
    return (f"{columns} x {rows} tiles of {tile_width}x{tile_height} covering "
            f"{canvas_width}x{canvas_height}, overlapping {overlap_across}px across and "
            f"{overlap_down}px down, fused every step inside one request. "
            f"Each step costs {columns * rows} model evaluations.")
