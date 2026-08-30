"""Deciding whether and how the hires stage renders as tiles.

The in-request hires pass upscales in latent space and never releases the first
stage's latent, which is why it holds detail. It is also bounded by what one
attention sequence fits in VRAM, so a large enough target cannot run that way at
all.

Tiling trades that latent continuity for reach: the first stage is decoded,
resampled up to the target, and repainted as overlapping tiles, each the size of
a pass already known to fit. Resampling supplies the pixels and the repaint puts
detail into them. It costs a VAE round trip and one request per tile, so it is
off by default and worth choosing only when the target is out of reach.
"""
from __future__ import annotations

from tiled_refine import covering_grid_tile_boxes, tile_start_positions_covering_length

HIRES_TILING_MODES = ("off", "on")
DEFAULT_HIRES_TILING_MODE = "off"
DEFAULT_HIRES_TILE_OVERLAP_PIXELS = 128


def hires_tile_size(params: dict) -> tuple[int, int]:
    """The tile the hires stage repaints in, defaulting to the main pass's size.

    The main pass has already rendered at that size in this job, so it is the one
    size known to fit. A target smaller than the tile in either axis shrinks the
    tile to it, since a tile larger than the canvas cannot be cropped from it.
    """
    tile_width = min(int(params.get("width", 0) or 0),
                     int(params.get("hires_width", 0) or 0))
    tile_height = min(int(params.get("height", 0) or 0),
                      int(params.get("hires_height", 0) or 0))
    return tile_width, tile_height


def hires_tile_overlap_pixels(params: dict, tile_length: int = 0) -> int:
    """How far neighbouring tiles share, in pixels of the hires target.

    The overlap is what the blend cross-fades across, so a non-positive value
    would leave a hard seam and is read as the default instead. When a tile length
    is given, the overlap is held below it so the tiling always advances.
    """
    try:
        overlap = int(params.get("hires_tile_overlap", DEFAULT_HIRES_TILE_OVERLAP_PIXELS) or 0)
    except (TypeError, ValueError):
        overlap = DEFAULT_HIRES_TILE_OVERLAP_PIXELS
    if overlap <= 0:
        overlap = DEFAULT_HIRES_TILE_OVERLAP_PIXELS
    if tile_length > 0:
        overlap = min(overlap, max(1, tile_length // 2))
    return overlap


def hires_tile_boxes_for_params(params: dict) -> list[tuple[int, int, int, int]]:
    """Crop boxes covering the hires target, in the order they will be rendered."""
    tile_width, tile_height = hires_tile_size(params)
    if tile_width <= 0 or tile_height <= 0:
        return []
    target_width = int(params.get("hires_width", 0) or 0)
    target_height = int(params.get("hires_height", 0) or 0)
    overlap = min(hires_tile_overlap_pixels(params, tile_width),
                  hires_tile_overlap_pixels(params, tile_height))
    return covering_grid_tile_boxes(target_width, target_height,
                                    tile_width, tile_height, overlap)


def renders_hires_by_tiling(params: dict) -> bool:
    """Whether this job's hires stage runs as tiles rather than in the request.

    A target needing a single tile is left to the ordinary hires path: tiling it
    would add a VAE round trip and buy nothing.
    """
    if not params.get("hires"):
        return False
    if str(params.get("hires_tiling", DEFAULT_HIRES_TILING_MODE)) != "on":
        return False
    return len(hires_tile_boxes_for_params(params)) > 1


def describe_hires_tiling_plan(params: dict) -> dict:
    """The grid the hires stage will render, or nothing when it will not tile."""
    if not renders_hires_by_tiling(params):
        return {}
    tile_width, tile_height = hires_tile_size(params)
    overlap = min(hires_tile_overlap_pixels(params, tile_width),
                  hires_tile_overlap_pixels(params, tile_height))
    return {
        "tile_count": len(hires_tile_boxes_for_params(params)),
        "columns": len(tile_start_positions_covering_length(
            int(params["hires_width"]), tile_width, overlap)),
        "rows": len(tile_start_positions_covering_length(
            int(params["hires_height"]), tile_height, overlap)),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "overlap_pixels": overlap,
    }
