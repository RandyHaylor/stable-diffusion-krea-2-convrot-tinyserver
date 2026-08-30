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

# How each tile's starting pixels are obtained.
#
# 'independent' cuts every tile from the same resampled canvas, so neighbours are
# repainted with no knowledge of each other and can disagree about anything that
# crosses their shared band. Cross-fading that disagreement turns a hard seam
# into a translucent ghost rather than removing it.
#
# 'anchored' writes each finished tile back before the next is cut, so a tile's
# starting pixels already contain its neighbour's refined content wherever they
# overlap. The repaint then has agreement to preserve rather than a conflict to
# invent. It costs nothing extra: the same number of passes, run in order.
HIRES_TILE_SOURCE_MODES = ("independent", "anchored")
DEFAULT_HIRES_TILE_SOURCE_MODE = "anchored"


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


def hires_upscale_factor(params: dict) -> float:
    """How far the target stretches the first stage, by its longer growth."""
    tile_width, tile_height = hires_tile_size(params)
    if tile_width <= 0 or tile_height <= 0:
        return 1.0
    return max(int(params.get("hires_width", 0) or 0) / tile_width,
               int(params.get("hires_height", 0) or 0) / tile_height)


def recommended_maximum_hires_denoise_for_tiling(params: dict) -> float:
    """The most denoise a tiled pass takes at this factor before tiles disagree.

    Resampling further leaves a blurrier source, which gives each tile more
    freedom to invent its own version of whatever crosses a shared band. Past
    some point the neighbours disagree and the cross-fade averages both into a
    ghost, so the usable denoise falls as the factor rises.

    This is a guide from three measured points, not a law: a doubling held
    together at 0.6, a tripling ghosted at 0.6 and was clean at 0.35. Treat it as
    the value to start below rather than a boundary anything is enforced at.
    """
    factor = hires_upscale_factor(params)
    if factor <= 1.0:
        return 0.75
    return max(0.2, min(0.75, 0.6 / (factor - 1.0)))


def anchors_each_hires_tile_to_its_neighbours(params: dict) -> bool:
    """Whether a finished tile is written back before the next one is cut."""
    requested = str(params.get("hires_tile_source", DEFAULT_HIRES_TILE_SOURCE_MODE))
    if requested not in HIRES_TILE_SOURCE_MODES:
        requested = DEFAULT_HIRES_TILE_SOURCE_MODE
    return requested == "anchored"


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
        "anchored": anchors_each_hires_tile_to_its_neighbours(params),
        "upscale_factor": hires_upscale_factor(params),
        "recommended_maximum_denoise":
            recommended_maximum_hires_denoise_for_tiling(params),
    }
