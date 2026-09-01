"""Sizing a source image that is sent straight to the hires stage.

Nothing is stretched. The source's own aspect is kept, scaled until its area
reaches the chosen budget, and then each side is rounded DOWN to the chosen
increment. Rounding down is what turns the fit into a small centre crop rather
than a squash, which is why the crop box is derived here too.

The budget is an area, not an edge: 1024 means 1024 x 1024 pixels' worth, so a
portrait source comes back taller than it is wide while costing the same
attention sequence as the square it replaces.
"""
from __future__ import annotations

import math

SOURCE_PIXEL_BUDGET_EDGES = [512, 768, 1024, 1216, 1536, 2024]
SOURCE_SIZE_INCREMENTS = [1, 8, 16, 32, 64]

DEFAULT_SOURCE_PIXEL_BUDGET_EDGE = 1024
# The VAE is /8, so anything below 8 leaves the runtime to round the request
# itself. 64 keeps both sides on the resolutions the model is used to.
DEFAULT_SOURCE_SIZE_INCREMENT = 64


def resolution_for_source_within_pixel_budget(source_width: int,
                                              source_height: int,
                                              pixel_budget_edge: int,
                                              size_increment: int) -> tuple[int, int]:
    """The size this source is encoded at: its own aspect, inside the budget."""
    source_width, source_height = max(1, int(source_width)), max(1, int(source_height))
    budget_pixels = max(1, int(pixel_budget_edge)) ** 2
    increment = max(1, int(size_increment))

    # The budget is a cap, never a target: enlarging here would invent pixels the
    # hires stage is there to add, so a source already inside the budget is kept.
    scale = min(1.0, math.sqrt(budget_pixels / (source_width * source_height)))
    scaled_width = source_width * scale
    scaled_height = source_height * scale

    width = max(increment, int(scaled_width // increment) * increment)
    height = max(increment, int(scaled_height // increment) * increment)
    return width, height


def cover_crop_box_for_resolution(source_width: int, source_height: int,
                                  target_width: int, target_height: int) -> tuple[int, int, int, int]:
    """The centred box to cut from the source so it fills the target without stretching.

    Returned as PIL's (left, upper, right, lower). The box always has the
    target's aspect, so resizing it afterwards only ever scales.
    """
    source_width, source_height = max(1, int(source_width)), max(1, int(source_height))
    target_width, target_height = max(1, int(target_width)), max(1, int(target_height))

    if source_width * target_height > target_width * source_height:
        # The source is wider than the target wants, so trim its sides.
        kept_width = round(source_height * target_width / target_height)
        kept_width = min(source_width, kept_width)
        left = (source_width - kept_width) // 2
        return left, 0, left + kept_width, source_height

    kept_height = round(source_width * target_height / target_width)
    kept_height = min(source_height, kept_height)
    upper = (source_height - kept_height) // 2
    return 0, upper, source_width, upper + kept_height
