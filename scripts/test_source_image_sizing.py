#!/usr/bin/env python3
"""Unit tests for sizing a source image that is sent straight to the hires stage."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_image_sizing import (  # noqa: E402
    DEFAULT_SOURCE_PIXEL_BUDGET_EDGE,
    DEFAULT_SOURCE_SIZE_INCREMENT,
    SOURCE_PIXEL_BUDGET_EDGES,
    SOURCE_SIZE_INCREMENTS,
    cover_crop_box_for_resolution,
    resolution_for_source_within_pixel_budget,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def main() -> int:
    check("the offered budgets are the ones the UI lists",
          SOURCE_PIXEL_BUDGET_EDGES == [512, 768, 1024, 1216, 1536, 2024],
          str(SOURCE_PIXEL_BUDGET_EDGES))
    check("the offered increments are the ones the UI lists",
          SOURCE_SIZE_INCREMENTS == [1, 8, 16, 32, 64],
          str(SOURCE_SIZE_INCREMENTS))
    check("the defaults are drawn from those lists",
          DEFAULT_SOURCE_PIXEL_BUDGET_EDGE in SOURCE_PIXEL_BUDGET_EDGES
          and DEFAULT_SOURCE_SIZE_INCREMENT in SOURCE_SIZE_INCREMENTS)

    check("a portrait source keeps its aspect rather than being squared",
          resolution_for_source_within_pixel_budget(832, 1216, 1024, 64) == (832, 1216),
          str(resolution_for_source_within_pixel_budget(832, 1216, 1024, 64)))

    width, height = resolution_for_source_within_pixel_budget(832, 1216, 1024, 1)
    check("the area never exceeds the budget",
          width * height <= 1024 * 1024,
          f"got {width}x{height} = {width * height} against {1024 * 1024}")
    check("an unrounded fit stays close to the source aspect",
          abs((width / height) - (832 / 1216)) < 0.01,
          f"got {width}x{height}")

    big_width, big_height = resolution_for_source_within_pixel_budget(4000, 3000, 1024, 64)
    check("an oversized source is scaled down to fit the budget",
          big_width * big_height <= 1024 * 1024,
          f"got {big_width}x{big_height}")
    check("a scaled down source lands on the increment",
          big_width % 64 == 0 and big_height % 64 == 0,
          f"got {big_width}x{big_height}")
    check("a landscape source stays landscape",
          big_width > big_height,
          f"got {big_width}x{big_height}")

    check("rounding to the increment only ever crops, never stretches past the budget",
          all(w * h <= edge * edge
              for edge in SOURCE_PIXEL_BUDGET_EDGES
              for increment in SOURCE_SIZE_INCREMENTS
              for w, h in [resolution_for_source_within_pixel_budget(832, 1216, edge, increment)]))

    check("a source smaller than the budget is never scaled up to fill it",
          resolution_for_source_within_pixel_budget(256, 256, 1024, 64) == (256, 256),
          str(resolution_for_source_within_pixel_budget(256, 256, 1024, 64)))
    check("a source already inside a generous budget is left exactly as it is",
          resolution_for_source_within_pixel_budget(832, 1216, 2024, 64) == (832, 1216),
          str(resolution_for_source_within_pixel_budget(832, 1216, 2024, 64)))
    check("the budget only ever shrinks, so it never exceeds the source's own area",
          all(w * h <= 832 * 1216
              for edge in SOURCE_PIXEL_BUDGET_EDGES
              for w, h in [resolution_for_source_within_pixel_budget(832, 1216, edge, 64)]))

    check("a tiny budget never rounds a side down to nothing",
          all(side >= increment
              for increment in SOURCE_SIZE_INCREMENTS
              for side in resolution_for_source_within_pixel_budget(4000, 100, 512, increment)),
          str(resolution_for_source_within_pixel_budget(4000, 100, 512, 64)))

    check("a square target crops a portrait source centrally rather than squashing it",
          cover_crop_box_for_resolution(832, 1216, 512, 512) == (0, 192, 832, 1024),
          str(cover_crop_box_for_resolution(832, 1216, 512, 512)))
    check("a matching aspect needs no crop at all",
          cover_crop_box_for_resolution(832, 1216, 416, 608) == (0, 0, 832, 1216),
          str(cover_crop_box_for_resolution(832, 1216, 416, 608)))
    check("a wide target crops a square source left and right",
          cover_crop_box_for_resolution(1000, 1000, 500, 250) == (0, 250, 1000, 750),
          str(cover_crop_box_for_resolution(1000, 1000, 500, 250)))

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for description in failures:
            print(f"  - {description}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
