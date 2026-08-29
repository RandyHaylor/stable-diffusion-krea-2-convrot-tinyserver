#!/usr/bin/env python3
"""Unit tests for WD14 tagging.

These cover the pure logic only: image downscaling, tag formatting, and prompt
composition. None of them load the tagger model, so the suite runs whether or
not the ONNX weights have been downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wd14_tagging import (  # noqa: E402
    MAX_TAGGED_PIXELS,
    downscale_image_to_pixel_budget,
    format_danbooru_tags_for_prompt,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def main() -> int:
    oversized = Image.new("RGB", (4000, 3000))
    downscaled = downscale_image_to_pixel_budget(oversized)
    check("an oversized image is downscaled within the pixel budget",
          downscaled.width * downscaled.height <= MAX_TAGGED_PIXELS,
          f"{downscaled.width}x{downscaled.height}")
    check("downscaling preserves the aspect ratio",
          abs((downscaled.width / downscaled.height) - (4000 / 3000)) < 0.01,
          f"{downscaled.width}x{downscaled.height}")

    already_small = Image.new("RGB", (512, 512))
    check("an image already within budget is returned unscaled",
          downscale_image_to_pixel_budget(already_small).size == (512, 512))
    check("an image exactly at the budget is not scaled",
          downscale_image_to_pixel_budget(Image.new("RGB", (1024, 1024))).size == (1024, 1024))

    check("underscores in tags become spaces",
          format_danbooru_tags_for_prompt(["cat_girl", "maid_headdress"])
          == "cat girl, maid headdress")
    check("tags are separated by a single comma and space",
          format_danbooru_tags_for_prompt(["a", "b", "c"]) == "a, b, c")
    check("an empty tag list formats to an empty string",
          format_danbooru_tags_for_prompt([]) == "")
    check("blank tags are dropped rather than leaving empty comma slots",
          format_danbooru_tags_for_prompt(["cat", "", "  ", "dog"]) == "cat, dog")
    check("escaped danbooru parentheses are left intact",
          format_danbooru_tags_for_prompt(["character_\\(series\\)"]) == "character \\(series\\)")

    print()
    if failures:
        print(f"{len(failures)} wd14 tagging check(s) failed")
        return 1
    print("all wd14 tagging checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
