#!/usr/bin/env python3
"""Unit tests for the hires tiling decisions.

Pure parameter reading and planning: no backend, no images. What is asserted here
is when tiling engages, how the target is divided, and that the plan it reports
matches the tiles that would actually be rendered.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hires_tiling import (  # noqa: E402
    DEFAULT_HIRES_TILE_OVERLAP_PIXELS,
    HIRES_TILE_SOURCE_MODES,
    HIRES_TILING_MODES,
    anchors_each_hires_tile_to_its_neighbours,
    describe_hires_tiling_plan,
    hires_tile_boxes_for_params,
    hires_tile_overlap_pixels,
    hires_tile_size,
    hires_tile_vision_ref_image_args,
    hires_tile_vision_weight,
    hires_tiling_hop_sizes,
    hires_upscale_factor,
    sends_each_hires_tile_to_the_vision_tower,
    MAXIMUM_TILING_HOP_FACTOR,
    recommended_maximum_hires_denoise_for_tiling,
    renders_hires_by_tiling,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def params(**overrides) -> dict:
    base = {"width": 832, "height": 1216, "hires": True,
            "hires_width": 1664, "hires_height": 2432, "hires_tiling": "on"}
    base.update(overrides)
    return base


def main() -> int:
    check("the tiling modes are the supported set",
          HIRES_TILING_MODES == ("off", "on"),
          f"got {HIRES_TILING_MODES}")

    check("tiling engages when it is on and the hires pass is enabled",
          renders_hires_by_tiling(params()))
    check("tiling is off by default, so existing jobs are untouched",
          not renders_hires_by_tiling({"width": 832, "height": 1216, "hires": True,
                                       "hires_width": 1664, "hires_height": 2432}),
          "the in-request latent hires must stay the default path")
    check("tiling never engages while the hires pass is disabled",
          not renders_hires_by_tiling(params(hires=False)),
          "there is no second pass to tile")
    check("an unrecognised tiling mode is treated as off",
          not renders_hires_by_tiling(params(hires_tiling="sometimes")))
    check("tiling does not engage when the target needs only one tile",
          not renders_hires_by_tiling(params(hires_width=832, hires_height=1216)),
          "a single tile is the plain hires pass with extra steps")

    check("the tile size defaults to the main pass resolution",
          hires_tile_size(params()) == (832, 1216),
          "the main pass already proved that size fits")
    check("the tile is never larger than the target it covers",
          hires_tile_size(params(hires_width=640, hires_height=900)) == (640, 900),
          "a tile bigger than the canvas would be cropped by the tiler")

    check("the overlap defaults to the shared constant",
          hires_tile_overlap_pixels({}) == DEFAULT_HIRES_TILE_OVERLAP_PIXELS)
    check("a chosen overlap is honoured",
          hires_tile_overlap_pixels({"hires_tile_overlap": 192}) == 192)
    check("a non-positive overlap falls back to the default",
          hires_tile_overlap_pixels({"hires_tile_overlap": 0})
          == DEFAULT_HIRES_TILE_OVERLAP_PIXELS,
          "zero overlap would leave nothing to blend across")
    check("an overlap that would leave no stride is clamped below the tile size",
          hires_tile_overlap_pixels({"hires_tile_overlap": 5000}, tile_length=832) < 832)

    boxes = hires_tile_boxes_for_params(params())
    check("an exact doubling needs three tiles per axis, not two",
          len(boxes) == 9,
          f"got {len(boxes)}: two 832px tiles cover 1664px only by touching, "
          f"which would leave no overlap to blend across")
    check("every hires tile is exactly the tile size",
          all(box[2] - box[0] == 832 and box[3] - box[1] == 1216 for box in boxes))
    check("no two hires tiles are placed on top of each other",
          len(set(boxes)) == len(boxes))
    check("a target the tile stride divides evenly needs only four tiles",
          len(hires_tile_boxes_for_params(params(hires_width=1536, hires_height=2304))) == 4,
          "choosing a stride-aligned target more than halves the passes")
    check("the hires tiles reach the exact target edges",
          max(box[2] for box in boxes) == 1664 and max(box[3] for box in boxes) == 2432,
          "the output must be the size the user asked for, not the tiles' multiple")

    wide_boxes = hires_tile_boxes_for_params(params(hires_width=2048, hires_height=2048))
    check("an awkward target is still covered exactly",
          max(box[2] for box in wide_boxes) == 2048
          and max(box[3] for box in wide_boxes) == 2048,
          f"{len(wide_boxes)} tiles")

    plan = describe_hires_tiling_plan(params())
    check("the plan reports the tile count that will actually be rendered",
          plan["tile_count"] == len(boxes),
          f"plan says {plan['tile_count']}, boxes are {len(boxes)}")
    check("the plan names the grid shape and the tile size",
          plan["columns"] == 3 and plan["rows"] == 3
          and plan["tile_width"] == 832 and plan["tile_height"] == 1216,
          f"got {plan}")
    check("the plan's grid shape multiplies out to its tile count",
          plan["columns"] * plan["rows"] == plan["tile_count"],
          "a notice promising one thing and rendering another would mislead")
    check("the plan is empty when tiling will not run",
          describe_hires_tiling_plan(params(hires_tiling="off")) == {},
          "no notice should be shown for a path that is not taken")

    check("the tile source modes are the supported set",
          HIRES_TILE_SOURCE_MODES == ("independent", "anchored"),
          f"got {HIRES_TILE_SOURCE_MODES}")
    check("tiles are anchored to their neighbours by default",
          anchors_each_hires_tile_to_its_neighbours({}),
          "cutting every tile from the same canvas is what produces ghosting")
    check("independent tile sources can be asked for",
          not anchors_each_hires_tile_to_its_neighbours(
              {"hires_tile_source": "independent"}))
    check("an unrecognised tile source falls back to the default",
          anchors_each_hires_tile_to_its_neighbours({"hires_tile_source": "guesswork"}))
    check("the plan reports whether tiles are anchored",
          describe_hires_tiling_plan(params())["anchored"] is True
          and describe_hires_tiling_plan(
              params(hires_tile_source="independent"))["anchored"] is False)

    check("the upscale factor is the larger of the two axes' growth",
          hires_upscale_factor(params()) == 2.0,
          f"got {hires_upscale_factor(params())}")
    check("a target no larger than the tile is not an upscale",
          hires_upscale_factor(params(hires_width=832, hires_height=1216)) == 1.0)
    check("a four tile grid allows the denoise it measured clean at",
          abs(recommended_maximum_hires_denoise_for_tiling(
              params(hires_width=1536, hires_height=2304)) - 0.6) < 0.01,
          "2x2 at 0.6 was measured clean")
    check("a crowded grid recommends the lower denoise it measured clean at",
          abs(recommended_maximum_hires_denoise_for_tiling(
              params(hires_width=2432, hires_height=3648)) - 0.35) < 0.01,
          "4x4 ghosted at 0.6 and was clean at 0.35")
    check("the recommendation is one of the two measured values, never interpolated",
          all(recommended_maximum_hires_denoise_for_tiling(
                  params(hires_width=width, hires_height=int(width * 1216 / 832)))
              in (0.35, 0.6)
              for width in (900, 1200, 1536, 1664, 2432, 3328, 6000)),
          "no point between them has been measured, so none is claimed")
    check("tile vision is off unless asked for",
          not sends_each_hires_tile_to_the_vision_tower({}),
          "it costs vision tokens in every tile's sequence")
    check("tile vision engages when switched on",
          sends_each_hires_tile_to_the_vision_tower({"hires_tile_vision": "on"}))
    check("an unrecognised tile vision mode is treated as off",
          not sends_each_hires_tile_to_the_vision_tower({"hires_tile_vision": "maybe"}))
    check("the tile vision weight defaults to neutral",
          hires_tile_vision_weight({}) == 1.0)
    check("a non-positive tile vision weight falls back to neutral",
          hires_tile_vision_weight({"hires_tile_vision_weight": -2}) == 1.0)
    check("a neutral tile vision weight sends no args, keeping the unweighted path",
          hires_tile_vision_ref_image_args({"hires_tile_vision": "on"}) == "")
    check("a weighted tile vision needs no positional padding, being one image",
          hires_tile_vision_ref_image_args({"hires_tile_vision": "on",
                                            "hires_tile_vision_weight": 0.5})
          == "vlm_image_token_weight=0.5",
          f"got {hires_tile_vision_ref_image_args({'hires_tile_vision': 'on', 'hires_tile_vision_weight': 0.5})!r}")
    check("a weight with tile vision off sends nothing, having no tokens to weight",
          hires_tile_vision_ref_image_args({"hires_tile_vision_weight": 0.5}) == "")

    check("more hops do not raise the recommendation",
          recommended_maximum_hires_denoise_for_tiling(
              params(hires_width=2432, hires_height=3648, hires_tiling_hops="doubling"))
          == recommended_maximum_hires_denoise_for_tiling(
              params(hires_width=2432, hires_height=3648, hires_tiling_hops="single")),
          "two hops each within a doubling still ghosted at 0.6")
    check("the plan carries the factor and the recommendation",
          describe_hires_tiling_plan(params())["upscale_factor"] == 2.0
          and describe_hires_tiling_plan(params())["recommended_maximum_denoise"] > 0)

    check("a doubling is reached in one hop, since one hop already suffices",
          hires_tiling_hop_sizes(params()) == [(1664, 2432)],
          f"got {hires_tiling_hop_sizes(params())}")
    check("single hop mode goes straight to the target however far it is",
          hires_tiling_hop_sizes(params(hires_width=2432, hires_height=3648,
                                        hires_tiling_hops="single"))
          == [(2432, 3648)])

    tripling_hops = hires_tiling_hop_sizes(params(hires_width=2432, hires_height=3648))
    check("a tripling is reached in two hops rather than one",
          len(tripling_hops) == 2,
          f"got {tripling_hops}")
    check("every hop lands on or below the hop factor",
          all(max(hop[0] / previous[0], hop[1] / previous[1])
              <= MAXIMUM_TILING_HOP_FACTOR + 0.001
              for previous, hop in zip([(832, 1216)] + tripling_hops, tripling_hops)),
          f"got {tripling_hops}")
    check("the last hop is exactly the requested target",
          tripling_hops[-1] == (2432, 3648),
          "an intermediate size must never become the output size")
    check("hops are strictly growing, so none is wasted",
          all(later[0] > earlier[0] and later[1] > earlier[1]
              for earlier, later in zip(tripling_hops, tripling_hops[1:])))

    for far_width in (2432, 3328, 5000, 8000):
        far_hops = hires_tiling_hop_sizes(params(hires_width=far_width,
                                                 hires_height=int(far_width * 1216 / 832)))
        check(f"a {far_width}px target keeps every hop within the factor",
              all(max(hop[0] / previous[0], hop[1] / previous[1])
                  <= MAXIMUM_TILING_HOP_FACTOR + 0.001
                  for previous, hop in zip([(832, 1216)] + far_hops, far_hops))
              and far_hops[-1][0] == far_width,
              f"{len(far_hops)} hop(s): {far_hops}")

    print()
    if failures:
        print(f"{len(failures)} hires tiling check(s) failed")
        return 1
    print("all hires tiling checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
