#!/usr/bin/env python3
"""Unit tests for the tiled diffusion sample args the UI settings turn into."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiled_diffusion import (  # noqa: E402
    LATENT_SCALE,
    DEFAULT_TILED_DIFFUSION_ROPE_OFFSET_MODE,
    TILED_DIFFUSION_GRIDS,
    actual_overlap_between_tiles,
    describe_tiled_diffusion_settings,
    largest_canvas_the_request_renders,
    renders_by_tiled_diffusion,
    tile_columns_and_rows,
    tile_size_covering_length,
    tile_start_positions_covering_length,
    tiled_diffusion_sample_args,
    describe_which_stages_run,
    stage_that_tiled_diffusion_applies_to,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def settings(**overrides) -> dict:
    params = {
        "tiled_diffusion": "on",
        "tiled_diffusion_grid": "2x2",
        "tiled_diffusion_overlap": 128,
        "tiled_diffusion_rope_offset": "off",
    }
    params.update(overrides)
    return params


def main() -> int:
    check("the latent scale matches the VAE's factor of eight", LATENT_SCALE == 8)
    check("the offered grids are the ones the UI lists",
          TILED_DIFFUSION_GRIDS == ["1x1", "1x2", "2x1", "2x2", "2x3", "3x2", "3x3"],
          str(TILED_DIFFUSION_GRIDS))

    check("a grid reads as columns then rows",
          tile_columns_and_rows(settings(tiled_diffusion_grid="2x3")) == (2, 3))
    check("an unparseable grid falls back to a single tile rather than raising",
          tile_columns_and_rows(settings(tiled_diffusion_grid="nonsense")) == (1, 1))

    check("the hires target is the canvas when the hires pass is on",
          largest_canvas_the_request_renders(
              {"width": 1024, "height": 1024, "hires": True,
               "hires_width": 1248, "hires_height": 1824}) == (1248, 1824))
    check("the main size is the canvas when the hires pass is off",
          largest_canvas_the_request_renders(
              {"width": 1024, "height": 1024, "hires": False,
               "hires_width": 2048, "hires_height": 2048}) == (1024, 1024))

    # Two tiles covering 1248 with 128 of overlap: (1248 + 128) / 2 = 688.
    check("a tile size is derived from the grid, the overlap and the length",
          tile_size_covering_length(1248, 2, 128) == 688,
          str(tile_size_covering_length(1248, 2, 128)))
    check("the derived tile lands on a whole latent cell",
          tile_size_covering_length(1248, 2, 128) % LATENT_SCALE == 0)
    check("three tiles across an exact doubling divide it evenly",
          tile_size_covering_length(2048, 3, 128) == 768,
          str(tile_size_covering_length(2048, 3, 128)))
    check("a single tile is the whole length",
          tile_size_covering_length(1824, 1, 128) == 1824)

    check("the derived tiles reproduce the requested overlap",
          actual_overlap_between_tiles(1248, 688, 2) == 128,
          str(actual_overlap_between_tiles(1248, 688, 2)))
    check("a single tile has no overlap to report",
          actual_overlap_between_tiles(1824, 1824, 1) == 0)

    check("tiled diffusion is off unless the setting asks for it",
          not renders_by_tiled_diffusion(settings(tiled_diffusion="off"), 2048, 2048))
    check("an absent setting is off rather than an error",
          not renders_by_tiled_diffusion({}, 2048, 2048))
    check("a 1x1 grid asks for no tiling at all",
          not renders_by_tiled_diffusion(settings(tiled_diffusion_grid="1x1"), 2048, 2048))
    check("a grid with more than one tile tiles",
          renders_by_tiled_diffusion(settings(tiled_diffusion_grid="1x2"), 1024, 2048))

    check("no sample args are emitted when tiled diffusion is off",
          tiled_diffusion_sample_args(settings(tiled_diffusion="off"), 1248, 1824) == "")
    check("no sample args are emitted for a single tile",
          tiled_diffusion_sample_args(settings(tiled_diffusion_grid="1x1"), 1248, 1824) == "")

    # 1248x1824 at 2x2 with 128 overlap: 688x976 in pixels, 86x122 in latents.
    check("each axis carries its own tile size, in latent units",
          tiled_diffusion_sample_args(settings(), 1248, 1824)
          == ("tiled_diffusion_tile_width=86,tiled_diffusion_tile_height=122"
              ",tiled_diffusion_overlap=16,tiled_diffusion_rope_offset=0"),
          tiled_diffusion_sample_args(settings(), 1248, 1824))
    check("rope offsets are requested only when the setting is on",
          tiled_diffusion_sample_args(settings(tiled_diffusion_rope_offset="on"), 1248, 1824)
          .endswith("tiled_diffusion_rope_offset=1"))
    check("rope offsets are on when the job says nothing about them",
          tiled_diffusion_sample_args({"tiled_diffusion": "on", "tiled_diffusion_grid": "2x2"},
                                      1248, 1824).endswith("tiled_diffusion_rope_offset=1"),
          f"default is {DEFAULT_TILED_DIFFUSION_ROPE_OFFSET_MODE}")

    zero_overlap_args = tiled_diffusion_sample_args(settings(tiled_diffusion_overlap=0), 1248, 1824)
    check("an overlap below the minimum still leaves a latent cell to blend across",
          int(zero_overlap_args.split("tiled_diffusion_overlap=")[1].split(",")[0]) >= 1,
          zero_overlap_args)
    check("an overlap that would swallow a tile is clamped rather than refused",
          tiled_diffusion_sample_args(settings(tiled_diffusion_overlap=4096), 1248, 1824) != "")

    check("an exact doubling needs three tiles per axis to overlap at all",
          len(tile_start_positions_covering_length(2048, 1024, 0)) == 2
          and len(tile_start_positions_covering_length(2048, 768, 128)) == 3)

    description = describe_tiled_diffusion_settings(settings(), 1248, 1824)
    check("the description names the grid", "2 x 2" in description, description)
    check("the description reports the derived tile size, not a setting",
          "688x976" in description, description)
    check("the description reports the overlap that actually results",
          "128px" in description, description)
    check("the description is empty when tiled diffusion will not engage",
          describe_tiled_diffusion_settings(settings(tiled_diffusion="off"), 1248, 1824) == "")
    check("a single tile says so rather than reporting a grid",
          "single tile" in describe_tiled_diffusion_settings(
              settings(tiled_diffusion_grid="1x1"), 1248, 1824),
          describe_tiled_diffusion_settings(settings(tiled_diffusion_grid="1x1"), 1248, 1824))

    # Tiling the primary stage under a hires stage spends model evaluations on a
    # pass whose output the hires stage resamples anyway.
    check("the hires stage is the only one tiled when it runs",
          stage_that_tiled_diffusion_applies_to(settings(hires=True)) == "hires")
    check("the primary stage tiles when it is the only stage",
          stage_that_tiled_diffusion_applies_to(settings(hires=False)) == "primary")
    check("no stage tiles when tiled diffusion is off",
          stage_that_tiled_diffusion_applies_to(
              settings(tiled_diffusion="off", hires=True)) == "")
    check("a single tile grid is not tiling any stage",
          stage_that_tiled_diffusion_applies_to(
              settings(tiled_diffusion_grid="1x1", hires=False)) == "")

    check("both stages are reported active for a hires job",
          describe_which_stages_run(settings(hires=True))
          == "primary stage: active, hires stage: active",
          describe_which_stages_run(settings(hires=True)))
    check("the hires stage is reported disabled when it is off",
          describe_which_stages_run(settings(hires=False))
          == "primary stage: active, hires stage: disabled",
          describe_which_stages_run(settings(hires=False)))
    check("the primary stage is reported skipped when the source replaces it",
          describe_which_stages_run(settings(hires=True,
                                             img2img_source_replaces_first_stage=True,
                                             source_image="a.png"))
          == "primary stage: skipped, hires stage: active",
          describe_which_stages_run(settings(hires=True,
                                             img2img_source_replaces_first_stage=True,
                                             source_image="a.png")))

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for failed in failures:
            print(f"  - {failed}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
