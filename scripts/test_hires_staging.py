#!/usr/bin/env python3
"""Unit tests for how the hires stage is staged against the main stage."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hires_staging import (
    renders_hires_from_existing_source,  # noqa: E402
    hires_settings_vary_from_main,
    lora_selections_differ,
    select_loras_for_stage,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def main() -> int:
    both_stages = {"filename": "turbo.safetensors", "strength": 1.0,
                   "use_in_main": True, "use_in_hires": True}
    main_only = {"filename": "identity.safetensors", "strength": 0.8,
                 "use_in_main": True, "use_in_hires": False}
    hires_only = {"filename": "detail.safetensors", "strength": 0.6,
                  "use_in_main": False, "use_in_hires": True}
    every_lora = [both_stages, main_only, hires_only]

    check("the main stage uses only the loras ticked for main",
          [lora["path"] for lora in select_loras_for_stage(every_lora, "main")]
          == ["turbo.safetensors", "identity.safetensors"])
    check("the hires stage uses only the loras ticked for hires",
          [lora["path"] for lora in select_loras_for_stage(every_lora, "hires")]
          == ["turbo.safetensors", "detail.safetensors"])
    check("a lora can be applied to the hires stage alone",
          [lora["path"] for lora in select_loras_for_stage([hires_only], "main")] == []
          and [lora["path"] for lora in select_loras_for_stage([hires_only], "hires")]
          == ["detail.safetensors"],
          "main-only, hires-only and both must all be expressible")
    check("strengths are carried through to the backend shape",
          select_loras_for_stage([main_only], "main")[0]
          == {"path": "identity.safetensors", "multiplier": 0.8, "is_high_noise": False})
    check("a lora ticked for neither stage is used by neither",
          select_loras_for_stage([{"filename": "a.safetensors", "strength": 1.0}], "main") == []
          and select_loras_for_stage([{"filename": "a.safetensors", "strength": 1.0}], "hires") == [])

    check("identical selections do not differ",
          not lora_selections_differ(select_loras_for_stage([both_stages], "main"),
                                     select_loras_for_stage([both_stages], "hires")))
    check("a lora dropped from the hires stage makes the selections differ",
          lora_selections_differ(select_loras_for_stage([both_stages, main_only], "main"),
                                 select_loras_for_stage([both_stages, main_only], "hires")))
    check("two empty selections do not differ",
          not lora_selections_differ([], []))
    check("the same loras in a different order do not differ",
          not lora_selections_differ(
              [{"path": "a", "multiplier": 1.0, "is_high_noise": False},
               {"path": "b", "multiplier": 0.5, "is_high_noise": False}],
              [{"path": "b", "multiplier": 0.5, "is_high_noise": False},
               {"path": "a", "multiplier": 1.0, "is_high_noise": False}]),
          "ordering is not meaningful to the backend, so it must not force a reload")
    check("the same lora at a different strength does differ",
          lora_selections_differ([{"path": "a", "multiplier": 1.0, "is_high_noise": False}],
                                 [{"path": "a", "multiplier": 0.5, "is_high_noise": False}]))

    check("matching selections keep the native single-request hires path",
          not hires_settings_vary_from_main(True, [both_stages]),
          "the native path stays in latent space and must remain the default")
    check("differing lora selections make the hires settings vary",
          hires_settings_vary_from_main(True, [both_stages, main_only]))
    check("a disabled hires pass never varies",
          not hires_settings_vary_from_main(False, [both_stages, main_only]))
    check("no loras at all keeps the native path",
          not hires_settings_vary_from_main(True, []))

    check("a hires prompt override makes the settings vary",
          hires_settings_vary_from_main(True, [both_stages], hires_prompt="sharp focus"),
          "one request carries one prompt, so a per-stage prompt forces its own request")
    check("a hires negative prompt override makes the settings vary",
          hires_settings_vary_from_main(True, [both_stages], hires_negative_prompt="blurry"))
    check("hires tags that differ from the main stage's make the settings vary",
          hires_settings_vary_from_main(True, [both_stages],
                                        main_tag_groups=["1girl"], hires_tag_groups=["1boy"]),
          "the two stages would need different prompts in one request")
    check("identical tag groups on both stages keep the native path",
          not hires_settings_vary_from_main(True, [both_stages],
                                            main_tag_groups=["1girl"], hires_tag_groups=["1girl"]),
          "the same prompt serves both stages, so no second request is needed")
    check("tags on the hires stage alone make the settings vary",
          hires_settings_vary_from_main(True, [both_stages], hires_tag_groups=["1boy"]))
    check("tags on the main stage alone make the settings vary",
          hires_settings_vary_from_main(True, [both_stages], main_tag_groups=["1girl"]))
    check("using the first stage's own tags makes the settings vary",
          hires_settings_vary_from_main(True, [both_stages], stage_one_tag_mode="append"),
          "those tags are only known after the first stage has run")
    check("not using the first stage's tags keeps the native path",
          not hires_settings_vary_from_main(True, [both_stages], stage_one_tag_mode="not_used"))
    check("a hires reference encode size makes the settings vary",
          hires_settings_vary_from_main(True, [both_stages], hires_reference_encode_size=512),
          "one request carries one ref_image_args, so shrinking references needs its own request")
    check("an auto reference encode size keeps the native path",
          not hires_settings_vary_from_main(True, [both_stages], hires_reference_encode_size=0))

    check("blank overrides and no tags keep the native path",
          not hires_settings_vary_from_main(True, [both_stages], hires_prompt="   ",
                                           hires_negative_prompt=""),
          "whitespace is not an override")

    check("the source replaces the first stage when the flag and a source are both set",
          renders_hires_from_existing_source({"img2img_source_replaces_first_stage": True,
                                              "source_image": "puppy.png"}))
    check("the flag alone refines nothing, so it does not replace the first stage",
          not renders_hires_from_existing_source({"img2img_source_replaces_first_stage": True,
                                                  "source_image": "   "}))
    check("a source without the flag leaves the first stage alone",
          not renders_hires_from_existing_source({"img2img_source_replaces_first_stage": False,
                                                  "source_image": "puppy.png"}))
    check("replacing the first stage does not depend on any tiling setting",
          renders_hires_from_existing_source({"img2img_source_replaces_first_stage": True,
                                              "source_image": "puppy.png",
                                              "tiled_diffusion": "off"}),
          "the hires stage continues a latent either way")

    print()
    if failures:
        print(f"{len(failures)} hires staging check(s) failed")
        return 1
    print("all hires staging checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
