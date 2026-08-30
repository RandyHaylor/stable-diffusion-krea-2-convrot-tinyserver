#!/usr/bin/env python3
"""Unit tests for Krea2 Edit request construction.

These exercise the pure functions directly, with no HTTP server and no
filesystem: the reference-image loader is injected.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from krea2_edit_request import (  # noqa: E402
    DEFAULT_SOURCE_VISION_GROUNDING_SIZE,
    NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT,
    SOURCE_VISION_GROUNDING_SIZES,
    build_source_vision_grounding_args,
    build_vision_only_ref_image_args,
    build_krea2_edit_ref_image_args,
    is_krea2_edit_enabled,
    krea2_edit_native_args_fields,
    krea2_edit_payload_fields,
    krea2_edit_reference_fit_mode,
    krea2_edit_references,
    source_vlm_image_token_weight,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def enabled_params(references=None, **overrides) -> dict:
    params = {
        "krea2_edit_enabled": True,
        "krea2_edit_references": ([{"filename": "scene.png", "ref_boost": 1}]
                                  if references is None else references),
        "grounding_px": 768,
    }
    params.update(overrides)
    return params


def fake_reference_image_loader(filename: str) -> str:
    return f"base64-of-{filename}"


def main() -> int:
    check("edit mode is off when the flag is absent",
          not is_krea2_edit_enabled({}))
    check("edit mode is off when the flag is false",
          not is_krea2_edit_enabled(enabled_params(krea2_edit_enabled=False)))
    check("edit mode is off when enabled with an empty reference list",
          not is_krea2_edit_enabled(enabled_params(references=[])),
          "an enabled checkbox with no images must not change the request")
    check("edit mode is on when enabled with at least one reference",
          is_krea2_edit_enabled(enabled_params()))

    tag_routed = enabled_params(references=[
        {"filename": "scene.png", "ref_boost": 1,
         "tags_to_stage_one": True, "tags_to_hires": False},
        {"filename": "person.png", "ref_boost": 1,
         "tags_to_stage_one": False, "tags_to_hires": True},
    ])
    check("each reference carries its own per-stage tag routing",
          [(reference["tags_to_stage_one"], reference["tags_to_hires"])
           for reference in krea2_edit_references(tag_routed)]
          == [(True, False), (False, True)],
          "a reference can feed tags to one stage without the other")
    check("a reference with no routing keys sends tags nowhere",
          krea2_edit_references(enabled_params(references=[{"filename": "scene.png"}]))[0]
          == {"filename": "scene.png", "ref_boost": 1.0, "tag_prompt_weight": 1.0,
              "tags_to_stage_one": False, "tags_to_hires": False},
          "ticking nothing must never cost a tagger run")

    check("a missing per-reference tag weight defaults to neutral",
          krea2_edit_references(enabled_params(references=[{"filename": "scene.png"}]))[0]
          ["tag_prompt_weight"] == 1.0)
    check("each reference carries its own tag weight",
          [reference["tag_prompt_weight"] for reference in krea2_edit_references(
              enabled_params(references=[{"filename": "scene.png", "tag_prompt_weight": 1.4},
                                         {"filename": "person.png", "tag_prompt_weight": 0.6}]))]
          == [1.4, 0.6])
    check("a non-positive tag weight falls back to neutral",
          krea2_edit_references(enabled_params(references=[
              {"filename": "scene.png", "tag_prompt_weight": 0}]))[0]["tag_prompt_weight"] == 1.0,
          "zero would silently erase the tags rather than weight them")

    check("references are empty when disabled",
          krea2_edit_references({}) == [])
    check("a reference without a filename is dropped",
          krea2_edit_references(enabled_params(references=[{"filename": "", "ref_boost": 4}])) == [],
          "an empty panel must not reach the backend")
    check("references keep the order the user arranged them in",
          [reference["filename"] for reference in krea2_edit_references(
              enabled_params(references=[{"filename": "scene.png", "ref_boost": 1},
                                         {"filename": "person.png", "ref_boost": 4}]))]
          == ["scene.png", "person.png"],
          "training order is scene first, subject second")
    check("an arbitrary number of references is supported",
          len(krea2_edit_references(enabled_params(references=[
              {"filename": f"reference-{index}.png", "ref_boost": 1} for index in range(5)]))) == 5)
    check("a missing per-reference boost defaults to neutral",
          krea2_edit_references(enabled_params(references=[{"filename": "scene.png"}]))[0]["ref_boost"] == 1.0)

    check("ref image args select the krea2_edit preset and carry grounding size",
          build_krea2_edit_ref_image_args(enabled_params())
          == "preset=krea2_edit,vlm_size=768",
          f"got {build_krea2_edit_ref_image_args(enabled_params())!r}")
    check("grounding size follows the requested value",
          build_krea2_edit_ref_image_args(enabled_params(grounding_px=384))
          == "preset=krea2_edit,vlm_size=384")
    check("a non-positive grounding size omits the vlm size entirely",
          build_krea2_edit_ref_image_args(enabled_params(grounding_px=0))
          == "preset=krea2_edit",
          "zero means native resolution, so no cap should be sent")

    check("the main stage sends no reference encode cap, leaving the preset's own budget",
          "vae_input_max_pixels" not in build_krea2_edit_ref_image_args(enabled_params()),
          "only the hires stage shrinks references, and only when asked")
    check("a hires reference encode size becomes a pixel-area budget",
          build_krea2_edit_ref_image_args(enabled_params(), reference_encode_size=512)
          == "preset=krea2_edit,vlm_size=768,vae_input_max_pixels=262144",
          "the runtime takes an area, so an edge length of N means an N*N budget")
    check("a larger encode size scales the budget quadratically",
          build_krea2_edit_ref_image_args(enabled_params(), reference_encode_size=1024)
          == "preset=krea2_edit,vlm_size=768,vae_input_max_pixels=1048576")
    check("a non-positive encode size means auto, so no cap is sent",
          "vae_input_max_pixels" not in build_krea2_edit_ref_image_args(
              enabled_params(), reference_encode_size=0))
    check("the encode cap sits alongside the boosts rather than replacing them",
          build_krea2_edit_ref_image_args(
              enabled_params(references=[{"filename": "scene.png", "ref_boost": 4}]),
              reference_encode_size=512)
          == "preset=krea2_edit,vlm_size=768,vae_input_max_pixels=262144,ref_boost=4")

    single_boosted = enabled_params(references=[{"filename": "scene.png", "ref_boost": 4}])
    check("a boosted single reference emits one ref_boost key",
          build_krea2_edit_ref_image_args(single_boosted)
          == "preset=krea2_edit,vlm_size=768,ref_boost=4",
          f"got {build_krea2_edit_ref_image_args(single_boosted)!r}")

    per_reference_boosts = enabled_params(references=[
        {"filename": "scene.png", "ref_boost": 1},
        {"filename": "person.png", "ref_boost": 4},
    ])
    check("each reference emits its own ref_boost key, in reference order",
          build_krea2_edit_ref_image_args(per_reference_boosts)
          == "preset=krea2_edit,vlm_size=768,ref_boost=1,ref_boost=4",
          f"got {build_krea2_edit_ref_image_args(per_reference_boosts)!r}")

    all_neutral = enabled_params(references=[
        {"filename": "scene.png", "ref_boost": 1},
        {"filename": "person.png", "ref_boost": 1},
    ])
    check("all-neutral boosts are omitted, leaving attention untouched",
          build_krea2_edit_ref_image_args(all_neutral) == "preset=krea2_edit,vlm_size=768",
          "neutral everywhere means the runtime should build no mask at all")

    check("a fractional boost keeps its precision",
          build_krea2_edit_ref_image_args(
              enabled_params(references=[{"filename": "scene.png", "ref_boost": 2.5}]))
          == "preset=krea2_edit,vlm_size=768,ref_boost=2.5")
    check("a non-positive boost is sent as neutral rather than rejected by the runtime",
          build_krea2_edit_ref_image_args(
              enabled_params(references=[{"filename": "a.png", "ref_boost": 0},
                                         {"filename": "b.png", "ref_boost": 4}]))
          == "preset=krea2_edit,vlm_size=768,ref_boost=1,ref_boost=4",
          "positions must stay aligned with the reference list")

    check("the reference fit mode defaults to fit when the user sent nothing",
          krea2_edit_reference_fit_mode(enabled_params()) == "fit")
    check("an explicit crop fit mode is kept",
          krea2_edit_reference_fit_mode(enabled_params(fit_mode="crop")) == "crop")
    check("an unrecognised fit mode falls back to the default rather than reaching the runtime",
          krea2_edit_reference_fit_mode(enabled_params(fit_mode="stretch")) == "fit",
          "the runtime would warn and ignore it anyway")

    check("the default fit mode is omitted, leaving the preset's own geometry in place",
          "fit_mode" not in build_krea2_edit_ref_image_args(enabled_params()),
          f"got {build_krea2_edit_ref_image_args(enabled_params())!r}")
    crop_mode = enabled_params(fit_mode="crop")
    check("crop fit mode is emitted as its own key",
          build_krea2_edit_ref_image_args(crop_mode)
          == "preset=krea2_edit,vlm_size=768,fit_mode=crop",
          f"got {build_krea2_edit_ref_image_args(crop_mode)!r}")
    crop_mode_with_boosts = enabled_params(fit_mode="crop", references=[
        {"filename": "scene.png", "ref_boost": 1},
        {"filename": "person.png", "ref_boost": 4},
    ])
    check("fit mode and per-reference boosts coexist",
          build_krea2_edit_ref_image_args(crop_mode_with_boosts)
          == "preset=krea2_edit,vlm_size=768,fit_mode=crop,ref_boost=1,ref_boost=4",
          f"got {build_krea2_edit_ref_image_args(crop_mode_with_boosts)!r}")
    check("the fit mode travels in the native args, never in the payload body",
          krea2_edit_native_args_fields(crop_mode)
          == {"ref_image_args": "preset=krea2_edit,vlm_size=768,fit_mode=crop"}
          and "fit_mode" not in krea2_edit_payload_fields(crop_mode, fake_reference_image_loader),
          f"got {krea2_edit_native_args_fields(crop_mode)}")

    check("a vision-only reference is kept out of the diffusion transformer",
          "pass_to_dit=false" in build_vision_only_ref_image_args(768),
          "these images exist to be read by the VLM, not to be attended to as latents")
    check("a vision-only reference still selects the krea2_edit preset",
          build_vision_only_ref_image_args(768)
          == "preset=krea2_edit,pass_to_dit=false,vlm_size=768",
          f"got {build_vision_only_ref_image_args(768)!r}")
    check("a non-positive grounding size omits the vlm size from vision-only args",
          build_vision_only_ref_image_args(0) == "preset=krea2_edit,pass_to_dit=false")
    check("krea2 edit references are never marked as vision-only",
          "pass_to_dit" not in build_krea2_edit_ref_image_args(enabled_params()),
          "edit references belong in the DiT; that is what edit mode is")

    check("no payload fields are produced when edit mode is off",
          krea2_edit_payload_fields({}, fake_reference_image_loader) == {})

    payload_fields = krea2_edit_payload_fields(per_reference_boosts, fake_reference_image_loader)
    check("every reference is sent as a base64 extra image, in order",
          payload_fields["extra_images"] == ["base64-of-scene.png", "base64-of-person.png"],
          f"got {payload_fields.get('extra_images')}")
    check("ref image args are not put in the payload body, which would drop them",
          "ref_image_args" not in payload_fields,
          "the sdapi route reads only named fields; ref_image_args must travel in the native args")
    check("ref image args travel in the native sd_cpp_extra_args block",
          krea2_edit_native_args_fields(per_reference_boosts)
          == {"ref_image_args": "preset=krea2_edit,vlm_size=768,ref_boost=1,ref_boost=4"},
          f"got {krea2_edit_native_args_fields(per_reference_boosts)}")
    check("no native args are produced when edit mode is off",
          krea2_edit_native_args_fields({}) == {})
    check("edit mode never sets init_images, which would make it img2img",
          "init_images" not in payload_fields,
          f"keys={sorted(payload_fields)}")
    check("edit mode never sets denoising_strength; the target starts as pure noise",
          "denoising_strength" not in payload_fields,
          f"keys={sorted(payload_fields)}")

    check("a missing source vision weight reads as neutral",
          source_vlm_image_token_weight({}) == NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT)
    check("a non-positive source vision weight falls back to neutral",
          source_vlm_image_token_weight({"img2img_source_vlm_image_token_weight": 0})
          == NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT,
          "zero would mean the vision tokens contribute nothing at all")

    weighted_vision = enabled_params(img2img_source_vlm_image_token_weight=0.5)
    check("a neutral source vision weight adds nothing to the ref image args",
          build_krea2_edit_ref_image_args(enabled_params(), source_gives_vision=True)
          == "preset=krea2_edit,vlm_size=768",
          "the neutral path must stay byte identical to before this knob existed")
    check("a source vision weight is placed after a neutral slot per reference",
          build_krea2_edit_ref_image_args(weighted_vision, source_gives_vision=True)
          == "preset=krea2_edit,vlm_size=768,vlm_image_token_weight=1,vlm_image_token_weight=0.5",
          "the runtime reads these positionally over references then vlm images")
    check("two references produce two neutral slots before the source weight",
          build_krea2_edit_ref_image_args(
              enabled_params(references=[{"filename": "scene.png"}, {"filename": "person.png"}],
                             img2img_source_vlm_image_token_weight=0.5),
              source_gives_vision=True)
          == ("preset=krea2_edit,vlm_size=768,"
              "vlm_image_token_weight=1,vlm_image_token_weight=1,vlm_image_token_weight=0.5"))
    check("a source that gives no vision contributes no weight, however it is set",
          build_krea2_edit_ref_image_args(weighted_vision, source_gives_vision=False)
          == "preset=krea2_edit,vlm_size=768",
          "an unattached image has no tokens to weight")

    vision_weight_without_edit_mode = {"img2img_source_vlm_image_token_weight": 0.5,
                                       "img2img_source_vision_grounding_size": "auto"}
    check("a source vision weight reaches the runtime with edit mode off",
          krea2_edit_native_args_fields(vision_weight_without_edit_mode,
                                        source_gives_vision=True)
          == {"ref_image_args": "vlm_image_token_weight=0.5"},
          "img2img with vision emits no other ref image args, so the weight needs its own")
    check("the edit mode off carrier names no preset, leaving the runtime default",
          "preset" not in krea2_edit_native_args_fields(
              vision_weight_without_edit_mode, source_gives_vision=True)["ref_image_args"],
          "naming a preset here would silently change which sizing rules apply")
    check("a neutral weight and auto grounding produce no native args at all",
          krea2_edit_native_args_fields({"img2img_source_vlm_image_token_weight": 1.0,
                                         "img2img_source_vision_grounding_size": "auto"},
                                        source_gives_vision=True) == {})
    check("a source giving no vision is neither sized nor weighted",
          krea2_edit_native_args_fields({"img2img_source_vlm_image_token_weight": 0.5},
                                        source_gives_vision=False) == {},
          "the default grounding size must not conjure args for an unattached image")

    check("auto is offered, so the runtime's own sizing stays reachable",
          "auto" in SOURCE_VISION_GROUNDING_SIZES,
          f"got {SOURCE_VISION_GROUNDING_SIZES}")
    check("every grounding size other than auto is a positive edge length",
          all(str(size).isdigit() and int(size) > 0
              for size in SOURCE_VISION_GROUNDING_SIZES if size != "auto"),
          f"got {SOURCE_VISION_GROUNDING_SIZES}")
    check("the default grounding size is one of the offered options",
          DEFAULT_SOURCE_VISION_GROUNDING_SIZE in SOURCE_VISION_GROUNDING_SIZES,
          f"default={DEFAULT_SOURCE_VISION_GROUNDING_SIZE}")
    check("an unspecified grounding size takes the default rather than the runtime's own",
          build_source_vision_grounding_args({}, source_gives_vision=True)
          == ["vlm_resize_mode=longest_side", f"vlm_size={DEFAULT_SOURCE_VISION_GROUNDING_SIZE}"],
          "a deliberate, predictable size is preferred over an inherited area band")

    check("auto grounding emits nothing, leaving the runtime's own sizing",
          build_source_vision_grounding_args(
              {"img2img_source_vision_grounding_size": "auto"},
              source_gives_vision=True) == [],
          "this is the path every img2img vision job took before this control existed")
    check("an unrecognised grounding size falls back to the default rather than being sent on",
          build_source_vision_grounding_args(
              {"img2img_source_vision_grounding_size": "enormous"},
              source_gives_vision=True)
          == ["vlm_resize_mode=longest_side", f"vlm_size={DEFAULT_SOURCE_VISION_GROUNDING_SIZE}"],
          "the runtime would only warn and ignore a value it does not know")
    check("a chosen grounding size sets a longest side rather than an area",
          build_source_vision_grounding_args(
              {"img2img_source_vision_grounding_size": "768"},
              source_gives_vision=True)
          == ["vlm_resize_mode=longest_side", "vlm_size=768"],
          "an explicit edge length is what makes the token count predictable")
    check("a source that gives no vision is not sized at all",
          build_source_vision_grounding_args(
              {"img2img_source_vision_grounding_size": "768"},
              source_gives_vision=False) == [],
          "nothing reaches the vision tower, so there is nothing to size")

    grounded_vision_without_edit_mode = {"img2img_source_vision_grounding_size": "512"}
    check("a grounding size reaches the runtime with edit mode off",
          krea2_edit_native_args_fields(grounded_vision_without_edit_mode,
                                        source_gives_vision=True)
          == {"ref_image_args": "vlm_resize_mode=longest_side,vlm_size=512"},
          f"got {krea2_edit_native_args_fields(grounded_vision_without_edit_mode, source_gives_vision=True)}")
    check("a grounding size and a vision weight travel together",
          krea2_edit_native_args_fields(
              {"img2img_source_vision_grounding_size": "512",
               "img2img_source_vlm_image_token_weight": 0.5},
              source_gives_vision=True)
          == {"ref_image_args": "vlm_resize_mode=longest_side,vlm_size=512,vlm_image_token_weight=0.5"})
    check("edit mode keeps its own grounding size for its references",
          build_krea2_edit_ref_image_args(
              enabled_params(img2img_source_vision_grounding_size="512"),
              source_gives_vision=True)
          == "preset=krea2_edit,vlm_size=768",
          "one request carries one vlm size; the edit preset's grounding wins")

    print()
    if failures:
        print(f"{len(failures)} krea2 edit request check(s) failed")
        return 1
    print("all krea2 edit request checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
