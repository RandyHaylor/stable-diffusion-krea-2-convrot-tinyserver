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
    build_krea2_edit_ref_image_args,
    is_krea2_edit_enabled,
    krea2_edit_native_args_fields,
    krea2_edit_payload_fields,
    krea2_edit_references,
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

    print()
    if failures:
        print(f"{len(failures)} krea2 edit request check(s) failed")
        return 1
    print("all krea2 edit request checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
