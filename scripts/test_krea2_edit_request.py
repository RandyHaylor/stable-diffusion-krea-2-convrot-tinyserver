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
    krea2_edit_reference_filenames,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def enabled_params(**overrides) -> dict:
    params = {
        "krea2_edit_enabled": True,
        "krea2_edit_reference_image": "scene.png",
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
    check("edit mode is off when enabled but no reference image is selected",
          not is_krea2_edit_enabled(enabled_params(krea2_edit_reference_image="")),
          "an enabled checkbox with no image must not change the request")
    check("edit mode is on when enabled with a reference image",
          is_krea2_edit_enabled(enabled_params()))

    check("reference filenames are empty when disabled",
          krea2_edit_reference_filenames({}) == [])
    check("a single reference filename is returned",
          krea2_edit_reference_filenames(enabled_params()) == ["scene.png"])
    check("scene is ordered before subject for two-image edits",
          krea2_edit_reference_filenames(
              enabled_params(krea2_edit_reference_image_b="person.png"))
          == ["scene.png", "person.png"],
          "training order is scene first, subject second")
    check("a subject without a scene is not sent alone",
          krea2_edit_reference_filenames(
              enabled_params(krea2_edit_reference_image="",
                             krea2_edit_reference_image_b="person.png")) == [])

    check("ref image args select the krea2_edit preset and carry grounding size",
          build_krea2_edit_ref_image_args(enabled_params())
          == "preset=krea2_edit,vlm_size=768",
          f"got {build_krea2_edit_ref_image_args(enabled_params())!r}")
    check("grounding size follows the requested value",
          build_krea2_edit_ref_image_args(enabled_params(grounding_px=384))
          == "preset=krea2_edit,vlm_size=384")
    check("a missing grounding size falls back to the trained default",
          build_krea2_edit_ref_image_args({"krea2_edit_enabled": True})
          == "preset=krea2_edit,vlm_size=768")
    check("a non-positive grounding size omits the vlm size entirely",
          build_krea2_edit_ref_image_args(enabled_params(grounding_px=0))
          == "preset=krea2_edit",
          "zero means native resolution, so no cap should be sent")

    check("no payload fields are produced when edit mode is off",
          krea2_edit_payload_fields({}, fake_reference_image_loader) == {})

    single_reference_fields = krea2_edit_payload_fields(
        enabled_params(), fake_reference_image_loader)
    check("the reference is sent as a base64 extra image",
          single_reference_fields["extra_images"] == ["base64-of-scene.png"],
          f"got {single_reference_fields.get('extra_images')}")
    check("ref image args are not put in the payload body, which would drop them",
          "ref_image_args" not in single_reference_fields,
          "the sdapi route reads only named fields; ref_image_args must travel in the native args")
    check("ref image args travel in the native sd_cpp_extra_args block",
          krea2_edit_native_args_fields(enabled_params())
          == {"ref_image_args": "preset=krea2_edit,vlm_size=768"},
          f"got {krea2_edit_native_args_fields(enabled_params())}")
    check("no native args are produced when edit mode is off",
          krea2_edit_native_args_fields({}) == {})
    check("edit mode never sets init_images, which would make it img2img",
          "init_images" not in single_reference_fields,
          f"keys={sorted(single_reference_fields)}")
    check("edit mode never sets denoising_strength; the target starts as pure noise",
          "denoising_strength" not in single_reference_fields,
          f"keys={sorted(single_reference_fields)}")

    two_reference_fields = krea2_edit_payload_fields(
        enabled_params(krea2_edit_reference_image_b="person.png"),
        fake_reference_image_loader)
    check("both references are sent in scene-then-subject order",
          two_reference_fields["extra_images"]
          == ["base64-of-scene.png", "base64-of-person.png"],
          f"got {two_reference_fields['extra_images']}")

    print()
    if failures:
        print(f"{len(failures)} krea2 edit request check(s) failed")
        return 1
    print("all krea2 edit request checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
