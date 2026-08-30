"""Krea2 Edit request construction.

Krea2 Edit is instruction-based editing, not img2img. The target latent starts
as pure noise; reference images reach the model only as conditioning, by two
paths at once: VAE latent tokens for appearance, and Qwen3-VL vision tokens for
semantics. Requests therefore use the txt2img endpoint and must never carry
init_images or denoising_strength, which would re-noise the reference instead.

The `krea2_edit` reference-image preset is what tells stable-diffusion.cpp to
condition references the way the identity-edit LoRA was trained. Krea2 otherwise
defaults to `krea2_ostris_edit`, which differs and degrades output silently.

References are ordered, and the order is meaningful: the LoRA was trained with
the scene first and the subject second. Each reference carries its own
ref_boost, which multiplies how hard the target attends to that reference.

The LoRA itself is not handled here: the user selects it like any other LoRA.
"""
from __future__ import annotations

from typing import Callable

REF_IMAGE_PRESET_NAME = "krea2_edit"
DEFAULT_GROUNDING_PIXELS = 768
NEUTRAL_REFERENCE_FIDELITY = 1.0
REFERENCE_FIT_MODES = ("fit", "crop")
DEFAULT_REFERENCE_FIT_MODE = "fit"


def is_krea2_edit_enabled(params: dict) -> bool:
    """True only when the user both enabled edit mode and supplied a reference."""
    return bool(params.get("krea2_edit_enabled")) and bool(krea2_edit_references(params))


def krea2_edit_references(params: dict) -> list[dict]:
    """The usable references, in order, each with its fidelity and tag routing.

    Panels the user left empty are dropped rather than sent as blanks. A boost
    that is missing or non-positive becomes neutral, which keeps every remaining
    reference at the position the user arranged it in. Tag routing is per stage
    and defaults to neither, so an untouched panel never costs a tagger run.
    """
    if not params.get("krea2_edit_enabled"):
        return []
    references = []
    for entry in params.get("krea2_edit_references", []):
        filename = str(entry.get("filename", "")).strip()
        if not filename:
            continue
        reference_fidelity = float(entry.get("ref_boost", NEUTRAL_REFERENCE_FIDELITY))
        if reference_fidelity <= 0:
            reference_fidelity = NEUTRAL_REFERENCE_FIDELITY
        references.append({"filename": filename,
                           "ref_boost": reference_fidelity,
                           "tags_to_stage_one": bool(entry.get("tags_to_stage_one")),
                           "tags_to_hires": bool(entry.get("tags_to_hires"))})
    return references


def krea2_edit_reference_fit_mode(params: dict) -> str:
    """How a reference is made to fit the target: 'fit' or 'crop'.

    Both modes resample the reference aspect-preserving; neither crops it. 'fit'
    additionally caps the reference's latent grid to the target grid and centres
    its RoPE positions on the target; 'crop' skips the cap and anchors positions
    at the origin. The identity-edit LoRA was trained on 'fit'.
    An unrecognised value falls back to the default rather than being sent on,
    since the runtime would only warn and ignore it.
    """
    requested_fit_mode = str(params.get("fit_mode", DEFAULT_REFERENCE_FIT_MODE)).strip().lower()
    if requested_fit_mode not in REFERENCE_FIT_MODES:
        return DEFAULT_REFERENCE_FIT_MODE
    return requested_fit_mode


def build_krea2_edit_ref_image_args(params: dict, reference_encode_size: int = 0) -> str:
    """The reference-image args: preset, VLM grounding size, fit mode, fidelity.

    The default fit mode is omitted so the preset's own geometry stands.

    `reference_encode_size` is an edge length, sent as the N*N pixel area the
    runtime budgets for encoding each reference. Zero leaves the preset's own
    budget in place. Reference tokens join the target's in one attention
    sequence whose cost grows with the square of that total, so shrinking
    references is what makes room at a large hires target.

    ref_boost is repeated once per reference, in reference order, because the
    runtime's key=value parser splits on both ',' and ';' and so cannot carry a
    delimited list inside a single value. All-neutral boosts are omitted so the
    runtime skips building an attention mask at all.
    """
    arguments = [f"preset={REF_IMAGE_PRESET_NAME}"]
    grounding_pixels = int(params.get("grounding_px", DEFAULT_GROUNDING_PIXELS))
    if grounding_pixels > 0:
        arguments.append(f"vlm_size={grounding_pixels}")
    if reference_encode_size > 0:
        arguments.append(f"vae_input_max_pixels={reference_encode_size * reference_encode_size}")
    fit_mode = krea2_edit_reference_fit_mode(params)
    if fit_mode != DEFAULT_REFERENCE_FIT_MODE:
        arguments.append(f"fit_mode={fit_mode}")
    references = krea2_edit_references(params)
    if any(reference["ref_boost"] != NEUTRAL_REFERENCE_FIDELITY for reference in references):
        arguments += [f"ref_boost={reference['ref_boost']:g}" for reference in references]
    return ",".join(arguments)


def build_vision_only_ref_image_args(grounding_pixels: int) -> str:
    """Reference args for an image attached solely so the VLM can read it.

    `extra_images` is the only request field that reaches the VLM, and the
    runtime otherwise VAE-encodes every entry into reference latents that the
    DiT attends to. Those latents cost sequence length in every sampling pass,
    including hires, which is not what attaching an image for its description is
    asking for; pass_to_dit=false leaves the diffusion transformer untouched.
    """
    arguments = [f"preset={REF_IMAGE_PRESET_NAME}", "pass_to_dit=false"]
    if grounding_pixels > 0:
        arguments.append(f"vlm_size={grounding_pixels}")
    return ",".join(arguments)


def krea2_edit_payload_fields(
    params: dict,
    load_reference_image_base64: Callable[[str], str],
) -> dict:
    """Request-body fields for a Krea2 Edit request; empty when edit mode is off.

    The compatibility endpoints read `extra_images` directly out of the body and
    turn each entry into a reference image.

    `load_reference_image_base64` resolves a reference filename to base64 image
    data, so this module stays independent of where those images are stored.
    """
    if not is_krea2_edit_enabled(params):
        return {}
    return {
        "extra_images": [load_reference_image_base64(reference["filename"])
                         for reference in krea2_edit_references(params)],
    }


def krea2_edit_native_args_fields(params: dict, reference_encode_size: int = 0) -> dict:
    """Fields for the native sd_cpp_extra_args block; empty when edit mode is off.

    `ref_image_args` is a native generation parameter. The compatibility
    endpoints parse only their own named fields out of the request body, so it
    reaches the runtime through the embedded native args instead.
    """
    if not is_krea2_edit_enabled(params):
        return {}
    return {"ref_image_args": build_krea2_edit_ref_image_args(params, reference_encode_size)}
