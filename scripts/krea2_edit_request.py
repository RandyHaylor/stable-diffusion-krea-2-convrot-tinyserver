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
ref_boost, which multiplies how hard the target attends to that reference, and
its own tag weight, which decides how hard any WD14 tags read from it pull
against the rest of the prompt.

A weight on the vision tokens themselves is built here too, since
`ref_image_args` is the only channel that reaches them. It applies to the
img2img source, which feeds the vision tower whether or not edit mode is on, so
these args are sometimes assembled for a request that has no references at all.

The LoRA itself is not handled here: the user selects it like any other LoRA.
"""
from __future__ import annotations

from typing import Callable

from prompt_composition import NEUTRAL_TAG_PROMPT_WEIGHT

REF_IMAGE_PRESET_NAME = "krea2_edit"
DEFAULT_GROUNDING_PIXELS = 768
NEUTRAL_REFERENCE_FIDELITY = 1.0
REFERENCE_FIT_MODES = ("fit", "crop")
DEFAULT_REFERENCE_FIT_MODE = "fit"

# How strongly the vision tower's reading of an image pulls on the result. Unlike
# ref_boost, which biases the DiT's attention to reference latents, this scales the
# hidden states of the image's own vision tokens.
NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT = 1.0

# Edge lengths the img2img source may be resized to before the vision tower reads
# it. The token count a vision image contributes follows its resized grid, so an
# explicit longest side is what makes that count predictable.
#
# 'auto' hands the sizing back to the runtime's own preset, which budgets by area
# and leaves anything already inside its band untouched, so the token count then
# tracks the source image's own dimensions. An explicit size is preferred by
# default; the runtime snaps whatever is asked for to its patch grid, so these are
# targets rather than exact output edges.
SOURCE_VISION_GROUNDING_SIZES = ("auto", "384", "512", "768", "1024", "1215", "1536", "2048")
DEFAULT_SOURCE_VISION_GROUNDING_SIZE = "1024"


def positive_weight_or_neutral(value, neutral_weight: float) -> float:
    """A user-supplied weight, with anything non-positive read as neutral.

    Zero and negatives would suppress or invert the contribution rather than
    weighting it, which is never what leaving a field alone should mean.
    """
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return neutral_weight
    return weight if weight > 0 else neutral_weight


def source_vlm_image_token_weight(params: dict) -> float:
    """How hard the img2img source's vision tokens pull, 1.0 for untouched."""
    return positive_weight_or_neutral(
        params.get("img2img_source_vlm_image_token_weight", NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT),
        NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT)


def build_source_vision_grounding_args(params: dict, source_gives_vision: bool) -> list[str]:
    """Args fixing the edge length the img2img source is read at, or none for auto.

    A longest-side resize is used rather than an area budget so one setting always
    means the same token count for a given aspect ratio. Those tokens join the
    prompt's own in the sequence the diffusion blocks attend over, which is why the
    size is worth controlling rather than inheriting.
    """
    if not source_gives_vision:
        return []
    requested_size = str(params.get("img2img_source_vision_grounding_size",
                                    DEFAULT_SOURCE_VISION_GROUNDING_SIZE)).strip()
    if requested_size not in SOURCE_VISION_GROUNDING_SIZES:
        requested_size = DEFAULT_SOURCE_VISION_GROUNDING_SIZE
    if requested_size == "auto":
        return []
    return ["vlm_resize_mode=longest_side", f"vlm_size={requested_size}"]


def is_krea2_edit_enabled(params: dict) -> bool:
    """True only when the user both enabled edit mode and supplied a reference."""
    return bool(params.get("krea2_edit_enabled")) and bool(krea2_edit_references(params))


def krea2_edit_references(params: dict) -> list[dict]:
    """The usable references, in order, each with its fidelity and tag routing.

    Panels the user left empty are dropped rather than sent as blanks. A boost
    that is missing or non-positive becomes neutral, which keeps every remaining
    reference at the position the user arranged it in. Tag routing is per stage
    and defaults to neither, so an untouched panel never costs a tagger run. The
    tag weight is per image rather than per stage, so both stages read this
    reference's tags at the same strength.
    """
    if not params.get("krea2_edit_enabled"):
        return []
    references = []
    for entry in params.get("krea2_edit_references", []):
        filename = str(entry.get("filename", "")).strip()
        if not filename:
            continue
        references.append({
            "filename": filename,
            "ref_boost": positive_weight_or_neutral(
                entry.get("ref_boost", NEUTRAL_REFERENCE_FIDELITY),
                NEUTRAL_REFERENCE_FIDELITY),
            "tag_prompt_weight": positive_weight_or_neutral(
                entry.get("tag_prompt_weight", NEUTRAL_TAG_PROMPT_WEIGHT),
                NEUTRAL_TAG_PROMPT_WEIGHT),
            "tags_to_stage_one": bool(entry.get("tags_to_stage_one")),
            "tags_to_hires": bool(entry.get("tags_to_hires")),
        })
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


def build_vlm_image_token_weight_args(params: dict,
                                      reference_count: int,
                                      source_gives_vision: bool) -> list[str]:
    """One `vlm_image_token_weight` per image the vision tower reads, in its order.

    The runtime assembles its vision images as reference images first and then the
    images attached for the tower alone, so weighting the img2img source means
    filling a neutral slot for each reference ahead of it. Nothing is emitted while
    every weight is neutral, which keeps the runtime on its unweighted path.
    """
    if not source_gives_vision:
        return []
    source_weight = source_vlm_image_token_weight(params)
    if source_weight == NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT:
        return []
    weights = [NEUTRAL_VLM_IMAGE_TOKEN_WEIGHT] * reference_count + [source_weight]
    return [f"vlm_image_token_weight={weight:g}" for weight in weights]


def build_krea2_edit_ref_image_args(params: dict,
                                    reference_encode_size: int = 0,
                                    source_gives_vision: bool = False) -> str:
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
    arguments += build_vlm_image_token_weight_args(params, len(references), source_gives_vision)
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


def krea2_edit_native_args_fields(params: dict,
                                  reference_encode_size: int = 0,
                                  source_gives_vision: bool = False) -> dict:
    """Fields for the native sd_cpp_extra_args block, or none when nothing needs them.

    `ref_image_args` is a native generation parameter. The compatibility
    endpoints parse only their own named fields out of the request body, so it
    reaches the runtime through the embedded native args instead.

    An img2img generation whose source feeds the vision tower has no edit
    references and so no other reason to send these args, but a weight on that
    source's vision tokens still has to travel somewhere. Such a request names no
    preset, leaving whichever one the runtime would have chosen on its own.
    """
    if not is_krea2_edit_enabled(params):
        arguments = (build_source_vision_grounding_args(params, source_gives_vision)
                     + build_vlm_image_token_weight_args(params, 0, source_gives_vision))
        return {"ref_image_args": ",".join(arguments)} if arguments else {}
    return {"ref_image_args": build_krea2_edit_ref_image_args(
        params, reference_encode_size, source_gives_vision)}
