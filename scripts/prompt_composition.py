"""Assembling the prompt text sent to the backend.

Two things get folded into a prompt: WD14 tag groups, one per tagged image, and
the hires stage's own prompt override. The hires stage always starts from the
main prompt as the user typed it, before any tags were added, so ticking tagging
on both stages cannot compound the tags twice.
"""
from __future__ import annotations

HIRES_PROMPT_MODES = ("append", "prepend", "replace")
DEFAULT_HIRES_PROMPT_MODE = "append"

# What the hires prompt does with tags read from the first stage's own output.
# Reference and source images route their own tags per stage, so this covers only
# the generated image, which has no panel of its own.
STAGE_ONE_TAG_MODES = ("not_used", "append", "prepend")
DEFAULT_STAGE_ONE_TAG_MODE = "not_used"

# What the hires stage is conditioned on besides its prompt. One request carries
# one attachment list, so these are exclusive: picking any source other than the
# edit references is also how those references are kept out of the hires
# attention sequence, where they are the largest single cost.
HIRES_VISION_SOURCES = ("img2img_source", "krea2edit_references",
                        "stage_one_output", "none")
DEFAULT_HIRES_VISION_SOURCE = "krea2edit_references"

# How strongly one image's tags pull, relative to the rest of the prompt. The
# runtime reads A1111 attention syntax, so a weight is expressed in the prompt
# text itself rather than as a request field.
NEUTRAL_TAG_PROMPT_WEIGHT = 1.0


def wrap_tag_group_in_attention_weight(tag_group: str, tag_prompt_weight: float) -> str:
    """One image's tags, weighted against the rest of the prompt.

    A neutral weight returns the group untouched, so the common case sends no
    attention syntax at all. Unescaped parentheses would read as nested attention
    groups, but danbooru tags arrive with theirs already escaped, so the group can
    be wrapped whole.
    """
    tag_group = tag_group.strip()
    if not tag_group or tag_prompt_weight == NEUTRAL_TAG_PROMPT_WEIGHT:
        return tag_group
    return f"({tag_group}:{tag_prompt_weight:g})"


def hires_stage_uses_krea2_edit_references(hires_vision_source: str) -> bool:
    """Whether the hires request should carry the Krea2 Edit references at all."""
    return hires_vision_source == "krea2edit_references"


def hires_vision_source_needs_the_first_stage_image(hires_vision_source: str) -> bool:
    """Whether this source requires the first stage's output to exist as a file."""
    return hires_vision_source == "stage_one_output"


def compose_prompt_with_tag_groups(prompt: str, tag_groups: list[str]) -> str:
    """Join a prompt and each image's tags with single comma separation.

    Each group is one image's tags, kept in image order. Either side may be
    empty: tagging with no prompt is a supported way to generate.
    """
    segments = [segment.strip().rstrip(",").strip()
                for segment in [prompt, *tag_groups]]
    return ", ".join(segment for segment in segments if segment)


def stage_one_tags_need_the_first_stage_image(stage_one_tag_mode: str) -> bool:
    """Whether this mode requires the first stage's output to exist as a file.

    Tagging the first stage's output means it has to be produced and saved before
    the hires request is built, which forces the low-res pass to run.
    """
    return stage_one_tag_mode in ("append", "prepend")


def apply_stage_one_tags(prompt: str,
                         stage_one_tag_groups: list[str],
                         stage_one_tag_mode: str) -> str:
    """Place tags read from the first stage's output around the hires prompt.

    Prepending puts the observed content ahead of the instruction, which reads as
    a description the hires pass elaborates on; appending leaves the instruction
    leading. An unrecognised mode contributes nothing rather than guessing.
    """
    if stage_one_tag_mode == "append":
        return compose_prompt_with_tag_groups(prompt, stage_one_tag_groups)
    if stage_one_tag_mode == "prepend":
        return compose_prompt_with_tag_groups("", [*stage_one_tag_groups, prompt])
    return prompt


def describe_missing_prompt(main_prompt: str,
                            main_tagging_enabled: bool,
                            hires_enabled: bool,
                            hires_prompt: str,
                            hires_prompt_mode: str) -> str | None:
    """Why the job cannot run for want of prompt text, or None when it can.

    Tagging counts as supplying the main prompt, since tags become the prompt.
    It cannot satisfy the hires replace mode, though: tags are only known once
    the images have been tagged at generation time, so a replace override with
    no text would send the hires pass an empty prompt.
    """
    if not main_prompt.strip() and not main_tagging_enabled:
        return "Prompt is required, or enable WD14 tagging to build one from the images"
    if hires_enabled and hires_prompt_mode == "replace" and not hires_prompt.strip():
        return "Hires prompt is required when the hires mode replaces the main prompt"
    return None


def compose_hires_prompt(base_prompt: str,
                         hires_prompt: str,
                         mode: str,
                         tag_groups: list[str]) -> str:
    """The hires stage's prompt: base, adjusted by the override, then tags.

    `base_prompt` is the main prompt before tagging, so the hires stage never
    inherits the main stage's tags. An empty override leaves the base alone even
    in replace mode, since blanking the prompt is never the intent of leaving a
    field empty.
    """
    base_prompt = base_prompt.strip()
    hires_prompt = hires_prompt.strip()

    if not hires_prompt:
        adjusted_prompt = base_prompt
    elif mode == "prepend":
        adjusted_prompt = compose_prompt_with_tag_groups(hires_prompt, [base_prompt])
    elif mode == "replace":
        adjusted_prompt = hires_prompt
    else:
        adjusted_prompt = compose_prompt_with_tag_groups(base_prompt, [hires_prompt])

    return compose_prompt_with_tag_groups(adjusted_prompt, tag_groups)
