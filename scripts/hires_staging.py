"""Deciding how the hires stage runs relative to the main stage.

One generation covers both passes. When the stages differ the job pauses between
them: the runtime holds the first stage's latent while the app decides what the
hires pass should be conditioned on, sends it back, and the refinement continues
from that latent. Nothing is re-encoded from a decoded image, so the latent
continuity that makes the in-request hires hold detail is never given up.

What varies therefore decides whether the job pauses, not whether it splits into
two requests. A differing LoRA selection is carried across that pause and applied
between the passes, which costs a LoRA reload but not the continuity.
"""
from __future__ import annotations


def renders_hires_from_existing_source(params: dict) -> bool:
    """Whether the img2img source stands in for a first stage, which is not sampled.

    The hires pass always continues the main pass's latent, so supplying the
    source as that latent is all this takes. A flag with no source has nothing to
    refine, so both are required.
    """
    if not params.get("img2img_source_replaces_first_stage"):
        return False
    return bool(str(params.get("source_image", "")).strip())


def select_loras_for_stage(extra_loras: list[dict], stage: str) -> list[dict]:
    """The backend-shaped LoRA list for one stage.

    Each stage has its own tick, so a LoRA can be main-only, hires-only or
    both. A LoRA left unticked for a stage is dropped from it rather than
    carried over from the other stage.
    """
    stage_tick = {"main": "use_in_main", "hires": "use_in_hires"}[stage]
    return [{"path": str(lora.get("filename", "")),
             "multiplier": float(lora.get("strength", 1.0)),
             "is_high_noise": False}
            for lora in extra_loras
            if lora.get(stage_tick)]


def lora_selections_differ(main_stage_loras: list[dict], hires_stage_loras: list[dict]) -> bool:
    """Whether the two stages need different weights loaded.

    Order is not meaningful to the backend, so it is normalized away; a
    difference in path or strength is what matters.
    """
    def comparable(loras: list[dict]) -> set:
        return {(lora["path"], lora["multiplier"], lora["is_high_noise"]) for lora in loras}

    return comparable(main_stage_loras) != comparable(hires_stage_loras)


def hires_settings_vary_from_main(hires_enabled: bool,
                                  extra_loras: list[dict],
                                  hires_prompt: str = "",
                                  hires_negative_prompt: str = "",
                                  main_tag_groups: list[str] | None = None,
                                  hires_tag_groups: list[str] | None = None,
                                  main_vision_images: list[str] | None = None,
                                  hires_vision_images: list[str] | None = None,
                                  hires_vision_source: str = "krea2edit_references",
                                  stage_one_tag_mode: str = "not_used",
                                  hires_reference_encode_size: int = 0) -> bool:
    """Whether the hires stage's settings differ from the main stage's.

    A difference means the job pauses between its passes so the hires stage can
    be given its own prompt, attachments and LoRAs. Tag groups only vary when the
    two stages would end up with different prompts, so routing the same image to
    both is free.
    """
    if not hires_enabled:
        return False
    if hires_reference_encode_size > 0:
        return True
    if stage_one_tag_mode != "not_used":
        return True
    if list(main_tag_groups or []) != list(hires_tag_groups or []):
        return True
    if list(main_vision_images or []) != list(hires_vision_images or []):
        return True
    if hires_vision_source != "krea2edit_references":
        return True
    if lora_selections_differ(select_loras_for_stage(extra_loras, "main"),
                              select_loras_for_stage(extra_loras, "hires")):
        return True
    return bool(hires_prompt.strip() or hires_negative_prompt.strip())
