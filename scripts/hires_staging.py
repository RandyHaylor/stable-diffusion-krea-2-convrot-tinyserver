"""Deciding how the hires stage runs relative to the main stage.

The runtime applies LoRAs per request, not per stage, so a hires pass that needs
a different LoRA selection cannot run inside the same request as the main pass.
When the selections match, the native in-request hires path is used: it upscales
in latent space and reuses the main pass's conditioning, which is why it holds
detail so much better than a round trip through the VAE. Running the hires stage
as its own request is therefore a deliberate downgrade, taken only when its
settings vary from the main stage.
"""
from __future__ import annotations


def select_loras_for_stage(extra_loras: list[dict], stage: str) -> list[dict]:
    """The backend-shaped LoRA list for one stage.

    The main stage uses every selected LoRA. The hires stage uses only those
    whose hires column is ticked, so an unticked LoRA is dropped rather than
    carried over.
    """
    return [{"path": str(lora.get("filename", "")),
             "multiplier": float(lora.get("strength", 1.0)),
             "is_high_noise": False}
            for lora in extra_loras
            if stage == "main" or lora.get("use_in_hires")]


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
                                  hires_tag_source: str = "none") -> bool:
    """Whether the hires stage's settings differ from the main stage's.

    One request carries one LoRA selection and one prompt, both shared by the
    two stages. Any variation therefore has to run as its own request, which
    reloads weights and costs the latent continuity the in-request hires enjoys.
    """
    if not hires_enabled:
        return False
    if lora_selections_differ(select_loras_for_stage(extra_loras, "main"),
                              select_loras_for_stage(extra_loras, "hires")):
        return True
    if hires_prompt.strip() or hires_negative_prompt.strip():
        return True
    return hires_tag_source != "none"
