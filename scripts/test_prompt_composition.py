#!/usr/bin/env python3
"""Unit tests for prompt assembly: tag groups and the hires prompt overrides."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_composition import (  # noqa: E402
    HIRES_PROMPT_MODES,
    STAGE_ONE_TAG_MODES,
    apply_stage_one_tags,
    compose_hires_prompt,
    compose_prompt_with_tag_groups,
    describe_missing_prompt,
    stage_one_tags_need_the_first_stage_image,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def main() -> int:
    check("tag groups are appended after the prompt, single comma separated",
          compose_prompt_with_tag_groups("a photo", ["cat girl", "maid headdress"])
          == "a photo, cat girl, maid headdress")
    check("an absent prompt yields tags only, with no leading comma",
          compose_prompt_with_tag_groups("", ["cat girl", "smile"]) == "cat girl, smile")
    check("a whitespace-only prompt is treated as absent",
          compose_prompt_with_tag_groups("   ", ["cat girl"]) == "cat girl")
    check("empty tag groups leave the prompt untouched",
          compose_prompt_with_tag_groups("a photo", []) == "a photo")
    check("an empty group between two populated ones does not double the comma",
          compose_prompt_with_tag_groups("a photo", ["cat girl", "", "smile"])
          == "a photo, cat girl, smile")
    check("a prompt already ending in a comma does not produce a double comma",
          compose_prompt_with_tag_groups("a photo,", ["cat girl"]) == "a photo, cat girl")
    check("groups from separate images stay comma separated from each other",
          compose_prompt_with_tag_groups("scene", ["1girl, smile", "1boy, hat"])
          == "scene, 1girl, smile, 1boy, hat")

    check("the three hires modes are the supported set",
          HIRES_PROMPT_MODES == ("append", "prepend", "replace"),
          f"got {HIRES_PROMPT_MODES}")

    check("append puts the hires text after the base prompt",
          compose_hires_prompt("a photo", "sharp focus", "append", []) == "a photo, sharp focus")
    check("prepend puts the hires text before the base prompt",
          compose_hires_prompt("a photo", "sharp focus", "prepend", []) == "sharp focus, a photo")
    check("replace drops the base prompt entirely",
          compose_hires_prompt("a photo", "sharp focus", "replace", []) == "sharp focus")

    check("an empty hires field leaves the base prompt alone in append mode",
          compose_hires_prompt("a photo", "", "append", []) == "a photo")
    check("an empty hires field leaves the base prompt alone in prepend mode",
          compose_hires_prompt("a photo", "   ", "prepend", []) == "a photo")
    check("an empty hires field in replace mode is gated before composition, not silently filled",
          compose_hires_prompt("a photo", "", "replace", []) == "a photo",
          "describe_missing_prompt refuses the job; composition stays predictable")

    check("a main prompt is required when nothing else supplies one",
          describe_missing_prompt("", False, False, "", "append") is not None)
    check("a main prompt is not required once tagging supplies the text",
          describe_missing_prompt("", True, False, "", "append") is None)
    check("a whitespace-only main prompt counts as absent",
          describe_missing_prompt("   ", False, False, "", "append") is not None)
    check("a real main prompt satisfies the gate",
          describe_missing_prompt("a photo", False, False, "", "append") is None)

    check("hires replace mode with an empty hires prompt is refused",
          describe_missing_prompt("a photo", False, True, "", "replace") is not None,
          "replacing with nothing would send an empty prompt to the hires pass")
    check("hires replace mode with text is accepted",
          describe_missing_prompt("a photo", False, True, "sharp focus", "replace") is None)
    check("hires append mode with an empty hires prompt is fine",
          describe_missing_prompt("a photo", False, True, "", "append") is None)
    check("the hires gate only applies when the hires pass is enabled",
          describe_missing_prompt("a photo", False, False, "", "replace") is None)
    check("hires replace with no text but hires tagging still needs a prompt",
          describe_missing_prompt("a photo", True, True, "", "replace") is not None,
          "tags are not known at queue time, so they cannot satisfy the gate")

    check("tags are added after the override is applied, in append mode",
          compose_hires_prompt("a photo", "sharp focus", "append", ["cat girl"])
          == "a photo, sharp focus, cat girl")
    check("tags are added after the override is applied, in prepend mode",
          compose_hires_prompt("a photo", "sharp focus", "prepend", ["cat girl"])
          == "sharp focus, a photo, cat girl")
    check("tags are added after the override is applied, in replace mode",
          compose_hires_prompt("a photo", "sharp focus", "replace", ["cat girl"])
          == "sharp focus, cat girl")
    check("tags alone are enough when there is no prompt at all",
          compose_hires_prompt("", "", "append", ["cat girl", "smile"]) == "cat girl, smile")

    check("the base prompt is used verbatim, never a tag-extended version",
          compose_hires_prompt("a photo, cat girl", "sharp focus", "append", ["cat girl"])
          == "a photo, cat girl, sharp focus, cat girl",
          "callers pass the pre-tag base; this function does not deduplicate")
    check("an unknown mode falls back to append rather than losing the text",
          compose_hires_prompt("a photo", "sharp focus", "nonsense", []) == "a photo, sharp focus")

    check("the three stage one tag modes are the supported set",
          STAGE_ONE_TAG_MODES == ("not_used", "append", "prepend"),
          f"got {STAGE_ONE_TAG_MODES}")

    stage_one_tags = ["1girl, kitchen, teacup"]

    check("the not_used mode leaves the hires prompt untouched",
          apply_stage_one_tags("a photo", stage_one_tags, "not_used") == "a photo")
    check("append puts the first stage's own tags after the hires prompt",
          apply_stage_one_tags("a photo", stage_one_tags, "append")
          == "a photo, 1girl, kitchen, teacup")
    check("prepend puts the first stage's own tags before the hires prompt",
          apply_stage_one_tags("a photo", stage_one_tags, "prepend")
          == "1girl, kitchen, teacup, a photo")
    check("several stage one groups keep their order in prepend mode",
          apply_stage_one_tags("a photo", ["1girl, smile", "kitchen"], "prepend")
          == "1girl, smile, kitchen, a photo")
    check("an untagged first stage leaves the prompt alone in every mode",
          apply_stage_one_tags("a photo", [], "append") == "a photo"
          and apply_stage_one_tags("a photo", [], "prepend") == "a photo")
    check("stage one tags alone are enough when there is no hires prompt",
          apply_stage_one_tags("", stage_one_tags, "append") == "1girl, kitchen, teacup"
          and apply_stage_one_tags("", stage_one_tags, "prepend") == "1girl, kitchen, teacup")
    check("an unknown mode contributes nothing rather than guessing",
          apply_stage_one_tags("a photo", stage_one_tags, "nonsense") == "a photo")

    check("only a mode that uses them needs the first stage image on disk",
          stage_one_tags_need_the_first_stage_image("append")
          and stage_one_tags_need_the_first_stage_image("prepend")
          and not stage_one_tags_need_the_first_stage_image("not_used"),
          "this decides whether the low-res pass has to be forced")

    print()
    if failures:
        print(f"{len(failures)} prompt composition check(s) failed")
        return 1
    print("all prompt composition checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
