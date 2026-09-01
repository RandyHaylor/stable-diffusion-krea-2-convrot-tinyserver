#!/usr/bin/env python3
"""Unit tests for making the steps setting mean the steps actually executed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executed_steps import (  # noqa: E402
    executed_steps_the_runtime_will_run,
    scheduled_steps_for_executed_steps,
    strength_is_a_noise_level,
)

failures: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS: {description}" + (f" {detail}" if detail else ""))
    else:
        failures.append(description)
        print(f"FAIL: {description}" + (f" {detail}" if detail else ""))


def main() -> int:
    check("a denoise of 0.4 asks the runtime for enough steps to execute eight",
          scheduled_steps_for_executed_steps(8, 0.4) == 20,
          str(scheduled_steps_for_executed_steps(8, 0.4)))
    check("a denoise of 0.25 asks for thirty two to execute eight",
          scheduled_steps_for_executed_steps(8, 0.25) == 32,
          str(scheduled_steps_for_executed_steps(8, 0.25)))
    check("a denoise of 0.6 rounds up rather than executing seven",
          scheduled_steps_for_executed_steps(8, 0.6) == 14,
          str(scheduled_steps_for_executed_steps(8, 0.6)))

    # The runtime decrements t_enc when it would equal the scheduled count, so a
    # full denoise needs one more scheduled step than it executes.
    check("a full denoise asks for one more than it executes",
          scheduled_steps_for_executed_steps(8, 1.0) == 9,
          str(scheduled_steps_for_executed_steps(8, 1.0)))
    check("a denoise above one is treated as full rather than reducing the count",
          scheduled_steps_for_executed_steps(8, 1.5) == 9,
          str(scheduled_steps_for_executed_steps(8, 1.5)))

    check("a denoise of zero leaves the count alone, since nothing is denoised",
          scheduled_steps_for_executed_steps(8, 0.0) == 8)
    check("a negative denoise is treated the same as zero",
          scheduled_steps_for_executed_steps(8, -0.5) == 8)

    check("the scaled count always executes exactly what was asked for",
          all(executed_steps_the_runtime_will_run(
                  scheduled_steps_for_executed_steps(wanted, denoise), denoise) == wanted
              for wanted in range(1, 33)
              for denoise in (0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5,
                              0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0)))

    check("an unscaled count under a partial denoise falls short, which is the bug",
          executed_steps_the_runtime_will_run(8, 0.4) == 3,
          str(executed_steps_the_runtime_will_run(8, 0.4)))

    check("strength read as a noise level is detected in free-form sample args",
          strength_is_a_noise_level("strength_as_noise_level=true"))
    check("the detection tolerates surrounding args and spacing",
          strength_is_a_noise_level("alpha=0.5, strength_as_noise_level = 1 ,beta=0.7"))
    check("an unrelated sample arg is not mistaken for it",
          not strength_is_a_noise_level("img2img_noise_multiplier=0"))
    check("empty sample args are not mistaken for it",
          not strength_is_a_noise_level(""))
    check("the count is left alone when strength is a noise level, since the "
          "runtime derives the step count from the sigma instead",
          scheduled_steps_for_executed_steps(
              8, 0.4, sample_args="strength_as_noise_level=true") == 8)

    if failures:
        print(f"\n{len(failures)} failing checks:")
        for description in failures:
            print(f"  - {description}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
