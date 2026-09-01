"""Making the steps setting mean the steps the model actually runs.

The runtime builds a sigma schedule of the requested length and then runs only
its tail: `t_enc = int(scheduled_steps * denoise)`. Asking for 8 steps at
denoise 0.4 therefore spends three evaluations, not eight.

The denoise is still what picks the starting sigma, which is the behaviour worth
keeping. So rather than changing the runtime, the requested count is scaled up
before it is sent, until the tail the runtime keeps is the count that was asked
for.
"""
from __future__ import annotations

import math
import re

# The runtime reads strength as a noise level when this sample arg is set, and
# then derives the step count from where that sigma falls in the schedule rather
# than from the denoise. Scaling the count would not produce the asked-for
# number of steps, so it is left alone.
_STRENGTH_AS_NOISE_LEVEL_PATTERN = re.compile(r"(^|,)\s*strength_as_noise_level\s*=", re.IGNORECASE)


def strength_is_a_noise_level(sample_args: str) -> bool:
    """Whether free-form sample args put the runtime on its noise-level branch."""
    return bool(_STRENGTH_AS_NOISE_LEVEL_PATTERN.search(str(sample_args or "")))


def executed_steps_the_runtime_will_run(scheduled_steps: int, denoise: float) -> int:
    """How many evaluations a request of this length and denoise actually spends.

    Mirrors the runtime's own truncation, including its refusal to let the tail
    be the whole schedule.
    """
    if denoise <= 0:
        return int(scheduled_steps)
    executed = int(int(scheduled_steps) * float(denoise))
    return min(executed, int(scheduled_steps) - 1) if executed >= int(scheduled_steps) else executed


def scheduled_steps_for_executed_steps(executed_steps: int,
                                       denoise: float,
                                       sample_args: str = "") -> int:
    """The step count to send so the runtime executes `executed_steps` of them."""
    executed_steps = int(executed_steps)
    denoise = float(denoise)
    if denoise <= 0 or strength_is_a_noise_level(sample_args):
        return executed_steps
    if denoise >= 1:
        return executed_steps + 1
    scheduled = math.ceil(executed_steps / denoise)
    # Integer truncation of the product can land one short of the target, so the
    # count is nudged up until the runtime's own arithmetic agrees.
    while executed_steps_the_runtime_will_run(scheduled, denoise) < executed_steps:
        scheduled += 1
    return scheduled
