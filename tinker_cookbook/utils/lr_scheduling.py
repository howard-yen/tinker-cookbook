import logging
import math

from typing import Literal

logger = logging.getLogger(__name__)


LRSchedule = Literal["linear", "cosine", "constant"]


def compute_schedule_lr_multiplier(
    lr_schedule: LRSchedule, step: int, total_steps: int, warmup_steps: int = 0,
) -> float:
    """
    What factor to multiply the base LR by due to the LR schedule.

    If warmup_steps > 0, the LR linearly ramps from 0 to the base learning
    rate over the first ``warmup_steps`` steps (multiplier goes 0 → 1). After
    warmup the decay schedule (linear, cosine, or constant) is applied over
    the remaining ``total_steps - warmup_steps`` steps, decaying from the peak.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

    # Warmup phase: linear ramp from 0 to 1
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps

    # Decay phase: schedule applied over the post-warmup range
    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps

    if lr_schedule == "linear":
        return 1 - decay_step / decay_steps if decay_steps > 0 else 1.0
    elif lr_schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps)) if decay_steps > 0 else 1.0
    elif lr_schedule == "constant":
        return 1.0
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")
