"""Learning-rate schedule: linear warmup followed by cosine annealing.

The paper (Table 2) trains with a warmup phase followed by cosine-annealed
decay of the learning rate, for both pretraining and fine-tuning. This module
is a small, dependency-free (`torch.optim.lr_scheduler.LambdaLR`-based)
implementation of that schedule, driven entirely by config values so it can
be disabled (``warmup_steps=0`` and ``total_steps<=warmup_steps``) without
touching the training loop.
"""
from __future__ import annotations

import math

import torch


def warmup_cosine_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0) -> float:
    """Multiplicative LR factor at ``step`` (0-indexed optimizer step).

    - Linear warmup from 0 -> 1 over ``warmup_steps`` steps.
    - Cosine decay from 1 -> ``min_lr_ratio`` over the remaining steps.
    - Constant at ``min_lr_ratio`` once ``step >= total_steps``.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / float(warmup_steps)
    remaining = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / float(remaining), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a per-optimizer-step ``LambdaLR`` implementing warmup+cosine.

    ``total_steps`` must be expressed in optimizer steps (not epochs); the
    caller (training script) is responsible for calling ``scheduler.step()``
    once per optimizer step.
    """
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), warmup_steps + 1)
    fn = lambda step: warmup_cosine_lambda(step, warmup_steps, total_steps, min_lr_ratio)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
