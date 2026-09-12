"""Exponential Moving Average (EMA) of model parameters.

The paper reports using an EMA of the encoder weights, evaluated/used at
validation and inference time (a common stabiliser for denoising-objective
training). This is a minimal, dependency-free implementation: it tracks a
shadow copy of every trainable float tensor and updates it with
``shadow = decay * shadow + (1 - decay) * param`` after every optimizer step.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.state_dict().items()
            if torch.is_floating_point(p)
        }
        self._non_float_keys = [
            name for name, p in model.state_dict().items() if not torch.is_floating_point(p)
        ]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for name, shadow_param in self.shadow.items():
            shadow_param.mul_(self.decay).add_(sd[name].detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: t.clone() for name, t in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for name, t in state_dict.items():
            self.shadow[name] = t.clone()

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model``'s floating-point parameters/buffers with the EMA shadow."""
        sd = model.state_dict()
        merged = dict(sd)
        merged.update(self.shadow)
        model.load_state_dict(merged)
