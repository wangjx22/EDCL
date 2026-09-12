"""Noise generator G_theta: predicts a per-atom, data-dependent noise scale sigma_i
from the clean-structure invariant atom features h_i (paper: adaptive/energy-aware
noise scale, softplus-parameterized to guarantee sigma_i > 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NoiseGenerator(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Softplus(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, dim] -> sigma: [N, 1], sigma > 0."""
        return self.net(h) + self.eps
