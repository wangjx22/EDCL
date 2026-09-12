"""Task heads on top of the shared equivariant encoder.

- DenoiseHead: EQUIVARIANT. Predicts the displacement vector Delta_hat_i that
  the noisy branch should have undergone, using the encoder's equivariant
  vector channel v_i (already O(3)-equivariant by construction) mixed with an
  invariant per-atom scalar gate. Output shape [N, 3], transforms like a
  vector under global rotations (verified in tests/test_equivariance.py).
- EnergyHead: INVARIANT. Per-atom scalar "energy" from invariant features h_i,
  pooled (mean) to a per-molecule scalar. Used only on the CLEAN branch.
- ProjectionHead: INVARIANT. Optional small MLP projection of h_graph before
  the contrastive loss (standard SimCLR-style projection head); identity by
  default (paper does not describe an explicit projection head; kept as an
  opt-in for stability, off by default so behaviour matches the paper).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenoiseHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """h: [N,dim] invariant, v: [N,3] equivariant -> Delta_hat: [N,3] equivariant."""
        scale = self.gate(h)  # [N,1] invariant scalar
        return scale * v


class EnergyHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N,dim] -> per-atom energy: [N,1]."""
        return self.net(h)


class ProjectionHead(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None, identity: bool = True):
        super().__init__()
        self.identity = identity
        out_dim = out_dim or dim
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, out_dim))

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        return h_graph if self.identity else self.net(h_graph)


class TaskHead(nn.Module):
    """Randomly-initialized fine-tuning head P_task described in the paper's
    fine-tuning protocol: lightweight MLP on top of the pretrained graph
    representation, trained end-to-end with the encoder."""

    def __init__(self, dim: int, num_targets: int = 1, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, num_targets),
        )

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        return self.net(h_graph)
