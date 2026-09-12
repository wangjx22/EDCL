"""Fine-tuning model: pretrained encoder F_phi + randomly-initialized task head
P_task, trained end-to-end (paper's fine-tuning protocol)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import EquivariantEncoder
from .heads import TaskHead


class EDCLFinetuneModel(nn.Module):
    def __init__(self, encoder: EquivariantEncoder, num_targets: int = 1, hidden: int = 128):
        super().__init__()
        self.encoder = encoder
        self.task_head = TaskHead(encoder.hidden_dim, num_targets=num_targets, hidden=hidden)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        num_targets: int = 1,
        map_location="cpu",
        hidden: int = 128,
    ):
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        encoder_cfg = ckpt["encoder_config"]
        encoder = EquivariantEncoder(**encoder_cfg)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        return cls(encoder, num_targets=num_targets, hidden=hidden)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        out = self.encoder(z, pos, batch)
        return self.task_head(out.h_graph)
