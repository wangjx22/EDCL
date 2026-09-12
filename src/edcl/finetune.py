"""Fine-tuning model: pretrained encoder F_phi + randomly-initialized task head
P_task, trained end-to-end (paper's fine-tuning protocol)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import EquivariantEncoder
from .heads import TaskHead


class EDCLFinetuneModel(nn.Module):
    """task_type is metadata only (not used inside forward): the head always
    emits raw un-squashed values ("logits" for classification, direct
    predictions for regression). The training script picks the matching
    loss (masked_bce_loss / masked_mse_loss, see edcl.metrics) based on
    this field -- keeping the model itself loss-agnostic."""

    def __init__(self, encoder: EquivariantEncoder, num_targets: int = 1, hidden: int = 128,
                 task_type: str = "regression"):
        super().__init__()
        if task_type not in ("regression", "binary_classification"):
            raise ValueError(f"unknown task_type={task_type!r}; expected 'regression' or 'binary_classification'")
        self.encoder = encoder
        self.task_head = TaskHead(encoder.hidden_dim, num_targets=num_targets, hidden=hidden)
        self.task_type = task_type

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        num_targets: int = 1,
        map_location="cpu",
        hidden: int = 128,
        use_ema: bool = True,
        task_type: str = "regression",
    ):
        """Load a pretraining checkpoint produced by ``train_pretrain.py``.

        If ``use_ema`` (default) and the checkpoint contains EMA-averaged
        encoder weights (``ema_encoder_state_dict``, written when
        ``train.ema_decay > 0`` in the pretrain config), those are used
        instead of the raw (last-step) weights — matching the paper's
        protocol of evaluating/fine-tuning from the EMA weights. Falls back
        to the raw weights if no EMA state is present.
        """
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        encoder_cfg = ckpt["encoder_config"]
        encoder = EquivariantEncoder(**encoder_cfg)
        if use_ema and "ema_encoder_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["ema_encoder_state_dict"])
        else:
            encoder.load_state_dict(ckpt["encoder_state_dict"])
        return cls(encoder, num_targets=num_targets, hidden=hidden, task_type=task_type)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        out = self.encoder(z, pos, batch)
        return self.task_head(out.h_graph)
