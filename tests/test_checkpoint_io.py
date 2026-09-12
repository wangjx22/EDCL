"""Tests for EDCLFinetuneModel.save_checkpoint / load_checkpoint (round-15
fix: previously train_finetune.py saved plain state_dict()s to best.pt,
which could not be reloaded without the caller separately re-deriving the
exact encoder architecture kwargs; scripts/predict.py needs a
self-describing checkpoint to do inference on new data without access to
the original training config)."""
from __future__ import annotations

import torch

from edcl.backbone import EquivariantEncoder
from edcl.finetune import EDCLFinetuneModel


def test_checkpoint_roundtrip_regression(tmp_path):
    encoder = EquivariantEncoder(hidden_dim=16, num_layers=1, num_rbf=8, max_neighbors=8)
    model = EDCLFinetuneModel(encoder, num_targets=3, hidden=8, task_type="regression")
    ckpt_path = tmp_path / "best.pt"
    model.save_checkpoint(str(ckpt_path))

    loaded = EDCLFinetuneModel.load_checkpoint(str(ckpt_path))
    assert loaded.num_targets == 3
    assert loaded.task_type == "regression"
    assert loaded.encoder.hidden_dim == 16

    # Weights must match exactly (not just architecture).
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_checkpoint_roundtrip_classification_and_forward_matches(tmp_path):
    torch.manual_seed(0)
    encoder = EquivariantEncoder(hidden_dim=16, num_layers=1, num_rbf=8, max_neighbors=8)
    model = EDCLFinetuneModel(encoder, num_targets=2, hidden=8, task_type="binary_classification")
    model.eval()
    ckpt_path = tmp_path / "best.pt"
    model.save_checkpoint(str(ckpt_path))

    loaded = EDCLFinetuneModel.load_checkpoint(str(ckpt_path))
    loaded.eval()

    z = torch.randint(1, 10, (6,))
    pos = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    with torch.no_grad():
        out1 = model(z, pos, batch)
        out2 = loaded(z, pos, batch)
    assert torch.allclose(out1, out2)


def test_load_checkpoint_rejects_legacy_bare_state_dict(tmp_path):
    encoder = EquivariantEncoder(hidden_dim=8, num_layers=1, num_rbf=4, max_neighbors=4)
    model = EDCLFinetuneModel(encoder, num_targets=1, hidden=4, task_type="regression")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), str(legacy_path))  # old (pre-fix) format

    try:
        EDCLFinetuneModel.load_checkpoint(str(legacy_path))
        assert False, "expected a clear error for legacy bare state_dict checkpoints"
    except (KeyError, ValueError, TypeError):
        pass
