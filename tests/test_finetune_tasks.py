"""Covers the two downstream task families the paper evaluates on
(MoleculeNet classification + QM9/ESOL-style regression), added after
round-5 verification found train_finetune.py hardcoded MSE-only."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import torch

from edcl import EDCLPretrainModel, EDCLConfig, EDCLFinetuneModel
from edcl.data import SyntheticMoleculeDataset, collate_molecules
from edcl.metrics import masked_mse_loss, masked_bce_loss, regression_metrics, classification_metrics


def _tiny_encoder():
    cfg = EDCLConfig(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    return EDCLPretrainModel(cfg).encoder


def test_masked_mse_loss_ignores_nan_targets():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.5, float("nan")], [float("nan"), 5.0]])
    loss = masked_mse_loss(pred, target)
    expected = ((1.0 - 1.5) ** 2 + (4.0 - 5.0) ** 2) / 2
    assert torch.isfinite(loss)
    assert abs(loss.item() - expected) < 1e-5


def test_masked_bce_loss_ignores_nan_and_matches_manual():
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.3]])
    target = torch.tensor([[1.0, 0.0], [1.0, float("nan")]])
    loss = masked_bce_loss(logits, target)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0, -1.0, 0.5]), torch.tensor([1.0, 0.0, 1.0])
    )
    assert torch.isfinite(loss)
    assert abs(loss.item() - manual.item()) < 1e-5


def test_masked_bce_loss_all_nan_returns_zero_not_nan():
    logits = torch.full((3, 1), 0.1)
    target = torch.full((3, 1), float("nan"))
    loss = masked_bce_loss(logits, target)
    assert torch.isfinite(loss)


def test_regression_metrics_mae_rmse():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 5.0])
    m = regression_metrics(pred, target)
    assert abs(m["mae"] - (0 + 0 + 2) / 3) < 1e-5
    assert abs(m["rmse"] - math.sqrt((0 + 0 + 4) / 3)) < 1e-5


def test_classification_metrics_auc_perfect_separation():
    logits = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    target = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    m = classification_metrics(logits, target)
    assert abs(m["auc"] - 1.0) < 1e-6
    assert m["num_valid_tasks"] == 1


def test_classification_metrics_single_class_column_skipped_not_crash():
    # column 0: all label 1 -> undefined AUC, must be skipped, not raise
    logits = torch.tensor([[0.1, -1.0], [0.2, 1.0], [0.3, -0.5], [0.4, 0.5]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    m = classification_metrics(logits, target)
    assert m["num_valid_tasks"] == 1
    assert math.isfinite(m["auc"])


def test_finetune_model_regression_end_to_end_step():
    torch.manual_seed(0)
    encoder = _tiny_encoder()
    model = EDCLFinetuneModel(encoder, num_targets=1, hidden=16, task_type="regression")
    ds = SyntheticMoleculeDataset(num_samples=6, num_targets=1, seed=1, label_type="regression")
    mb = collate_molecules([ds[i] for i in range(len(ds))])
    pred = model(mb.z, mb.pos, mb.batch)
    loss = masked_mse_loss(pred, mb.y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_finetune_model_classification_end_to_end_step_with_missing_labels():
    torch.manual_seed(0)
    encoder = _tiny_encoder()
    model = EDCLFinetuneModel(encoder, num_targets=3, hidden=16, task_type="binary_classification")
    ds = SyntheticMoleculeDataset(
        num_samples=8, num_targets=3, seed=2, label_type="binary_classification", nan_label_prob=0.3
    )
    mb = collate_molecules([ds[i] for i in range(len(ds))])
    assert torch.isnan(mb.y).any(), "test fixture should exercise missing-label masking"
    pred = model(mb.z, mb.pos, mb.batch)
    loss = masked_bce_loss(pred, mb.y)
    assert torch.isfinite(loss)
    loss.backward()
    metrics = classification_metrics(pred.detach(), mb.y)
    assert 0 <= metrics["num_valid_tasks"] <= 3
    if metrics["num_valid_tasks"] > 0:
        assert 0.0 <= metrics["auc"] <= 1.0


def test_from_pretrained_defaults_to_regression_task_type(tmp_path):
    torch.manual_seed(0)
    cfg = EDCLConfig(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    pretrain = EDCLPretrainModel(cfg)
    ckpt_path = str(tmp_path / "encoder.pt")
    torch.save({
        "encoder_state_dict": pretrain.encoder.state_dict(),
        "encoder_config": dict(num_elements=119, hidden_dim=16, num_layers=2,
                                num_rbf=32, cutoff=5.0, max_neighbors=8),
    }, ckpt_path)
    ft = EDCLFinetuneModel.from_pretrained(ckpt_path, num_targets=2, hidden=16, use_ema=False)
    assert ft.task_type == "regression"
    ft_cls = EDCLFinetuneModel.from_pretrained(
        ckpt_path, num_targets=2, hidden=16, use_ema=False, task_type="binary_classification"
    )
    assert ft_cls.task_type == "binary_classification"
