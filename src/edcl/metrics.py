"""Loss functions and evaluation metrics for downstream fine-tuning.

Supports the paper's two families of downstream tasks it claims to
evaluate on (MoleculeNet + QM9): MoleculeNet is majority binary/
multi-label classification (BACE, BBBP, ClinTox, HIV, MUV, PCBA, SIDER,
Tox21, ToxCast, ...) while QM9/ESOL/FreeSolv/Lipophilicity are
regression. MoleculeNet splits commonly contain missing labels (NaN) for
some (molecule, task) pairs in multi-task classification -- both the
loss and metrics below mask those out rather than crashing or biasing
the loss, matching standard MoleculeNet fine-tuning practice.
"""
from __future__ import annotations

import torch


def _valid_mask(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over regression targets, ignoring NaN entries (missing labels)."""
    mask = _valid_mask(target)
    if not mask.any():
        return pred.sum() * 0.0
    diff = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
    return (diff * mask).sum() / mask.sum().clamp_min(1)


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits over (possibly multi-label) binary targets in {0,1},
    ignoring NaN entries (missing labels) -- standard MoleculeNet protocol."""
    mask = _valid_mask(target)
    if not mask.any():
        return logits.sum() * 0.0
    safe_target = torch.nan_to_num(target, nan=0.0)
    elem = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, safe_target, reduction="none"
    )
    return (elem * mask).sum() / mask.sum().clamp_min(1)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mask = _valid_mask(target)
    if not mask.any():
        return {"mae": float("nan"), "rmse": float("nan")}
    diff = (pred - target)[mask]
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()
    return {"mae": mae, "rmse": rmse}


def _roc_auc_1d(y_true: torch.Tensor, y_score: torch.Tensor) -> float | None:
    """Rank-based (Mann-Whitney U) ROC-AUC for one column, no sklearn dep.

    Returns None if the column has only one class present (AUC undefined),
    so callers can skip it rather than reporting a meaningless 0.5/NaN.
    """
    mask = torch.isfinite(y_true)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float)
    # average ranks for ties
    sorted_scores = y_score[order]
    sorted_ranks = ranks[order]
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = sorted_ranks[i:j + 1].mean()
            sorted_ranks[i:j + 1] = avg
        i = j + 1
    ranks[order] = sorted_ranks
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    """Mean ROC-AUC across task columns that have both classes present in
    this batch/split. Falls back gracefully (reports NaN, does not crash)
    if every column is single-class, which is common on tiny smoke-test
    batches."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
        target = target.unsqueeze(1)
    aucs = []
    for col in range(logits.shape[1]):
        auc = _roc_auc_1d(target[:, col], logits[:, col])
        if auc is not None:
            aucs.append(auc)
    mean_auc = sum(aucs) / len(aucs) if aucs else float("nan")
    return {"auc": mean_auc, "num_valid_tasks": len(aucs)}
