#!/usr/bin/env python3
"""Fine-tuning CLI: loads a pretrained encoder checkpoint (from
train_pretrain.py) and trains encoder + task head end-to-end for a
downstream property-prediction task.

Supports both task families the paper claims to evaluate on:
- model.task_type: "regression"  -> MSE loss, reports MAE/RMSE (QM9, ESOL, ...)
- model.task_type: "binary_classification" -> masked BCE-with-logits loss
  (NaN-safe, for MoleculeNet's multi-label missing-value splits), reports
  mean ROC-AUC across valid label columns (BACE, BBBP, ClinTox, Tox21, ...)

Two data modes (see docs/data.md):
1. Paper-faithful scaffold split (recommended, matches Table 1 protocol):
   set data.train_path / data.val_path / data.test_path to the three files
   produced by `scripts/make_scaffold_split.py`. These are used verbatim,
   with NO re-splitting, so scaffold separation is preserved end to end.
   A held-out test-set evaluation is run once at the end of training using
   the best validation checkpoint, and written to
   `<ckpt_dir>/test_metrics.json`.
2. Legacy/smoke mode: set data.path to a single .pt file (or leave it null
   to fall back to the synthetic dataset) and data.val_fraction controls an
   internal *random* train/val split. No test-set evaluation is performed
   in this mode since there is no held-out test split to score.

Usage:
    python scripts/train_finetune.py --config configs/finetune.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edcl import EDCLFinetuneModel
from edcl.data import SyntheticMoleculeDataset, collate_molecules, load_pt_dataset
from edcl.schedule import build_warmup_cosine_scheduler
from edcl.metrics import masked_bce_loss, masked_mse_loss, classification_metrics, regression_metrics


def build_datasets(cfg: dict):
    """Returns (train_ds, val_ds, test_ds). test_ds is None unless the
    paper-faithful 3-path mode (data.train_path/val_path[/test_path]) is
    used."""
    data_cfg = cfg["data"]
    num_targets = cfg["model"].get("num_targets", 1)

    train_path = data_cfg.get("train_path")
    val_path = data_cfg.get("val_path")
    if train_path and val_path:
        train_ds = load_pt_dataset(train_path, require_y=True, expected_num_targets=num_targets)
        val_ds = load_pt_dataset(val_path, require_y=True, expected_num_targets=num_targets)
        test_path = data_cfg.get("test_path")
        test_ds = (
            load_pt_dataset(test_path, require_y=True, expected_num_targets=num_targets)
            if test_path
            else None
        )
        return train_ds, val_ds, test_ds

    # Legacy single-file / synthetic mode: internal random split, no test set.
    path = data_cfg.get("path")
    if path:
        dataset = load_pt_dataset(path, require_y=True, expected_num_targets=num_targets)
    else:
        task_type = cfg["model"].get("task_type", "regression")
        label_type = "binary_classification" if task_type == "binary_classification" else "regression"
        dataset = SyntheticMoleculeDataset(
            num_samples=data_cfg.get("num_synthetic_samples", 256),
            num_targets=num_targets,
            label_type=label_type,
        )
    val_frac = float(data_cfg.get("val_fraction", 0.1))
    if len(dataset) < 2:
        raise ValueError("fine-tuning requires at least 2 samples for a non-empty train/validation split")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("data.val_fraction must be strictly between 0 and 1")
    n_val = min(len(dataset) - 1, max(1, int(len(dataset) * val_frac)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return train_ds, val_ds, None


def evaluate(model, loader, loss_fn, task_type, device):
    model.eval()
    losses = []
    all_pred, all_y = [], []
    with torch.no_grad():
        for mb in loader:
            mb = mb.to(device)
            pred = model(mb.z, mb.pos, mb.batch)
            losses.append(loss_fn(pred, mb.y).item())
            all_pred.append(pred)
            all_y.append(mb.y)
    loss = sum(losses) / max(1, len(losses))
    all_pred = torch.cat(all_pred, dim=0)
    all_y = torch.cat(all_y, dim=0)
    if task_type == "binary_classification":
        metrics = classification_metrics(all_pred, all_y)
    else:
        metrics = regression_metrics(all_pred, all_y)
    metrics["loss"] = loss
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    train_ds, val_ds, test_ds = build_datasets(cfg)
    scaffold_mode = test_ds is not None or (
        cfg["data"].get("train_path") and cfg["data"].get("val_path")
    )

    bs = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_molecules)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_molecules)
    test_loader = (
        DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_molecules)
        if test_ds is not None
        else None
    )

    num_targets = cfg["model"].get("num_targets", 1)
    head_hidden = cfg["model"].get("hidden", 128)
    task_type = cfg["model"].get("task_type", "regression")
    if task_type not in ("regression", "binary_classification"):
        raise ValueError(f"model.task_type must be 'regression' or 'binary_classification', got {task_type!r}")
    ckpt = cfg.get("pretrained_ckpt")
    use_ema = bool(cfg.get("use_ema", True))
    if ckpt and os.path.exists(ckpt):
        model = EDCLFinetuneModel.from_pretrained(
            ckpt, num_targets=num_targets, hidden=head_hidden, use_ema=use_ema, task_type=task_type
        )
        print(f"loaded pretrained encoder from {ckpt}" + (" (EMA weights)" if use_ema else " (raw weights)"))
    else:
        from edcl.backbone import EquivariantEncoder
        encoder = EquivariantEncoder()
        model = EDCLFinetuneModel(
            encoder, num_targets=num_targets, hidden=head_hidden, task_type=task_type
        )
        print("WARNING: no pretrained checkpoint found, training encoder from scratch")
    model = model.to(device)

    tcfg = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_warmup_cosine_scheduler(
        opt,
        warmup_steps=warmup_epochs * steps_per_epoch,
        total_steps=tcfg["epochs"] * steps_per_epoch,
        min_lr_ratio=tcfg.get("min_lr_ratio", 0.0),
    )
    loss_fn = masked_bce_loss if task_type == "binary_classification" else masked_mse_loss
    ckpt_dir = tcfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")

    step = 0
    best_val = float("inf")
    for epoch in range(tcfg["epochs"]):
        model.train()
        for mb in train_loader:
            mb = mb.to(device)
            pred = model(mb.z, mb.pos, mb.batch)
            loss = loss_fn(pred, mb.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            scheduler.step()
            if step % tcfg.get("log_every", 20) == 0:
                print(f"epoch={epoch} step={step} lr={scheduler.get_last_lr()[0]:.2e} train_loss={loss.item():.4f}")
            step += 1

        metrics = evaluate(model, val_loader, loss_fn, task_type, device)
        if task_type == "binary_classification":
            print(f"epoch={epoch} val_loss={metrics['loss']:.4f} val_auc={metrics['auc']:.4f} (valid_tasks={metrics['num_valid_tasks']})")
        else:
            print(f"epoch={epoch} val_loss={metrics['loss']:.4f} val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f}")

        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            model.save_checkpoint(best_path)
            print(f"new best val_loss={best_val:.4f}, checkpoint saved")

    if test_loader is not None:
        # Reload the best-on-validation checkpoint before scoring the
        # held-out (scaffold-disjoint) test set, matching the paper's
        # evaluation protocol.
        if os.path.exists(best_path):
            model = EDCLFinetuneModel.load_checkpoint(best_path, map_location=device).to(device)
            print(f"reloaded best checkpoint from {best_path} for test evaluation")
        test_metrics = evaluate(model, test_loader, loss_fn, task_type, device)
        if task_type == "binary_classification":
            print(f"TEST  loss={test_metrics['loss']:.4f} test_auc={test_metrics['auc']:.4f} (valid_tasks={test_metrics['num_valid_tasks']})")
        else:
            print(f"TEST  loss={test_metrics['loss']:.4f} test_mae={test_metrics['mae']:.4f} test_rmse={test_metrics['rmse']:.4f}")
        summary_path = os.path.join(ckpt_dir, "test_metrics.json")
        with open(summary_path, "w") as f:
            json.dump({k: (v if isinstance(v, (int, float)) else float(v)) for k, v in test_metrics.items()}, f, indent=2)
        print(f"wrote test metrics -> {summary_path}")
    elif scaffold_mode:
        print("WARNING: 3-path mode requested but no data.test_path given; skipping test-set evaluation")
    else:
        print("(no data.test_path/held-out test set configured; skipping final test evaluation. "
              "Use data.train_path/val_path/test_path from scripts/make_scaffold_split.py for a "
              "paper-faithful scaffold-split evaluation with a held-out test metric.)")


if __name__ == "__main__":
    main()
