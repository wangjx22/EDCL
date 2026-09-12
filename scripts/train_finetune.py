#!/usr/bin/env python3
"""Fine-tuning CLI: loads a pretrained encoder checkpoint (from
train_pretrain.py) and trains encoder + task head end-to-end for a
downstream property-prediction task.

Supports both task families the paper claims to evaluate on:
- model.task_type: "regression"  -> MSE loss, reports MAE/RMSE (QM9, ESOL, ...)
- model.task_type: "binary_classification" -> masked BCE-with-logits loss
  (NaN-safe, for MoleculeNet's multi-label missing-value splits), reports
  mean ROC-AUC across valid label columns (BACE, BBBP, ClinTox, Tox21, ...)

Usage:
    python scripts/train_finetune.py --config configs/finetune.yaml
"""
from __future__ import annotations

import argparse
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


def build_dataset(cfg: dict):
    path = cfg["data"].get("path")
    num_targets = cfg["model"].get("num_targets", 1)
    if path:
        return load_pt_dataset(
            path, require_y=True, expected_num_targets=num_targets
        )
    task_type = cfg["model"].get("task_type", "regression")
    label_type = "binary_classification" if task_type == "binary_classification" else "regression"
    return SyntheticMoleculeDataset(
        num_samples=cfg["data"].get("num_synthetic_samples", 256),
        num_targets=num_targets,
        label_type=label_type,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    dataset = build_dataset(cfg)
    val_frac = float(cfg["data"].get("val_fraction", 0.1))
    if len(dataset) < 2:
        raise ValueError("fine-tuning requires at least 2 samples for a non-empty train/validation split")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("data.val_fraction must be strictly between 0 and 1")
    n_val = min(len(dataset) - 1, max(1, int(len(dataset) * val_frac)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    bs = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_molecules)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_molecules)

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
    os.makedirs(tcfg["ckpt_dir"], exist_ok=True)

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

        model.eval()
        val_losses = []
        all_pred, all_y = [], []
        with torch.no_grad():
            for mb in val_loader:
                mb = mb.to(device)
                pred = model(mb.z, mb.pos, mb.batch)
                val_losses.append(loss_fn(pred, mb.y).item())
                all_pred.append(pred)
                all_y.append(mb.y)
        val_loss = sum(val_losses) / max(1, len(val_losses))
        all_pred = torch.cat(all_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)
        if task_type == "binary_classification":
            metrics = classification_metrics(all_pred, all_y)
            print(f"epoch={epoch} val_loss={val_loss:.4f} val_auc={metrics['auc']:.4f} (valid_tasks={metrics['num_valid_tasks']})")
        else:
            metrics = regression_metrics(all_pred, all_y)
            print(f"epoch={epoch} val_loss={val_loss:.4f} val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(tcfg["ckpt_dir"], "best.pt"))
            print(f"new best val_loss={val_loss:.4f}, checkpoint saved")


if __name__ == "__main__":
    main()
