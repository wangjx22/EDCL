#!/usr/bin/env python3
"""Pretraining CLI for EDCL (Equivariant Denoising Contrastive Learning).

Usage:
    python scripts/train_pretrain.py --config configs/pretrain.yaml

Falls back to ``SyntheticMoleculeDataset`` when ``data.path`` is null in the
config, so this script is fully runnable end-to-end without any external
dataset (useful for CI / smoke-testing the whole pipeline).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edcl import EDCLPretrainModel, EDCLConfig, EDCLLossWeights
from edcl.data import SyntheticMoleculeDataset, collate_molecules, load_pt_dataset


def build_dataset(cfg: dict):
    path = cfg["data"].get("path")
    if path:
        require_energy = float(cfg.get("weights", {}).get("beta", 10.0)) != 0.0
        return load_pt_dataset(
            path,
            require_y=require_energy,
            expected_num_targets=1 if require_energy else None,
        )
    return SyntheticMoleculeDataset(
        num_samples=cfg["data"].get("num_synthetic_samples", 512),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0), collate_fn=collate_molecules,
    )

    mcfg = cfg["model"]
    wcfg = cfg["weights"]
    config = EDCLConfig(
        num_elements=mcfg.get("num_elements", 119),
        hidden_dim=mcfg.get("hidden_dim", 128),
        num_layers=mcfg.get("num_layers", 4),
        cutoff=mcfg.get("cutoff", 5.0),
        max_neighbors=mcfg.get("max_neighbors", 32),
        omega=mcfg.get("omega", 0.1),
        r_min=mcfg.get("r_min", 0.8),
        tau=mcfg.get("tau", 0.1),
        weights=EDCLLossWeights(alpha=wcfg["alpha"], beta=wcfg["beta"], lam=wcfg["lam"]),
    )
    model = EDCLPretrainModel(config).to(device)

    tcfg = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    os.makedirs(tcfg["ckpt_dir"], exist_ok=True)

    step = 0
    for epoch in range(tcfg["epochs"]):
        for mb in loader:
            mb = mb.to(device)
            target_energy = mb.y.view(-1) if mb.y is not None else None
            total, logs = model(mb.z, mb.pos, mb.batch, target_energy=target_energy)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            if step % tcfg.get("log_every", 50) == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in logs.items())
                print(f"epoch={epoch} step={step} {msg}")
            step += 1

        if (epoch + 1) % tcfg.get("ckpt_every_epoch", 10) == 0 or epoch == tcfg["epochs"] - 1:
            ckpt_path = os.path.join(tcfg["ckpt_dir"], "last.pt")
            torch.save({
                "encoder_state_dict": model.encoder.state_dict(),
                "encoder_config": dict(
                    num_elements=config.num_elements, hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers, cutoff=config.cutoff,
                    max_neighbors=config.max_neighbors,
                ),
                "epoch": epoch,
            }, ckpt_path)
            print(f"saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
