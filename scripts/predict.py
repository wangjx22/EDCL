#!/usr/bin/env python3
"""Inference CLI: load a fine-tuned checkpoint (written by
EDCLFinetuneModel.save_checkpoint, e.g. <ckpt_dir>/best.pt from
train_finetune.py) and predict on new molecules given as SMILES in a CSV.

Usage:
    python scripts/predict.py --checkpoint runs/finetune/best.pt \
        --input molecules.csv --smiles_col smiles --output predictions.csv

Input CSV needs one column of SMILES strings (name configurable via
--smiles_col, default "smiles"); any other columns are passed through
unchanged into the output CSV. Each SMILES is 3-D-embedded with RDKit
(same code path as scripts/make_scaffold_split.py, via edcl.featurize) --
rows that fail to embed get empty predictions and a printed warning,
rather than crashing the whole run.

Output columns are named "pred_0", "pred_1", ... for a multi-task
checkpoint (or just "pred" for a single-task one). For a
binary_classification checkpoint, predictions are probabilities (raw
logits passed through sigmoid); for regression they are the direct
predicted values.

Kept dependency-light (stdlib csv, no pandas) to match the rest of this
repo (see pyproject.toml: torch/numpy/PyYAML only).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from edcl import EDCLFinetuneModel
from edcl.featurize import smiles_to_atoms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a fine-tuned checkpoint (best.pt)")
    parser.add_argument("--input", required=True, help="CSV file with a SMILES column")
    parser.add_argument("--output", required=True, help="where to write predictions CSV")
    parser.add_argument("--smiles_col", default="smiles", help="name of the SMILES column in --input (default: smiles)")
    parser.add_argument("--seed", type=int, default=0, help="RDKit conformer-embedding seed")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda (default: cpu)")
    parser.add_argument("--batch_size", type=int, default=64, help="inference batch size (default: 64)")
    args = parser.parse_args(argv)

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or args.smiles_col not in reader.fieldnames:
            raise SystemExit(
                f"--smiles_col={args.smiles_col!r} not found in {args.input}; "
                f"columns are {reader.fieldnames}"
            )
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    model = EDCLFinetuneModel.load_checkpoint(args.checkpoint, map_location=args.device).to(args.device)
    model.eval()

    smiles_list = [row[args.smiles_col] for row in rows]
    atoms_list = [smiles_to_atoms(s, seed=args.seed) for s in smiles_list]

    num_targets = model.num_targets
    preds = [[None] * num_targets for _ in smiles_list]

    valid_idx = [i for i, a in enumerate(atoms_list) if a is not None]
    with torch.no_grad():
        for start in range(0, len(valid_idx), args.batch_size):
            batch_idx = valid_idx[start:start + args.batch_size]
            if not batch_idx:
                continue
            zs = [atoms_list[i]["z"] for i in batch_idx]
            poss = [atoms_list[i]["pos"] for i in batch_idx]
            batch = torch.cat([torch.full((z.shape[0],), b, dtype=torch.long) for b, z in enumerate(zs)])
            z = torch.cat(zs).to(args.device)
            pos = torch.cat(poss).to(args.device)
            batch = batch.to(args.device)
            out = model(z, pos, batch)
            if model.task_type == "binary_classification":
                out = torch.sigmoid(out)
            out = out.detach().cpu()
            for row_idx, i in enumerate(batch_idx):
                preds[i] = out[row_idx].tolist()

    pred_cols = ["pred"] if num_targets == 1 else [f"pred_{t}" for t in range(num_targets)]
    out_fieldnames = fieldnames + pred_cols
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row, pred in zip(rows, preds):
            out_row = dict(row)
            for col, val in zip(pred_cols, pred):
                out_row[col] = "" if val is None else val
            writer.writerow(out_row)

    n_failed = sum(a is None for a in atoms_list)
    print(f"wrote {len(rows)} predictions to {args.output}"
          + (f" ({n_failed} SMILES failed embedding -> blank rows)" if n_failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
