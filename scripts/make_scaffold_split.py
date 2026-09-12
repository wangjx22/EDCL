"""End-to-end example: raw `smiles,label` CSV -> scaffold-split `.pt` files.

This wires together the pieces documented separately in docs/data.md
(the `z`/`pos`/`y` sample format) and `edcl.splits.scaffold_split`
(the MoleculeNet scaffold-split protocol from the paper's Table 1) so a
user does not have to write their own glue code to reproduce the paper's
classification benchmarks from a plain CSV of molecules.

Usage:
    pip install rdkit
    python scripts/make_scaffold_split.py --csv my_data.csv --out_dir splits/

The CSV must have a SMILES column (default name "smiles") and one or more
label columns (default: a single column named "label"; pass
`--label_col task1,task2,...` for multi-task datasets such as Tox21
(12 tasks), ToxCast (~600), SIDER (27), MUV (17), ClinTox (2) or PCBA
(~128) -- the MoleculeNet norm is that most rows have some tasks
unmeasured, encoded as an empty cell; those become `NaN` in `y` and are
automatically excluded from the loss/metrics by
`edcl.metrics.masked_bce_loss` / `classification_metrics` (see
`docs/data.md`). 3-D coordinates are generated with RDKit's ETKDG embedder
+ a quick MMFF94 optimization; molecules that fail to embed (rare, e.g.
disconnected fragments) are skipped with a warning and excluded from every
split so indices stay consistent.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import torch


def _parse_label_cols(label_col: str) -> list[str]:
    cols = [c.strip() for c in label_col.split(",") if c.strip()]
    if not cols:
        raise ValueError(f"--label_col must name at least one column, got {label_col!r}")
    return cols


def _read_csv(path: str, smiles_col: str, label_cols: list[str]):
    """Read SMILES + one-or-more label columns.

    Missing/blank/non-numeric label cells become `float('nan')` (the
    MoleculeNet "unmeasured task" convention); every other numeric cell is
    parsed as-is (so both regression targets and {0,1} classification
    labels work unchanged). Returns `labels` as a list of per-row lists,
    one float per requested column (length 1 for the common single-task
    case).
    """
    smiles, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if smiles_col not in reader.fieldnames:
            raise ValueError(
                f"CSV column '{smiles_col}' not found; available columns: {reader.fieldnames}"
            )
        missing_cols = [c for c in label_cols if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"CSV label column(s) {missing_cols} not found; available columns: {reader.fieldnames}"
            )
        for row in reader:
            smiles.append(row[smiles_col].strip())
            row_labels = []
            for col in label_cols:
                raw = row[col].strip() if row[col] is not None else ""
                if raw == "":
                    row_labels.append(float("nan"))
                else:
                    try:
                        row_labels.append(float(raw))
                    except ValueError:
                        row_labels.append(float("nan"))
            labels.append(row_labels)
    if not smiles:
        raise ValueError(f"No rows read from {path}")
    n_missing = sum(math.isnan(v) for row in labels for v in row)
    if n_missing:
        print(f"Note: {n_missing} missing label value(s) across {len(label_cols)} task column(s) -> NaN (masked out).")
    return smiles, labels


def _smiles_to_sample(smiles: str, label: list[float], seed: int):
    """Embed a 3-D conformer for `smiles` and build a {z, pos, y} sample.

    `label` is a list of one-or-more per-task floats (NaN = unmeasured);
    stored as `y` with shape `[num_tasks]` so single- and multi-task CSVs
    share the same code path.

    Returns None (with a warning) if embedding fails, so the caller can
    skip the molecule without breaking index alignment.

    Delegates the actual RDKit embedding to `edcl.featurize.smiles_to_atoms`
    (shared with `scripts/predict.py` so the two paths cannot drift apart).
    """
    from edcl.featurize import smiles_to_atoms

    atoms = smiles_to_atoms(smiles, seed=seed)
    if atoms is None:
        return None
    atoms["y"] = torch.tensor(label, dtype=torch.float32)
    return atoms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Input CSV with a SMILES column and one or more label columns")
    parser.add_argument("--out_dir", required=True, help="Directory to write train.pt/val.pt/test.pt into")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument(
        "--label_col",
        default="label",
        help="Single column name, or a comma-separated list for multi-task "
             "datasets (e.g. --label_col NR-AR,NR-AR-LBD,... for Tox21). "
             "Blank/unparseable cells become NaN (masked out of loss/metrics).",
    )
    parser.add_argument("--frac_train", type=float, default=0.8)
    parser.add_argument("--frac_val", type=float, default=0.1)
    parser.add_argument("--frac_test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="RDKit conformer-embedding seed")
    args = parser.parse_args(argv)

    try:
        from edcl.splits import scaffold_split
    except ImportError:
        print(
            "ERROR: this script requires the optional RDKit dependency.\n"
            "Install it with `pip install rdkit` and retry.",
            file=sys.stderr,
        )
        return 1

    label_cols = _parse_label_cols(args.label_col)
    smiles, labels = _read_csv(args.csv, args.smiles_col, label_cols)
    print(f"Read {len(smiles)} rows from {args.csv} ({len(label_cols)} task column(s): {label_cols})")

    samples, kept_smiles = [], []
    for smi, lab in zip(smiles, labels):
        sample = _smiles_to_sample(smi, lab, args.seed)
        if sample is not None:
            samples.append(sample)
            kept_smiles.append(smi)
    n_skipped = len(smiles) - len(samples)
    if n_skipped:
        print(f"Skipped {n_skipped} molecule(s) that failed to parse/embed.")
    if not samples:
        print("ERROR: no molecules survived 3-D embedding; nothing to write.", file=sys.stderr)
        return 1

    train_idx, val_idx, test_idx = scaffold_split(
        kept_smiles,
        frac_train=args.frac_train,
        frac_val=args.frac_val,
        frac_test=args.frac_test,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        subset = [samples[i] for i in idx]
        out_path = out_dir / f"{name}.pt"
        torch.save(subset, out_path)
        print(f"Wrote {len(subset)} samples -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
