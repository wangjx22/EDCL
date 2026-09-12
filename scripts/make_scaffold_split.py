"""End-to-end example: raw `smiles,label` CSV -> scaffold-split `.pt` files.

This wires together the pieces documented separately in docs/data.md
(the `z`/`pos`/`y` sample format) and `edcl.splits.scaffold_split`
(the MoleculeNet scaffold-split protocol from the paper's Table 1) so a
user does not have to write their own glue code to reproduce the paper's
classification benchmarks from a plain CSV of molecules.

Usage:
    pip install rdkit
    python scripts/make_scaffold_split.py --csv my_data.csv --out_dir splits/

The CSV must have a SMILES column (default name "smiles") and a label
column (default name "label"). 3-D coordinates are generated with RDKit's
ETKDG embedder + a quick MMFF94 optimization; molecules that fail to embed
(rare, e.g. disconnected fragments) are skipped with a warning and excluded
from every split so indices stay consistent.
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import torch


def _read_csv(path: str, smiles_col: str, label_col: str):
    smiles, labels = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if smiles_col not in reader.fieldnames:
            raise ValueError(
                f"CSV column '{smiles_col}' not found; available columns: {reader.fieldnames}"
            )
        if label_col not in reader.fieldnames:
            raise ValueError(
                f"CSV column '{label_col}' not found; available columns: {reader.fieldnames}"
            )
        for row in reader:
            smiles.append(row[smiles_col].strip())
            labels.append(float(row[label_col]))
    if not smiles:
        raise ValueError(f"No rows read from {path}")
    return smiles, labels


def _smiles_to_sample(smiles: str, label: float, seed: int):
    """Embed a 3-D conformer for `smiles` and build a {z, pos, y} sample.

    Returns None (with a warning) if embedding fails, so the caller can
    skip the molecule without breaking index alignment.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"Could not parse SMILES, skipping: {smiles!r}")
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        warnings.warn(f"3-D embedding failed, skipping: {smiles!r}")
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # optimization is best-effort; unoptimized embedding is still valid

    conf = mol.GetConformer()
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.float32)
    return {"z": z, "pos": pos, "y": y}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="Input CSV with a SMILES column and a label column")
    parser.add_argument("--out_dir", required=True, help="Directory to write train.pt/val.pt/test.pt into")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--label_col", default="label")
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

    smiles, labels = _read_csv(args.csv, args.smiles_col, args.label_col)
    print(f"Read {len(smiles)} rows from {args.csv}")

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
