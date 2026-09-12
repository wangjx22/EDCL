"""Closes the gap found by round-9 verification: make_scaffold_split.py
produces train/val/test .pt files that train_finetune.py previously could
not consume (it only did a random split over one file, and never scored a
held-out test set). This exercises the real, full pipeline end-to-end:

    tiny CSV -> scripts/make_scaffold_split.py -> train.pt/val.pt/test.pt
             -> scripts/train_finetune.py (3-path mode) -> test_metrics.json

and asserts the whole thing runs and reports a genuine test-set metric.
Skips (instead of failing) if RDKit is not installed, since it is an
optional dependency documented in docs/data.md.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCRIPTS = os.path.join(ROOT, "scripts")

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rdkit") is None,
    reason="make_scaffold_split.py requires the optional rdkit dependency",
)

# 15 ring-bearing molecules, each with a *distinct* Bemis-Murcko scaffold
# (unlike acyclic molecules, which all collapse to the same empty-string
# "no scaffold" bucket). This keeps scaffold_split's greedy group-packing
# from dumping everything into one bucket, so with 0.6/0.2/0.2 fractions
# the tiny toy dataset still yields a non-empty train/val/test split.
CSV_ROWS = [
    ("c1ccccc1", 0.1),          # benzene
    ("c1ccncc1", 0.2),          # pyridine
    ("c1cncnc1", 0.3),          # pyrimidine
    ("c1ccoc1", 0.4),           # furan
    ("c1ccsc1", 0.5),           # thiophene
    ("c1cc[nH]c1", 0.6),        # pyrrole
    ("c1cnc[nH]1", 0.7),        # imidazole
    ("c1ccc2[nH]ccc2c1", 0.8),  # indole
    ("c1ccc2ccccc2c1", 0.9),    # naphthalene
    ("c1ccc2ncccc2c1", 1.0),    # quinoline
    ("C1CCCCC1", 1.1),          # cyclohexane
    ("C1CCCC1", 1.2),           # cyclopentane
    ("C1CCNCC1", 1.3),          # piperidine
    ("C1COCCN1", 1.4),          # morpholine
    ("c1ccc2c(c1)cccn2", 1.5),  # isoquinoline-ish
]


def _write_csv(path):
    with open(path, "w") as f:
        f.write("smiles,label\n")
        for smi, lab in CSV_ROWS:
            f.write(f"{smi},{lab}\n")


def test_scaffold_split_feeds_finetune_three_path_mode(tmp_path):
    csv_path = tmp_path / "toy.csv"
    split_dir = tmp_path / "splits"
    ckpt_dir = tmp_path / "ckpt"
    _write_csv(csv_path)

    split_cmd = [
        sys.executable, os.path.join(SCRIPTS, "make_scaffold_split.py"),
        "--csv", str(csv_path), "--out_dir", str(split_dir),
        "--frac_train", "0.6", "--frac_val", "0.2", "--frac_test", "0.2",
        "--seed", "0",
    ]
    r = subprocess.run(split_cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"make_scaffold_split.py failed:\n{r.stdout}\n{r.stderr}"
    for name in ("train.pt", "val.pt", "test.pt"):
        assert (split_dir / name).exists(), f"missing {name}: {r.stdout}"

    cfg_text = textwrap.dedent(f"""
    seed: 42
    device: cpu
    data:
      train_path: {split_dir / 'train.pt'}
      val_path: {split_dir / 'val.pt'}
      test_path: {split_dir / 'test.pt'}
      batch_size: 4
    pretrained_ckpt: null
    use_ema: false
    model:
      num_targets: 1
      hidden: 8
      task_type: regression
    train:
      epochs: 1
      lr: 1.0e-3
      weight_decay: 0.0
      grad_clip: 5.0
      log_every: 50
      ckpt_dir: {ckpt_dir}
      warmup_epochs: 0
      min_lr_ratio: 1.0
    """)
    cfg_path = tmp_path / "ft.yaml"
    cfg_path.write_text(cfg_text)

    ft_cmd = [sys.executable, os.path.join(SCRIPTS, "train_finetune.py"), "--config", str(cfg_path)]
    r = subprocess.run(ft_cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"train_finetune.py (3-path mode) failed:\n{r.stdout}\n{r.stderr}"
    assert "TEST" in r.stdout, f"no held-out test evaluation printed:\n{r.stdout}"

    metrics_path = ckpt_dir / "test_metrics.json"
    assert metrics_path.exists(), "test_metrics.json was not written"
    metrics = json.loads(metrics_path.read_text())
    assert "mae" in metrics and "rmse" in metrics
