# Dataset format

Both training CLIs accept `data.path` pointing to a file written with
`torch.save`. The file must contain a non-empty `list` (or tuple) of sample
dictionaries. Each sample has:

| Key | Required | Type and shape | Meaning |
|---|---:|---|---|
| `z` | yes | `torch.long`, `[N]` | Positive atomic numbers |
| `pos` | yes | floating tensor, `[N, 3]` | Finite Cartesian coordinates in Å |
| `y` | see below | floating tensor with at least one value | Graph-level regression target(s) |

`N` must be positive and identical for `z` and `pos`. Every labeled sample in
a batch must have the same number of targets. Mixing labeled and unlabeled
samples is rejected instead of silently discarding labels.

Fine-tuning always requires `y`, with exactly `model.num_targets` values per
sample. Pretraining's energy head is scalar, so it requires exactly one `y`
value per sample when the energy-loss weight `weights.beta` is non-zero (the
default is `10.0`). To run an explicitly unsupervised ablation on structures
without energy labels, set `weights.beta: 0.0`; denoising, KL, and contrastive
terms remain active.

Example:

```python
import torch

samples = [
    {
        "z": torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        "pos": torch.tensor(
            [[0.0, 0.0, 0.0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6],
             [-0.6, 0.6, -0.6], [0.6, -0.6, -0.6]],
            dtype=torch.float32,
        ),
        "y": torch.tensor([-40.1], dtype=torch.float32),
    }
]
torch.save(samples, "molecules.pt")
```

The loader validates the complete file eagerly on CPU and reports the failing
sample index before training starts. Dataset acquisition and chemistry-specific
normalization remain external to this repository so their provenance can be
managed by the user.

## Scaffold splitting (MoleculeNet protocol)

The paper reports MoleculeNet classification results using a **scaffold
split** (Table 1: "Scaffold AUC-ROC"), not a random split: molecules sharing
a Bemis-Murcko scaffold are kept in the same split so structurally related
molecules cannot leak between train/val/test. `edcl.splits.scaffold_split`
reproduces this protocol given the dataset's SMILES strings (in the same
order as the samples in your `.pt` file):

```python
from edcl.splits import scaffold_split

train_idx, val_idx, test_idx = scaffold_split(
    smiles_list, frac_train=0.8, frac_val=0.1, frac_test=0.1
)
```

The returned index lists can be used to slice your sample list before saving
separate `train.pt`/`val.pt`/`test.pt` files. This function requires the
optional `rdkit` dependency (`pip install rdkit`); it is not needed for
QM9/QM7-style random splits or for the core `z`/`pos`/`y` training pipeline.

`scripts/make_scaffold_split.py` wires the above together end to end: it
reads a raw `smiles,label[,label2,...]` CSV, builds `z`/`pos`/`y` samples
with RDKit-generated 3D conformers, applies `scaffold_split`, and writes
`train.pt`/`val.pt`/`test.pt` to an output directory:

```bash
python scripts/make_scaffold_split.py --csv my_dataset.csv --out_dir data/my_dataset_split \
    --frac_train 0.8 --frac_val 0.1 --frac_test 0.1
```

For multi-task datasets (Tox21, ToxCast, SIDER, MUV, ClinTox, PCBA), pass
`--label_col` as a comma-separated list matching your CSV's task columns;
empty cells become `NaN` and are excluded from the loss/AUC by
`edcl.metrics.masked_bce_loss`/`classification_metrics` (the same
missing-label convention MoleculeNet uses):

```bash
python scripts/make_scaffold_split.py --csv tox21.csv --out_dir data/tox21_split \
    --label_col NR-AR,NR-AR-LBD,NR-AhR,SR-p53
```

## Feeding a scaffold split into `train_finetune.py`

Point `data.train_path` / `data.val_path` / `data.test_path` in
`configs/finetune.yaml` at the three files above instead of the legacy
single `data.path`:

```yaml
data:
  train_path: data/my_dataset_split/train.pt
  val_path: data/my_dataset_split/val.pt
  test_path: data/my_dataset_split/test.pt   # optional; enables final test-set scoring
  batch_size: 32
```

In this mode `train_finetune.py` uses the three files **verbatim** (no
internal re-shuffling), so the scaffold separation from `make_scaffold_split.py`
is preserved end to end. If `test_path` is set, once training finishes the
best-on-validation checkpoint is reloaded and scored once on the held-out
test set; the metrics are printed and written to
`<train.ckpt_dir>/test_metrics.json`, matching the paper's train/val/test
evaluation protocol (Table 1). `data.path`/`data.val_fraction` (a single file
with an internal random split, or the built-in synthetic dataset) remain
available as a quick smoke-test mode with no held-out test evaluation.
