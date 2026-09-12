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
