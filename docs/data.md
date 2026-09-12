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
normalization/splitting remain external to this repository so their provenance
can be managed by the user.
