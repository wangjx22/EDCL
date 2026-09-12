"""
Molecular graph batching and dataset utilities.

Real datasets are loaded from a validated ``.pt`` file; see ``docs/data.md``.
``SyntheticMoleculeDataset`` is intentionally only a deterministic smoke-test
fixture and is not a scientific benchmark dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


@dataclass
class MoleculeBatch:
    """A batched collection of molecules.

    Attributes
    ----------
    z : LongTensor [N]        atomic numbers for every atom in the batch
    pos : FloatTensor [N, 3]  3D coordinates (Angstrom)
    batch : LongTensor [N]    graph index each atom belongs to, in [0, B)
    y : Optional FloatTensor [B, T]  downstream regression targets (fine-tuning only)
    num_graphs : int
    """

    z: torch.Tensor
    pos: torch.Tensor
    batch: torch.Tensor
    y: Optional[torch.Tensor] = None

    @property
    def num_graphs(self) -> int:
        return int(self.batch.max().item()) + 1 if self.batch.numel() > 0 else 0

    def to(self, device):
        return MoleculeBatch(
            z=self.z.to(device),
            pos=self.pos.to(device),
            batch=self.batch.to(device),
            y=None if self.y is None else self.y.to(device),
        )

    def clone(self):
        return MoleculeBatch(
            z=self.z.clone(),
            pos=self.pos.clone(),
            batch=self.batch.clone(),
            y=None if self.y is None else self.y.clone(),
        )


def validate_sample(
    sample: dict,
    *,
    require_y: bool = False,
    expected_num_targets: Optional[int] = None,
    index: Optional[int] = None,
) -> None:
    """Validate one serialized molecule and raise an actionable error."""
    where = f"sample {index}" if index is not None else "sample"
    if not isinstance(sample, dict):
        raise TypeError(f"{where} must be a dict, got {type(sample).__name__}")
    missing = {"z", "pos"} - sample.keys()
    if missing:
        raise ValueError(f"{where} is missing required key(s): {sorted(missing)}")

    z, pos = sample["z"], sample["pos"]
    if not isinstance(z, torch.Tensor) or not isinstance(pos, torch.Tensor):
        raise TypeError(f"{where} keys 'z' and 'pos' must be torch tensors")
    if z.dtype != torch.long or z.ndim != 1:
        raise ValueError(f"{where}['z'] must have dtype torch.long and shape [N]")
    if not pos.is_floating_point() or pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{where}['pos'] must be floating point with shape [N, 3]")
    if z.shape[0] == 0:
        raise ValueError(f"{where} must contain at least one atom")
    if pos.shape[0] != z.shape[0]:
        raise ValueError(
            f"{where} has {z.shape[0]} atomic numbers but {pos.shape[0]} positions"
        )
    if (z <= 0).any():
        raise ValueError(f"{where}['z'] must contain positive atomic numbers")
    if not torch.isfinite(pos).all():
        raise ValueError(f"{where}['pos'] contains NaN or infinity")

    y = sample.get("y")
    if require_y and y is None:
        raise ValueError(f"{where} requires a non-null 'y' target")
    if y is not None:
        if not isinstance(y, torch.Tensor) or not y.is_floating_point():
            raise TypeError(f"{where}['y'] must be a floating-point torch tensor")
        if y.ndim > 1:
            raise ValueError(f"{where}['y'] must be a scalar or one-dimensional target vector")
        if y.numel() == 0:
            raise ValueError(f"{where}['y'] must contain at least one target")
        if expected_num_targets is not None and y.numel() != expected_num_targets:
            raise ValueError(
                f"{where}['y'] must contain exactly {expected_num_targets} target(s), "
                f"got {y.numel()}"
            )
        if torch.isinf(y).any():
            raise ValueError(f"{where}['y'] contains infinity")


def load_pt_dataset(
    path: Union[str, PathLike],
    *,
    require_y: bool = False,
    expected_num_targets: Optional[int] = None,
) -> List[dict]:
    """Load and eagerly validate the documented list-of-dicts ``.pt`` format."""
    samples = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(samples, (list, tuple)):
        raise TypeError("dataset file must contain a list or tuple of sample dicts")
    if len(samples) == 0:
        raise ValueError("dataset file contains no samples")
    samples = list(samples)
    for index, sample in enumerate(samples):
        validate_sample(
            sample,
            require_y=require_y,
            expected_num_targets=expected_num_targets,
            index=index,
        )
    return samples


def collate_molecules(examples: Sequence[dict]) -> MoleculeBatch:
    """Collate validated molecule dicts without silently discarding labels."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    for index, example in enumerate(examples):
        validate_sample(example, index=index)

    label_flags = [example.get("y") is not None for example in examples]
    if any(label_flags) and not all(label_flags):
        raise ValueError("a batch cannot mix labeled and unlabeled samples")

    zs, poss, batches, ys = [], [], [], []
    target_width = None
    for graph_index, example in enumerate(examples):
        n_atoms = example["z"].shape[0]
        zs.append(example["z"])
        poss.append(example["pos"])
        batches.append(torch.full((n_atoms,), graph_index, dtype=torch.long))
        if all(label_flags):
            y = example["y"].reshape(1, -1)
            if target_width is None:
                target_width = y.shape[1]
            elif y.shape[1] != target_width:
                raise ValueError("all labels in a batch must have the same number of targets")
            ys.append(y)
    return MoleculeBatch(
        z=torch.cat(zs, dim=0),
        pos=torch.cat(poss, dim=0),
        batch=torch.cat(batches, dim=0),
        y=torch.cat(ys, dim=0) if all(label_flags) else None,
    )


class SyntheticMoleculeDataset(Dataset):
    """Randomly generated small molecules for unit tests and CI smoke runs.

    Not physically meaningful, but exercises every shape/broadcast path in
    the model (variable atom counts, batching, equivariance) without
    requiring any dataset download.
    """

    def __init__(self, num_samples: int = 64, min_atoms: int = 4, max_atoms: int = 12,
                 num_targets: int = 1, seed: int = 0, label_type: str = "regression",
                 nan_label_prob: float = 0.0):
        """label_type: "regression" -> continuous y ~ N(0,1);
        "binary_classification" -> y in {0., 1.}, with `nan_label_prob`
        fraction of entries randomly set to NaN to exercise masked-loss /
        missing-label handling (mirrors MoleculeNet's sparse label matrices).
        """
        if label_type not in ("regression", "binary_classification"):
            raise ValueError(f"label_type must be 'regression' or 'binary_classification', got {label_type!r}")
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        for _ in range(num_samples):
            n = int(torch.randint(min_atoms, max_atoms + 1, (1,), generator=g).item())
            z = torch.randint(1, 10, (n,), generator=g)
            pos = torch.randn(n, 3, generator=g) * 1.5
            if label_type == "regression":
                y = torch.randn(num_targets, generator=g)
            else:
                y = (torch.rand(num_targets, generator=g) > 0.5).float()
                if nan_label_prob > 0:
                    drop = torch.rand(num_targets, generator=g) < nan_label_prob
                    y = torch.where(drop, torch.full_like(y, float("nan")), y)
            self.samples.append({"z": z, "pos": pos, "y": y})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
