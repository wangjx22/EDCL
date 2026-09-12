"""
Minimal molecular graph batch container + collation, and a synthetic dataset
used for unit tests / smoke tests when no real 3D dataset (PCQM4Mv2 / OC20 /
QM9) is available locally.

Real datasets: point ``EDCLPretrainDataset``/``EDCLFinetuneDataset`` at a
``.pt`` file produced by ``scripts/prepare_data.py`` (see ``docs/data.md``);
this module does not itself perform any network download, per the project
convention of keeping data acquisition out of library code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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


def collate_molecules(examples: List[dict]) -> MoleculeBatch:
    """Collate a list of ``{"z": [n], "pos": [n,3], "y": [t]?}`` dicts into a MoleculeBatch."""
    zs, poss, batches, ys = [], [], [], []
    has_y = all(("y" in e and e["y"] is not None) for e in examples)
    for gidx, ex in enumerate(examples):
        n = ex["z"].shape[0]
        zs.append(ex["z"])
        poss.append(ex["pos"])
        batches.append(torch.full((n,), gidx, dtype=torch.long))
        if has_y:
            ys.append(ex["y"].view(1, -1))
    return MoleculeBatch(
        z=torch.cat(zs, dim=0),
        pos=torch.cat(poss, dim=0),
        batch=torch.cat(batches, dim=0),
        y=torch.cat(ys, dim=0) if has_y else None,
    )


class SyntheticMoleculeDataset(Dataset):
    """Randomly generated small molecules for unit tests and CI smoke runs.

    Not physically meaningful, but exercises every shape/broadcast path in
    the model (variable atom counts, batching, equivariance) without
    requiring any dataset download.
    """

    def __init__(self, num_samples: int = 64, min_atoms: int = 4, max_atoms: int = 12,
                 num_targets: int = 1, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        for _ in range(num_samples):
            n = int(torch.randint(min_atoms, max_atoms + 1, (1,), generator=g).item())
            z = torch.randint(1, 10, (n,), generator=g)
            pos = torch.randn(n, 3, generator=g) * 1.5
            y = torch.randn(num_targets, generator=g)
            self.samples.append({"z": z, "pos": pos, "y": y})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
