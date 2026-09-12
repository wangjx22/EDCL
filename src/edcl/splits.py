"""
Scaffold-based dataset splitting for MoleculeNet-style benchmarks.

The EDCL paper (Table 1) reports MoleculeNet classification results using a
"Scaffold" split rather than a random split: molecules are grouped by their
Bemis-Murcko scaffold, and scaffold groups are assigned to train/val/test so
that molecules sharing a scaffold never leak across splits. This is the
standard protocol used by MoleculeNet / DeepChem and is required for a fair,
reproducible comparison to the paper's numbers.

This module is intentionally decoupled from ``data.py``: it only needs a
list of SMILES strings (index-aligned with whatever ``.pt`` dataset the user
has prepared) and returns index arrays. RDKit is an optional dependency
(only required if scaffold splitting is actually used) and is imported
lazily so the rest of the package works without it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

__all__ = ["compute_scaffold", "scaffold_split"]


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when rdkit absent
        raise ImportError(
            "Scaffold splitting requires RDKit, which is not installed. "
            "Install it with `pip install rdkit` (or `rdkit-pypi` on older "
            "platforms) and retry."
        ) from exc
    return Chem, MurckoScaffold


def compute_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return the Bemis-Murcko scaffold SMILES for a single molecule.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    include_chirality : bool
        Whether to preserve stereochemistry in the scaffold (MoleculeNet's
        canonical scaffold-split implementation defaults to False).

    Returns
    -------
    str
        The scaffold SMILES. Molecules that fail to parse or have no ring
        system fall back to the empty string ``""`` so they are grouped
        together deterministically (matching DeepChem's behaviour) rather
        than raising and aborting the whole split.
    """
    Chem, MurckoScaffold = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception:
        return ""
    return scaffold


def scaffold_split(
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    include_chirality: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic greedy scaffold split (DeepChem/MoleculeNet convention).

    Molecules are grouped by Bemis-Murcko scaffold. Scaffold groups are
    sorted by descending group size (ties broken by first-occurrence index
    for full determinism) and greedily packed into train, then val, then
    test, so that no scaffold group is split across sets.

    Parameters
    ----------
    smiles_list : Sequence[str]
        SMILES strings, index-aligned with the dataset to be split.
    frac_train, frac_val, frac_test : float
        Target fractions; must sum to ~1.0 (validated). ``frac_test`` gets
        whatever remains after train/val are filled, matching DeepChem.
    include_chirality : bool
        See :func:`compute_scaffold`.

    Returns
    -------
    (train_idx, val_idx, test_idx) : tuple of lists of int
        Index lists into ``smiles_list``, disjoint and covering all inputs.
    """
    if len(smiles_list) == 0:
        raise ValueError("smiles_list must be non-empty")
    total = frac_train + frac_val + frac_test
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"frac_train + frac_val + frac_test must sum to 1.0, got {total}"
        )
    if min(frac_train, frac_val, frac_test) < 0:
        raise ValueError("fractions must be non-negative")

    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = compute_scaffold(smi, include_chirality=include_chirality)
        scaffold_to_indices[scaffold].append(idx)

    # Sort groups by (descending size, ascending first index) for determinism.
    groups = sorted(
        scaffold_to_indices.values(), key=lambda g: (-len(g), g[0])
    )

    n_total = len(smiles_list)
    n_train_cutoff = frac_train * n_total
    n_val_cutoff = (frac_train + frac_val) * n_total

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for group in groups:
        if len(train_idx) + len(group) <= n_train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= n_val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    assert len(train_idx) + len(val_idx) + len(test_idx) == n_total
    return train_idx, val_idx, test_idx
