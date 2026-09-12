"""Tests for scaffold-based dataset splitting (edcl.splits).

Skips gracefully if RDKit is not installed in the test environment, since
RDKit is an optional dependency (see requirements.txt).
"""
import pytest

rdkit = pytest.importorskip("rdkit", reason="rdkit not installed (optional dep)")

from edcl.splits import compute_scaffold, scaffold_split


# A small, deterministic set of SMILES with two distinct scaffolds:
# - benzene-based: c1ccccc1X  (4 molecules)
# - cyclohexane-based: C1CCCCC1X (2 molecules)
BENZENE_DERIVS = [
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1CCC",
    "c1ccccc1N",
]
CYCLOHEXANE_DERIVS = [
    "C1CCCCC1C",
    "C1CCCCC1N",
]
SMILES_LIST = BENZENE_DERIVS + CYCLOHEXANE_DERIVS


def test_compute_scaffold_groups_related_molecules():
    scaffolds = [compute_scaffold(s) for s in SMILES_LIST]
    # All benzene derivatives share one scaffold, all cyclohexane share another.
    assert len(set(scaffolds[:4])) == 1
    assert len(set(scaffolds[4:])) == 1
    assert scaffolds[0] != scaffolds[4]


def test_compute_scaffold_invalid_smiles_is_empty_string():
    assert compute_scaffold("not_a_smiles!!!") == ""


def test_scaffold_split_no_scaffold_leakage():
    train_idx, val_idx, test_idx = scaffold_split(
        SMILES_LIST, frac_train=0.6, frac_val=0.2, frac_test=0.2
    )
    # Partition covers everything exactly once.
    all_idx = sorted(train_idx + val_idx + test_idx)
    assert all_idx == list(range(len(SMILES_LIST)))

    scaffolds = [compute_scaffold(s) for s in SMILES_LIST]
    train_scaffolds = {scaffolds[i] for i in train_idx}
    val_scaffolds = {scaffolds[i] for i in val_idx}
    test_scaffolds = {scaffolds[i] for i in test_idx}
    # No scaffold group is split across two different sets.
    assert not (train_scaffolds & val_scaffolds)
    assert not (train_scaffolds & test_scaffolds)
    assert not (val_scaffolds & test_scaffolds)


def test_scaffold_split_is_deterministic():
    r1 = scaffold_split(SMILES_LIST, 0.6, 0.2, 0.2)
    r2 = scaffold_split(SMILES_LIST, 0.6, 0.2, 0.2)
    assert r1 == r2


def test_scaffold_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        scaffold_split(SMILES_LIST, 0.5, 0.5, 0.5)


def test_scaffold_split_rejects_empty_input():
    with pytest.raises(ValueError):
        scaffold_split([], 0.8, 0.1, 0.1)


def test_scaffold_split_larger_scaffold_group_first():
    # The larger benzene scaffold group (4 molecules) is considered first by
    # the greedy packer and fits within a generous train cutoff, while the
    # smaller cyclohexane group (2 molecules) overflows train's cutoff and
    # is pushed to test (DeepChem-style greedy-by-size, cumulative-cutoff
    # packing; a group goes to the first bucket whose cutoff it still fits).
    train_idx, val_idx, test_idx = scaffold_split(
        SMILES_LIST, frac_train=0.7, frac_val=0.2, frac_test=0.1
    )
    assert set(train_idx) == {0, 1, 2, 3}
    assert set(test_idx) == {4, 5}
    assert val_idx == []
