"""Shared SMILES -> 3-D-conformer featurization used by both the data-prep
script (``scripts/make_scaffold_split.py``) and the inference script
(``scripts/predict.py``), so the two paths cannot silently drift apart.

Produces the same ``{"z": LongTensor[N], "pos": FloatTensor[N, 3]}`` atom
representation the model consumes elsewhere in the codebase (see
``docs/data.md``).
"""
from __future__ import annotations

import warnings

import torch


def smiles_to_atoms(smiles: str, seed: int = 0):
    """Embed a 3-D conformer for ``smiles`` with RDKit (ETKDGv3 + MMFF).

    Returns ``{"z": LongTensor[N], "pos": FloatTensor[N, 3]}`` on success,
    or ``None`` (with a ``warnings.warn``) if the SMILES cannot be parsed
    or 3-D embedding fails -- callers should skip the molecule rather than
    crash, since embedding failures are common on odd/invalid inputs and
    should not abort a whole-file batch job.
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
    return {"z": z, "pos": pos}
