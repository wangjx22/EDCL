import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from edcl import EDCLConfig, EDCLLossWeights, EDCLPretrainModel
from edcl.data import collate_molecules, load_pt_dataset, validate_sample


def _sample(*, y=True):
    sample = {
        "z": torch.tensor([6, 1], dtype=torch.long),
        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
    }
    if y:
        sample["y"] = torch.tensor([-1.0])
    return sample


def test_collate_rejects_empty_and_mixed_labels():
    with pytest.raises(ValueError, match="empty batch"):
        collate_molecules([])
    with pytest.raises(ValueError, match="mix labeled and unlabeled"):
        collate_molecules([_sample(y=True), _sample(y=False)])


def test_validate_rejects_zero_atoms_and_mismatched_lengths():
    with pytest.raises(ValueError, match="at least one atom"):
        validate_sample({"z": torch.empty(0, dtype=torch.long), "pos": torch.empty(0, 3)})
    bad = _sample()
    bad["pos"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="atomic numbers"):
        validate_sample(bad)


def test_load_pt_dataset_validates_targets_and_top_level(tmp_path):
    missing_target = tmp_path / "missing-target.pt"
    torch.save([_sample(y=False)], missing_target)
    with pytest.raises(ValueError, match="requires a non-null 'y'"):
        load_pt_dataset(missing_target, require_y=True)

    wrong_target_count = tmp_path / "wrong-target-count.pt"
    sample = _sample()
    sample["y"] = torch.tensor([1.0, 2.0])
    torch.save([sample], wrong_target_count)
    with pytest.raises(ValueError, match=r"exactly 1 target\(s\).+got 2"):
        load_pt_dataset(
            wrong_target_count, require_y=True, expected_num_targets=1
        )

    wrong_container = tmp_path / "wrong-container.pt"
    torch.save({"z": torch.tensor([1])}, wrong_container)
    with pytest.raises(TypeError, match="list or tuple"):
        load_pt_dataset(wrong_container)


def test_nonzero_energy_weight_requires_target():
    cfg = EDCLConfig(hidden_dim=16, num_layers=1, max_neighbors=8)
    model = EDCLPretrainModel(cfg)
    sample = collate_molecules([_sample(y=False), _sample(y=False)])
    with pytest.raises(ValueError, match="target_energy is required"):
        model(sample.z, sample.pos, sample.batch)


def test_zero_energy_weight_allows_unlabeled_ablation():
    cfg = EDCLConfig(
        hidden_dim=16,
        num_layers=1,
        max_neighbors=8,
        weights=EDCLLossWeights(beta=0.0),
    )
    model = EDCLPretrainModel(cfg)
    sample = collate_molecules([_sample(y=False), _sample(y=False)])
    loss, logs = model(sample.z, sample.pos, sample.batch)
    assert torch.isfinite(loss)
    assert logs["loss/energy"].item() == 0.0
