import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from edcl import EDCLPretrainModel, EDCLConfig, EDCLFinetuneModel
from edcl.data import collate_molecules, SyntheticMoleculeDataset


def _synthetic_batch():
    z = torch.randint(1, 10, (12,))
    pos = torch.randn(12, 3)
    batch = torch.tensor([0] * 5 + [1] * 7)
    return z, pos, batch


def test_pretrain_forward_backward_finite_and_grads():
    torch.manual_seed(0)
    cfg = EDCLConfig(hidden_dim=32, num_layers=2, cutoff=5.0, max_neighbors=16)
    model = EDCLPretrainModel(cfg)
    z, pos, batch = _synthetic_batch()
    loss, logs = model(z, pos, batch)
    assert torch.isfinite(loss)
    loss.backward()
    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params = sum(1 for _ in model.parameters())
    # allow the (opt-in, identity-by-default) projection head params to have no grad
    assert n_params_with_grad >= n_params - 8


def test_synthetic_dataset_and_collate():
    ds = SyntheticMoleculeDataset(num_samples=5, seed=0)
    samples = [ds[i] for i in range(len(ds))]
    mb = collate_molecules(samples)
    assert mb.z.shape[0] == mb.pos.shape[0] == mb.batch.shape[0]
    assert int(mb.batch.max().item()) == 4


def test_finetune_model_forward():
    torch.manual_seed(0)
    cfg = EDCLConfig(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    pretrain = EDCLPretrainModel(cfg)
    ft = EDCLFinetuneModel(pretrain.encoder, num_targets=1)
    z, pos, batch = _synthetic_batch()
    out = ft(z, pos, batch)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()


def test_save_and_load_encoder_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = EDCLConfig(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    pretrain = EDCLPretrainModel(cfg)
    ckpt_path = str(tmp_path / "encoder.pt")
    torch.save({
        "encoder_state_dict": pretrain.encoder.state_dict(),
        "encoder_config": dict(num_elements=119, hidden_dim=16, num_layers=2,
                                num_rbf=32, cutoff=5.0, max_neighbors=8),
    }, ckpt_path)
    ft = EDCLFinetuneModel.from_pretrained(ckpt_path, num_targets=2)
    z, pos, batch = _synthetic_batch()
    out = ft(z, pos, batch)
    assert out.shape == (2, 2)
