import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from edcl.backbone import EquivariantEncoder
from edcl.heads import DenoiseHead


def _random_rotation(proper=True):
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if proper and torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _make_batch():
    torch.manual_seed(1)
    z = torch.randint(1, 10, (9,))
    pos = torch.randn(9, 3)
    batch = torch.tensor([0] * 4 + [1] * 5)
    return z, pos, batch


def test_translation_invariance_h():
    z, pos, batch = _make_batch()
    enc = EquivariantEncoder(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    enc.eval()
    t = torch.randn(3)
    o1 = enc(z, pos, batch)
    o2 = enc(z, pos + t, batch)
    assert (o1.h - o2.h).abs().max().item() < 1e-4
    assert (o1.h_graph - o2.h_graph).abs().max().item() < 1e-4


def test_rotation_invariance_h_and_equivariance_v():
    z, pos, batch = _make_batch()
    enc = EquivariantEncoder(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    dh = DenoiseHead(dim=16)
    enc.eval(); dh.eval()
    R = _random_rotation(proper=True)
    o1 = enc(z, pos, batch)
    o2 = enc(z, pos @ R.T, batch)
    assert (o1.h - o2.h).abs().max().item() < 1e-4
    assert (o1.h_graph - o2.h_graph).abs().max().item() < 1e-4
    assert (o1.v @ R.T - o2.v).abs().max().item() < 1e-4
    d1 = dh(o1.h, o1.v) @ R.T
    d2 = dh(o2.h, o2.v)
    assert (d1 - d2).abs().max().item() < 1e-4


def test_reflection_equivariance():
    """O(3) (improper rotations / reflections) should also hold since the
    backbone only uses norms and linear combinations of relative vectors."""
    z, pos, batch = _make_batch()
    enc = EquivariantEncoder(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    enc.eval()
    R = _random_rotation(proper=False)  # det = -1
    o1 = enc(z, pos, batch)
    o2 = enc(z, pos @ R.T, batch)
    assert (o1.h - o2.h).abs().max().item() < 1e-4
    assert (o1.v @ R.T - o2.v).abs().max().item() < 1e-4


def test_permutation_invariance_h_graph():
    """Reordering atoms within a graph should not change pooled graph features."""
    torch.manual_seed(2)
    z = torch.randint(1, 10, (6,))
    pos = torch.randn(6, 3)
    batch = torch.zeros(6, dtype=torch.long)
    perm = torch.randperm(6)

    enc = EquivariantEncoder(hidden_dim=16, num_layers=2, cutoff=5.0, max_neighbors=8)
    enc.eval()
    o1 = enc(z, pos, batch)
    o2 = enc(z[perm], pos[perm], batch[perm])
    assert (o1.h_graph - o2.h_graph).abs().max().item() < 1e-4
