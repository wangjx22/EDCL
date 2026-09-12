import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from edcl.validity import sample_valid_perturbation


def _min_pairwise_dist(pos):
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    return d.min().item()


def test_perturbation_respects_min_distance():
    torch.manual_seed(3)
    # 2 molecules, deliberately packed close together so naive noise would violate r_min
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0],
        [5.0, 0.0, 0.0], [5.9, 0.0, 0.0],
    ])
    batch = torch.tensor([0, 0, 0, 1, 1])
    sigma = torch.ones(5, 1) * 0.3
    r_min = 0.7
    pos_tilde, eps = sample_valid_perturbation(pos, sigma, batch, r_min=r_min, max_resamples=50)
    for g in [0, 1]:
        sub = pos_tilde[batch == g]
        if sub.shape[0] > 1:
            assert _min_pairwise_dist(sub) >= r_min - 1e-3


def test_perturbation_is_centered_noise():
    """With very small sigma and generous r_min, output should be close to input."""
    torch.manual_seed(4)
    pos = torch.randn(6, 3) * 3.0  # spread out, unlikely to collide
    batch = torch.zeros(6, dtype=torch.long)
    sigma = torch.ones(6, 1) * 1e-4
    pos_tilde, eps = sample_valid_perturbation(pos, sigma, batch, r_min=0.01, max_resamples=5)
    assert (pos_tilde - pos).abs().max().item() < 1e-2
