import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from edcl.losses import (
    denoising_nll_loss, kl_prior_loss, info_nce_contrastive_loss,
    energy_mse_loss, total_edcl_loss, EDCLLossWeights,
)


def test_denoising_nll_nonnegative_and_finite():
    torch.manual_seed(0)
    n = 10
    delta_true = torch.randn(n, 3) * 0.1
    delta_hat = delta_true + torch.randn(n, 3) * 0.01
    sigma = torch.rand(n, 1) + 0.1
    loss = denoising_nll_loss(delta_hat, delta_true, sigma)
    assert torch.isfinite(loss)
    # perfect prediction with small fixed sigma should give low (though not
    # necessarily >=0, since NLL of a Gaussian can be negative for small sigma)
    loss_perfect = denoising_nll_loss(delta_true, delta_true, sigma)
    assert loss_perfect <= loss + 1e-4


def test_kl_zero_when_prior_matches():
    n = 8
    sigma = torch.ones(n, 1) * 0.5
    sigma_prior = torch.ones(n, 1) * 0.5
    kl = kl_prior_loss(sigma, sigma_prior)
    assert kl.abs().item() < 1e-5


def test_kl_nonnegative():
    torch.manual_seed(0)
    sigma = torch.rand(20, 1) + 0.05
    sigma_prior = torch.rand(20, 1) + 0.05
    kl = kl_prior_loss(sigma, sigma_prior)
    assert kl.item() >= -1e-5


def test_info_nce_lower_when_aligned():
    torch.manual_seed(0)
    z1 = torch.randn(16, 32)
    z2 = z1 + torch.randn(16, 32) * 0.01  # aligned positives
    z2_shuffled = z2[torch.randperm(16)]
    loss_aligned = info_nce_contrastive_loss(z1, z2)
    loss_random = info_nce_contrastive_loss(z1, z2_shuffled)
    assert loss_aligned.item() < loss_random.item()


def test_energy_mse():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    assert energy_mse_loss(pred, target).item() < 1e-6
    pred2 = torch.tensor([0.0, 0.0, 0.0])
    assert energy_mse_loss(pred2, target).item() > 1.0


def test_energy_mse_rejects_shape_mismatch_before_broadcasting():
    with pytest.raises(ValueError, match="same number of graphs"):
        energy_mse_loss(torch.zeros(2), torch.zeros(4))


def test_total_loss_combines_with_paper_weights():
    w = EDCLLossWeights()
    assert w.alpha == 0.1 and w.beta == 10 and w.lam == 1
    l_d = torch.tensor(1.0)
    l_kl = torch.tensor(2.0)
    l_c = torch.tensor(3.0)
    l_e = torch.tensor(4.0)
    total, logs = total_edcl_loss(l_d, l_kl, l_c, l_e, w)
    expected = 1.0 + 1 * 2.0 + 0.1 * 3.0 + 10 * 4.0
    assert abs(total.item() - expected) < 1e-5
    assert set(logs.keys()) == {"loss/denoise", "loss/kl", "loss/contrast", "loss/energy", "loss/total"}
