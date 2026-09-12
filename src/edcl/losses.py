"""Loss functions, one function per paper equation, plus the combined
EDCL objective (paper Eq. 17):

    L_total = L_denoise + lambda * L_KL + alpha * L_contrast + beta * L_energy

with the paper's stated coefficients alpha=0.1, beta=10, lambda=1 as defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .ops import scatter_mean


def denoising_nll_loss(delta_hat: torch.Tensor, delta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Eq. (10)/(12): per-atom weighted MSE, mean over all atoms in the batch.
    delta_hat, delta: [N,3]; sigma: [N,1] (or [N])."""
    sigma2 = (sigma.view(-1) ** 2).clamp(min=1e-12)
    sq_err = ((delta_hat - delta) ** 2).sum(dim=-1)  # [N]
    return (sq_err / sigma2).mean()


def kl_prior_loss(sigma: torch.Tensor, omega: float = 0.1) -> torch.Tensor:
    """Eq. (14): KL( N(0, sigma_i^2) || N(0, omega^2) ), mean over atoms."""
    sigma = sigma.view(-1).clamp(min=1e-12)
    kl = torch.log(omega / sigma) + (sigma ** 2) / (2 * omega ** 2) - 0.5
    return kl.mean()


def info_nce_contrastive_loss(g_tilde: torch.Tensor, g_clean: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Eq. (4): asymmetric InfoNCE — each perturbed-graph embedding must be
    matched to its OWN clean-graph embedding among all clean embeddings in
    the batch (single direction, cosine similarity)."""
    z_tilde = F.normalize(g_tilde, dim=-1)
    z_clean = F.normalize(g_clean, dim=-1)
    logits = z_tilde @ z_clean.t() / tau  # [B,B]
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def energy_mse_loss(pred_energy: torch.Tensor, target_energy: torch.Tensor) -> torch.Tensor:
    """Eq. (16): simple MSE between predicted and reference molecular energy.
    Applied to the CLEAN branch only (caller's responsibility, see model.py)."""
    pred_energy = pred_energy.view(-1)
    target_energy = target_energy.view(-1)
    if pred_energy.shape != target_energy.shape:
        raise ValueError(
            "predicted and target energy must contain the same number of graphs, "
            f"got {pred_energy.numel()} and {target_energy.numel()}"
        )
    return F.mse_loss(pred_energy, target_energy)


@dataclass
class EDCLLossWeights:
    alpha: float = 0.1   # contrastive weight
    beta: float = 10.0   # energy weight
    lam: float = 1.0     # KL weight


def total_edcl_loss(l_denoise: torch.Tensor, l_kl: torch.Tensor, l_contrast: torch.Tensor,
                     l_energy: torch.Tensor, weights: EDCLLossWeights = EDCLLossWeights()):
    total = l_denoise + weights.lam * l_kl + weights.alpha * l_contrast + weights.beta * l_energy
    return total, {
        "loss/denoise": l_denoise.detach(),
        "loss/kl": l_kl.detach(),
        "loss/contrast": l_contrast.detach(),
        "loss/energy": l_energy.detach(),
        "loss/total": total.detach(),
    }
