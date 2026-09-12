"""Full EDCL pretraining model: dual-branch (clean + perturbed) forward pass
sharing one encoder F_phi, wiring together noise generation with validity
constraint, denoising, energy and contrastive heads, and the combined loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import EquivariantEncoder
from .noise_generator import NoiseGenerator
from .heads import DenoiseHead, EnergyHead, ProjectionHead
from .losses import (EDCLLossWeights, denoising_nll_loss, kl_prior_loss,
                      info_nce_contrastive_loss, energy_mse_loss, total_edcl_loss)
from .validity import sample_valid_perturbation


@dataclass
class EDCLConfig:
    num_elements: int = 119
    hidden_dim: int = 128
    num_layers: int = 4
    cutoff: float = 5.0
    max_neighbors: int = 32
    omega: float = 0.1      # prior std for KL, Eq.14
    r_min: float = 0.8      # validity constraint, Angstrom
    tau: float = 0.1        # InfoNCE temperature
    weights: EDCLLossWeights = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = EDCLLossWeights()


class EDCLPretrainModel(nn.Module):
    def __init__(self, config: EDCLConfig = EDCLConfig()):
        super().__init__()
        self.config = config
        self.encoder = EquivariantEncoder(
            num_elements=config.num_elements, hidden_dim=config.hidden_dim,
            num_layers=config.num_layers, cutoff=config.cutoff, max_neighbors=config.max_neighbors,
        )
        self.noise_gen = NoiseGenerator(config.hidden_dim)
        self.denoise_head = DenoiseHead(config.hidden_dim)
        self.energy_head = EnergyHead(config.hidden_dim)
        self.proj_head = ProjectionHead(config.hidden_dim, identity=True)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor,
                target_energy: torch.Tensor | None = None):
        cfg = self.config
        # --- clean branch ---
        out_clean = self.encoder(z, pos, batch)
        sigma = self.noise_gen(out_clean.h)  # [N,1], data-dependent per-atom scale, Eq.(7)-ish

        # --- sample physically-valid perturbation, Eq.(1) + validity constraint ---
        pos_tilde, eps = sample_valid_perturbation(pos, sigma, batch, r_min=cfg.r_min)
        delta = pos_tilde - pos  # realised displacement Delta_i

        # --- perturbed branch (shared encoder) ---
        out_noisy = self.encoder(z, pos_tilde, batch)

        # --- denoising head predicts displacement from the noisy representation ---
        delta_hat = self.denoise_head(out_noisy.h, out_noisy.v)

        l_denoise = denoising_nll_loss(delta_hat, delta, sigma)
        l_kl = kl_prior_loss(sigma, omega=cfg.omega)

        g_clean = self.proj_head(out_clean.h_graph)
        g_noisy = self.proj_head(out_noisy.h_graph)
        l_contrast = info_nce_contrastive_loss(g_noisy, g_clean, tau=cfg.tau)

        pred_energy_atom = self.energy_head(out_clean.h)  # clean branch ONLY
        from .ops import scatter_mean
        num_graphs = int(batch.max().item()) + 1
        pred_energy = scatter_mean(pred_energy_atom, batch, dim_size=num_graphs).view(-1)
        if target_energy is None:
            if cfg.weights.beta != 0:
                raise ValueError(
                    "target_energy is required when the energy-loss weight beta is non-zero"
                )
            # Keep the beta=0 ablation differentiable without pretending that
            # an unavailable target contributes an observed zero loss.
            l_energy = pred_energy.sum() * 0.0
        else:
            l_energy = energy_mse_loss(pred_energy, target_energy)

        total, logs = total_edcl_loss(l_denoise, l_kl, l_contrast, l_energy, cfg.weights)
        return total, logs

    def encode(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        """Inference-time / fine-tuning-time encoder call (clean structure only)."""
        return self.encoder(z, pos, batch)
