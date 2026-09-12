"""Physical validity constraint on the sampled perturbation (paper: perturbed
structures must remain chemically plausible; reject/resample atoms that end
up closer than r_min to any other atom in the same molecule).
"""
from __future__ import annotations

import torch

from .ops import scatter_sum


def _min_pairwise_dist_per_graph(pos: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """O(N^2) min pairwise distance per graph (fine for small molecules)."""
    device = pos.device
    out = torch.full((num_graphs,), float("inf"), device=device)
    for g in range(num_graphs):
        mask = batch == g
        p = pos[mask]
        if p.shape[0] < 2:
            continue
        d = torch.cdist(p, p)
        d.fill_diagonal_(float("inf"))
        out[g] = d.min()
    return out


def sample_valid_perturbation(pos: torch.Tensor, sigma: torch.Tensor, batch: torch.Tensor,
                               r_min: float = 0.8, max_resamples: int = 5,
                               generator: torch.Generator | None = None):
    """Sample epsilon ~ N(0,I), form pos_tilde = pos + sigma*epsilon, and reject/resample
    (per-graph, whole-molecule redraw) any molecule whose resulting minimum pairwise
    atomic distance falls below ``r_min`` Angstrom, up to ``max_resamples`` attempts.
    Molecules still invalid after all attempts keep their last (least-bad) sample —
    this bounds worst-case cost while remaining correct in the common case.

    Returns
    -------
    pos_tilde : [N,3] perturbed coordinates
    epsilon   : [N,3] the noise actually used (so callers can compute Delta = pos_tilde-pos)
    """
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    best_pos_tilde = None
    best_eps = None
    best_mind = None
    for attempt in range(max_resamples):
        eps = torch.randn(pos.shape, device=pos.device, generator=generator)
        pos_tilde = pos + sigma * eps
        mind = _min_pairwise_dist_per_graph(pos_tilde, batch, num_graphs)
        if best_pos_tilde is None:
            best_pos_tilde, best_eps, best_mind = pos_tilde, eps, mind
        else:
            improve = mind > best_mind
            # replace per-graph where this attempt is better
            node_improve = improve[batch]
            best_pos_tilde = torch.where(node_improve.unsqueeze(-1), pos_tilde, best_pos_tilde)
            best_eps = torch.where(node_improve.unsqueeze(-1), eps, best_eps)
            best_mind = torch.where(improve, mind, best_mind)
        if bool((best_mind >= r_min).all()):
            break
    return best_pos_tilde, best_eps
