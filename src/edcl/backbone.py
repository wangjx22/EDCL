"""
SE(3)/E(3)-equivariant molecular encoder.

DESIGN NOTE (deviation from the paper, documented in README):
The paper's encoder F_phi is a full Equiformer V2 (spherical harmonics up to
degree L_max, Wigner-D equivariant tensor products, see Table 9). That
architecture requires `e3nn` and a substantial implementation effort. Given
the project's time budget and the unavailability of `e3nn`/`torch_geometric`
in this environment, we implement a lighter but *genuinely* SE(3)/E(3)
equivariant backbone in the style of Satorras et al. "E(n) Equivariant Graph
Neural Networks" (EGNN, ICML 2021):

  - Invariant scalar node features h_i are updated using only rotation/
    translation INVARIANT quantities (pairwise distances via Gaussian RBF,
    node embeddings) -> this yields an SE(3)-INVARIANT h_i and graph pooling.
  - An auxiliary vector channel v_i (used only for the denoising head, which
    must predict a 3D EQUIVARIANT displacement) is updated as a learned
    linear combination of relative position vectors (r_i - r_j) weighted by
    invariant edge scalars. This is the standard EGNN trick that guarantees
    exact rotation/reflection equivariance and translation invariance
    (relative vectors are translation-invariant, and a linear combination of
    vectors with invariant coefficients is manifestly equivariant to O(3)).

This is a strict subset of what Equiformer V2 can represent (only L=0 and
L=1 signals, no L>=2 tensors), but it satisfies the same symmetry
constraints and is verified numerically in `tests/test_equivariance.py`.
The `Backbone` interface is small enough that a real Equiformer V2 (e.g.
imported from `fairchem`) can be substituted with no changes to
`model.py` / `losses.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ops import gaussian_rbf, scatter_mean, scatter_sum, radius_graph


def _mlp(in_dim, hidden, out_dim, act=nn.SiLU):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, out_dim),
    )


@dataclass
class EncoderOutput:
    h: torch.Tensor            # [N, dim] invariant atom-level features
    v: torch.Tensor            # [N, 3]   equivariant atom-level vector features
    h_graph: torch.Tensor      # [B, dim] invariant graph-level (pooled) features


class EGNNLayer(nn.Module):
    """One message-passing layer: invariant scalar update + equivariant vector update."""

    def __init__(self, dim: int, num_rbf: int = 32, cutoff: float = 10.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.edge_mlp = _mlp(2 * dim + num_rbf, dim, dim)
        self.node_mlp = _mlp(2 * dim, dim, dim)
        self.vector_gate_mlp = _mlp(dim, dim, 1)  # scalar coefficient per edge for vector update

    def forward(self, h, v, pos, edge_index):
        src, dst = edge_index  # message dst <- src
        if src.numel() == 0:
            return h, v
        rel = pos[src] - pos[dst]                      # [E,3] translation-invariant relative vector
        dist = rel.norm(dim=-1)                         # [E]   rotation/translation-invariant
        rbf = gaussian_rbf(dist, num_basis=self.num_rbf, cutoff=self.cutoff)  # [E, num_rbf] invariant
        edge_in = torch.cat([h[src], h[dst], rbf], dim=-1)
        m_ij = self.edge_mlp(edge_in)                   # [E, dim] invariant edge message
        agg = scatter_sum(m_ij, dst, dim_size=h.shape[0])
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # Equivariant vector update: linear combo of relative vectors with invariant coeffs.
        coeff = self.vector_gate_mlp(m_ij)               # [E, 1] invariant scalar per edge
        unit = rel / (dist.clamp(min=1e-6).unsqueeze(-1))
        v_msg = coeff * unit                             # [E, 3] equivariant (rotates with rel)
        v_agg = scatter_sum(v_msg, dst, dim_size=h.shape[0])
        v_new = v + v_agg
        return h_new, v_new


class EquivariantEncoder(nn.Module):
    """Shared encoder F_phi used for BOTH the clean and perturbed branches (weight-tied)."""

    def __init__(self, num_elements: int = 119, hidden_dim: int = 128, num_layers: int = 4,
                 num_rbf: int = 32, cutoff: float = 10.0, max_neighbors: int = 32):
        super().__init__()
        dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_elements, dim)
        self.layers = nn.ModuleList([
            EGNNLayer(dim, num_rbf=num_rbf, cutoff=cutoff) for _ in range(num_layers)
        ])
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> EncoderOutput:
        h = self.embedding(z)
        v = torch.zeros_like(pos)
        edge_index = radius_graph(pos, batch, cutoff=self.cutoff, max_neighbors=self.max_neighbors)
        for layer in self.layers:
            h, v = layer(h, v, pos, edge_index)
        h = self.out_norm(h)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        h_graph = scatter_mean(h, batch, dim_size=num_graphs)
        return EncoderOutput(h=h, v=v, h_graph=h_graph)
