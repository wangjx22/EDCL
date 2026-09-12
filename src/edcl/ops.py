"""
Lightweight graph-batching primitives.

We deliberately avoid a hard dependency on ``torch_geometric`` /
``torch_scatter`` (they are heavy, hard to install offline, and pull in
compiled extensions that need to match the exact torch/CUDA build).  Instead
we implement the handful of scatter-reduce operations we actually need with
plain ``torch.index_add_`` / ``torch.Tensor.scatter_reduce_`` calls.  A
project that already has ``torch_geometric`` installed can trivially swap
these for the optimized CUDA kernels without changing any call sites (the
function signatures are compatible with ``torch_geometric.utils.scatter``).
"""
from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum-reduce ``src`` (shape [N, *]) into ``dim_size`` buckets given by ``index`` (shape [N])."""
    shape = (dim_size,) + tuple(src.shape[1:])
    out = src.new_zeros(shape)
    index_exp = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, index_exp, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-reduce ``src`` into ``dim_size`` buckets given by ``index``."""
    summed = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones(src.shape[0], device=src.device, dtype=src.dtype), index, dim_size)
    count = count.clamp(min=1.0)
    view_shape = (dim_size,) + (1,) * (src.dim() - 1)
    return summed / count.view(view_shape)


def gaussian_rbf(distances: torch.Tensor, num_basis: int = 32, cutoff: float = 10.0) -> torch.Tensor:
    """Expand scalar distances into a Gaussian radial-basis-function feature vector.

    Follows the standard SchNet-style RBF expansion used as the "Gaussian RBF
    + 2-layer MLP" edge-distance encoding described in the paper's
    architecture table (Table 9).
    """
    device = distances.device
    centers = torch.linspace(0.0, cutoff, num_basis, device=device)
    width = centers[1] - centers[0] if num_basis > 1 else torch.tensor(1.0, device=device)
    diff = distances.unsqueeze(-1) - centers.view(*([1] * distances.dim()), num_basis)
    return torch.exp(-0.5 * (diff / (width + 1e-8)) ** 2)


def full_pairwise_edges(batch_index: torch.Tensor, self_loops: bool = False):
    """Build all-pairs (i, j) edge indices within each graph of a batch.

    Returns a ``(2, E)`` LongTensor ``edge_index`` such that ``edge_index[0]``
    are source nodes and ``edge_index[1]`` are target nodes, restricted to
    node pairs that belong to the same graph (same ``batch_index`` value).
    Suitable for the small molecules (tens of atoms) typical of QM9 /
    PCQM4Mv2-scale pretraining; for larger systems a radius-graph should be
    used instead (see ``radius_graph`` below).
    """
    device = batch_index.device
    n = batch_index.shape[0]
    idx = torch.arange(n, device=device)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    same_graph = batch_index[src] == batch_index[dst]
    if not self_loops:
        same_graph &= src != dst
    return torch.stack([src[same_graph], dst[same_graph]], dim=0)


def radius_graph(pos: torch.Tensor, batch_index: torch.Tensor, cutoff: float, max_neighbors: int = 32):
    """Simple O(N^2) radius-graph builder (fine for small molecules used here)."""
    edge_index = full_pairwise_edges(batch_index, self_loops=False)
    src, dst = edge_index
    dist = (pos[src] - pos[dst]).norm(dim=-1)
    keep = dist <= cutoff
    edge_index = edge_index[:, keep]
    dist = dist[keep]
    if max_neighbors is not None and edge_index.shape[1] > 0:
        # cap number of neighbours per destination node to bound cost on dense regions
        order = torch.argsort(dist)
        edge_index = edge_index[:, order]
        dist = dist[order]
        dst_sorted = edge_index[1]
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=pos.device)
        counts = {}
        keep_list = []
        for i in range(edge_index.shape[1]):
            d = int(dst_sorted[i])
            c = counts.get(d, 0)
            if c < max_neighbors:
                keep_list.append(i)
                counts[d] = c + 1
        keep_idx = torch.tensor(keep_list, dtype=torch.long, device=pos.device)
        edge_index = edge_index[:, keep_idx]
    return edge_index
