"""Functional-connectivity graph construction for brain networks.

Provides the connectivity metrics compared in this study:

- :func:`pearson_connectivity` — the classical baseline, a dense correlation
  matrix over region time series.
- :func:`rbf_connectivity` — a non-linear classical baseline.
- :func:`fidelity_connectivity` — quantum state fidelity |<psi_i|psi_j>|^2
  between per-region quantum embeddings.

and the sparsification and adaptive-edge machinery that turns any of them into
a graph the attention network can propagate over.

Quantum fidelity is used to *initialise* the topology. During training the
edge weights are recomputed from the measured expectation values by
:class:`AdaptiveConnectomeEdges`, which is far cheaper than re-deriving full
statevector overlaps at every step and keeps the adjacency differentiable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from qagta.quantum.fidelity import pairwise_fidelity


def pearson_connectivity(timeseries: np.ndarray) -> np.ndarray:
    """Dense Pearson correlation matrix from a ``(T, n_roi)`` BOLD matrix.

    Regions with zero variance (occasionally produced by parcellation at the
    brain edge) would divide by zero, so they are correlated as 0 rather than
    left as NaN.
    """
    series = timeseries.T
    centred = series - series.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    normed = centred / safe
    matrix = normed @ normed.T
    dead = (norms.ravel() == 0)
    matrix[dead, :] = 0.0
    matrix[:, dead] = 0.0
    return matrix


def rbf_connectivity(features: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Gaussian (RBF) similarity between region feature vectors."""
    diff = features[:, None, :] - features[None, :, :]
    squared = (diff**2).sum(-1)
    if gamma is None:
        median = np.median(squared[squared > 0]) if (squared > 0).any() else 1.0
        gamma = 1.0 / max(median, 1e-8)
    return np.exp(-gamma * squared)


def fidelity_connectivity(states: torch.Tensor) -> torch.Tensor:
    """Quantum fidelity adjacency |<psi_i|psi_j>|^2 between region states."""
    return pairwise_fidelity(states)


def knn_sparsify(
    adjacency: torch.Tensor, k: int = 20, exclude_self: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the top-``k`` connections per node.

    Returns ``(edge_index, edge_weight)`` for a directed graph, where each node
    retains its ``k`` strongest outgoing edges. Sparsification is what keeps
    propagation on functionally meaningful paths instead of letting a dense,
    noise-driven matrix wash the signal out.
    """
    n = adjacency.shape[0]
    weights = adjacency.clone()
    if exclude_self:
        weights.fill_diagonal_(float("-inf"))

    k = min(k, n - 1 if exclude_self else n)
    top_weights, top_idx = torch.topk(weights, k=k, dim=1)

    src = torch.arange(n, device=adjacency.device).repeat_interleave(k)
    dst = top_idx.reshape(-1)
    edge_weight = top_weights.reshape(-1)

    finite = torch.isfinite(edge_weight)
    return torch.stack([src[finite], dst[finite]], dim=0), edge_weight[finite]


def graph_density(edge_index: torch.Tensor, n_nodes: int) -> float:
    """Average out-degree of the sparsified graph."""
    return float(edge_index.shape[1]) / n_nodes


class AdaptiveConnectomeEdges(nn.Module):
    """Differentiable edge weights over a fixed candidate topology.

    Implements the training-time edge kernel

        W_ij = ReLU(alpha * cos(z_i, z_j) + beta * MLP(z_i || z_j))

    where the candidate pairs come from fidelity-initialised k-NN
    sparsification. ``alpha`` and ``beta`` are trainable, so the balance
    between the geometric similarity of the quantum latents and a learned
    non-linear correction is optimised against the classification loss.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.pair_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Edge weights for the given candidate edges, shape ``(E,)``."""
        src, dst = edge_index[0], edge_index[1]
        normed = F.normalize(latents, p=2, dim=1)
        cosine = (normed[src] * normed[dst]).sum(-1)
        learned = self.pair_mlp(torch.cat([latents[src], latents[dst]], dim=-1)).squeeze(-1)
        return F.relu(self.alpha * cosine + self.beta * learned)


def mean_average_distance(features: torch.Tensor) -> float:
    """Mean pairwise cosine distance between node features.

    The standard over-smoothing diagnostic: as depth grows, representations
    collapse toward a common vector and this value falls toward zero. Higher
    means node-level distinctiveness has been preserved.
    """
    normed = F.normalize(features, p=2, dim=1)
    distance = 1.0 - normed @ normed.T
    n = distance.shape[0]
    off_diagonal = distance[~torch.eye(n, dtype=torch.bool, device=distance.device)]
    return float(off_diagonal.mean())
