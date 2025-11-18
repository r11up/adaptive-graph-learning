"""Adaptive edge learning: dynamic, differentiable graph topology from latents.

Edge weights between latent embeddings ``z_i`` and ``z_j`` are produced by a
parametric kernel that mixes several similarity notions:

    W_ij = a * Sim(z_i, z_j)          # cosine similarity
         + b * Learnable(z_i, z_j)    # dense non-linear pair transform
         + c * Attn(z_i, z_j)         # attention-derived coefficient
         + d * Fidelity(psi_i, psi_j) # optional quantum-native similarity

The mixing coefficients ``(a, b, c, d)`` are trainable (softmax-normalised
logits), so the balance between the similarity notions is itself learned
during optimisation. Candidate edges come from a k-nearest-neighbour
sparsification of the similarity structure; weights of the retained edges
stay differentiable, so gradients from the downstream graph objective flow
back into both the kernel and the (quantum) encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qagta.quantum.fidelity import pairwise_fidelity


@dataclass
class EdgeLearnerOutput:
    """Result of one adaptive edge construction pass."""

    edge_index: torch.Tensor  # (2, E) long
    edge_weight: torch.Tensor  # (E,) float, differentiable
    mixing: torch.Tensor  # normalised kernel coefficients (3 or 4,)


class AdaptiveEdgeLearner(nn.Module):
    """Computes a sparse, differentiable adjacency from latent embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 16,
        k_neighbors: int = 5,
        threshold: float = 0.4,
        use_fidelity: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.use_fidelity = use_fidelity

        n_terms = 4 if use_fidelity else 3
        # Softmax over these logits yields the kernel mixing coefficients.
        self.mixing_logits = nn.Parameter(torch.zeros(n_terms))

        self.pair_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.attn = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def mixing(self) -> torch.Tensor:
        return F.softmax(self.mixing_logits, dim=0)

    @staticmethod
    def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(embeddings, p=2, dim=1)
        return normed @ normed.T

    def _candidate_pairs(self, similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k-NN candidate edges (excluding self) from the similarity matrix."""
        n = similarity.shape[0]
        k = min(self.k_neighbors + 1, n)
        _, knn = torch.topk(similarity, k=k, dim=1)
        src = torch.arange(n, device=similarity.device).repeat_interleave(k)
        dst = knn.reshape(-1)
        keep = src != dst
        return src[keep], dst[keep]

    def forward(
        self,
        embeddings: torch.Tensor,
        statevectors: torch.Tensor | None = None,
    ) -> EdgeLearnerOutput:
        """Build edges over a batch of latent embeddings.

        Parameters
        ----------
        embeddings:
            Latent node features ``z`` of shape ``(N, D)``.
        statevectors:
            Optional prepared quantum states ``(N, 2**n)`` used for the
            fidelity term. Ignored when ``use_fidelity`` is False.
        """
        n = embeddings.shape[0]
        similarity = self.cosine_similarity_matrix(embeddings)
        src, dst = self._candidate_pairs(similarity)

        pair_feats = torch.cat([embeddings[src], embeddings[dst]], dim=-1)
        sim_term = similarity[src, dst]
        learn_term = self.pair_mlp(pair_feats).squeeze(-1)
        attn_term = torch.sigmoid(self.attn(pair_feats).squeeze(-1))

        terms = [sim_term, learn_term, attn_term]
        if self.use_fidelity:
            if statevectors is not None:
                fid = pairwise_fidelity(statevectors)
                terms.append(fid[src, dst])
            else:
                # Fidelity requested but unavailable (e.g. estimator backend):
                # substitute a neutral constant so mixing stays well-defined.
                terms.append(torch.full_like(sim_term, 0.5))

        mixing = self.mixing
        weight = sum(m * t for m, t in zip(mixing, terms, strict=True))

        mask = weight > self.threshold
        if int(mask.sum()) == 0:
            # Degenerate case: keep a ring so the graph stays connected.
            src = torch.arange(n, device=embeddings.device)
            dst = (src + 1) % n
            weight = torch.ones(n, device=embeddings.device)
        else:
            src, dst, weight = src[mask], dst[mask], weight[mask]

        edge_index = torch.stack([src, dst], dim=0)
        return EdgeLearnerOutput(edge_index=edge_index, edge_weight=weight, mixing=mixing)

    def dense_adjacency(
        self,
        embeddings: torch.Tensor,
        statevectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense ``(N, N)`` differentiable adjacency (for analysis/visualisation)."""
        out = self.forward(embeddings, statevectors)
        n = embeddings.shape[0]
        adjacency = torch.zeros(n, n, device=embeddings.device, dtype=out.edge_weight.dtype)
        adjacency[out.edge_index[0], out.edge_index[1]] = out.edge_weight
        return adjacency
