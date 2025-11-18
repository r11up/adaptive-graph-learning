"""Decision module fusing graph-propagated features with quantum latents."""

from __future__ import annotations

import torch
from torch import nn


class DecisionModule(nn.Module):
    """Gated fusion of graph embeddings and the original quantum latents.

    The graph encoder output captures relational context; the raw latent
    keeps node-local quantum information. A learned per-dimension gate
    balances the two, yielding the final representation used for scoring.

    ``reconstruct`` maps that representation back to the latent space. It
    provides the label-free training signal for the graph stage: the fused
    representation must stay faithful to the quantum latent it came from,
    so relational context is added without discarding the quantum features.
    """

    def __init__(self, graph_dim: int, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.graph_dim = graph_dim
        self.latent_dim = latent_dim

        self.fusion = nn.Sequential(
            nn.Linear(graph_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, graph_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(graph_dim + latent_dim, graph_dim),
            nn.Sigmoid(),
        )
        self.latent_proj = nn.Linear(latent_dim, graph_dim)
        self.latent_head = nn.Linear(graph_dim, latent_dim)

    def forward(self, graph_emb: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([graph_emb, latent], dim=-1)
        fused = self.fusion(combined)
        gate = self.gate(combined)
        return gate * fused + (1.0 - gate) * self.latent_proj(latent)

    def reconstruct(self, fused: torch.Tensor) -> torch.Tensor:
        """Project a fused representation back into the latent space."""
        return self.latent_head(fused)
