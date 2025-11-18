"""Dynamic graph construction from quantum latent embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from qagta.graph.adaptive_edges import AdaptiveEdgeLearner


class DynamicGraphConstructor(nn.Module):
    """Turns a batch of latent embeddings into a graph ``Data`` object.

    A residual refinement network first sharpens the embeddings, the
    adaptive edge learner then infers a sparse weighted topology, and
    self-loops are added for propagation stability. The topology is
    recomputed on every forward pass, so the graph evolves with the
    (quantum) latent states rather than being fixed up front.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 16,
        k_neighbors: int = 5,
        threshold: float = 0.4,
        use_fidelity: bool = True,
    ) -> None:
        super().__init__()
        self.edge_learner = AdaptiveEdgeLearner(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            k_neighbors=k_neighbors,
            threshold=threshold,
            use_fidelity=use_fidelity,
        )
        self.refiner = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        statevectors: torch.Tensor | None = None,
    ) -> Data:
        refined = self.refiner(embeddings) + embeddings
        edges = self.edge_learner(refined, statevectors)
        edge_index, edge_weight = add_self_loops(
            edges.edge_index,
            edge_attr=edges.edge_weight,
            fill_value=1.0,
            num_nodes=embeddings.shape[0],
        )
        return Data(
            x=refined,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(-1),
        )
