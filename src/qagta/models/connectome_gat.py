"""Graph-attention classifier over quantum-derived brain connectomes.

Architecture:

    GAT layer 1   latent_dim -> hidden, multi-head attention
    activation    ELU
    dropout       p = 0.6 (aggressive, given the sample size relative to capacity)
    GAT layer 2   hidden -> n_classes
    read-out      global mean pooling over the region nodes
    classifier    softmax over the pooled graph embedding

Attention coefficients are modulated by the learned quantum edge weights, so
messages propagate along the topology the quantum kernel surfaced rather than
uniformly over a dense correlation graph.

An isotropic GCN variant with the same depth and width is provided for the
ablation that isolates the contribution of attention specifically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class ConnectomeGAT(nn.Module):
    """Two-layer graph attention classifier with quantum-weighted edges."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        n_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(latent_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout, edge_dim=1
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_node_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight

        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        node_features = F.elu(h)

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        pooled = global_mean_pool(node_features, batch)
        logits = self.classifier(pooled)

        if return_node_features:
            return logits, node_features
        return logits


class ConnectomeGCN(nn.Module):
    """Isotropic GCN with matched depth and width, for the attention ablation."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        n_classes: int = 2,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(latent_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor | None = None,
        return_node_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = edge_weight.squeeze(-1) if edge_weight.dim() > 1 else edge_weight
        # GCN normalisation needs non-negative weights.
        weight = weight.clamp_min(0)

        h = F.relu(self.conv1(x, edge_index, edge_weight=weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        node_features = F.relu(self.conv2(h, edge_index, edge_weight=weight))

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        logits = self.classifier(global_mean_pool(node_features, batch))

        if return_node_features:
            return logits, node_features
        return logits
