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
from torch_geometric.nn import GATConv, GCNConv, global_add_pool, global_max_pool, global_mean_pool


class ConnectomeGAT(nn.Module):
    """Two-layer graph attention classifier with quantum-weighted edges."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        n_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.6,
        readout: str = "mean",
        n_nodes: int = 200,
    ) -> None:
        """``readout`` selects how node features become a graph embedding.

        ``mean`` averages over regions, which discards *which* region carries a
        pattern — the whole point of a connectome. The alternatives preserve
        regional identity to different degrees:

        ``attention``  learns a weight per region and takes a weighted sum, so
                       discriminative regions can dominate the embedding.
        ``flatten``    concatenates regions in fixed atlas order, preserving
                       identity exactly. Widest read-out, and the closest
                       analogue to what the correlation SVM sees.
        ``stats``      concatenates mean, max and standard deviation over
                       regions — cheap, and keeps distributional information a
                       plain mean throws away.
        """
        super().__init__()
        self.dropout = dropout
        self.readout = readout
        self.n_nodes = n_nodes
        self.conv1 = GATConv(latent_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout, edge_dim=1
        )
        if readout == "attention":
            self.attention_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            readout_dim = hidden_dim
        elif readout == "flatten":
            # Fixed atlas ordering makes region i the same feature block for
            # every subject, which is what preserves identity.
            readout_dim = hidden_dim * n_nodes
            self.compress = nn.Sequential(
                nn.Linear(readout_dim, hidden_dim * 4), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
            )
            readout_dim = hidden_dim
        elif readout == "stats":
            readout_dim = hidden_dim * 3
        elif readout == "mean":
            readout_dim = hidden_dim
        else:
            raise ValueError(f"unknown readout {readout!r}")

        self.classifier = nn.Linear(readout_dim, n_classes)

    def _pool(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return global_mean_pool(features, batch)
        if self.readout == "stats":
            mean = global_mean_pool(features, batch)
            maximum = global_max_pool(features, batch)
            # std via E[x^2] - E[x]^2, computed with the same pooling op.
            second = global_mean_pool(features**2, batch)
            std = (second - mean**2).clamp_min(1e-8).sqrt()
            return torch.cat([mean, maximum, std], dim=-1)
        if self.readout == "attention":
            weights = self.attention_gate(features)
            weights = weights - global_max_pool(weights, batch)[batch]  # stable softmax
            exponent = weights.exp()
            denominator = global_add_pool(exponent, batch)[batch].clamp_min(1e-8)
            return global_add_pool(features * (exponent / denominator), batch)

        # flatten: reshape each graph's nodes into one fixed-order vector
        n_graphs = int(batch.max()) + 1
        width = features.shape[-1]
        flat = features.new_zeros(n_graphs, self.n_nodes * width)
        for g in range(n_graphs):
            nodes = features[batch == g]
            take = min(nodes.shape[0], self.n_nodes)
            flat[g, : take * width] = nodes[:take].reshape(-1)
        return self.compress(flat)

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
        logits = self.classifier(self._pool(node_features, batch))

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
