"""Graph attention network with over-smoothing mitigation.

Deep message passing tends to collapse node embeddings toward a common
representation ("over-smoothing"), which suppresses exactly the sparse,
anomalous nodes this pipeline is meant to surface. Two countermeasures are
built in:

- multi-head attention conditioned on the learned edge weights, so
  propagation is selective rather than uniform, and
- learnable weighted aggregation over per-layer skip projections, so the
  final representation can retain shallow (less-smoothed) features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 8,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.hidden_channels = hidden_channels

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=1,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=1,
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        for layer in range(num_layers):
            width = hidden_channels if layer == num_layers - 1 else hidden_channels * heads
            self.skip_projections.append(nn.Linear(width, hidden_channels))

        # Learnable aggregation over layer outputs (anti-over-smoothing).
        self.skip_logits = nn.Parameter(torch.zeros(num_layers))
        self.dropout = nn.Dropout(dropout)

    @property
    def out_channels(self) -> int:
        return self.hidden_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skips = []
        for conv, norm, proj in zip(
            self.convs, self.norms, self.skip_projections, strict=True
        ):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)
            skips.append(proj(x))

        weights = F.softmax(self.skip_logits, dim=0)
        return sum(w * s for w, s in zip(weights, skips, strict=True))
