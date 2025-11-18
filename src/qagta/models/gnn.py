"""Baseline message-passing network (no attention) for ablation comparisons."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE stack with mean-aggregated skip connections.

    Serves as the fixed-propagation baseline: it consumes the same
    dynamically constructed graph as the attention encoder but propagates
    features uniformly over neighbourhoods, without attention weighting.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_channels = hidden_channels

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
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
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            skips.append(x)
        return torch.stack(skips).mean(dim=0)
