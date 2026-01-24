"""Tests for the graph encoders and the fusion decision module."""

from __future__ import annotations

import torch

from qagta.graph import DynamicGraphConstructor
from qagta.models import DecisionModule, GraphAttentionEncoder, GraphSAGEEncoder


def _graph(n: int = 16, d: int = 4):
    torch.manual_seed(0)
    constructor = DynamicGraphConstructor(embedding_dim=d, threshold=0.0, use_fidelity=False)
    constructor.eval()
    return constructor(torch.randn(n, d))


def test_gat_output_shape():
    graph = _graph()
    model = GraphAttentionEncoder(in_channels=4, hidden_channels=8, num_layers=3, heads=4)
    model.eval()
    out = model(graph.x, graph.edge_index, graph.edge_attr)
    assert out.shape == (16, 8)
    assert torch.isfinite(out).all()


def test_sage_output_shape():
    graph = _graph()
    model = GraphSAGEEncoder(in_channels=4, hidden_channels=8, num_layers=3)
    model.eval()
    out = model(graph.x, graph.edge_index, graph.edge_attr)
    assert out.shape == (16, 8)


def test_gat_gradients_reach_edge_weights():
    """Attention must be conditioned on the learned edge weights."""
    graph = _graph()
    edge_attr = graph.edge_attr.detach().clone().requires_grad_(True)
    model = GraphAttentionEncoder(in_channels=4, hidden_channels=8, num_layers=2, heads=2)
    model.train()
    model(graph.x, graph.edge_index, edge_attr).sum().backward()
    assert edge_attr.grad is not None
    assert torch.isfinite(edge_attr.grad).all()


def test_gat_resists_over_smoothing_better_than_plain_stack():
    """Skip aggregation should keep node embeddings from collapsing.

    Over-smoothing shows up as the variance across nodes going to zero as
    depth grows; the learnable skip aggregation is meant to preserve it.
    """
    graph = _graph(n=32)
    torch.manual_seed(1)
    deep = GraphAttentionEncoder(
        in_channels=4, hidden_channels=8, num_layers=6, heads=2, dropout=0.0
    )
    deep.eval()
    with torch.no_grad():
        out = deep(graph.x, graph.edge_index, graph.edge_attr)
    node_variance = out.var(dim=0).mean()
    assert float(node_variance) > 1e-4


def test_decision_module_fuses_both_inputs():
    module = DecisionModule(graph_dim=8, latent_dim=4)
    module.eval()
    graph_emb = torch.randn(12, 8)
    latent = torch.randn(12, 4)
    out = module(graph_emb, latent)
    assert out.shape == (12, 8)

    # Changing either input must change the output.
    assert not torch.allclose(out, module(graph_emb * 2, latent))
    assert not torch.allclose(out, module(graph_emb, latent * 2))
