"""Tests for adaptive edge learning and dynamic graph construction."""

from __future__ import annotations

import torch

from qagta.graph import AdaptiveEdgeLearner, DynamicGraphConstructor
from qagta.quantum import StatevectorSimulator


def _latents(n: int = 12, d: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, d)


def test_edge_index_is_well_formed():
    learner = AdaptiveEdgeLearner(embedding_dim=4, k_neighbors=3, threshold=0.0, use_fidelity=False)
    out = learner(_latents())
    assert out.edge_index.shape[0] == 2
    assert out.edge_index.shape[1] == out.edge_weight.shape[0]
    assert out.edge_index.dtype == torch.long
    assert int(out.edge_index.max()) < 12
    # No self-loops at this stage; they are added by the constructor.
    assert not bool((out.edge_index[0] == out.edge_index[1]).any())


def test_kernel_mixing_coefficients_are_a_simplex():
    learner = AdaptiveEdgeLearner(embedding_dim=4, use_fidelity=True)
    mixing = learner.mixing.detach()
    assert mixing.shape == (4,)
    assert abs(float(mixing.sum()) - 1.0) < 1e-5
    assert torch.all(mixing > 0)


def test_threshold_controls_sparsity():
    latents = _latents()
    loose = AdaptiveEdgeLearner(embedding_dim=4, k_neighbors=5, threshold=0.0, use_fidelity=False)
    strict = AdaptiveEdgeLearner(embedding_dim=4, k_neighbors=5, threshold=0.9, use_fidelity=False)
    strict.load_state_dict(loose.state_dict())
    assert strict(latents).edge_index.shape[1] <= loose(latents).edge_index.shape[1]


def test_edge_weights_are_differentiable():
    learner = AdaptiveEdgeLearner(embedding_dim=4, threshold=0.0, use_fidelity=False)
    latents = _latents().requires_grad_(True)
    learner(latents).edge_weight.sum().backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert learner.mixing_logits.grad is not None


def test_fidelity_term_uses_quantum_states():
    """With fidelity enabled the weights must depend on the statevectors."""
    sim = StatevectorSimulator(n_qubits=4, reps=2)
    angles_a = torch.rand(10, 4) * 6.28
    angles_b = torch.rand(10, 4) * 6.28
    states_a = sim.prepare_state(angles_a)
    states_b = sim.prepare_state(angles_b)

    learner = AdaptiveEdgeLearner(embedding_dim=4, threshold=0.0, use_fidelity=True)
    latents = _latents(10, 4)
    weights_a = learner(latents, states_a).edge_weight
    weights_b = learner(latents, states_b).edge_weight
    assert not torch.allclose(weights_a, weights_b)


def test_degenerate_threshold_falls_back_to_connected_ring():
    learner = AdaptiveEdgeLearner(embedding_dim=4, threshold=10.0, use_fidelity=False)
    out = learner(_latents(8, 4))
    assert out.edge_index.shape[1] == 8


def test_dense_adjacency_shape():
    learner = AdaptiveEdgeLearner(embedding_dim=4, threshold=0.0, use_fidelity=False)
    adjacency = learner.dense_adjacency(_latents(9, 4))
    assert adjacency.shape == (9, 9)


def test_constructor_adds_self_loops_and_preserves_nodes():
    constructor = DynamicGraphConstructor(embedding_dim=4, threshold=0.0, use_fidelity=False)
    latents = _latents(10, 4)
    graph = constructor(latents)

    assert graph.x.shape == (10, 4)
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    self_loops = int((graph.edge_index[0] == graph.edge_index[1]).sum())
    assert self_loops == 10


def test_topology_evolves_with_latents():
    """Different latent states must yield a different graph."""
    constructor = DynamicGraphConstructor(embedding_dim=4, threshold=0.35, use_fidelity=False)
    constructor.eval()
    torch.manual_seed(3)
    graph_a = constructor(torch.randn(20, 4))
    graph_b = constructor(torch.randn(20, 4) * 3.0)
    same_shape = graph_a.edge_index.shape == graph_b.edge_index.shape
    assert not same_shape or not torch.equal(graph_a.edge_index, graph_b.edge_index)
