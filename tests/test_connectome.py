"""Tests for connectome construction, the ABIDE loader and LSO evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qagta.data.abide import compress_region_timeseries
from qagta.graph.connectome import (
    AdaptiveConnectomeEdges,
    graph_density,
    knn_sparsify,
    mean_average_distance,
    pearson_connectivity,
    rbf_connectivity,
)
from qagta.models.connectome_gat import ConnectomeGAT, ConnectomeGCN
from qagta.pipelines.connectome_pipeline import EncodedCohort
from qagta.training.lso import leave_site_out

# --------------------------------------------------------------- connectivity

def test_pearson_matches_numpy_on_healthy_data():
    rng = np.random.default_rng(0)
    series = rng.normal(size=(120, 15))
    assert np.allclose(pearson_connectivity(series), np.corrcoef(series.T), atol=1e-6)


def test_pearson_handles_zero_variance_regions():
    """A flat ROI must yield zeros, not NaNs that poison the whole graph."""
    rng = np.random.default_rng(0)
    series = rng.normal(size=(100, 10))
    series[:, 4] = 3.0  # constant region
    matrix = pearson_connectivity(series)
    assert not np.isnan(matrix).any()
    assert np.allclose(matrix[4], 0.0)


def test_rbf_connectivity_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    matrix = rbf_connectivity(rng.normal(size=(12, 6)))
    assert np.allclose(matrix, matrix.T, atol=1e-6)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-6)


# -------------------------------------------------------------- sparsification

def test_knn_keeps_exactly_k_edges_per_node():
    torch.manual_seed(0)
    adjacency = torch.rand(30, 30)
    edge_index, edge_weight = knn_sparsify(adjacency, k=5)
    assert edge_index.shape == (2, 150)
    assert edge_weight.shape == (150,)
    counts = torch.bincount(edge_index[0], minlength=30)
    assert torch.all(counts == 5)


def test_knn_excludes_self_loops():
    adjacency = torch.eye(20) * 10 + torch.rand(20, 20) * 0.1
    edge_index, _ = knn_sparsify(adjacency, k=3)
    assert not bool((edge_index[0] == edge_index[1]).any())


def test_knn_keeps_the_strongest_edges():
    adjacency = torch.zeros(5, 5)
    adjacency[0, 3] = 0.9
    adjacency[0, 1] = 0.8
    edge_index, _ = knn_sparsify(adjacency, k=2)
    kept = set(edge_index[1, edge_index[0] == 0].tolist())
    assert kept == {3, 1}


def test_graph_density_reports_average_degree():
    edge_index, _ = knn_sparsify(torch.rand(40, 40), k=20)
    assert graph_density(edge_index, 40) == pytest.approx(20.0)


# ------------------------------------------------------------- adaptive edges

def test_adaptive_edges_are_non_negative_and_differentiable():
    torch.manual_seed(0)
    learner = AdaptiveConnectomeEdges(latent_dim=8)
    latents = torch.randn(20, 8, requires_grad=True)
    edge_index, _ = knn_sparsify(torch.rand(20, 20), k=4)

    weights = learner(latents, edge_index)
    assert weights.shape == (edge_index.shape[1],)
    assert torch.all(weights >= 0)  # ReLU kernel

    weights.sum().backward()
    assert latents.grad is not None
    assert learner.alpha.grad is not None and learner.beta.grad is not None


# ------------------------------------------------------------------ over-smoothing

def test_mad_is_zero_for_identical_nodes_and_positive_otherwise():
    collapsed = torch.ones(10, 4)
    assert mean_average_distance(collapsed) == pytest.approx(0.0, abs=1e-6)
    torch.manual_seed(0)
    assert mean_average_distance(torch.randn(10, 4)) > 0.1


# ------------------------------------------------------------------------ PCA

def test_pca_compression_shape_and_variance():
    rng = np.random.default_rng(0)
    series = rng.normal(size=(180, 200))  # (T, n_roi)
    features, retained = compress_region_timeseries(series, n_components=16)
    assert features.shape == (200, 16)
    assert 0.0 < retained <= 1.0


def test_pca_pads_when_scan_is_shorter_than_requested_components():
    rng = np.random.default_rng(0)
    features, _ = compress_region_timeseries(rng.normal(size=(9, 50)), n_components=16)
    assert features.shape == (50, 16)  # padded, so every subject has equal width


def test_pca_rejects_non_2d_input():
    with pytest.raises(ValueError):
        compress_region_timeseries(np.zeros((5,)), n_components=4)


# ---------------------------------------------------------------- classifiers

def _toy_cohort(n_subjects=24, n_roi=30, latent_dim=8, k=5, seed=0):
    torch.manual_seed(seed)
    latents, edges, weights = [], [], []
    labels = torch.tensor([i % 2 for i in range(n_subjects)])
    for i in range(n_subjects):
        # Give the two classes different feature means so the task is learnable.
        z = torch.randn(n_roi, latent_dim) + (1.5 if labels[i] == 1 else -1.5)
        adjacency = torch.rand(n_roi, n_roi)
        edge_index, edge_weight = knn_sparsify(adjacency, k=k)
        latents.append(z)
        edges.append(edge_index)
        weights.append(edge_weight)
    sites = np.array([f"SITE{i % 3}" for i in range(n_subjects)])
    return EncodedCohort(
        latents=torch.stack(latents), edge_index=torch.stack(edges),
        edge_weight=torch.stack(weights), labels=labels, sites=sites,
    )


@pytest.mark.parametrize("model_cls", [ConnectomeGAT, ConnectomeGCN])
def test_classifiers_output_graph_level_logits(model_cls):
    cohort = _toy_cohort()
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([cohort.graph(i) for i in range(6)])
    model = model_cls(latent_dim=8, hidden_dim=16)
    model.eval()
    logits, nodes = model(
        batch.x, batch.edge_index, batch.edge_attr.squeeze(-1), batch.batch,
        return_node_features=True,
    )
    assert logits.shape == (6, 2)  # one prediction per graph, not per node
    assert nodes.shape[0] == batch.x.shape[0]
    assert torch.isfinite(logits).all()


def test_cohort_graph_roundtrip_and_save(tmp_path):
    cohort = _toy_cohort()
    graph = cohort.graph(0)
    assert graph.x.shape == (30, 8)
    assert graph.y.shape == (1,)

    path = tmp_path / "cohort.pt"
    cohort.save(path)
    restored = EncodedCohort.load(path)
    assert len(restored) == len(cohort)
    assert torch.equal(restored.labels, cohort.labels)


# ------------------------------------------------------------------------ LSO

def test_leave_site_out_holds_each_site_out_exactly_once():
    cohort = _toy_cohort(n_subjects=36)
    result = leave_site_out(
        cohort, name="toy", epochs=2, min_test_size=2, verbose=False, measure_mad=False
    )
    assert len(result.folds) == 3
    assert {f.site for f in result.folds} == {"SITE0", "SITE1", "SITE2"}
    assert sum(f.n_test for f in result.folds) == 36


def test_lso_metrics_are_in_range_and_ci_reported():
    cohort = _toy_cohort(n_subjects=36)
    result = leave_site_out(
        cohort, name="toy", epochs=2, min_test_size=2, verbose=False, measure_mad=False
    )
    for metric in ("f1", "accuracy", "specificity"):
        mean, half = result.mean_ci(metric)
        assert 0.0 <= mean <= 1.0
        assert half >= 0.0
    assert "f1=" in result.summary()


def test_lso_skips_sites_with_a_single_class():
    cohort = _toy_cohort(n_subjects=24)
    cohort.labels = torch.zeros(24, dtype=torch.long)  # degenerate: one class only
    result = leave_site_out(
        cohort, name="toy", epochs=1, min_test_size=2, verbose=False, measure_mad=False
    )
    assert result.folds == []
