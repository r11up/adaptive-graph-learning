"""End-to-end pipeline and data-handling tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qagta import PipelineConfig, QuantumAdaptiveGraphPipeline
from qagta.data import generate_multivariate_series, split_normal_anomaly
from qagta.training.evaluate import comparison_table, evaluate_embeddings


@pytest.fixture(scope="module")
def split():
    df = generate_multivariate_series(n_samples=90, n_features=6, seed=0)
    return split_normal_anomaly(df.drop(columns=["attack"]).to_numpy(), df["attack"].to_numpy())


def _fast_config(**overrides) -> PipelineConfig:
    config = PipelineConfig()
    config.quantum.n_qubits = 3
    config.training.encoder_epochs = 3
    config.training.graph_epochs = 3
    for key, value in overrides.items():
        section, field = key.split(".")
        setattr(getattr(config, section), field, value)
    return config


def test_synthetic_data_is_scaled_and_labelled(split):
    assert split.x_train.min() >= 0.0 and split.x_train.max() <= 1.0
    assert set(np.unique(split.y_test)) <= {0, 1}
    assert split.y_test.sum() > 0
    assert split.n_features == 6


def test_pipeline_fit_and_embed(split):
    pipeline = QuantumAdaptiveGraphPipeline(_fast_config(), input_dim=split.n_features)
    pipeline.fit(split.x_train, verbose=False)

    embeddings = pipeline.embed(split.x_test)
    assert embeddings.shape[0] == split.x_test.shape[0]
    assert torch.isfinite(embeddings).all()

    latents = pipeline.ablation_embed(split.x_test)
    assert latents.shape == (split.x_test.shape[0], 3)


def test_pipeline_training_reduces_reconstruction_loss(split):
    pipeline = QuantumAdaptiveGraphPipeline(
        _fast_config(**{"training.encoder_epochs": 15}), input_dim=split.n_features
    )
    pipeline.fit(split.x_train, verbose=False)
    losses = pipeline.history["encoder_loss"]
    assert losses[-1] < losses[0]


def test_pipeline_evaluate_returns_valid_metrics(split):
    pipeline = QuantumAdaptiveGraphPipeline(_fast_config(), input_dim=split.n_features)
    pipeline.fit(split.x_train, verbose=False)
    result = pipeline.evaluate(split.x_test, split.y_test)

    for key in ("accuracy", "precision", "recall", "f1", "balanced_accuracy"):
        assert 0.0 <= result.metrics[key] <= 1.0
    assert -1.0 <= result.metrics["mcc"] <= 1.0
    assert result.confusion.shape == (2, 2)
    assert result.confusion.sum() == len(split.y_test)
    assert "f1" in result.summary()


def test_predict_returns_binary_labels(split):
    pipeline = QuantumAdaptiveGraphPipeline(_fast_config(), input_dim=split.n_features)
    pipeline.fit(split.x_train, verbose=False)
    preds = pipeline.predict(split.x_test)
    assert preds.shape == (split.x_test.shape[0],)
    assert set(np.unique(preds)) <= {0, 1}


def test_sage_variant_runs(split):
    pipeline = QuantumAdaptiveGraphPipeline(
        _fast_config(**{"model.encoder": "sage"}), input_dim=split.n_features
    )
    pipeline.fit(split.x_train, verbose=False)
    assert pipeline.embed(split.x_test).shape[0] == split.x_test.shape[0]


def test_joint_quantum_parameter_shift_updates_circuit(split):
    config = _fast_config(
        **{"training.joint_quantum": True, "training.quantum_gradient": "parameter_shift"}
    )
    pipeline = QuantumAdaptiveGraphPipeline(config, input_dim=split.n_features)
    before = pipeline.encoder.circuit.weights.detach().clone()
    pipeline.fit(split.x_train, verbose=False)
    assert not torch.allclose(before, pipeline.encoder.circuit.weights.detach())


def test_embed_before_fit_raises(split):
    pipeline = QuantumAdaptiveGraphPipeline(_fast_config(), input_dim=split.n_features)
    with pytest.raises(RuntimeError):
        pipeline.embed(split.x_test)


def test_unknown_graph_encoder_rejected():
    with pytest.raises(ValueError):
        QuantumAdaptiveGraphPipeline(_fast_config(**{"model.encoder": "mlp"}), input_dim=5)


def test_comparison_table_lists_every_configuration(split):
    rng = np.random.default_rng(0)
    results = [
        evaluate_embeddings(
            rng.normal(size=(20, 4)), rng.normal(size=(15, 4)),
            np.array([0] * 8 + [1] * 7), name=name,
        )
        for name in ("baseline", "full system")
    ]
    table = comparison_table(results)
    assert "baseline" in table and "full system" in table


def test_config_yaml_roundtrip(tmp_path):
    config = PipelineConfig()
    config.quantum.n_qubits = 5
    config.model.encoder = "sage"
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    loaded = PipelineConfig.from_yaml(path)
    assert loaded.quantum.n_qubits == 5
    assert loaded.model.encoder == "sage"
