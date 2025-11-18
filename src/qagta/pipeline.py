"""End-to-end orchestration of the quantum-assisted adaptive graph pipeline.

Wires the subsystems together behind a scikit-learn-like interface::

    pipeline = QuantumAdaptiveGraphPipeline(config, input_dim=split.n_features)
    pipeline.fit(split.x_train)
    result = pipeline.evaluate(split.x_test, split.y_test)

``embed`` exposes the final fused representations, and ``ablation_embed``
returns the quantum-latent-only representation for baseline comparisons.
"""

from __future__ import annotations

import numpy as np
import torch

from qagta.config import PipelineConfig
from qagta.graph.constructor import DynamicGraphConstructor
from qagta.models.decision import DecisionModule
from qagta.models.gat import GraphAttentionEncoder
from qagta.models.gnn import GraphSAGEEncoder
from qagta.quantum.encoder import QuantumEncoder
from qagta.training.evaluate import EvaluationResult, evaluate_embeddings
from qagta.training.trainer import train_graph_stage, train_quantum_encoder


def _build_quantum_encoder(config: PipelineConfig, input_dim: int) -> torch.nn.Module:
    if config.quantum.backend == "qiskit":
        from qagta.quantum.qiskit_backend import QiskitQuantumEncoder

        return QiskitQuantumEncoder(
            input_dim=input_dim,
            n_qubits=config.quantum.n_qubits,
            reps=config.quantum.reps,
            entanglement=config.quantum.entanglement,
            decoder_hidden=config.quantum.decoder_hidden,
        )
    return QuantumEncoder(
        input_dim=input_dim,
        n_qubits=config.quantum.n_qubits,
        reps=config.quantum.reps,
        entanglement=config.quantum.entanglement,
        decoder_hidden=config.quantum.decoder_hidden,
    )


class QuantumAdaptiveGraphPipeline:
    """Quantum encoding -> adaptive topology -> attention propagation -> scoring."""

    def __init__(self, config: PipelineConfig | None = None, input_dim: int = 10) -> None:
        self.config = config or PipelineConfig()
        self.input_dim = input_dim

        torch.manual_seed(self.config.training.seed)
        np.random.seed(self.config.training.seed)

        self.encoder = _build_quantum_encoder(self.config, input_dim)
        latent_dim = self.encoder.latent_dim

        self.constructor = DynamicGraphConstructor(
            embedding_dim=latent_dim,
            hidden_dim=self.config.graph.hidden_dim,
            k_neighbors=self.config.graph.k_neighbors,
            threshold=self.config.graph.edge_threshold,
            use_fidelity=self._fidelity_enabled(),
        )
        if self.config.model.encoder == "gat":
            self.graph_model: torch.nn.Module = GraphAttentionEncoder(
                in_channels=latent_dim,
                hidden_channels=self.config.model.hidden_channels,
                num_layers=self.config.model.num_layers,
                heads=self.config.model.heads,
                dropout=self.config.model.dropout,
            )
        elif self.config.model.encoder == "sage":
            self.graph_model = GraphSAGEEncoder(
                in_channels=latent_dim,
                hidden_channels=self.config.model.hidden_channels,
                num_layers=self.config.model.num_layers,
                dropout=self.config.model.dropout,
            )
        else:
            raise ValueError(f"Unknown graph encoder: {self.config.model.encoder!r}")

        self.decision = DecisionModule(
            graph_dim=self.config.model.hidden_channels,
            latent_dim=latent_dim,
            hidden_dim=self.config.model.decision_hidden,
        )

        self._fitted = False
        self._x_train: torch.Tensor | None = None
        self.history: dict[str, list[float]] = {}

    def _fidelity_enabled(self) -> bool:
        if not self.config.graph.use_fidelity:
            return False
        return getattr(self.encoder, "supports_statevector", True)

    # ------------------------------------------------------------------ fit

    def fit(
        self, x_train: np.ndarray | torch.Tensor, verbose: bool = True
    ) -> QuantumAdaptiveGraphPipeline:
        """Two-stage hybrid training on normal-only data."""
        x = torch.as_tensor(np.asarray(x_train), dtype=torch.float32)
        self._x_train = x
        cfg = self.config.training

        if verbose:
            print("== stage 1: quantum encoder pre-training ==")
        self.history["encoder_loss"] = train_quantum_encoder(
            self.encoder,
            x,
            epochs=cfg.encoder_epochs,
            lr=cfg.encoder_lr,
            verbose=verbose,
        )

        if verbose:
            print("== stage 2: adaptive graph + propagation training ==")
        self.history["graph_loss"] = train_graph_stage(
            self.encoder,
            self.constructor,
            self.graph_model,
            self.decision,
            x,
            epochs=cfg.graph_epochs,
            lr=cfg.graph_lr,
            joint_quantum=cfg.joint_quantum,
            quantum_gradient=cfg.quantum_gradient,
            quantum_lr=cfg.quantum_lr,
            use_fidelity=self._fidelity_enabled(),
            latent_mask=cfg.latent_mask,
            verbose=verbose,
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------- embedding

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._fidelity_enabled():
            latent, states = self.encoder.encode(x, return_state=True)
            return latent, states
        return self.encoder.encode(x), None

    @torch.no_grad()
    def embed(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Final fused embeddings for a batch of samples."""
        self._check_fitted()
        self._eval_mode()
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        latent, states = self._encode(x)
        graph = self.constructor(latent, states)
        propagated = self.graph_model(graph.x, graph.edge_index, graph.edge_attr)
        return self.decision(propagated, latent)

    @torch.no_grad()
    def ablation_embed(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Quantum-latent-only embeddings (no graph stage), for baselines."""
        self._eval_mode()
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        latent, _ = self._encode(x)
        return latent

    # ------------------------------------------------------------- inference

    def evaluate(
        self,
        x_test: np.ndarray | torch.Tensor,
        y_test: np.ndarray,
        name: str = "quantum-adaptive-graph",
    ) -> EvaluationResult:
        """Score a labelled test set with the full pipeline."""
        self._check_fitted()
        assert self._x_train is not None
        return evaluate_embeddings(
            self.embed(self._x_train),
            self.embed(x_test),
            y_test,
            name=name,
            nu=self.config.training.ocsvm_nu,
        )

    def predict(self, x_test: np.ndarray | torch.Tensor) -> np.ndarray:
        """Binary anomaly predictions (1 = anomaly) for unlabelled data."""
        self._check_fitted()
        assert self._x_train is not None
        from sklearn.svm import OneClassSVM

        detector = OneClassSVM(
            kernel="rbf", gamma="scale", nu=self.config.training.ocsvm_nu
        )
        detector.fit(self.embed(self._x_train).numpy())
        raw = detector.predict(self.embed(x_test).numpy())
        return (raw == -1).astype(int)

    # -------------------------------------------------------------- helpers

    def _eval_mode(self) -> None:
        for module in (self.encoder, self.constructor, self.graph_model, self.decision):
            module.eval()

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Pipeline is not fitted yet; call fit() first.")
