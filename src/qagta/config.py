"""Configuration for the end-to-end pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class QuantumConfig:
    backend: str = "native"  # "native" (built-in simulator) or "qiskit"
    n_qubits: int = 4
    reps: int = 2
    entanglement: str = "full"  # "full" or "linear"
    decoder_hidden: int = 8


@dataclass
class GraphConfig:
    hidden_dim: int = 16
    k_neighbors: int = 5
    edge_threshold: float = 0.4
    use_fidelity: bool = True


@dataclass
class ModelConfig:
    encoder: str = "gat"  # "gat" or "sage"
    hidden_channels: int = 8
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.3
    decision_hidden: int = 32


@dataclass
class TrainingConfig:
    encoder_epochs: int = 30
    encoder_lr: float = 0.01
    graph_epochs: int = 40
    graph_lr: float = 0.005
    joint_quantum: bool = False
    quantum_gradient: str = "autograd"  # "autograd" or "parameter_shift"
    quantum_lr: float = 0.01
    latent_mask: float = 0.5  # fraction of self-features masked during fusion
    ocsvm_nu: float = 0.05
    seed: int = 42


@dataclass
class PipelineConfig:
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            quantum=QuantumConfig(**raw.get("quantum", {})),
            graph=GraphConfig(**raw.get("graph", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as handle:
            yaml.safe_dump(asdict(self), handle, sort_keys=False)
