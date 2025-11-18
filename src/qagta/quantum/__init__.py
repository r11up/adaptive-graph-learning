"""Quantum subsystem: variational encoder, simulator and similarity metrics."""

from qagta.quantum.encoder import QuantumEncoder
from qagta.quantum.fidelity import fidelity, pairwise_fidelity
from qagta.quantum.simulator import StatevectorSimulator

__all__ = [
    "QuantumEncoder",
    "StatevectorSimulator",
    "fidelity",
    "pairwise_fidelity",
]
