"""Optional Qiskit backend for the quantum encoder.

Provides the same interface as :class:`qagta.quantum.encoder.QuantumEncoder`
but executes the circuit through Qiskit's ``EstimatorQNN`` /
``TorchConnector`` stack (ZZ feature map + RealAmplitudes ansatz, Pauli-Z
observables). Useful for running on Qiskit simulators or real hardware
backends; gradients of the quantum parameters are computed by Qiskit's
gradient framework (parameter-shift based) rather than autograd.

Qiskit is an optional dependency::

    pip install "qagta[qiskit]"

Note: statevector access (for fidelity similarity) is not exposed by the
estimator primitive, so this backend reports fidelity as unavailable and
the edge learner falls back to classical similarity terms.
"""

from __future__ import annotations

import math

import torch
from torch import nn

try:  # pragma: no cover - exercised only when qiskit is installed
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QISKIT_AVAILABLE = False


class QiskitQuantumEncoder(nn.Module):
    """Drop-in Qiskit-backed replacement for :class:`QuantumEncoder`."""

    supports_statevector = False

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        reps: int = 2,
        entanglement: str = "full",
        decoder_hidden: int = 8,
    ) -> None:
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit backend requested but qiskit / qiskit-machine-learning "
                "are not installed. Install with: pip install 'qagta[qiskit]'"
            )
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        self.angle_head = nn.Linear(input_dim, n_qubits)

        feature_map = ZZFeatureMap(
            feature_dimension=n_qubits, reps=1, entanglement=entanglement
        )
        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)
        circuit = QuantumCircuit(n_qubits)
        circuit.compose(feature_map, range(n_qubits), inplace=True)
        circuit.compose(ansatz, range(n_qubits), inplace=True)

        observables = [
            SparsePauliOp.from_list([("I" * k + "Z" + "I" * (n_qubits - 1 - k), 1.0)])
            for k in range(n_qubits)
        ]
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
        )
        self.qnn = TorchConnector(qnn)

        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, input_dim),
            nn.Sigmoid(),
        )

    @property
    def latent_dim(self) -> int:
        return self.n_qubits

    def encode_angles(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(self.angle_head(x)) + 1.0) * math.pi

    def encode(self, x: torch.Tensor, return_state: bool = False) -> torch.Tensor:
        if return_state:
            raise NotImplementedError(
                "The Qiskit estimator backend does not expose statevectors; "
                "use the native simulator backend for fidelity-based edges."
            )
        return self.qnn(self.encode_angles(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z
