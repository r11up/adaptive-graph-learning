"""Variational quantum encoder producing latent embeddings from temporal features.

The encoder is the first learnable stage of the pipeline:

1. a classical projection maps the (normalised) input window to one angle
   per qubit, squashed into the ``[0, 2*pi]`` encoding domain,
2. the angles drive a parameterised quantum circuit, and
3. Pauli-Z expectation values of the prepared state form the latent vector
   ``z``; the raw statevector can also be returned for fidelity-based
   similarity computation downstream.

A lightweight classical decoder is attached so the module can be
pre-trained as an autoencoder with a reconstruction objective.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qagta.quantum.simulator import StatevectorSimulator


class QuantumEncoder(nn.Module):
    """Quantum autoencoder subsystem: classical head, quantum core, classical tail."""

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        reps: int = 2,
        entanglement: str = "full",
        decoder_hidden: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        self.angle_head = nn.Linear(input_dim, n_qubits)
        # Angle encoding is only informative if each qubit's rotation varies
        # across the batch. Without normalisation the learned projection
        # readily collapses onto one or two qubits, leaving the rest at a
        # near-constant angle: the prepared states become indistinguishable,
        # every pairwise fidelity approaches 1 and the quantum similarity
        # term carries no signal. Standardising per qubit keeps all of them
        # spanning a useful arc of the rotation domain.
        self.angle_norm = nn.BatchNorm1d(n_qubits)
        self.circuit = StatevectorSimulator(
            n_qubits=n_qubits, reps=reps, entanglement=entanglement
        )
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
        """Map inputs to rotation angles in ``[0, 2*pi]``."""
        return (torch.tanh(self.angle_norm(self.angle_head(x))) + 1.0) * math.pi

    def encode(
        self, x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return latent embeddings ``z`` (and optionally the statevectors)."""
        angles = self.encode_angles(x)
        return self.circuit(angles, return_state=return_state)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoencoder pass: returns ``(reconstruction, latent)``."""
        z = self.encode(x)
        return self.decoder(z), z
