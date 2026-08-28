"""Variational Quantum Classifier, following the Qiskit reference design.

The VQC of the Qiskit Machine Learning tutorials: a ZZFeatureMap encoding, a
RealAmplitudes variational ansatz, and a parity read-out mapping the measured
bitstring distribution onto class probabilities.

    encoding    H on every qubit, RZ(2 x_k), then RZ(2(pi-x_i)(pi-x_j)) on
                entangled pairs — the second-order terms that distinguish a ZZ
                map from a product encoding
    ansatz      RealAmplitudes: alternating trainable RY layers and CNOT
                entanglers, reps + 1 rotation layers in total
    read-out    parity of the computational basis state, summed into two class
                probabilities

Implemented in the batched simulator for the same reason as the QCNN: Qiskit's
sampler-based VQC evaluates one sample per execution and optimises with COBYLA,
which does not scale to thousands of subjects across dozens of folds. This
version is differentiable and trains by gradient descent.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qagta.quantum.simulator import _apply_ry, _apply_rz, _cnot_permutation


class VQC(nn.Module):
    """Variational quantum classifier with parity read-out."""

    def __init__(
        self,
        n_qubits: int = 8,
        reps: int = 2,
        feature_reps: int = 1,
        entanglement: str = "linear",
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_reps = feature_reps
        self.dim = 2**n_qubits

        if entanglement == "full":
            pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        else:
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        self.pairs = pairs
        for control, target in pairs:
            self.register_buffer(
                f"_cx_{control}_{target}",
                _cnot_permutation(n_qubits, control, target, torch.device("cpu")),
            )

        # RealAmplitudes: reps + 1 layers of one RY angle per qubit.
        self.theta = nn.Parameter(0.1 * torch.randn((reps + 1) * n_qubits))

        # Parity of each basis index decides which class it contributes to.
        indices = torch.arange(self.dim)
        parity = torch.zeros(self.dim)
        for bit in range(n_qubits):
            parity += ((indices >> bit) & 1).float()
        self.register_buffer("_parity", (parity % 2).long())

    def _cx(self, state, control, target):
        return state.index_select(1, getattr(self, f"_cx_{control}_{target}"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return two class logits per sample, shape ``(batch, 2)``."""
        if x.dim() != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"expected (batch, {self.n_qubits}), got {tuple(x.shape)}")

        batch = x.shape[0]
        state = torch.zeros((batch, self.dim), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0
        half_pi = torch.full((batch,), math.pi / 2, device=x.device)
        pi = torch.full((batch,), math.pi, device=x.device)

        # --- ZZ feature map ---------------------------------------------
        for _ in range(self.feature_reps):
            for q in range(self.n_qubits):
                state = _apply_rz(state, self.n_qubits, q, pi)
                state = _apply_ry(state, self.n_qubits, q, half_pi)
                state = _apply_rz(state, self.n_qubits, q, 2.0 * x[:, q])
            for control, target in self.pairs:
                state = self._cx(state, control, target)
                interaction = 2.0 * (math.pi - x[:, control]) * (math.pi - x[:, target])
                state = _apply_rz(state, self.n_qubits, target, interaction)
                state = self._cx(state, control, target)

        # --- RealAmplitudes ansatz ---------------------------------------
        weights = self.theta.reshape(self.reps + 1, self.n_qubits)
        for layer in range(self.reps):
            for q in range(self.n_qubits):
                state = _apply_ry(state, self.n_qubits, q, weights[layer, q].expand(batch))
            for control, target in self.pairs:
                state = self._cx(state, control, target)
        for q in range(self.n_qubits):
            state = _apply_ry(state, self.n_qubits, q, weights[self.reps, q].expand(batch))

        # --- parity read-out ----------------------------------------------
        probs = state.real**2 + state.imag**2
        even = probs[:, self._parity == 0].sum(dim=1)
        odd = probs[:, self._parity == 1].sum(dim=1)
        # Log-probabilities act as logits; clamped so an empty branch cannot
        # produce a non-finite loss.
        return torch.log(torch.stack([even, odd], dim=1).clamp_min(1e-9))
