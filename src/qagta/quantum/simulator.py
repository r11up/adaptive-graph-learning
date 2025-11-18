"""Differentiable statevector simulator for the variational quantum encoder.

Implements just enough of gate-model quantum computing to support the
encoder used by the pipeline:

- single-qubit RY rotations with per-sample (batched) angles,
- CNOT entanglement (linear or full connectivity),
- an angle-encoding feature map,
- a RealAmplitudes-style hardware-efficient ansatz,
- Pauli-Z expectation values, and
- access to the prepared statevectors (for fidelity-based similarity).

Everything is written in pure PyTorch on complex tensors, so gradients flow
through the circuit via autograd. This makes the quantum stage co-trainable
with the classical stages without an external quantum SDK; an optional
Qiskit backend with the same interface lives in
:mod:`qagta.quantum.qiskit_backend`.

Convention: qubit 0 is the most-significant bit of the computational basis
index, i.e. the state tensor of shape ``(batch, 2**n)`` is viewed as
``(batch, 2, 2, ..., 2)`` with qubit 0 on the left.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn


def _apply_ry(state: torch.Tensor, n_qubits: int, qubit: int, theta: torch.Tensor) -> torch.Tensor:
    """Apply RY(theta) on ``qubit``. ``theta`` is scalar or shape ``(batch,)``."""
    batch = state.shape[0]
    left = 2**qubit
    right = 2 ** (n_qubits - qubit - 1)
    s = state.reshape(batch, left, 2, right)

    theta = torch.as_tensor(theta, dtype=torch.float32, device=state.device)
    if theta.dim() == 0:
        theta = theta.expand(batch)
    cos = torch.cos(theta / 2).reshape(batch, 1, 1)
    sin = torch.sin(theta / 2).reshape(batch, 1, 1)

    s0 = s[:, :, 0, :]
    s1 = s[:, :, 1, :]
    out0 = cos * s0 - sin * s1
    out1 = sin * s0 + cos * s1
    return torch.stack((out0, out1), dim=2).reshape(batch, -1)


def _cnot_permutation(
    n_qubits: int, control: int, target: int, device: torch.device
) -> torch.Tensor:
    """Basis-index permutation implementing CNOT(control, target)."""
    dim = 2**n_qubits
    indices = torch.arange(dim, device=device)
    control_bit = (indices >> (n_qubits - control - 1)) & 1
    flipped = indices ^ (1 << (n_qubits - target - 1))
    return torch.where(control_bit.bool(), flipped, indices)


def _entangler_pairs(n_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    if entanglement == "full":
        return list(itertools.combinations(range(n_qubits), 2))
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    raise ValueError(f"Unknown entanglement scheme: {entanglement!r}")


class StatevectorSimulator(nn.Module):
    """Prepares |psi(x, theta)> and measures Pauli-Z expectations.

    The circuit is::

        [feature map] x reps_in   : RY(x_k) on every qubit + entangling CNOTs
        [ansatz]      x reps      : RY(theta) on every qubit + entangling CNOTs
        final RY(theta) layer

    which mirrors an angle-encoding feature map followed by a
    RealAmplitudes-style variational ansatz.
    """

    def __init__(
        self,
        n_qubits: int,
        reps: int = 2,
        input_reps: int = 1,
        entanglement: str = "full",
    ) -> None:
        super().__init__()
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.reps = reps
        self.input_reps = input_reps
        self.entanglement = entanglement
        self.dim = 2**n_qubits

        # (reps + 1) RY layers of n_qubits parameters each, RealAmplitudes-style.
        self.weights = nn.Parameter(0.1 * torch.randn((reps + 1) * n_qubits))

        pairs = _entangler_pairs(n_qubits, entanglement)
        self._cnot_perms: list[torch.Tensor] = []
        for control, target in pairs:
            perm = _cnot_permutation(n_qubits, control, target, torch.device("cpu"))
            self.register_buffer(f"_perm_{control}_{target}", perm)
            self._cnot_perms.append(getattr(self, f"_perm_{control}_{target}"))

        # Precomputed +-1 signs of Z_k over the computational basis.
        indices = torch.arange(self.dim)
        signs = torch.stack(
            [1.0 - 2.0 * ((indices >> (n_qubits - k - 1)) & 1).float() for k in range(n_qubits)]
        )
        self.register_buffer("_z_signs", signs)  # (n_qubits, dim)

    @property
    def num_weights(self) -> int:
        return self.weights.numel()

    def _entangle(self, state: torch.Tensor) -> torch.Tensor:
        for perm in self._cnot_perms:
            state = state.index_select(1, perm)
        return state

    def prepare_state(
        self, angles: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the circuit and return statevectors of shape ``(batch, 2**n)``.

        Parameters
        ----------
        angles:
            Encoding angles, shape ``(batch, n_qubits)``; expected in
            ``[0, 2*pi]`` (any real value is valid).
        weights:
            Optional override of the ansatz parameters (used by the
            parameter-shift rule); defaults to the trained ``self.weights``.
        """
        if angles.dim() != 2 or angles.shape[1] != self.n_qubits:
            raise ValueError(f"angles must have shape (batch, {self.n_qubits})")
        if weights is None:
            weights = self.weights

        batch = angles.shape[0]
        state = torch.zeros((batch, self.dim), dtype=torch.complex64, device=angles.device)
        state[:, 0] = 1.0

        for _ in range(self.input_reps):
            for q in range(self.n_qubits):
                state = _apply_ry(state, self.n_qubits, q, angles[:, q])
            state = self._entangle(state)

        w = weights.reshape(self.reps + 1, self.n_qubits)
        for layer in range(self.reps):
            for q in range(self.n_qubits):
                state = _apply_ry(state, self.n_qubits, q, w[layer, q])
            state = self._entangle(state)
        for q in range(self.n_qubits):
            state = _apply_ry(state, self.n_qubits, q, w[self.reps, q])

        return state

    def expectations(self, state: torch.Tensor) -> torch.Tensor:
        """Pauli-Z expectation <psi|Z_k|psi> per qubit, shape ``(batch, n_qubits)``."""
        probs = state.real**2 + state.imag**2
        return probs @ self._z_signs.T

    def forward(
        self,
        angles: torch.Tensor,
        weights: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = self.prepare_state(angles, weights)
        z = self.expectations(state)
        if return_state:
            return z, state
        return z
