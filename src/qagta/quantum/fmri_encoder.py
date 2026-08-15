"""Ring-entangled variational encoder for per-region fMRI features.

Implements the circuit applied to each brain region, one qubit per PCA
component:

    Layer 1 (encoding)     RY(theta_k * x_k) on every qubit, theta_k trainable
    Layer 2 (entanglement) ring of CNOTs, control k -> target (k+1) mod n,
                           including the wrap-around gate closing the ring
    Layer 3 (variational)  trainable RZ(phi_z) and RX(phi_x) per qubit
    Read-out               <psi|Z_k|psi> for each qubit -> latent z in R^n

The full statevector is retained alongside the expectation values, since the
graph-construction stage needs it for fidelity.

Two backends are provided behind one interface:

- :class:`RingEntangledEncoder` — batched PyTorch statevector simulation.
  All regions of a subject are evolved as one batched tensor, and gradients
  come from autograd. This is the default: the workload is ~200 executions
  of the same circuit per subject, which vectorises almost perfectly.
- :class:`PennyLaneRingEncoder` — PennyLane ``lightning.qubit`` with adjoint
  differentiation, executing region by region. Slower here, but it is the
  reference implementation and the path to shot-based or hardware backends.

The two agree to simulator precision; ``tests/test_fmri_encoder.py`` pins that.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qagta.quantum.simulator import _apply_rx, _apply_ry, _apply_rz, _cnot_permutation


class RingEntangledEncoder(nn.Module):
    """Batched PyTorch implementation of the per-region ansatz."""

    supports_statevector = True

    def __init__(self, n_qubits: int = 16, input_scale: float = math.pi) -> None:
        super().__init__()
        if n_qubits < 2:
            raise ValueError("the entangling ring needs at least 2 qubits")
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.input_scale = input_scale

        # Layer 1: trainable per-qubit gain on the encoding angle.
        self.encode_scale = nn.Parameter(torch.ones(n_qubits))
        # Layer 3: trainable variational rotations.
        self.phi_z = nn.Parameter(0.01 * torch.randn(n_qubits))
        self.phi_x = nn.Parameter(0.01 * torch.randn(n_qubits))

        for k in range(n_qubits):
            target = (k + 1) % n_qubits
            self.register_buffer(
                f"_ring_{k}", _cnot_permutation(n_qubits, k, target, torch.device("cpu"))
            )

        indices = torch.arange(self.dim)
        signs = torch.stack(
            [1.0 - 2.0 * ((indices >> (n_qubits - k - 1)) & 1).float() for k in range(n_qubits)]
        )
        self.register_buffer("_z_signs", signs)

    @property
    def latent_dim(self) -> int:
        return self.n_qubits

    def encode_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation angles ``theta_k * x_k * pi`` for a ``(batch, n_qubits)`` input."""
        if x.dim() != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(
                f"expected input of shape (batch, {self.n_qubits}), got {tuple(x.shape)}"
            )
        return x * self.encode_scale * self.input_scale

    def prepare_state(self, x: torch.Tensor) -> torch.Tensor:
        """Evolve |0...0> through the circuit for a batch of regions."""
        angles = self.encode_angles(x)
        batch = angles.shape[0]

        state = torch.zeros((batch, self.dim), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0

        for k in range(self.n_qubits):  # layer 1
            state = _apply_ry(state, self.n_qubits, k, angles[:, k])
        for k in range(self.n_qubits):  # layer 2, ring closes at k = n-1
            state = state.index_select(1, getattr(self, f"_ring_{k}"))
        for k in range(self.n_qubits):  # layer 3
            state = _apply_rz(state, self.n_qubits, k, self.phi_z[k])
            state = _apply_rx(state, self.n_qubits, k, self.phi_x[k])
        return state

    def expectations(self, state: torch.Tensor) -> torch.Tensor:
        probs = state.real**2 + state.imag**2
        return probs @ self._z_signs.T

    def forward(
        self, x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = self.prepare_state(x)
        z = self.expectations(state)
        return (z, state) if return_state else z

    def encode(self, x: torch.Tensor, return_state: bool = False):
        return self.forward(x, return_state=return_state)


class PennyLaneRingEncoder(nn.Module):
    """PennyLane reference backend using adjoint differentiation.

    Mirrors :class:`RingEntangledEncoder` gate for gate. Executes one region
    per circuit evaluation, so it is markedly slower than the batched
    simulator for a 200-node graph; it exists as the paper-faithful reference
    and as the route to shot-based or hardware execution.
    """

    supports_statevector = True

    def __init__(
        self, n_qubits: int = 16, input_scale: float = math.pi, device_name: str = "lightning.qubit"
    ) -> None:
        super().__init__()
        import pennylane as qml

        self.n_qubits = n_qubits
        self.input_scale = input_scale
        self.encode_scale = nn.Parameter(torch.ones(n_qubits))
        self.phi_z = nn.Parameter(0.01 * torch.randn(n_qubits))
        self.phi_x = nn.Parameter(0.01 * torch.randn(n_qubits))

        dev = qml.device(device_name, wires=n_qubits)

        def circuit_ops(angles, phi_z, phi_x):
            for k in range(n_qubits):
                qml.RY(angles[k], wires=k)
            for k in range(n_qubits):
                qml.CNOT(wires=[k, (k + 1) % n_qubits])
            for k in range(n_qubits):
                qml.RZ(phi_z[k], wires=k)
                qml.RX(phi_x[k], wires=k)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def expval_node(angles, phi_z, phi_x):
            circuit_ops(angles, phi_z, phi_x)
            return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]

        # Statevector read-out has no adjoint gradient; it is used for the
        # one-off fidelity initialisation, so backprop is not required.
        @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="torch")
        def state_node(angles, phi_z, phi_x):
            circuit_ops(angles, phi_z, phi_x)
            return qml.state()

        self._expval_node = expval_node
        self._state_node = state_node

    @property
    def latent_dim(self) -> int:
        return self.n_qubits

    def encode_angles(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.encode_scale * self.input_scale

    def forward(
        self, x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        angles = self.encode_angles(x)
        rows = [
            torch.stack(self._expval_node(angles[i], self.phi_z, self.phi_x))
            for i in range(angles.shape[0])
        ]
        z = torch.stack(rows).float()
        if not return_state:
            return z
        states = torch.stack(
            [
                self._state_node(angles[i].detach(), self.phi_z.detach(), self.phi_x.detach())
                for i in range(angles.shape[0])
            ]
        ).to(torch.complex64)
        return z, states

    def encode(self, x: torch.Tensor, return_state: bool = False):
        return self.forward(x, return_state=return_state)


def build_encoder(backend: str = "torch", n_qubits: int = 16) -> nn.Module:
    """Construct the per-region encoder for the requested backend."""
    if backend == "torch":
        return RingEntangledEncoder(n_qubits=n_qubits)
    if backend == "pennylane":
        return PennyLaneRingEncoder(n_qubits=n_qubits)
    raise ValueError(f"unknown quantum backend: {backend!r} (expected 'torch' or 'pennylane')")
