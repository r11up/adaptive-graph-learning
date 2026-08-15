"""Quantum kernel methods operating between *subjects*.

The connectome pipeline uses fidelity between brain *regions* to build a graph.
This module uses the same primitive at a different level: fidelity between whole
*subjects*, giving a kernel matrix

    K_ij = |<psi(x_i)|psi(x_j)>|^2

that drops directly into a kernel SVM. That is the quantum kernel of Havlicek et
al. (Nature 567:209, 2019) and Schuld & Killoran (PRL 122:040504, 2019), and it
is the setting where quantum feature maps have shown measurable advantage over
classical kernels on neuroimaging data.

Why this framing has a better shot than region-level topology: fidelity
concentrates toward zero as the register widens, so it needs a *narrow* register
to stay discriminative. A subject-level kernel needs only enough qubits to
encode a compressed subject descriptor — 4 to 8 — where fidelity retains real
dynamic range. Region-level topology over 200 nodes pushed toward wider
registers, which is exactly where the metric dies.

Fairness note: the classical comparator must see the *same* features. A quantum
kernel on 8 PCA components compared against a classical kernel on 19,900 raw
correlations measures dimensionality, not quantum advantage.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from qagta.quantum.simulator import _apply_ry, _apply_rz, _cnot_permutation


class QuantumFeatureMap(nn.Module):
    """Non-parametric feature map for kernel estimation.

    A ZZ-style map: Hadamard-like superposition via RY(pi/2), single-qubit
    RZ rotations carrying the features, and entangling RZ interactions on
    connected pairs carrying products of features. The second-order terms are
    what make the induced kernel hard to reproduce classically; a purely
    single-qubit map factorises and buys nothing.

    Has no trainable parameters — the kernel is fixed by the encoding, which is
    what makes it usable with a standard SVM.
    """

    def __init__(
        self, n_qubits: int = 8, reps: int = 2, entanglement: str = "linear", scale: float = 1.0
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.reps = reps
        self.scale = scale
        self.dim = 2**n_qubits

        if entanglement == "full":
            pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        elif entanglement == "linear":
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        else:
            raise ValueError(f"unknown entanglement {entanglement!r}")
        self.pairs = pairs
        for control, target in pairs:
            self.register_buffer(
                f"_cx_{control}_{target}",
                _cnot_permutation(n_qubits, control, target, torch.device("cpu")),
            )

    def _cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        return state.index_select(1, getattr(self, f"_cx_{control}_{target}"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(batch, n_qubits)`` feature block to statevectors."""
        if x.dim() != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"expected (batch, {self.n_qubits}), got {tuple(x.shape)}")

        batch = x.shape[0]
        state = torch.zeros((batch, self.dim), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0
        angles = x * self.scale

        for _ in range(self.reps):
            for k in range(self.n_qubits):  # superposition
                state = _apply_ry(state, self.n_qubits, k, torch.full((batch,), math.pi / 2))
            for k in range(self.n_qubits):  # first-order terms
                state = _apply_rz(state, self.n_qubits, k, 2.0 * angles[:, k])
            for control, target in self.pairs:  # second-order interaction terms
                state = self._cnot(state, control, target)
                interaction = 2.0 * (math.pi - angles[:, control]) * (math.pi - angles[:, target])
                state = _apply_rz(state, self.n_qubits, target, interaction)
                state = self._cnot(state, control, target)
        return state


@torch.no_grad()
def quantum_kernel_matrix(
    x_left: np.ndarray,
    x_right: np.ndarray,
    feature_map: QuantumFeatureMap,
    block: int = 512,
) -> np.ndarray:
    """Fidelity kernel ``|<psi(x_i)|psi(x_j)>|^2`` between two feature sets.

    Computed by preparing every state once and taking overlaps as a matrix
    product, rather than re-running a circuit per pair — the pairwise form is
    what makes naive quantum-kernel estimation quadratically expensive.
    """
    left = feature_map(torch.as_tensor(np.asarray(x_left), dtype=torch.float32))
    right = feature_map(torch.as_tensor(np.asarray(x_right), dtype=torch.float32))

    out = np.empty((left.shape[0], right.shape[0]), dtype=np.float32)
    for start in range(0, left.shape[0], block):
        chunk = left[start : start + block]
        overlaps = chunk.conj() @ right.T
        out[start : start + block] = (overlaps.abs() ** 2).cpu().numpy()
    return out
