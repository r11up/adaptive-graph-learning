"""Projected quantum kernels and kernel-target alignment.

Two tools aimed squarely at the exponential concentration measured in
FINDING 01.

**Projected quantum kernels** (Huang et al., Nat. Commun. 12:2631, 2021).
Instead of comparing full statevectors, which concentrate as 2^n outruns the
number of distinct states, compare their *reduced density matrices*. Projecting
onto one-qubit marginals keeps the comparison in a fixed low-dimensional space
no matter how wide the register, so the kernel retains dynamic range where a
fidelity kernel would collapse:

    K_ij = exp(-gamma * sum_k || rho_k(x_i) - rho_k(x_j) ||_F^2)

A one-qubit reduced density matrix is fully determined by its Bloch vector
(<X>, <Y>, <Z>), and the Frobenius distance between two of them is half the
squared Euclidean distance between Bloch vectors — so the whole kernel reduces
to Pauli expectation values, which are cheap.

**Kernel-target alignment** (Cristianini et al., NeurIPS 2001). Measures how
well a kernel's similarity structure matches the labels, without training a
classifier. Selecting on alignment rather than validation accuracy is standard
in the quantum-kernel literature and is far cheaper than an inner CV loop.
"""

from __future__ import annotations

import numpy as np
import torch


def bloch_vectors(states: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Per-qubit Bloch vectors ``(batch, n_qubits, 3)`` from statevectors.

    A one-qubit reduced density matrix is ``(I + xX + yY + zZ)/2``, so the
    triple ``(<X>, <Y>, <Z>)`` determines it completely.
    """
    batch = states.shape[0]
    out = torch.empty(batch, n_qubits, 3, device=states.device)

    for qubit in range(n_qubits):
        left = 2**qubit
        right = 2 ** (n_qubits - qubit - 1)
        block = states.reshape(batch, left, 2, right)
        a = block[:, :, 0, :]  # amplitude with qubit = |0>
        b = block[:, :, 1, :]  # amplitude with qubit = |1>

        cross = (a.conj() * b).sum(dim=(1, 2))
        out[:, qubit, 0] = 2.0 * cross.real          # <X>
        out[:, qubit, 1] = -2.0 * cross.imag         # <Y>
        out[:, qubit, 2] = (a.abs() ** 2).sum(dim=(1, 2)) - (b.abs() ** 2).sum(dim=(1, 2))  # <Z>
    return out


@torch.no_grad()
def projected_kernel_matrix(
    states_left: torch.Tensor,
    states_right: torch.Tensor,
    n_qubits: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Projected quantum kernel from one-qubit reduced density matrices.

    ``|| rho_a - rho_b ||_F^2 = 0.5 * || bloch_a - bloch_b ||^2``, so the
    exponent is a plain squared Euclidean distance over stacked Bloch vectors.
    """
    left = bloch_vectors(states_left, n_qubits).reshape(states_left.shape[0], -1)
    right = bloch_vectors(states_right, n_qubits).reshape(states_right.shape[0], -1)
    squared = torch.cdist(left, right, p=2) ** 2
    return torch.exp(-gamma * 0.5 * squared).cpu().numpy()


def kernel_target_alignment(kernel: np.ndarray, labels: np.ndarray) -> float:
    """Centred kernel-target alignment in ``[-1, 1]``; higher is better.

    Compares the kernel against the ideal kernel ``y y^T`` built from +-1
    labels. Centring matters: without it a kernel with a large constant offset
    scores well regardless of how it orders the data.
    """
    y = np.where(np.asarray(labels) > 0, 1.0, -1.0).reshape(-1, 1)
    ideal = y @ y.T

    n = kernel.shape[0]
    centring = np.eye(n) - np.ones((n, n)) / n
    centred = centring @ kernel @ centring

    numerator = float((centred * ideal).sum())
    denominator = float(np.linalg.norm(centred) * np.linalg.norm(ideal))
    return numerator / denominator if denominator > 0 else 0.0
