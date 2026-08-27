"""Validate the quantum kernel against PennyLane's lightning.qubit reference.

The kernel is computed with a batched PyTorch statevector implementation rather
than by invoking a quantum SDK per pair, because pairwise kernel estimation is
quadratic and the batched form reduces it to one matrix product. That is only
legitimate if it reproduces a reference simulator exactly, so the equivalence is
pinned here rather than assumed.

lightning.qubit is PennyLane's compiled C++ statevector backend and runs
natively on Apple Silicon with no CUDA requirement, which makes it a practical
reference on this hardware.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix

qml = pytest.importorskip("pennylane", reason="PennyLane not installed")


def _lightning_states(x: np.ndarray, n_qubits: int, reps: int) -> np.ndarray:
    """Prepare the same ZZ-style feature map on lightning.qubit."""
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(sample):
        for _ in range(reps):
            for k in range(n_qubits):
                qml.RY(np.pi / 2, wires=k)
            for k in range(n_qubits):
                qml.RZ(2.0 * sample[k], wires=k)
            for k in range(n_qubits - 1):
                qml.CNOT(wires=[k, k + 1])
                qml.RZ(
                    2.0 * (np.pi - sample[k]) * (np.pi - sample[k + 1]), wires=k + 1
                )
                qml.CNOT(wires=[k, k + 1])
        return qml.state()

    return np.stack([np.asarray(circuit(sample)) for sample in x])


@pytest.mark.parametrize("n_qubits,reps", [(4, 1), (6, 2)])
def test_feature_map_matches_lightning(n_qubits, reps):
    feature_map = QuantumFeatureMap(
        n_qubits=n_qubits, reps=reps, entanglement="linear", bandwidth=1.0
    )
    x = np.random.default_rng(0).uniform(0, np.pi, size=(8, n_qubits))

    reference = _lightning_states(x, n_qubits, reps)
    with torch.no_grad():
        mine = feature_map(torch.as_tensor(x, dtype=torch.float32)).numpy()

    overlap = np.abs((reference.conj() * mine).sum(axis=1))
    assert overlap.min() > 1 - 1e-5, f"states diverge: min overlap {overlap.min()}"


def test_kernel_matrix_matches_lightning():
    n_qubits, reps = 5, 2
    feature_map = QuantumFeatureMap(n_qubits=n_qubits, reps=reps, entanglement="linear")
    x = np.random.default_rng(1).uniform(0, np.pi, size=(10, n_qubits))

    reference = _lightning_states(x, n_qubits, reps)
    expected = np.abs(reference.conj() @ reference.T) ** 2
    measured = quantum_kernel_matrix(x, x, feature_map)

    assert np.abs(measured - expected).max() < 1e-4


def test_bandwidth_scales_kernel_concentration():
    """Small bandwidth keeps states close; large bandwidth drives overlaps down.

    This is the mechanism behind the bandwidth hyperparameter: it controls how
    far the feature map spreads states, and therefore whether the kernel
    retains dynamic range or concentrates toward zero.
    """
    x = np.random.default_rng(2).uniform(0, np.pi, size=(24, 6))
    off_diagonal = []
    for bandwidth in (0.05, 1.0):
        feature_map = QuantumFeatureMap(n_qubits=6, reps=2, bandwidth=bandwidth)
        kernel = quantum_kernel_matrix(x, x, feature_map)
        mask = ~np.eye(len(x), dtype=bool)
        off_diagonal.append(kernel[mask].mean())

    assert off_diagonal[0] > off_diagonal[1], (
        f"narrow bandwidth should retain higher overlap: {off_diagonal}"
    )
