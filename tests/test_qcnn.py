"""Pin the QCNN implementation against Qiskit's reference circuits.

The QCNN here is re-implemented in a batched PyTorch simulator rather than
executed through Qiskit, because Qiskit's EstimatorQNN evaluates one sample per
circuit run and trains gradient-free, which does not scale to thousands of
subjects across dozens of folds. That substitution is only legitimate if the
two agree numerically, so the equivalence is asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qagta.quantum.qcnn import QCNN, QCNNClassifier

qiskit = pytest.importorskip("qiskit", reason="Qiskit not installed")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402


def _conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def _pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def _qiskit_reference(model: QCNN, sample: torch.Tensor) -> float:
    """Rebuild the same circuit in Qiskit and return the read-out expectation."""
    n = model.n_qubits
    circuit = QuantumCircuit(n)
    for _ in range(model.feature_reps):
        for q in range(n):
            circuit.h(q)
            circuit.p(2.0 * float(sample[q]), q)

    for stage, (active, sources, sinks) in enumerate(model.schedule):
        conv_w = model.weights[2 * stage].detach().numpy()
        pool_w = model.weights[2 * stage + 1].detach().numpy()

        index = 0
        for q0, q1 in zip(active[0::2], active[1::2], strict=False):
            circuit.compose(_conv_circuit(conv_w[index : index + 3]), [q0, q1], inplace=True)
            index += 3
        if len(active) > 2:
            for q0, q1 in zip(active[1::2], active[2::2] + [active[0]], strict=False):
                circuit.compose(_conv_circuit(conv_w[index : index + 3]), [q0, q1], inplace=True)
                index += 3

        index = 0
        for source, sink in zip(sources, sinks, strict=True):
            circuit.compose(_pool_circuit(pool_w[index : index + 3]), [source, sink], inplace=True)
            index += 3

    observable = SparsePauliOp.from_list([("Z" + "I" * (n - 1), 1)])
    return float(np.real(Statevector(circuit).expectation_value(observable)))


@pytest.mark.parametrize("n_qubits", [4, 8])
def test_matches_qiskit_reference(n_qubits):
    torch.manual_seed(0)
    model = QCNN(n_qubits=n_qubits, seed=0)
    samples = torch.rand(3, n_qubits) * np.pi

    with torch.no_grad():
        mine = model(samples).numpy()
    reference = np.array([_qiskit_reference(model, s) for s in samples])

    assert np.abs(mine - reference).max() < 1e-4, (
        f"QCNN diverges from Qiskit: {np.abs(mine - reference).max():.2e}"
    )


def test_expectation_is_physical():
    model = QCNN(n_qubits=4, seed=1)
    with torch.no_grad():
        z = model(torch.rand(16, 4) * np.pi)
    assert torch.all(z <= 1 + 1e-5) and torch.all(z >= -1 - 1e-5)


def test_register_halves_at_each_stage():
    """8 qubits should pool 8 -> 4 -> 2 -> 1, giving three stages."""
    model = QCNN(n_qubits=8)
    assert len(model.schedule) == 3
    assert [len(active) for active, _, _ in model.schedule] == [8, 4, 2]


def test_gradients_reach_every_layer():
    model = QCNN(n_qubits=4, seed=0)
    model(torch.rand(4, 4)).sum().backward()
    for i, param in enumerate(model.weights):
        assert param.grad is not None, f"no gradient for layer {i}"
        assert torch.isfinite(param.grad).all()


def test_classifier_emits_two_logits():
    model = QCNNClassifier(n_qubits=4, seed=0)
    logits = model(torch.rand(6, 4))
    assert logits.shape == (6, 2)
    assert torch.isfinite(logits).all()


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        QCNN(n_qubits=6)
