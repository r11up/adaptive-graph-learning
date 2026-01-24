"""Tests for the quantum simulator, encoder and fidelity metrics."""

from __future__ import annotations

import math

import pytest
import torch

from qagta.quantum import QuantumEncoder, StatevectorSimulator, pairwise_fidelity
from qagta.quantum.simulator import _apply_ry


def test_statevector_is_normalised():
    sim = StatevectorSimulator(n_qubits=3, reps=2)
    angles = torch.rand(5, 3) * 2 * math.pi
    state = sim.prepare_state(angles)
    norms = (state.abs() ** 2).sum(dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_expectations_within_physical_range():
    sim = StatevectorSimulator(n_qubits=4, reps=2)
    angles = torch.rand(8, 4) * 2 * math.pi
    z = sim(angles)
    assert z.shape == (8, 4)
    assert torch.all(z <= 1.0 + 1e-5)
    assert torch.all(z >= -1.0 - 1e-5)


def test_ry_rotation_matches_analytic_result():
    """RY(pi) on |0> gives |1>, so <Z> flips from +1 to -1."""
    state = torch.zeros(1, 2, dtype=torch.complex64)
    state[0, 0] = 1.0
    rotated = _apply_ry(state, n_qubits=1, qubit=0, theta=torch.tensor(math.pi))
    assert abs(rotated[0, 1].abs().item() - 1.0) < 1e-5
    assert rotated[0, 0].abs().item() < 1e-5


def test_entanglement_creates_correlated_state():
    """RY(pi/2) on qubit 0 followed by CNOT yields a Bell state.

    ``reps=0`` leaves a single (zeroed) ansatz rotation layer and no ansatz
    entangler, so exactly one CNOT acts on the encoded state.
    """
    sim = StatevectorSimulator(n_qubits=2, reps=0, entanglement="linear")
    with torch.no_grad():
        sim.weights.zero_()
    angles = torch.tensor([[math.pi / 2, 0.0]])
    state = sim.prepare_state(angles)
    probs = (state.abs() ** 2)[0]
    # Population sits on |00> and |11>; the |01>/|10> branches stay empty.
    assert probs[1].item() < 1e-4
    assert probs[2].item() < 1e-4
    assert probs[0].item() > 0.4 and probs[3].item() > 0.4


def test_gradients_flow_through_circuit():
    sim = StatevectorSimulator(n_qubits=3, reps=2)
    angles = (torch.rand(4, 3) * 2 * math.pi).requires_grad_(True)
    sim(angles).sum().backward()
    assert sim.weights.grad is not None
    assert torch.isfinite(sim.weights.grad).all()
    assert angles.grad is not None


def test_fidelity_properties():
    sim = StatevectorSimulator(n_qubits=3, reps=2)
    angles = torch.rand(6, 3) * 2 * math.pi
    states = sim.prepare_state(angles)
    fid = pairwise_fidelity(states)

    assert fid.shape == (6, 6)
    assert torch.allclose(torch.diagonal(fid), torch.ones(6), atol=1e-5)
    assert torch.allclose(fid, fid.T, atol=1e-5)
    assert torch.all(fid >= -1e-6) and torch.all(fid <= 1.0 + 1e-5)


def test_encoder_angles_in_encoding_domain():
    encoder = QuantumEncoder(input_dim=7, n_qubits=4)
    angles = encoder.encode_angles(torch.rand(10, 7))
    assert torch.all(angles >= 0.0)
    assert torch.all(angles <= 2 * math.pi + 1e-5)


def test_encoding_does_not_collapse_onto_few_qubits():
    """Every qubit must receive a genuinely varying rotation angle.

    Regression test: without angle normalisation the learned projection
    collapses onto one or two qubits, the remaining qubits sit at a
    near-constant angle, all prepared states look alike and the quantum
    similarity term stops carrying information.
    """
    torch.manual_seed(0)
    encoder = QuantumEncoder(input_dim=10, n_qubits=4)
    encoder.train()
    # Low-rank input, the case that provokes the collapse.
    factors = torch.randn(64, 2)
    x = torch.sigmoid(factors @ torch.randn(2, 10))
    angle_std = encoder.encode_angles(x).std(dim=0)
    assert torch.all(angle_std > 0.1), f"angles collapsed: {angle_std}"


def test_distinct_inputs_give_distinguishable_states():
    """Fidelity between different samples must be well below 1."""
    torch.manual_seed(0)
    encoder = QuantumEncoder(input_dim=6, n_qubits=4)
    encoder.train()
    _, states = encoder.encode(torch.rand(32, 6), return_state=True)
    fid = pairwise_fidelity(states)
    off_diagonal = fid[~torch.eye(32, dtype=torch.bool)]
    assert float(off_diagonal.mean()) < 0.9


def test_encoder_forward_shapes():
    encoder = QuantumEncoder(input_dim=7, n_qubits=4)
    recon, latent = encoder(torch.rand(10, 7))
    assert recon.shape == (10, 7)
    assert latent.shape == (10, 4)


def test_encoder_returns_state_when_requested():
    encoder = QuantumEncoder(input_dim=5, n_qubits=3)
    latent, state = encoder.encode(torch.rand(4, 5), return_state=True)
    assert latent.shape == (4, 3)
    assert state.shape == (4, 8)
    assert state.dtype == torch.complex64


def test_invalid_entanglement_rejected():
    with pytest.raises(ValueError):
        StatevectorSimulator(n_qubits=2, entanglement="triangular")
