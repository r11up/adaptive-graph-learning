"""Tests for the ring-entangled per-region encoder and its two backends."""

from __future__ import annotations

import math

import pytest
import torch

from qagta.quantum.fmri_encoder import RingEntangledEncoder, build_encoder
from qagta.quantum.simulator import _apply_rx, _apply_rz

pennylane = pytest.importorskip("pennylane", reason="PennyLane backend not installed")


def test_state_is_normalised():
    encoder = RingEntangledEncoder(n_qubits=6)
    state = encoder.prepare_state(torch.rand(8, 6))
    assert torch.allclose((state.abs() ** 2).sum(1), torch.ones(8), atol=1e-5)


def test_expectations_physical_and_shaped():
    encoder = RingEntangledEncoder(n_qubits=6)
    z = encoder(torch.rand(5, 6))
    assert z.shape == (5, 6)
    assert torch.all(z <= 1 + 1e-5) and torch.all(z >= -1 - 1e-5)


def test_rz_is_a_phase_and_preserves_z_expectation():
    """RZ is diagonal, so it must not change <Z> or the norm."""
    state = torch.zeros(1, 2, dtype=torch.complex64)
    state[0, 0] = 1 / math.sqrt(2)
    state[0, 1] = 1 / math.sqrt(2)
    rotated = _apply_rz(state, 1, 0, torch.tensor(0.7))
    probs_before = state.abs() ** 2
    probs_after = rotated.abs() ** 2
    assert torch.allclose(probs_before, probs_after, atol=1e-6)


def test_rx_flips_zero_state_at_pi():
    """RX(pi)|0> = -i|1>, so all population moves to |1>."""
    state = torch.zeros(1, 2, dtype=torch.complex64)
    state[0, 0] = 1.0
    rotated = _apply_rx(state, 1, 0, torch.tensor(math.pi))
    assert rotated[0, 1].abs().item() == pytest.approx(1.0, abs=1e-5)
    assert rotated[0, 0].abs().item() == pytest.approx(0.0, abs=1e-5)


def test_ring_entanglement_wraps_around():
    """The ring must include the closing CNOT from the last qubit back to q0."""
    encoder = RingEntangledEncoder(n_qubits=4)
    assert hasattr(encoder, "_ring_3")
    # q3 -> q0 flips bit 0 of the basis index when q3 is set.
    perm = encoder._ring_3
    assert int(perm[0b0001]) == 0b1001


def test_gradients_reach_every_circuit_parameter():
    encoder = RingEntangledEncoder(n_qubits=6)
    encoder(torch.rand(4, 6)).sum().backward()
    for name in ("encode_scale", "phi_z", "phi_x"):
        grad = getattr(encoder, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name


def test_input_dimension_is_validated():
    encoder = RingEntangledEncoder(n_qubits=8)
    with pytest.raises(ValueError):
        encoder(torch.rand(3, 5))


def test_ring_needs_at_least_two_qubits():
    with pytest.raises(ValueError):
        RingEntangledEncoder(n_qubits=1)


def test_torch_and_pennylane_backends_agree():
    """The fast batched simulator must match the PennyLane reference.

    The whole runtime argument for defaulting to the batched backend rests on
    it being numerically equivalent to adjoint differentiation on
    lightning.qubit, so that equivalence is pinned here rather than assumed.
    """
    from qagta.quantum.fmri_encoder import PennyLaneRingEncoder

    torch.manual_seed(0)
    n = 6
    fast = RingEntangledEncoder(n_qubits=n)
    reference = PennyLaneRingEncoder(n_qubits=n)
    reference.load_state_dict(fast.state_dict(), strict=False)

    x = torch.rand(10, n)
    z_fast, state_fast = fast(x, return_state=True)
    z_ref, state_ref = reference(x, return_state=True)

    assert torch.allclose(z_fast, z_ref, atol=1e-5)
    overlap = (state_fast.conj() * state_ref).sum(1).abs()
    assert torch.all(overlap > 1 - 1e-5)


def test_backend_gradients_agree():
    from qagta.quantum.fmri_encoder import PennyLaneRingEncoder

    torch.manual_seed(1)
    n = 5
    fast = RingEntangledEncoder(n_qubits=n)
    reference = PennyLaneRingEncoder(n_qubits=n)
    reference.load_state_dict(fast.state_dict(), strict=False)

    x = torch.rand(6, n)
    fast(x).sum().backward()
    reference(x).sum().backward()
    for name in ("encode_scale", "phi_z", "phi_x"):
        assert torch.allclose(
            getattr(fast, name).grad, getattr(reference, name).grad, atol=1e-4
        ), name


def test_build_encoder_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_encoder("qiskit")
