"""The parameter-shift rule must reproduce exact analytic gradients."""

from __future__ import annotations

import math

import torch

from qagta.quantum.simulator import StatevectorSimulator
from qagta.training.parameter_shift import (
    apply_parameter_shift_step,
    expectation_jacobian,
    quantum_weight_gradient,
)


def test_parameter_shift_matches_autograd():
    torch.manual_seed(0)
    sim = StatevectorSimulator(n_qubits=3, reps=2)
    angles = torch.rand(4, 3) * 2 * math.pi

    # Autograd reference on sum_k <Z_k>: dL/dtheta_j = sum_bk dz_bk/dtheta_j.
    sim.zero_grad()
    sim(angles).sum().backward()
    autograd_grad = sim.weights.grad.clone()

    jac = expectation_jacobian(sim, angles)
    shift_grad = jac.sum(dim=(0, 1))

    assert torch.allclose(autograd_grad, shift_grad, atol=1e-4)


def test_chain_rule_gradient_matches_autograd():
    torch.manual_seed(1)
    sim = StatevectorSimulator(n_qubits=2, reps=1)
    angles = torch.rand(3, 2) * 2 * math.pi
    weight = torch.rand(3, 2)

    sim.zero_grad()
    (sim(angles) * weight).sum().backward()
    reference = sim.weights.grad.clone()

    chained = quantum_weight_gradient(sim, angles, grad_latent=weight)
    assert torch.allclose(reference, chained, atol=1e-4)


def test_jacobian_shape():
    sim = StatevectorSimulator(n_qubits=3, reps=2)
    jac = expectation_jacobian(sim, torch.rand(5, 3))
    assert jac.shape == (5, 3, sim.num_weights)


def test_parameter_shift_step_reduces_loss():
    """A descent step on <Z_0> should decrease it."""
    torch.manual_seed(2)
    sim = StatevectorSimulator(n_qubits=2, reps=1)
    angles = torch.rand(6, 2) * 2 * math.pi

    def objective() -> float:
        with torch.no_grad():
            return float(sim(angles)[:, 0].sum())

    before = objective()
    grad_latent = torch.zeros(6, 2)
    grad_latent[:, 0] = 1.0  # dL/dz for L = sum_b z_b0
    apply_parameter_shift_step(sim, angles, grad_latent, lr=0.1)
    assert objective() < before
