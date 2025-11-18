"""Parameter-shift rule for gradients of the variational circuit weights.

For a circuit whose parameterised gates are Pauli rotations, the exact
partial derivative of any expectation value f(theta) with respect to a
rotation parameter is

    df/dtheta_j = ( f(theta + pi/2 * e_j) - f(theta - pi/2 * e_j) ) / 2

This module provides that rule for the native simulator, both as a
Jacobian utility and as a hook that lets a downstream (classical) loss
drive updates of the quantum weights without relying on autograd through
the circuit — the update path used when the circuit runs on hardware or a
shot-based backend. On the exact simulator it matches autograd gradients,
which is verified in the test suite.
"""

from __future__ import annotations

import math

import torch

from qagta.quantum.simulator import StatevectorSimulator


@torch.no_grad()
def expectation_jacobian(
    circuit: StatevectorSimulator,
    angles: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jacobian d<Z_k>/d theta_j via the parameter-shift rule.

    Returns a tensor of shape ``(batch, n_qubits, num_weights)``.
    """
    if weights is None:
        weights = circuit.weights.detach()
    batch = angles.shape[0]
    jac = torch.zeros(batch, circuit.n_qubits, weights.numel(), device=angles.device)
    shift = math.pi / 2
    for j in range(weights.numel()):
        shifted = weights.clone()
        shifted[j] += shift
        plus = circuit(angles, weights=shifted)
        shifted[j] -= 2 * shift
        minus = circuit(angles, weights=shifted)
        jac[:, :, j] = (plus - minus) / 2
    return jac


def quantum_weight_gradient(
    circuit: StatevectorSimulator,
    angles: torch.Tensor,
    grad_latent: torch.Tensor,
) -> torch.Tensor:
    """Chain-rule combination dL/dtheta = sum_bk dL/dz_bk * dz_bk/dtheta.

    Parameters
    ----------
    circuit:
        The variational circuit whose weights are being optimised.
    angles:
        Encoding angles used in the forward pass, shape ``(batch, n_qubits)``.
    grad_latent:
        Gradient of the training loss with respect to the latent
        embeddings, shape ``(batch, n_qubits)`` (obtained via autograd on
        the classical stages).
    """
    jac = expectation_jacobian(circuit, angles)
    return torch.einsum("bk,bkj->j", grad_latent, jac)


def apply_parameter_shift_step(
    circuit: StatevectorSimulator,
    angles: torch.Tensor,
    grad_latent: torch.Tensor,
    lr: float,
) -> None:
    """In-place SGD step on the circuit weights using parameter-shift gradients."""
    grad = quantum_weight_gradient(circuit, angles, grad_latent)
    with torch.no_grad():
        circuit.weights -= lr * grad
