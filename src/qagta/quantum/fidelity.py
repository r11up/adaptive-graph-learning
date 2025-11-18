"""Quantum-native similarity metrics between latent states."""

from __future__ import annotations

import torch


def pairwise_fidelity(states: torch.Tensor) -> torch.Tensor:
    """Pairwise fidelity ``F_ij = |<psi_i|psi_j>|^2`` between pure states.

    Parameters
    ----------
    states:
        Complex statevectors of shape ``(batch, dim)``. They are assumed to
        be normalised (the simulator produces normalised states).

    Returns
    -------
    torch.Tensor
        Real matrix of shape ``(batch, batch)`` with entries in ``[0, 1]``,
        ``F_ii = 1`` on the diagonal. Differentiable.
    """
    if states.dim() != 2:
        raise ValueError("states must have shape (batch, dim)")
    overlaps = states.conj() @ states.T
    return overlaps.abs() ** 2


def fidelity(state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
    """Fidelity between two batches of pure states, element-wise."""
    overlap = (state_a.conj() * state_b).sum(dim=-1)
    return overlap.abs() ** 2
