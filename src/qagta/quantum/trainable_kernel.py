"""Trainable quantum embedding kernels, and a matched classical comparator.

The kernels used so far are *non-parametric*: a fixed feature map with a single
bandwidth. FINDING 12 showed why that ties a classical RBF — with one width
parameter, both sweep the same family of Gram matrices, so tuning them to the
same effective dimension gives the same performance.

A trainable quantum embedding kernel (Hubregtsen et al., Phys. Rev. A
106:042431, 2022) removes that restriction. The embedding carries free
parameters optimised against kernel-target alignment,

    KTA(K, y) = <K_c, y y^T>_F / (||K_c||_F ||y y^T||_F)

so the geometry of the feature space adapts to the task instead of being fixed
in advance. This is the sharpest available answer to "why quantum rather than
classical": a fixed classical kernel has one degree of freedom in its width,
while a trainable embedding has many, and they act on the *shape* of the
similarity rather than only its scale.

That argument only holds if the classical comparator is allowed the same
freedom. :class:`TrainableClassicalKernel` therefore learns a linear metric
under the identical objective and optimiser — an RBF kernel on a learned
projection, which is standard metric learning. Comparing a *trained* quantum
kernel against an *untrained* classical one would attribute the training to
the quantum part.

Data re-uploading (Perez-Salinas et al., Quantum 4:226, 2020) is available on
the quantum side: interleaving data and trainable layers raises the order of
Fourier terms the embedding can express. It also degrades trainability, so
depth is a parameter rather than a default.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from qagta.quantum.simulator import _apply_ry, _apply_rz, _cnot_permutation


def kernel_target_alignment(kernel: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Centred kernel-target alignment, differentiable.

    Centring matters: an uncentred kernel with a large constant offset scores
    well regardless of how it orders the data, so an uncentred objective can be
    maximised without improving separation at all.
    """
    y = torch.where(labels > 0, 1.0, -1.0).to(kernel.dtype).reshape(-1, 1)
    ideal = y @ y.T

    n = kernel.shape[0]
    centring = torch.eye(n, dtype=kernel.dtype, device=kernel.device) - 1.0 / n
    centred = centring @ kernel @ centring

    numerator = (centred * ideal).sum()
    denominator = centred.norm() * ideal.norm()
    return numerator / denominator.clamp_min(1e-12)


class TrainableQuantumKernel(nn.Module):
    """Quantum embedding kernel with trainable rotation and entangling weights.

    Structure per layer:

        RY(pi/2)                        superposition
        RZ(2 * w_k * x_k + b_k)         data-dependent, trainable scale and shift
        CNOT / RZ(v_kj) / CNOT          trainable two-qubit interaction

    ``reuploading`` repeats the data-dependent layer with independent
    parameters, which is the re-uploading construction: each repetition raises
    the accessible Fourier order of the embedding.

    The per-feature weights ``w`` generalise a single global bandwidth — the
    non-parametric kernel is the special case where every ``w`` is tied and no
    other parameter is free.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        layers: int = 2,
        entanglement: str = "linear",
        init_bandwidth: float = 0.15,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.layers = layers
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

        # Initialised at the bandwidth FINDING 12 identified as the usable
        # window, so training starts inside the useful regime rather than in
        # the near-identity region where gradients are uninformative.
        self.weight = nn.Parameter(torch.full((layers, n_qubits), float(init_bandwidth)))
        self.bias = nn.Parameter(torch.zeros(layers, n_qubits))
        self.coupling = nn.Parameter(0.1 * torch.randn(layers, len(pairs)))

    def _cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        return state.index_select(1, getattr(self, f"_cx_{control}_{target}"))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare |psi(x)> for a batch of inputs, shape ``(batch, 2**n)``."""
        if x.dim() != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"expected (batch, {self.n_qubits}), got {tuple(x.shape)}")

        batch = x.shape[0]
        state = torch.zeros((batch, self.dim), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0

        for layer in range(self.layers):
            for k in range(self.n_qubits):
                state = _apply_ry(
                    state, self.n_qubits, k,
                    torch.full((batch,), math.pi / 2, device=x.device),
                )
            for k in range(self.n_qubits):
                angle = 2.0 * (self.weight[layer, k] * x[:, k] + self.bias[layer, k])
                state = _apply_rz(state, self.n_qubits, k, angle)
            for p, (control, target) in enumerate(self.pairs):
                state = self._cnot(state, control, target)
                interaction = self.coupling[layer, p] * (
                    (math.pi - x[:, control]) * (math.pi - x[:, target])
                )
                state = _apply_rz(state, self.n_qubits, target, interaction)
                state = self._cnot(state, control, target)
        return state

    def kernel(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        """Fidelity kernel between two batches; differentiable end to end."""
        left = self.embed(x_left)
        right = self.embed(x_right)
        return (left.conj() @ right.T).abs() ** 2


class TrainableClassicalKernel(nn.Module):
    """RBF kernel on a learned linear metric — the matched classical comparator.

    Learns ``K_ij = exp(-||A x_i - A x_j||^2)`` with ``A`` free, which is
    standard metric learning. It is given the same objective, optimiser and
    step budget as the quantum kernel, so the comparison isolates the embedding
    rather than rewarding whichever side was allowed to train.
    """

    def __init__(self, n_features: int, rank: int | None = None, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        rank = rank or n_features
        self.projection = nn.Parameter(
            torch.eye(rank, n_features) + 0.01 * torch.randn(rank, n_features)
        )
        self.log_scale = nn.Parameter(torch.zeros(1))

    def kernel(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        left = x_left @ self.projection.T
        right = x_right @ self.projection.T
        squared = torch.cdist(left, right, p=2) ** 2
        return torch.exp(-self.log_scale.exp() * squared)


def train_kernel(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    steps: int = 60,
    lr: float = 0.05,
    batch_size: int = 256,
    seed: int = 0,
    verbose: bool = False,
) -> list[float]:
    """Maximise kernel-target alignment on the training set.

    Alignment is computed on random subsets rather than the full Gram matrix:
    the matrix is quadratic in sample count, and subsampling keeps each step
    affordable while remaining an unbiased view of the objective.
    """
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(x_train), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    history = []
    for step in range(steps):
        idx = rng.choice(len(x), size=min(batch_size, len(x)), replace=False)
        sub_x, sub_y = x[idx], y[idx]

        optimizer.zero_grad()
        kernel = model.kernel(sub_x, sub_x)
        loss = -kernel_target_alignment(kernel, sub_y)
        loss.backward()
        optimizer.step()

        history.append(-float(loss.detach()))
        if verbose and (step + 1) % 20 == 0:
            print(f"    step {step + 1}/{steps}  alignment={history[-1]:.4f}", flush=True)
    return history


@torch.no_grad()
def gram_matrix(model: nn.Module, x_left: np.ndarray, x_right: np.ndarray,
                block: int = 512) -> np.ndarray:
    """Kernel matrix between two sets, evaluated in blocks."""
    left = torch.as_tensor(np.asarray(x_left), dtype=torch.float32)
    right = torch.as_tensor(np.asarray(x_right), dtype=torch.float32)
    out = np.empty((len(left), len(right)), dtype=np.float32)
    for start in range(0, len(left), block):
        chunk = left[start : start + block]
        out[start : start + block] = model.kernel(chunk, right).cpu().numpy()
    return out
