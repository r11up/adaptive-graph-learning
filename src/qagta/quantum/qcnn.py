"""Quantum Convolutional Neural Network, following the Qiskit reference design.

Implements the QCNN of Cong, Choi and Lukin (Nat. Phys. 15:1273, 2019) in the
form given by the Qiskit Machine Learning tutorial: a ZFeatureMap encoding
followed by alternating convolution and pooling layers, read out as a Pauli-Z
expectation on the final surviving qubit.

The circuits are gate-for-gate identical to the reference:

    conv_circuit(p)   RZ(-pi/2) on q1; CX(1,0); RZ(p0) on q0; RY(p1) on q1;
                      CX(0,1); RY(p2) on q1; CX(1,0); RZ(pi/2) on q0
    pool_circuit(p)   as above without the trailing CX and RZ

    conv_layer        conv_circuit on even pairs, then on odd pairs with
                      wrap-around, so each layer mixes all neighbours
    pool_layer        conv-style unitary from each source onto its sink,
                      halving the active register

Why re-implement rather than call Qiskit directly: Qiskit's EstimatorQNN
evaluates one sample per circuit execution and trains with a gradient-free
optimiser, which is fine for the tutorial's 4-qubit toy problem but not for
2400 subjects across 25 folds. This version evolves the whole batch as one
tensor and is differentiable, so it trains by gradient descent in seconds
rather than hours.

Equivalence to the Qiskit reference is asserted numerically in
``tests/test_qcnn.py`` rather than assumed.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qagta.quantum.simulator import _apply_ry, _apply_rz, _cnot_permutation


def _cx(state: torch.Tensor, perms: dict, control: int, target: int) -> torch.Tensor:
    return state.index_select(1, perms[(control, target)])


class QCNN(nn.Module):
    """Batched, differentiable QCNN classifier.

    Parameters mirror the reference: three parameters per two-qubit block,
    ``num_qubits * 3`` per convolution layer and ``num_qubits // 2 * 3`` per
    pooling layer. The register halves at each pooling stage, so an 8-qubit
    model runs conv(8) -> pool(8->4) -> conv(4) -> pool(4->2) -> conv(2) ->
    pool(2->1) and reads out qubit 7.
    """

    def __init__(self, n_qubits: int = 8, feature_reps: int = 2, seed: int = 0,
                 reupload: int = 0) -> None:
        super().__init__()
        if n_qubits & (n_qubits - 1) or n_qubits < 2:
            raise ValueError("n_qubits must be a power of two and at least 2")
        torch.manual_seed(seed)

        self.n_qubits = n_qubits
        self.feature_reps = feature_reps
        self.dim = 2**n_qubits
        # Data re-uploading (Perez-Salinas et al., Quantum 4:226, 2020):
        # re-inject the features between convolution stages, each time with its
        # own trainable scale. Each re-upload raises the order of Fourier terms
        # the circuit can express in the data, which is capacity a single
        # encoding at the input cannot reach. It also deepens the circuit, so
        # trainability degrades and depth is a parameter rather than a default.
        self.reupload = reupload

        # Qiskit numbers qubit 0 as the least significant bit; this simulator
        # numbers qubit 0 as the most significant. Every index below is written
        # in Qiskit's convention and mapped through _q, so the circuits can be
        # read directly against the reference implementation.
        perms = {}
        for a in range(n_qubits):
            for b in range(n_qubits):
                if a != b:
                    perms[(a, b)] = _cnot_permutation(
                        n_qubits, self._q(a, n_qubits), self._q(b, n_qubits),
                        torch.device("cpu"),
                    )
                    self.register_buffer(f"_cx_{a}_{b}", perms[(a, b)])
        self._pairs = list(perms)

        # Layer schedule: active qubits halve after each pooling stage.
        self.schedule = []
        active = list(range(n_qubits))
        while len(active) > 1:
            half = len(active) // 2
            sources, sinks = active[:half], active[half:]
            self.schedule.append((list(active), sources, sinks))
            active = sinks

        params = []
        for active, sources, _ in self.schedule:
            params.append(nn.Parameter(0.1 * torch.randn(len(active) * 3)))  # conv
            params.append(nn.Parameter(0.1 * torch.randn(len(sources) * 3)))  # pool
        self.weights = nn.ParameterList(params)

        # One trainable gain per re-upload, per qubit. Initialised small so a
        # re-uploading model starts close to the plain QCNN and can only depart
        # from it if that helps.
        if reupload > 0:
            self.reupload_scale = nn.Parameter(0.1 * torch.ones(reupload, n_qubits))

        # Reference observable is Z on the highest-numbered Qiskit qubit.
        indices = torch.arange(self.dim)
        internal = self._q(n_qubits - 1, n_qubits)
        sign = 1.0 - 2.0 * ((indices >> (n_qubits - internal - 1)) & 1).float()
        self.register_buffer("_z_sign", sign)

    @staticmethod
    def _q(index: int, n_qubits: int) -> int:
        """Map a Qiskit qubit index to this simulator's internal index."""
        return n_qubits - 1 - index

    def _perms(self) -> dict:
        return {(a, b): getattr(self, f"_cx_{a}_{b}") for a, b in self._pairs}

    def _conv_block(self, state, perms, q0, q1, params):
        """The reference two-qubit convolution unitary."""
        n = self.n_qubits
        batch = state.shape[0]
        i0, i1 = self._q(q0, n), self._q(q1, n)
        full = torch.full((batch,), -math.pi / 2, device=state.device)
        state = _apply_rz(state, n, i1, full)
        state = _cx(state, perms, q1, q0)
        state = _apply_rz(state, n, i0, params[0].expand(batch))
        state = _apply_ry(state, n, i1, params[1].expand(batch))
        state = _cx(state, perms, q0, q1)
        state = _apply_ry(state, n, i1, params[2].expand(batch))
        state = _cx(state, perms, q1, q0)
        return _apply_rz(state, n, i0, torch.full((batch,), math.pi / 2, device=state.device))

    def _pool_block(self, state, perms, source, sink, params):
        """Convolution unitary truncated after the second RY, as in the reference."""
        n = self.n_qubits
        batch = state.shape[0]
        src, snk = self._q(source, n), self._q(sink, n)
        state = _apply_rz(state, n, snk, torch.full((batch,), -math.pi / 2, device=state.device))
        state = _cx(state, perms, sink, source)
        state = _apply_rz(state, n, src, params[0].expand(batch))
        state = _apply_ry(state, n, snk, params[1].expand(batch))
        state = _cx(state, perms, source, sink)
        return _apply_ry(state, n, snk, params[2].expand(batch))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ZFeatureMap: Hadamard then phase 2*x on each qubit, repeated."""
        batch = x.shape[0]
        state = torch.zeros((batch, self.dim), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1.0
        half_pi = torch.full((batch,), math.pi / 2, device=x.device)
        pi = torch.full((batch,), math.pi, device=x.device)
        for _ in range(self.feature_reps):
            for q in range(self.n_qubits):
                internal = self._q(q, self.n_qubits)
                # H = RZ(pi) then RY(pi/2), up to a global phase that <Z> ignores.
                state = _apply_rz(state, self.n_qubits, internal, pi)
                state = _apply_ry(state, self.n_qubits, internal, half_pi)
                state = _apply_rz(state, self.n_qubits, internal, 2.0 * x[:, q])
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the read-out expectation in ``[-1, 1]``, shape ``(batch,)``."""
        if x.dim() != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(f"expected (batch, {self.n_qubits}), got {tuple(x.shape)}")

        perms = self._perms()
        state = self.feature_map(x)

        for stage, (active, sources, sinks) in enumerate(self.schedule):
            conv_w = self.weights[2 * stage]
            pool_w = self.weights[2 * stage + 1]

            # Re-inject the data before each stage after the first, on the
            # qubits still active at that depth.
            if self.reupload > 0 and 0 < stage <= self.reupload:
                gains = self.reupload_scale[stage - 1]
                for q in active:
                    internal = self._q(q, self.n_qubits)
                    state = _apply_rz(
                        state, self.n_qubits, internal, 2.0 * gains[q] * x[:, q]
                    )

            index = 0
            for q0, q1 in zip(active[0::2], active[1::2], strict=False):
                state = self._conv_block(state, perms, q0, q1, conv_w[index : index + 3])
                index += 3
            if len(active) > 2:
                rotated = active[1::2]
                partners = active[2::2] + [active[0]]
                for q0, q1 in zip(rotated, partners, strict=False):
                    state = self._conv_block(state, perms, q0, q1, conv_w[index : index + 3])
                    index += 3

            index = 0
            for source, sink in zip(sources, sinks, strict=True):
                state = self._pool_block(state, perms, source, sink, pool_w[index : index + 3])
                index += 3

        probs = state.real**2 + state.imag**2
        return probs @ self._z_sign


class QCNNClassifier(nn.Module):
    """QCNN read-out mapped to two class logits by a trainable affine layer.

    The bare expectation is a single number in ``[-1, 1]``; the affine map
    turns it into logits so the model trains with cross-entropy and class
    weighting, matching how the classical baselines are trained.
    """

    def __init__(self, n_qubits: int = 8, seed: int = 0, reupload: int = 0) -> None:
        super().__init__()
        self.qcnn = QCNN(n_qubits=n_qubits, seed=seed, reupload=reupload)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.qcnn(x)
        logit = self.scale * z + self.bias
        return torch.stack([-logit, logit], dim=1)
