"""Quantum Graph Transformer: attention scores computed by a joint circuit.

Every quantum construct evaluated earlier in this study scores a pair of
regions by the overlap of two *separately prepared* states. FINDING 01 measured
what that costs: as the register widens the overlap concentrates and the score
stops discriminating. FINDING 19 added a second limit specific to that form ---
anything applied identically to both states cancels, so a variational ansatz
placed after the encoding cannot influence the score at all.

A graph transformer avoids both by construction. Query and key are encoded into
*one* register and allowed to interact through entangling gates before
measurement, so the attention logit is

    a_ij = <psi(x_i, x_j)| Z_0 |psi(x_i, x_j)>

rather than |<psi(x_i)|psi(x_j)>|^2. Because the two nodes share a circuit, the
trainable layers sit between data-dependent operations on both, and neither
cancellation nor pairwise concentration applies. This is the mechanism behind
the quantum graph transformer and quantum self-attention literature, in the
form closest to the classical attention it replaces.

Design choices, and their cost:

- ``node_qubits`` qubits per node, so a pair occupies ``2 * node_qubits``. The
  joint register is what makes the interaction possible and is also the
  dominant cost: a pair circuit is exponential in ``2 * node_qubits``.
- Attention is computed over a k-nearest-neighbour graph rather than all-to-all.
  All-to-all on 200 regions is 39,800 ordered pairs per subject; at k=10 it is
  2,000, and the sparsification is the same one the classical baseline uses, so
  the comparison stays matched.
- Aggregation is classical. The quantum component is the attention mechanism
  itself, which isolates what is being tested: replacing the classical
  attention of a GAT with a quantum one, everything else held fixed.

Matched classical comparator: :class:`ClassicalGraphTransformer` below, an
identical architecture whose attention logits come from a small MLP on the same
concatenated node pair, with a comparable parameter count.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qagta.quantum.simulator import _apply_ry, _apply_rz, _cnot_permutation


class QuantumAttention(nn.Module):
    """Attention logits from a joint two-node circuit.

    The circuit is, for a pair ``(i, j)`` with features ``u`` and ``v``:

        RY(w_q * u) on qubits 0..n-1,  RY(w_k * v) on qubits n..2n-1
        Hadamard on every qubit
        CZ ring across the full 2n register        <- couples query and key
        [ RZ(theta) ; RY(phi) ] per qubit          <- trainable, interleaved
        second data layer RY(w2_q * u), RY(w2_k * v)
        CZ ring
        RY(pi/2) on every qubit                    <- basis change
        read out <Z_0 Z_n>                         <- query-key correlator

    Three details are load-bearing, and the first two were found by measuring
    gradients rather than by inspection:

    - The second data layer uses RY, not RZ. RZ, CZ and the Z observable are
      all diagonal in the computational basis, so a diagonal layer placed
      immediately before a Z measurement is exactly inert -- its parameters
      measured gradients of order 1e-10.
    - A basis-changing rotation follows the final CZ, so the phases that CZ
      writes become observable amplitudes rather than unmeasured phase.
    - The observable is the correlator Z_0 Z_n, one qubit from each node,
      rather than Z_0 alone. Reading a single query qubit left the key
      influencing the score only through the ring, giving a logit spread of
      0.006 across different keys -- an attention mechanism that barely
      attends.

    The second data layer also keeps the trainable block from cancelling:
    without it the post-encoding stack would sit on one side of the
    measurement and reduce to a fixed rotation (FINDING 19).
    """

    def __init__(self, node_qubits: int = 3, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.node_qubits = node_qubits
        self.n = 2 * node_qubits
        self.dim = 2**self.n

        self.w_q = nn.Parameter(torch.ones(node_qubits))
        self.w_k = nn.Parameter(torch.ones(node_qubits))
        self.w2_q = nn.Parameter(0.5 * torch.ones(node_qubits))
        self.w2_k = nn.Parameter(0.5 * torch.ones(node_qubits))
        self.theta = nn.Parameter(0.1 * torch.randn(self.n))
        self.phi = nn.Parameter(0.1 * torch.randn(self.n))

        # CZ is diagonal: a sign flip on basis states where both qubits are 1.
        idx = torch.arange(self.dim)
        signs = []
        for a in range(self.n):
            b = (a + 1) % self.n
            bit_a = (idx >> (self.n - a - 1)) & 1
            bit_b = (idx >> (self.n - b - 1)) & 1
            signs.append(1.0 - 2.0 * (bit_a & bit_b).float())
        self.register_buffer("_cz", torch.stack(signs).prod(dim=0))

        # Correlator Z_0 Z_n: one qubit from the query half, one from the key
        # half, so the score cannot be insensitive to either node.
        z_q = 1.0 - 2.0 * ((idx >> (self.n - 1)) & 1).float()
        z_k = 1.0 - 2.0 * ((idx >> (self.n - node_qubits - 1)) & 1).float()
        self.register_buffer("_z0", z_q * z_k)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Attention logits for ``(pairs, node_qubits)`` query/key features."""
        pairs = u.shape[0]
        nq = self.node_qubits
        state = torch.zeros((pairs, self.dim), dtype=torch.complex64, device=u.device)
        state[:, 0] = 1.0

        for q in range(nq):
            state = _apply_ry(state, self.n, q, self.w_q[q] * u[:, q])
            state = _apply_ry(state, self.n, nq + q, self.w_k[q] * v[:, q])

        half_pi = torch.full((pairs,), math.pi / 2, device=u.device)
        pi = torch.full((pairs,), math.pi, device=u.device)
        for q in range(self.n):  # Hadamard = RZ(pi) then RY(pi/2)
            state = _apply_rz(state, self.n, q, pi)
            state = _apply_ry(state, self.n, q, half_pi)

        state = state * self._cz
        for q in range(self.n):
            state = _apply_rz(state, self.n, q, self.theta[q].expand(pairs))
            state = _apply_ry(state, self.n, q, self.phi[q].expand(pairs))

        for q in range(nq):
            state = _apply_ry(state, self.n, q, self.w2_q[q] * u[:, q])
            state = _apply_ry(state, self.n, nq + q, self.w2_k[q] * v[:, q])
        state = state * self._cz
        for q in range(self.n):
            state = _apply_ry(state, self.n, q, half_pi)

        probs = state.real**2 + state.imag**2
        return probs @ self._z0


class ClassicalAttention(nn.Module):
    """Matched classical comparator: an MLP on the same concatenated pair."""

    def __init__(self, node_qubits: int = 3, hidden: int = 8, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(2 * node_qubits, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([u, v], dim=1)).squeeze(-1))


class GraphTransformer(nn.Module):
    """Graph transformer whose attention module is swappable.

    ``attention='quantum'`` uses :class:`QuantumAttention`; ``'classical'`` uses
    :class:`ClassicalAttention`. Everything else --- projection, aggregation,
    read-out, classifier --- is shared, so a difference between the two arms is
    attributable to the attention mechanism and nothing else.
    """

    def __init__(self, in_features: int, node_qubits: int = 3,
                 attention: str = "quantum", hidden: int = 32, seed: int = 0,
                 n_roi: int | None = None) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.node_qubits = node_qubits
        # Learned per-region embedding. Without it the network is permutation
        # symmetric over regions, so it cannot express "this particular region
        # carries the signal" -- on a synthetic task whose label depends on one
        # region, both arms scored chance until this was added. A fixed atlas
        # gives regions a stable identity, so encoding that identity is
        # legitimate here in a way it would not be for an unordered graph.
        self.pos = (nn.Parameter(0.01 * torch.randn(n_roi, node_qubits))
                    if n_roi else None)
        self.project = nn.Linear(in_features, node_qubits)
        self.norm = nn.BatchNorm1d(node_qubits)
        self.attention_kind = attention
        self.attend = (QuantumAttention(node_qubits, seed=seed)
                       if attention == "quantum"
                       else ClassicalAttention(node_qubits, seed=seed))
        self.value = nn.Linear(node_qubits, node_qubits)
        self.head = nn.Sequential(
            nn.Linear(3 * node_qubits, hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """``x``: (batch, n_roi, in_features);  ``edge_index``: (2, n_edges).

        Three details matter for whether anything is learnable, and all three
        were established by testing the model on synthetic graphs whose label
        depends on a single region. The original form scored chance on that
        task while learning a global-mean task fine, which located the fault in
        the read-out rather than in the attention.

        - **Self-loops.** Without an edge from a node to itself, a node's own
          features reach its representation only after being mixed with its
          neighbours', so a signal confined to one region is diluted before it
          is ever read.
        - **Residual.** The aggregate is added to the node's own projection
          rather than replacing it, so message passing refines the node
          representation instead of overwriting it.
        - **A read-out that is not a single mean.** Averaging 200 regions into
          ``node_qubits`` numbers divides a localised signal by the number of
          regions. Concatenating mean, max and standard deviation keeps the
          extremes, which is where a few discriminative regions show up.

        Both arms receive all three, so the comparison still isolates the
        attention module.
        """
        batch, n_roi, _ = x.shape
        h = self.project(x.reshape(batch * n_roi, -1))
        h = self.norm(h).reshape(batch, n_roi, self.node_qubits)
        if self.pos is not None:
            h = h + self.pos.unsqueeze(0)
        h = torch.tanh(h) * (math.pi / 2)

        src, dst = edge_index[0], edge_index[1]
        u = h[:, src, :].reshape(-1, self.node_qubits)
        v = h[:, dst, :].reshape(-1, self.node_qubits)
        logits = self.attend(u, v).reshape(batch, len(src))

        # Softmax over the incoming edges of each destination node, computed
        # by scatter rather than a Python loop over nodes.
        m = logits.max(dim=1, keepdim=True).values
        e = torch.exp(logits - m)
        denom = torch.zeros(batch, n_roi, device=x.device).index_add_(1, dst, e)
        weights = e / (denom[:, dst] + 1e-9)

        messages = self.value(h)[:, src, :] * weights.unsqueeze(-1)
        aggregated = torch.zeros_like(h).index_add_(1, dst, messages)

        # Residual: refine the node's own representation rather than replace it.
        z = aggregated + h

        # Read-out preserving the extremes, not only the average.
        pooled = torch.cat([z.mean(dim=1), z.max(dim=1).values, z.std(dim=1)], dim=1)
        return self.head(pooled)
