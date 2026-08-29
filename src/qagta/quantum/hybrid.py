"""Hybrid models: a learned projection feeding a quantum or classical head.

FINDING 11 measured the binding constraint on the quantum arm. At eight
features the quantum kernel and a classical RBF are statistically
indistinguishable (0.622 against 0.616 on ABIDE), while the same classical
kernel given all 2000 selected features reaches 0.649. The gap is the feature
budget, not the model, and the register width caps that budget.

The models here attack that directly. Instead of a t-test choosing which eight
connections the circuit sees, a linear map ``R^d -> R^n_qubits`` is learned
*jointly with the circuit*, end to end. The quantum stage then receives an
optimised eight-dimensional summary of every connection rather than eight raw
ones, without needing more qubits.

The same projection is offered to the classical heads. A learned projection
feeding a classical network is an equally valid architecture, and if the
quantum arm only wins when the classical arm is denied the same mechanism,
that is an artefact of the comparison rather than a property of the model.
Every hybrid here therefore has a classical twin with an identical projection,
identical parameter budget where possible, and identical training.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qagta.quantum.qcnn import QCNN
from qagta.quantum.variational import VQC


class LearnedProjection(nn.Module):
    """Linear map from the full feature vector to circuit input angles.

    The output must be bounded, because angle encoding stops being injective
    once angles wrap. A plain sigmoid does that but also squashes the spread:
    standardised inputs land in the flat tails, and every subject receives a
    similar angle. FINDING 01 measured what that costs — when encoding angles
    lose spread, prepared states become indistinguishable and the quantum stage
    carries no information. Restoring angle spread there was worth 0.62 to 0.97
    AUC.

    So the projection is normalised per dimension before bounding, which keeps
    each qubit's angle using the full domain regardless of how the linear map
    scales its row.

    A bias-free map is used so the projection cannot simply shift every subject
    onto the same angle and let the head do all the work.
    """

    def __init__(self, in_features: int, n_qubits: int, seed: int = 0,
                 warm_start: torch.Tensor | None = None) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, n_qubits, bias=False)
        nn.init.normal_(self.linear.weight, std=1.0 / math.sqrt(in_features))
        self.norm = nn.BatchNorm1d(n_qubits)

        if warm_start is not None:
            # Warm start at the fixed-selection solution: row k reads feature
            # warm_start[k] and little else. Random init leaves the quantum
            # head optimising a 2000-parameter projection through a bounded
            # expectation value, which it does poorly — measured at 0.528
            # accuracy against 0.610 for the same circuit on fixed features.
            # Starting from the known-good selection means the learned
            # projection can only depart from it if that helps.
            with torch.no_grad():
                self.linear.weight.mul_(0.01)
                for k, feature in enumerate(warm_start[:n_qubits]):
                    self.linear.weight[k, int(feature)] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise, then map to [0, pi] with tanh. tanh is used rather than a
        # sigmoid on the raw projection because the normalisation has already
        # centred and scaled each dimension, so the input sits in tanh's
        # responsive region instead of its tails.
        # FINDING 22: the circuit encodes RZ(2x), so the projection must land in
        # [0, pi/2] rather than [0, pi] -- the latter spans a full phase period
        # and maps the two ends of the range onto the same state.
        return (torch.tanh(self.norm(self.linear(x))) + 1.0) * (math.pi / 4)


class HybridQCNN(nn.Module):
    """Learned projection into the Qiskit-reference QCNN."""

    def __init__(self, in_features: int, n_qubits: int = 8, seed: int = 0,
                 warm_start=None) -> None:
        super().__init__()
        self.projection = LearnedProjection(in_features, n_qubits, seed=seed,
                                            warm_start=warm_start)
        self.qcnn = QCNN(n_qubits=n_qubits, seed=seed)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.scale * self.qcnn(self.projection(x)) + self.bias
        return torch.stack([-logit, logit], dim=1)


class HybridVQC(nn.Module):
    """Learned projection into the variational classifier."""

    def __init__(self, in_features: int, n_qubits: int = 8, reps: int = 2,
                 seed: int = 0, warm_start=None) -> None:
        super().__init__()
        self.projection = LearnedProjection(in_features, n_qubits, seed=seed,
                                            warm_start=warm_start)
        self.vqc = VQC(n_qubits=n_qubits, reps=reps, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vqc(self.projection(x))


class HybridClassical(nn.Module):
    """The matched classical twin: same projection, classical head.

    ``head`` selects what sits after the projection:

    ``mlp``     two-layer network on the projected vector
    ``cnn``     1-D convolutional network, the counterpart to QCNN
    ``linear``  a single linear layer, the minimal control — it establishes how
                much of any gain comes from the projection alone rather than
                from whatever follows it

    The bottleneck width matches the quantum register, so both arms compress
    the input to the same dimensionality before their head sees it.
    """

    def __init__(self, in_features: int, n_qubits: int = 8, head: str = "mlp",
                 hidden: int = 32, seed: int = 0, warm_start=None) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.projection = LearnedProjection(in_features, n_qubits, seed=seed,
                                            warm_start=warm_start)
        self.head_kind = head

        if head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(n_qubits, hidden), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(hidden, 2),
            )
        elif head == "cnn":
            channels = 8
            self.head = nn.Sequential(
                nn.Conv1d(1, channels, 3, padding=1), nn.ReLU(),
                nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, 2),
            )
        elif head == "linear":
            self.head = nn.Linear(n_qubits, 2)
        else:
            raise ValueError(f"unknown head {head!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        if self.head_kind == "cnn":
            return self.head(projected.unsqueeze(1))
        return self.head(projected)
