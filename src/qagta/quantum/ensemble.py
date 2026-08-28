"""Plan B: ensembles of narrow quantum models over disjoint feature blocks.

The register width caps how many features one circuit can see, and FINDING 01
shows it cannot simply be widened — fidelity collapses, and a 16-qubit circuit
over a thousand subjects is impractical on this hardware in any case.

An ensemble sidesteps the cap rather than lifting it. The top ``k * n_qubits``
features are split into ``k`` disjoint blocks, one narrow model is trained per
block, and their outputs are combined. The register stays at ``n_qubits`` while
the ensemble as a whole sees ``k`` times as many features.

Two properties make this a fairer test than it first appears:

- Blocks are disjoint, so no feature is counted twice and the ensemble's
  advantage cannot come from re-weighting the same information.
- The identical construction is available classically. A classical ensemble
  over the same blocks is the comparator; a single classical model given all
  ``k * n_qubits`` features at once is the reference ceiling, since it sees the
  same information without the block partition.

If the quantum ensemble beats the classical ensemble, that is about the model.
If both beat their single-model versions, that is about the ensemble, and says
nothing about quantum.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def make_blocks(ranked_features: np.ndarray, n_blocks: int, block_size: int) -> list[np.ndarray]:
    """Split ranked features into disjoint blocks, strongest first.

    Features arrive ordered by discriminative strength, so block 0 receives the
    strongest ``block_size`` features, block 1 the next, and so on. Interleaving
    them instead would give every block similar strength but would also mix
    weak features into the first model, which is the one most likely to carry
    the ensemble.
    """
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        block = ranked_features[start : start + block_size]
        if len(block) == block_size:
            blocks.append(block)
    return blocks


class BlockEnsemble(nn.Module):
    """Ensemble of per-block models with a learned combination weight.

    Members are trained jointly on the shared classification loss rather than
    independently, so the combiner can down-weight a block that carries little
    signal instead of averaging it in regardless.
    """

    def __init__(self, members: list[nn.Module], learn_weights: bool = True) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.logit_weights = nn.Parameter(
            torch.zeros(len(members)), requires_grad=learn_weights
        )

    def forward(self, blocks: list[torch.Tensor]) -> torch.Tensor:
        if len(blocks) != len(self.members):
            raise ValueError(f"expected {len(self.members)} blocks, got {len(blocks)}")
        weights = F.softmax(self.logit_weights, dim=0)
        stacked = torch.stack(
            [member(block) for member, block in zip(self.members, blocks, strict=True)]
        )
        return (weights.view(-1, 1, 1) * stacked).sum(dim=0)

    @property
    def member_weights(self) -> np.ndarray:
        """Learned contribution of each block, for inspection."""
        with torch.no_grad():
            return F.softmax(self.logit_weights, dim=0).cpu().numpy()
