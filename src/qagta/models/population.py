"""Quantum Population Graph (QPG) — subjects as nodes, quantum fidelity as edges.

This is a different architecture from the region-level pipeline, and the
difference is the point.

Prior approaches in this repository place brain *regions* on the graph:

    RQT (Region-level Quantum Topology)
        nodes = 200 brain regions of one subject
        edges = quantum fidelity between regional states
        read-out = pooling over regions -> one prediction per subject

    SQK (Subject-level Quantum Kernel)
        no graph; quantum fidelity between subjects forms an SVM kernel

RQT fails for a reason that is now measured rather than guessed: per-region
temporal features carry no diagnostic signal (FINDING 06, AUC 0.459 across all
200 regions against 0.693 for pairwise correlations). The discriminative
information in rs-fMRI lives in the correlation *between* regions, and RQT puts
that information into the graph structure, where message passing cannot recover
it, while handing the network node features that lack it.

QPG inverts the assignment, following the population-graph paradigm that
dominates classical ABIDE results (Parisot et al., MedIA 48:117, 2018):

    nodes    = subjects
    features = that subject's full connectivity profile — the representation
               that demonstrably carries signal
    edges    = quantum fidelity between subjects, optionally gated by
               phenotypic agreement
    task     = transductive node classification over the population

The quantum contribution moves from the node features to the edges, which is
where a fidelity kernel is a natural fit and where it stays in the narrow
register that FINDING 01 shows it needs. Node features are unconstrained by
qubit count, so the compression penalty that dominated FINDING 08 no longer
applies to them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def phenotypic_affinity(
    sex: np.ndarray | None,
    age: np.ndarray | None,
    site: np.ndarray | None = None,
    age_tolerance: float = 2.0,
    use_site: bool = False,
) -> np.ndarray:
    """Pairwise phenotypic agreement, as in the population-graph literature.

    Each available attribute contributes 1 when two subjects agree and 0 when
    they do not; age agrees when the gap is within ``age_tolerance`` years. The
    result is the count of agreeing attributes, used to gate imaging similarity
    so that edges connect comparable subjects.

    ``use_site`` is off by default. Site agreement is a strong signal and is
    standard in the original formulation, but under Leave-Site-Out the held-out
    site is unseen during training, so a site term would encourage the model to
    rely on exactly the structure the protocol withholds.
    """
    parts = []
    if sex is not None:
        parts.append((sex[:, None] == sex[None, :]).astype(float))
    if age is not None:
        parts.append((np.abs(age[:, None] - age[None, :]) <= age_tolerance).astype(float))
    if use_site and site is not None:
        parts.append((site[:, None] == site[None, :]).astype(float))

    if not parts:
        return np.ones((len(sex or age), len(sex or age)), dtype=float)
    return np.sum(parts, axis=0)


def build_population_graph(
    similarity: np.ndarray,
    affinity: np.ndarray | None = None,
    k_neighbors: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparsify a subject-by-subject similarity into a population graph.

    The edge weight is ``similarity * affinity`` — imaging similarity gated by
    phenotypic agreement — then reduced to each node's ``k`` strongest
    neighbours. Sparsification matters here for the same reason it does at
    region level: a dense population graph propagates every subject into every
    other and erases the local structure the classifier needs.

    Returns ``(edge_index, edge_weight)``; the graph is symmetrised, since
    subject similarity has no direction.
    """
    weights = similarity.copy()
    if affinity is not None:
        weights = weights * affinity
    np.fill_diagonal(weights, -np.inf)  # no self-loops before top-k

    n = weights.shape[0]
    k = min(k_neighbors, n - 1)
    neighbours = np.argpartition(-weights, kth=k - 1, axis=1)[:, :k]

    rows = np.repeat(np.arange(n), k)
    cols = neighbours.reshape(-1)
    values = weights[rows, cols]

    keep = np.isfinite(values)
    rows, cols, values = rows[keep], cols[keep], values[keep]

    # Symmetrise: an edge kept from either direction is kept for both.
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    val = np.concatenate([values, values])

    edge_index = torch.as_tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.as_tensor(val, dtype=torch.float32)
    return edge_index, edge_weight


class PopulationGCN(nn.Module):
    """Transductive node classifier over a population graph.

    Every subject is a node in one graph. The loss is evaluated only on
    training nodes, while message passing sees the whole population — the
    semi-supervised setting the population-graph literature uses, and the
    reason a held-out subject benefits from its neighbours' labels indirectly
    through propagation.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        n_classes: int = 2,
        dropout: float = 0.3,
        layers: int = 2,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_features, hidden))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, hidden))
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        # GCN normalisation requires non-negative weights.
        weight = edge_weight.clamp_min(0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=weight)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
