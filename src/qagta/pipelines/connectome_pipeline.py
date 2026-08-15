"""Quantum encoding and graph caching for the ABIDE connectome study.

Encoding every region of every subject through the variational circuit is the
dominant cost of the pipeline (~0.6 s per subject on CPU for a 200-region,
16-qubit configuration). Because the circuit parameters are shared across
subjects, the encoding can be done once for a given parameter set and cached,
after which Leave-Site-Out cross-validation over the classifier is cheap.

This module provides that cache, plus the per-subject graph assembly that
turns cached latents into a sparsified, fidelity-initialised graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from qagta.data.abide import AbideDataset
from qagta.graph.connectome import fidelity_connectivity, knn_sparsify


@dataclass
class EncodedCohort:
    """Quantum latents and fidelity-derived topology for a whole cohort."""

    latents: torch.Tensor  # (n_subjects, n_roi, latent_dim)
    edge_index: torch.Tensor  # (n_subjects, 2, E) — fixed E from k-NN
    edge_weight: torch.Tensor  # (n_subjects, E) fidelity at initialisation
    labels: torch.Tensor  # (n_subjects,)
    sites: np.ndarray  # (n_subjects,)

    def __len__(self) -> int:
        return self.latents.shape[0]

    def graph(self, index: int) -> Data:
        return Data(
            x=self.latents[index],
            edge_index=self.edge_index[index],
            edge_attr=self.edge_weight[index].unsqueeze(-1),
            y=self.labels[index].reshape(1),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latents": self.latents,
                "edge_index": self.edge_index,
                "edge_weight": self.edge_weight,
                "labels": self.labels,
                "sites": self.sites,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> EncodedCohort:
        blob = torch.load(path, weights_only=False)
        return cls(**blob)


@torch.no_grad()
def encode_cohort(
    dataset: AbideDataset,
    encoder: torch.nn.Module,
    k_neighbors: int = 20,
    verbose: bool = True,
    progress_every: int = 50,
) -> EncodedCohort:
    """Encode every subject and build its fidelity-initialised sparse graph.

    The statevectors are used here, once, to define the topology; they are not
    retained, since training recomputes edge weights from the (much smaller)
    expectation-value latents.
    """
    encoder.eval()
    latents, edge_indices, edge_weights = [], [], []

    for i, subject in enumerate(dataset.subjects):
        x = torch.as_tensor(subject.features, dtype=torch.float32)
        z, states = encoder.encode(x, return_state=True)

        adjacency = fidelity_connectivity(states)
        edge_index, edge_weight = knn_sparsify(adjacency, k=k_neighbors)

        latents.append(z)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)

        if verbose and ((i + 1) % progress_every == 0 or i + 1 == len(dataset)):
            print(f"  encoded {i + 1}/{len(dataset)} subjects", flush=True)

    return EncodedCohort(
        latents=torch.stack(latents),
        edge_index=torch.stack(edge_indices),
        edge_weight=torch.stack(edge_weights),
        labels=torch.as_tensor(dataset.labels, dtype=torch.long),
        sites=dataset.sites,
    )
