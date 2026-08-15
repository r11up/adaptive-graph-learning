"""Subject descriptors for kernel methods.

PCA on flattened correlation vectors is the generic default, but it discards
the network structure a quantum feature map might exploit — principal
components of 19,900 pairwise correlations carry no explicit notion of hubs,
segregation, or modular organisation.

This module adds graph-theoretic descriptors computed from each subject's
connectivity matrix. They are low-dimensional by construction (a few dozen
numbers rather than 19,900), which suits a narrow qubit register, and each
carries an interpretable network meaning.
"""

from __future__ import annotations

import numpy as np

from qagta.graph.connectome import pearson_connectivity


def graph_descriptor(timeseries: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Graph-theoretic summary of one subject's connectome.

    Computes weighted and binarised network measures over the correlation
    matrix, then summarises each nodal measure by its distribution rather than
    keeping it per-node — so the descriptor stays comparable across subjects
    and short enough to encode on a handful of qubits.
    """
    matrix = pearson_connectivity(timeseries)
    np.fill_diagonal(matrix, 0.0)
    weights = np.abs(matrix)
    binary = (weights > threshold).astype(float)

    strength = weights.sum(axis=1)          # weighted degree
    degree = binary.sum(axis=1)             # binary degree

    # Local clustering on the binarised graph.
    triangles = np.diag(binary @ binary @ binary)
    possible = degree * (degree - 1)
    clustering = np.divide(triangles, possible, out=np.zeros_like(triangles),
                           where=possible > 0)

    # Spectral summary: leading eigenvalues of the weighted matrix describe
    # global integration without needing a full graph library.
    try:
        eigenvalues = np.linalg.eigvalsh(weights)[-5:]
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(5)

    def summarise(values: np.ndarray) -> list[float]:
        return [float(values.mean()), float(values.std()),
                float(np.percentile(values, 25)), float(np.percentile(values, 75)),
                float(values.max())]

    features = (
        summarise(strength)
        + summarise(degree)
        + summarise(clustering)
        + [float(weights.mean()), float(weights.std()),
           float((weights > threshold).mean()),          # density
           float(np.percentile(weights, 90))]
        + [float(v) for v in eigenvalues]
    )
    return np.nan_to_num(np.asarray(features, dtype=np.float32))


def build_descriptors(timeseries_list, kind: str = "correlation") -> np.ndarray:
    """Stack per-subject descriptors of the requested kind.

    ``correlation`` returns the flattened upper triangle (high-dimensional,
    the classical default). ``graph`` returns the network summary above.
    ``both`` concatenates them.
    """
    rows = []
    for series in timeseries_list:
        if kind == "correlation":
            matrix = pearson_connectivity(series)
            rows.append(matrix[np.triu_indices_from(matrix, k=1)])
        elif kind == "graph":
            rows.append(graph_descriptor(series))
        elif kind == "both":
            matrix = pearson_connectivity(series)
            rows.append(np.concatenate([
                matrix[np.triu_indices_from(matrix, k=1)], graph_descriptor(series)
            ]))
        else:
            raise ValueError(f"unknown descriptor kind {kind!r}")
    return np.asarray(rows, dtype=np.float32)
