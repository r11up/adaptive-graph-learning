"""Classical baselines evaluated under the identical Leave-Site-Out protocol.

Four comparators representing standard practice in fMRI connectivity analysis:

- SVM (linear) and SVM (RBF) on flattened correlation matrices, and
- GCN on a fixed Pearson-correlation graph, and on an RBF-similarity graph.

The GCN baselines consume the *same* PCA node features as the proposed model,
so the only thing that differs is how the topology is built. That isolates the
contribution of the connectivity metric rather than confounding it with a
different node representation.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qagta.data.abide import AbideDataset
from qagta.graph.connectome import knn_sparsify, pearson_connectivity, rbf_connectivity
from qagta.pipelines.connectome_pipeline import EncodedCohort
from qagta.training.lso import FoldResult, LSOResult, _metrics, train_fold


def correlation_features(dataset: AbideDataset, timeseries: list[np.ndarray]) -> np.ndarray:
    """Flattened upper-triangle correlation vectors, one row per subject."""
    rows = []
    for series in timeseries:
        matrix = pearson_connectivity(series)
        upper = np.triu_indices_from(matrix, k=1)
        rows.append(matrix[upper])
    return np.asarray(rows, dtype=np.float32)


def svm_leave_site_out(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    kernel: str = "linear",
    min_test_size: int = 10,
    verbose: bool = True,
) -> LSOResult:
    """SVM baseline on flattened correlation features."""
    result = LSOResult(name=f"SVM ({kernel})")
    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue

        model = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel, C=1.0, gamma="scale", class_weight="balanced"),
        )
        model.fit(features[train_idx], labels[train_idx])
        y_pred = model.predict(features[test_idx])
        f1, accuracy, specificity, sensitivity = _metrics(labels[test_idx], y_pred)
        result.folds.append(
            FoldResult(site=site, n_test=len(test_idx), f1=f1, accuracy=accuracy,
                       specificity=specificity, sensitivity=sensitivity)
        )
        if verbose:
            print(f"  {site:<12} n={len(test_idx):<4} f1={f1:.3f} acc={accuracy:.3f}", flush=True)
    return result


def build_classical_cohort(
    dataset: AbideDataset,
    timeseries: list[np.ndarray],
    metric: str = "pearson",
    k_neighbors: int = 20,
) -> EncodedCohort:
    """Build a cohort whose topology comes from a classical similarity metric.

    Node features are the same PCA vectors used by the quantum pipeline; only
    the adjacency differs, which is what the comparison is meant to isolate.
    """
    latents, edge_indices, edge_weights = [], [], []
    for subject, series in zip(dataset.subjects, timeseries, strict=True):
        features = torch.as_tensor(subject.features, dtype=torch.float32)
        if metric == "pearson":
            adjacency = np.abs(pearson_connectivity(series))  # strength, not direction
        elif metric == "rbf":
            adjacency = rbf_connectivity(subject.features)
        else:
            raise ValueError(f"unknown metric {metric!r}")

        edge_index, edge_weight = knn_sparsify(
            torch.as_tensor(adjacency, dtype=torch.float32), k=k_neighbors
        )
        latents.append(features)
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)

    return EncodedCohort(
        latents=torch.stack(latents),
        edge_index=torch.stack(edge_indices),
        edge_weight=torch.stack(edge_weights),
        labels=torch.as_tensor(dataset.labels, dtype=torch.long),
        sites=dataset.sites,
    )


def gcn_leave_site_out(
    cohort: EncodedCohort, name: str, min_test_size: int = 10, verbose: bool = True, **fold_kwargs
) -> LSOResult:
    """GCN baseline over a fixed classical topology."""
    fold_kwargs.setdefault("model_type", "gcn")
    result = LSOResult(name=name)
    sites = np.asarray(cohort.sites)
    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < min_test_size or len(np.unique(cohort.labels[test_idx])) < 2:
            continue
        fold = train_fold(cohort, train_idx, test_idx, **fold_kwargs)
        fold.site = site
        result.folds.append(fold)
        if verbose:
            print(f"  {site:<12} n={fold.n_test:<4} f1={fold.f1:.3f} "
                  f"acc={fold.accuracy:.3f}", flush=True)
    return result


def permutation_test(
    cohort: EncodedCohort,
    observed_f1: float,
    n_permutations: int = 100,
    seed: int = 0,
    verbose: bool = True,
    **lso_kwargs,
) -> tuple[float, np.ndarray]:
    """Empirical p-value for an observed F1 under label permutation.

    Labels are shuffled *within* the cohort and the full LSO evaluation is
    re-run, giving a null distribution that preserves the site structure. The
    returned p-value is the fraction of permutations reaching at least the
    observed score.
    """
    from qagta.training.lso import leave_site_out

    rng = np.random.default_rng(seed)
    null_scores = []
    original = cohort.labels.clone()
    try:
        for i in range(n_permutations):
            cohort.labels = original[torch.as_tensor(rng.permutation(len(original)))]
            result = leave_site_out(cohort, name=f"perm{i}", verbose=False, **lso_kwargs)
            null_scores.append(result.mean_ci("f1")[0])
            if verbose and (i + 1) % 10 == 0:
                print(f"  permutation {i + 1}/{n_permutations} "
                      f"null mean={np.mean(null_scores):.3f}", flush=True)
    finally:
        cohort.labels = original

    null = np.array(null_scores)
    p_value = float((null >= observed_f1).sum() + 1) / (n_permutations + 1)
    return p_value, null
