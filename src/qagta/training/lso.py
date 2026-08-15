"""Leave-Site-Out cross-validation and the classical baselines.

Splitting by acquisition site rather than at random is what makes the estimate
meaningful for ABIDE: scanner and protocol differences between sites are a
well-known confound, and a random split lets a model partly memorise them.
Under LSO, every fold trains on all sites but one and tests on the held-out
site, so what is being measured is generalisation to an unseen scanner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch_geometric.data import Batch

from qagta.graph.connectome import AdaptiveConnectomeEdges, mean_average_distance
from qagta.models.connectome_gat import ConnectomeGAT
from qagta.pipelines.connectome_pipeline import EncodedCohort


@dataclass
class FoldResult:
    site: str
    n_test: int
    f1: float
    accuracy: float
    specificity: float
    sensitivity: float
    mad: float = float("nan")


@dataclass
class LSOResult:
    name: str
    folds: list[FoldResult] = field(default_factory=list)

    def _values(self, metric: str) -> np.ndarray:
        return np.array([getattr(f, metric) for f in self.folds], dtype=float)

    def mean_ci(self, metric: str) -> tuple[float, float]:
        """Mean and 95% confidence half-width across folds."""
        values = self._values(metric)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return float("nan"), float("nan")
        mean = float(values.mean())
        if len(values) < 2:
            return mean, 0.0
        half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
        return mean, half

    def summary(self) -> str:
        parts = []
        for metric in ("f1", "accuracy", "specificity"):
            mean, half = self.mean_ci(metric)
            parts.append(f"{metric}={mean:.3f}+-{half:.3f}")
        return f"{self.name:<28}" + "  ".join(parts)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    return float(f1), float(accuracy), float(specificity), float(sensitivity)


def train_fold(
    cohort: EncodedCohort,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int = 30,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    hidden_dim: int = 32,
    heads: int = 4,
    dropout: float = 0.6,
    batch_size: int = 16,
    model_type: str = "gat",
    seed: int = 0,
    measure_mad: bool = True,
) -> FoldResult:
    """Train the classifier on one LSO fold and evaluate on the held-out site."""
    torch.manual_seed(seed)
    latent_dim = cohort.latents.shape[-1]

    if model_type == "gat":
        model: torch.nn.Module = ConnectomeGAT(
            latent_dim=latent_dim, hidden_dim=hidden_dim, heads=heads, dropout=dropout
        )
    elif model_type == "gcn":
        from qagta.models.connectome_gat import ConnectomeGCN

        model = ConnectomeGCN(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"unknown model_type {model_type!r}")

    edge_learner = AdaptiveConnectomeEdges(latent_dim=latent_dim)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(edge_learner.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Class imbalance varies by site; weight the loss by inverse frequency.
    train_labels = cohort.labels[train_idx]
    counts = torch.bincount(train_labels, minlength=2).float()
    class_weight = (counts.sum() / (2 * counts.clamp_min(1))).to(torch.float32)

    def build_batch(indices: np.ndarray) -> Batch:
        return Batch.from_data_list([cohort.graph(int(i)) for i in indices])

    rng = np.random.default_rng(seed)
    model.train()
    edge_learner.train()
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            if len(chunk) < 2:  # BatchNorm-free, but a 1-graph batch is still noisy
                continue
            batch = build_batch(chunk)
            optimizer.zero_grad()
            weights = edge_learner(batch.x, batch.edge_index)
            logits = model(batch.x, batch.edge_index, weights, batch.batch)
            loss = F.cross_entropy(logits, batch.y, weight=class_weight)
            loss.backward()
            optimizer.step()

    model.eval()
    edge_learner.eval()
    preds, mad_values = [], []
    with torch.no_grad():
        for start in range(0, len(test_idx), batch_size):
            chunk = test_idx[start : start + batch_size]
            batch = build_batch(chunk)
            weights = edge_learner(batch.x, batch.edge_index)
            logits, node_features = model(
                batch.x, batch.edge_index, weights, batch.batch, return_node_features=True
            )
            preds.append(logits.argmax(dim=1).cpu().numpy())
            if measure_mad:
                # MAD on the first graph of the batch only: the diagnostic is
                # per-graph, and pooling across graphs would conflate them.
                first = node_features[batch.batch == 0]
                mad_values.append(mean_average_distance(first))

    y_pred = np.concatenate(preds)
    y_true = cohort.labels[test_idx].numpy()
    f1, accuracy, specificity, sensitivity = _metrics(y_true, y_pred)

    return FoldResult(
        site="",
        n_test=len(test_idx),
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        sensitivity=sensitivity,
        mad=float(np.mean(mad_values)) if mad_values else float("nan"),
    )


def leave_site_out(
    cohort: EncodedCohort,
    name: str = "proposed",
    min_test_size: int = 10,
    verbose: bool = True,
    **fold_kwargs,
) -> LSOResult:
    """Run Leave-Site-Out cross-validation over every acquisition site."""
    result = LSOResult(name=name)
    sites = np.asarray(cohort.sites)

    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < min_test_size or len(np.unique(cohort.labels[test_idx])) < 2:
            if verbose:
                print(f"  skipping site {site}: {len(test_idx)} subjects, "
                      f"{len(np.unique(cohort.labels[test_idx]))} classes")
            continue

        fold = train_fold(cohort, train_idx, test_idx, **fold_kwargs)
        fold.site = site
        result.folds.append(fold)
        if verbose:
            print(f"  {site:<12} n={fold.n_test:<4} f1={fold.f1:.3f} "
                  f"acc={fold.accuracy:.3f} spec={fold.specificity:.3f}", flush=True)

    return result
