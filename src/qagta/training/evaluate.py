"""Anomaly scoring and evaluation utilities.

Final embeddings are scored with a one-class SVM fitted on normal-only
training embeddings; a comprehensive metric suite supports comparing the
full pipeline against ablated baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.svm import OneClassSVM


@dataclass
class EvaluationResult:
    name: str
    metrics: dict[str, float]
    predictions: np.ndarray
    scores: np.ndarray
    confusion: np.ndarray
    roc: tuple[np.ndarray, np.ndarray]

    def summary(self) -> str:
        keys = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "auc_roc",
        ]
        parts = [f"{k}={self.metrics[k]:.4f}" for k in keys]
        return f"{self.name}: " + "  ".join(parts)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def evaluate_embeddings(
    train_embeddings: torch.Tensor | np.ndarray,
    test_embeddings: torch.Tensor | np.ndarray,
    y_test: np.ndarray,
    name: str = "model",
    nu: float = 0.05,
) -> EvaluationResult:
    """Fit a one-class SVM on normal-only embeddings and score the test set."""
    train = _to_numpy(train_embeddings)
    test = _to_numpy(test_embeddings)
    y_test = np.asarray(y_test)

    detector = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    detector.fit(train)
    raw = detector.predict(test)
    preds = (raw == -1).astype(int)  # -1 (outlier) -> anomaly class 1
    scores = -detector.decision_function(test)  # higher = more anomalous

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = recall_score(y_test, preds, zero_division=0)

    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_roc = auc(fpr, tpr)
        auc_pr = average_precision_score(y_test, scores)
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        auc_roc = auc_pr = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_test, preds, zero_division=0),
        "f2": fbeta_score(y_test, preds, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_test, preds),
        "kappa": cohen_kappa_score(y_test, preds),
        "geometric_mean": float(np.sqrt(recall * specificity)),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    return EvaluationResult(
        name=name,
        metrics=metrics,
        predictions=preds,
        scores=scores,
        confusion=cm,
        roc=(fpr, tpr),
    )


def comparison_table(results: list[EvaluationResult]) -> str:
    """Plain-text comparison table across evaluated configurations."""
    columns = [
        ("accuracy", "acc"),
        ("balanced_accuracy", "bal_acc"),
        ("precision", "prec"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("mcc", "mcc"),
        ("auc_roc", "auc_roc"),
    ]
    header = f"{'model':<28}" + "".join(f"{label:>10}" for _, label in columns)
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.name:<28}" + "".join(f"{r.metrics[key]:>10.4f}" for key, _ in columns)
        )
    return "\n".join(lines)
