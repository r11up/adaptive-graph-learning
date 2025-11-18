"""Dataset loading and preparation for anomaly detection.

The pipeline is trained one-class style: only normal samples are used for
fitting, and the held-out set mixes unseen normal samples with anomalies.
Features are min-max normalised (a prerequisite for the bounded rotation
encoding of the quantum stage).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DatasetSplit:
    """Normal-only training features plus a mixed, labelled test set."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


def split_normal_anomaly(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    train_fraction: float = 0.7,
) -> DatasetSplit:
    """Scale features and build the one-class train/test split."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(features)

    normal = scaled[labels == 0]
    anomalous = scaled[labels == 1]
    if len(normal) < 2:
        raise ValueError("Need at least 2 normal samples to build a split")

    split_at = int(train_fraction * len(normal))
    x_train = normal[:split_at]
    held_out_normal = normal[split_at:]
    x_test = np.vstack([held_out_normal, anomalous])
    y_test = np.hstack(
        [np.zeros(len(held_out_normal), dtype=int), np.ones(len(anomalous), dtype=int)]
    )
    names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        feature_names=names,
    )


def load_csv_dataset(
    path: str | Path,
    label_column: str = "attack",
    drop_columns: tuple[str, ...] = ("node_id",),
    train_fraction: float = 0.7,
) -> DatasetSplit:
    """Load a labelled CSV (features + binary label column) into a split.

    Works for flow/telemetry exports and windowed time-series feature
    tables alike: every non-label, non-dropped column is treated as a
    feature; the label column holds 0 (normal) / 1 (anomaly).
    """
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"Label column {label_column!r} not found in {path}")
    labels = df[label_column].to_numpy().astype(int)
    to_drop = [c for c in (*drop_columns, label_column) if c in df.columns]
    feature_df = df.drop(columns=to_drop)
    return split_normal_anomaly(
        feature_df.to_numpy(dtype=float),
        labels,
        feature_names=list(feature_df.columns),
        train_fraction=train_fraction,
    )
