"""Synthetic multivariate time-series generator with injected anomalies.

Produces data in the same shape as the real-world targets of the pipeline
(sensor/flow telemetry, physiological recordings): several correlated
channels with periodic structure and autoregressive noise, where anomalous
windows break the normal regime through bursts, level shifts or
correlation breakdown. Useful for demos, tests and CI, where the actual
datasets cannot be shipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_multivariate_series(
    n_samples: int = 400,
    n_features: int = 10,
    anomaly_fraction: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate windowed feature vectors with binary anomaly labels.

    Returns a DataFrame with ``n_features`` feature columns and an
    ``attack`` label column (0 = normal, 1 = anomaly), mirroring the CSV
    layout expected by :func:`qagta.data.loaders.load_csv_dataset`.
    """
    rng = np.random.default_rng(seed)
    n_anomalies = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies

    # Normal regime: low-rank correlated channels + smooth periodic drift.
    mixing = rng.normal(size=(3, n_features))
    t = np.arange(n_normal)
    factors = np.stack(
        [
            np.sin(2 * np.pi * t / 50 + rng.uniform(0, 2 * np.pi)),
            np.cos(2 * np.pi * t / 120 + rng.uniform(0, 2 * np.pi)),
            rng.normal(scale=0.5, size=n_normal).cumsum() / np.sqrt(n_normal),
        ],
        axis=1,
    )
    normal = factors @ mixing + rng.normal(scale=0.3, size=(n_normal, n_features))

    # Anomalous regime: three failure modes, chosen per sample.
    anomalies = np.empty((n_anomalies, n_features))
    modes = rng.integers(0, 3, size=n_anomalies)
    for i, mode in enumerate(modes):
        base = normal[rng.integers(0, n_normal)]
        if mode == 0:  # burst on a subset of channels
            channels = rng.choice(n_features, size=max(1, n_features // 3), replace=False)
            sample = base.copy()
            sample[channels] += rng.uniform(2.5, 5.0) * rng.choice([-1, 1])
        elif mode == 1:  # level shift across all channels
            sample = base + rng.uniform(1.5, 3.0)
        else:  # correlation breakdown: channels decouple into pure noise
            sample = rng.normal(scale=1.5, size=n_features)
        anomalies[i] = sample

    features = np.vstack([normal, anomalies])
    labels = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_anomalies, dtype=int)])

    order = rng.permutation(n_samples)
    df = pd.DataFrame(
        features[order], columns=[f"feature_{i}" for i in range(n_features)]
    )
    df["attack"] = labels[order]
    return df
