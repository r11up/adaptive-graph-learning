#!/usr/bin/env python3
"""Generate a synthetic multivariate time-series dataset with anomalies.

Example:
    python scripts/generate_data.py --samples 400 --features 10 --out data/synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qagta.data import generate_multivariate_series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--anomaly-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic.csv"))
    args = parser.parse_args()

    df = generate_multivariate_series(
        n_samples=args.samples,
        n_features=args.features,
        anomaly_fraction=args.anomaly_fraction,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_anom = int(df["attack"].sum())
    print(f"Wrote {len(df)} samples ({n_anom} anomalous) to {args.out}")


if __name__ == "__main__":
    main()
