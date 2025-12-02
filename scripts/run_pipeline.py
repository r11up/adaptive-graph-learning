#!/usr/bin/env python3
"""Train and evaluate the full pipeline, with ablation baselines.

Runs three configurations on the same data and prints a comparison:

- quantum latents only (no graph stage),
- quantum latents + adaptive graph + SAGE propagation,
- quantum latents + adaptive graph + attention propagation (full system).

Examples:
    # Self-contained demo on synthetic data
    python scripts/run_pipeline.py --synthetic

    # Your own CSV (feature columns + a binary label column)
    python scripts/run_pipeline.py --data data.csv --label-column attack

    # Custom configuration
    python scripts/run_pipeline.py --synthetic --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from qagta import PipelineConfig, QuantumAdaptiveGraphPipeline
from qagta.data import generate_multivariate_series, load_csv_dataset, split_normal_anomaly
from qagta.training.evaluate import comparison_table, evaluate_embeddings


def load_split(args: argparse.Namespace):
    if args.synthetic:
        df = generate_multivariate_series(
            n_samples=args.samples, n_features=args.features, seed=args.seed
        )
        features = df.drop(columns=["attack"]).to_numpy()
        return split_normal_anomaly(features, df["attack"].to_numpy())
    if args.data is None:
        raise SystemExit("Provide --data <csv> or use --synthetic")
    return load_csv_dataset(args.data, label_column=args.label_column)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, help="CSV with features + label column")
    parser.add_argument("--label-column", default="attack")
    parser.add_argument("--synthetic", action="store_true", help="Use generated data")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--config", type=Path, help="YAML pipeline configuration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    split = load_split(args)
    verbose = not args.quiet
    print(
        f"data: {split.x_train.shape[0]} train (normal), "
        f"{split.x_test.shape[0]} test ({int(split.y_test.sum())} anomalous), "
        f"{split.n_features} features"
    )

    base_config = (
        PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    )
    base_config.training.seed = args.seed

    results = []

    # Full system: adaptive graph + attention propagation.
    gat_config = copy.deepcopy(base_config)
    gat_config.model.encoder = "gat"
    print("\n### training: quantum + adaptive graph + attention (full system)")
    gat_pipeline = QuantumAdaptiveGraphPipeline(gat_config, input_dim=split.n_features)
    gat_pipeline.fit(split.x_train, verbose=verbose)

    # Baseline 1: quantum latents only (shares the trained encoder).
    results.append(
        evaluate_embeddings(
            gat_pipeline.ablation_embed(split.x_train),
            gat_pipeline.ablation_embed(split.x_test),
            split.y_test,
            name="quantum latents only",
            nu=base_config.training.ocsvm_nu,
        )
    )

    # Baseline 2: adaptive graph with non-attentive propagation.
    sage_config = copy.deepcopy(base_config)
    sage_config.model.encoder = "sage"
    print("\n### training: quantum + adaptive graph + SAGE (baseline)")
    sage_pipeline = QuantumAdaptiveGraphPipeline(sage_config, input_dim=split.n_features)
    sage_pipeline.fit(split.x_train, verbose=verbose)
    results.append(
        sage_pipeline.evaluate(split.x_test, split.y_test, name="adaptive graph + SAGE")
    )

    results.append(
        gat_pipeline.evaluate(split.x_test, split.y_test, name="adaptive graph + GAT")
    )

    print("\n" + comparison_table(results))
    best = max(results, key=lambda r: r.metrics["f1"])
    print(f"\nbest by F1: {best.name} (f1={best.metrics['f1']:.4f})")

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "comparison.txt"
    report_path.write_text(comparison_table(results) + "\n")
    print(f"saved: {report_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for result in results:
            fpr, tpr = result.roc
            axes[0].plot(fpr, tpr, label=f"{result.name} ({result.metrics['auc_roc']:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set_xlabel("false positive rate")
        axes[0].set_ylabel("true positive rate")
        axes[0].set_title("ROC")
        axes[0].legend(fontsize=8)

        names = [r.name for r in results]
        f1s = [r.metrics["f1"] for r in results]
        axes[1].bar(range(len(results)), f1s, color=["#888", "#5a9", "#36c"])
        axes[1].set_xticks(range(len(results)))
        axes[1].set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
        axes[1].set_ylabel("F1")
        axes[1].set_title("F1 by configuration")

        fig.tight_layout()
        plot_path = args.out / "comparison.png"
        fig.savefig(plot_path, dpi=200)
        print(f"saved: {plot_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
