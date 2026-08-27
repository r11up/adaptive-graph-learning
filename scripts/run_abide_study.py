#!/usr/bin/env python3
"""Run the ABIDE connectome study: quantum-adaptive graph vs classical baselines.

Pipeline: load ABIDE ROI time series -> per-region PCA -> quantum encoding ->
fidelity-initialised k-NN topology -> graph attention classifier, evaluated
with Leave-Site-Out cross-validation against four classical baselines.

Quantum encoding is cached to disk, since it depends only on the circuit
parameters and dominates runtime; re-running the evaluation reuses the cache.

Examples:
    # quick check on a subset
    python scripts/run_abide_study.py --limit 120 --epochs 10

    # full cohort
    python scripts/run_abide_study.py --epochs 30

    # add the permutation test (slow)
    python scripts/run_abide_study.py --permutations 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qagta.data.abide import load_abide
from qagta.graph.connectome import graph_density
from qagta.pipelines.connectome_pipeline import EncodedCohort, encode_cohort
from qagta.quantum.fmri_encoder import build_encoder
from qagta.training.baselines import (
    build_classical_cohort,
    correlation_features,
    gcn_leave_site_out,
    permutation_test,
    svm_leave_site_out,
)
from qagta.training.lso import leave_site_out


def load_timeseries(dataset, root: Path, pipeline: str, strategy: str, atlas: str):
    """Re-read raw ROI time series (needed by the correlation baselines)."""
    series_dir = root / "ABIDE_pcp" / pipeline / strategy
    return [np.loadtxt(series_dir / f"{s.file_id}_{atlas}.1D") for s in dataset.subjects]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--n-qubits", type=int, default=16)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--topology", default="mixed", choices=["mixed", "fidelity", "cosine"],
                        help="how the candidate topology is seeded")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--readout", default="mean",
                        choices=["mean", "attention", "flatten", "stats"],
                        help="graph read-out; mean discards regional identity")
    parser.add_argument("--limit", type=int, help="use only the first N subjects")
    parser.add_argument("--backend", default="torch", choices=["torch", "pennylane"])
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("results/abide_encoded.pt"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ABIDE connectome study: quantum-adaptive graph vs classical baselines")
    print("=" * 78)

    dataset = load_abide(
        root=args.data_root, pipeline=args.pipeline, strategy=args.strategy,
        atlas=args.atlas, n_components=args.n_qubits, limit=args.limit,
    )
    print(dataset.summary())

    # ---- quantum encoding (cached) -------------------------------------
    if args.cache.exists() and not args.refresh_cache:
        cohort = EncodedCohort.load(args.cache)
        if len(cohort) != len(dataset):
            print(f"cache holds {len(cohort)} subjects but {len(dataset)} loaded; re-encoding")
            cohort = None
        else:
            print(f"loaded cached encoding from {args.cache}")
    else:
        cohort = None

    if cohort is None:
        encoder = build_encoder(args.backend, n_qubits=args.n_qubits)
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"\nencoding {len(dataset)} subjects x {dataset.n_roi} regions "
              f"({args.n_qubits} qubits, {n_params} circuit parameters, backend={args.backend})")
        start = time.perf_counter()
        cohort = encode_cohort(dataset, encoder, k_neighbors=args.k_neighbors,
                               topology=args.topology)
        print(f"encoding took {time.perf_counter() - start:.1f}s")
        cohort.save(args.cache)
        print(f"cached -> {args.cache}")

    density = graph_density(cohort.edge_index[0], cohort.latents.shape[1])
    print(f"quantum graph: {cohort.latents.shape[1]} nodes, avg degree {density:.1f}, "
      f"topology={args.topology}")

    results = []

    # ---- classical baselines -------------------------------------------
    if not args.skip_baselines:
        print("\n" + "-" * 78)
        print("classical baselines")
        print("-" * 78)
        timeseries = load_timeseries(
            dataset, args.data_root, args.pipeline, args.strategy, args.atlas
        )
        corr = correlation_features(dataset, timeseries)
        print(f"correlation feature vectors: {corr.shape}")

        for kernel in ("linear", "rbf"):
            print(f"\nSVM ({kernel}) on flattened correlation matrices:")
            results.append(
                svm_leave_site_out(corr, dataset.labels, dataset.sites, kernel=kernel)
            )

        for metric, label in (("pearson", "GCN + Pearson"), ("rbf", "GCN + RBF")):
            print(f"\n{label}:")
            classical = build_classical_cohort(
                dataset, timeseries, metric=metric, k_neighbors=args.k_neighbors
            )
            results.append(
                gcn_leave_site_out(
                    classical, name=label, epochs=args.epochs, lr=args.lr,
                    hidden_dim=args.hidden_dim, dropout=args.dropout, seed=args.seed,
                )
            )

    # ---- proposed framework --------------------------------------------
    print("\n" + "-" * 78)
    print("proposed: quantum fidelity topology + adaptive edges + GAT")
    print("-" * 78)
    proposed = leave_site_out(
        cohort, name="Proposed (quantum + GAT)", epochs=args.epochs, lr=args.lr,
        hidden_dim=args.hidden_dim, heads=args.heads, dropout=args.dropout,
        model_type="gat", readout=args.readout, seed=args.seed,
    )
    results.append(proposed)

    print("\nablation: quantum topology + GCN (isolates the attention mechanism)")
    results.append(
        leave_site_out(
            cohort, name="Ablation (quantum + GCN)", epochs=args.epochs, lr=args.lr,
            hidden_dim=args.hidden_dim, dropout=args.dropout, model_type="gcn",
            seed=args.seed, verbose=False,
        )
    )

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("LEAVE-SITE-OUT RESULTS (mean +- 95% CI across site folds)")
    print("=" * 78)
    header = f"{'model':<28}{'F1':>16}{'accuracy':>16}{'specificity':>16}"
    print(header)
    print("-" * len(header))
    for result in results:
        row = f"{result.name:<28}"
        for metric in ("f1", "accuracy", "specificity"):
            mean, half = result.mean_ci(metric)
            row += f"{mean:>10.3f}+-{half:<5.3f}"
        print(row)

    mad_mean, _ = proposed.mean_ci("mad")
    print(f"\nover-smoothing diagnostic (MAD, higher = more distinct): {mad_mean:.3f}")

    payload = {
        "n_subjects": len(dataset),
        "n_sites": len(set(dataset.sites.tolist())),
        "config": vars(args) | {"data_root": str(args.data_root), "cache": str(args.cache),
                                "out": str(args.out)},
        "results": {
            r.name: {
                m: {"mean": r.mean_ci(m)[0], "ci95": r.mean_ci(m)[1]}
                for m in ("f1", "accuracy", "specificity", "sensitivity")
            }
            | {"folds": [vars(f) for f in r.folds]}
            for r in results
        },
    }

    if args.permutations:
        observed = proposed.mean_ci("f1")[0]
        print(f"\npermutation test ({args.permutations} shuffles) against F1={observed:.3f}")
        p_value, null = permutation_test(
            cohort, observed, n_permutations=args.permutations, seed=args.seed,
            epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim,
            dropout=args.dropout, model_type="gat",
        )
        print(f"null mean F1={null.mean():.3f}  observed={observed:.3f}  p={p_value:.4f}")
        payload["permutation"] = {
            "observed_f1": observed, "p_value": p_value,
            "null_mean": float(null.mean()), "n_permutations": args.permutations,
        }

    out_path = args.out / "abide_lso_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
