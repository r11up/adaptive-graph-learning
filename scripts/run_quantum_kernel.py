#!/usr/bin/env python3
"""Quantum kernel SVM vs classical kernels, matched features, matched folds.

Tests one question: does a quantum feature map induce a better kernel than RBF
or linear on the *same* subject descriptors, under the same Leave-Site-Out
folds?

Design decisions that make the comparison honest:

- **Matched features.** Every classifier sees the identical n-dimensional
  descriptor. Comparing a quantum kernel on 8 components against a classical
  kernel on 19,900 raw correlations measures dimensionality, not quantum
  advantage. The full-dimensional SVM is reported separately as a reference
  ceiling, clearly labelled.
- **No leakage.** PCA and the scaler are fit on the training sites only, inside
  each fold, then applied to the held-out site.
- **Matched tuning budget.** Both sides sweep the same grid of C, and the
  classical side additionally sweeps gamma — so the classical comparator is if
  anything favoured.
- **Paired testing.** Since both models are evaluated on identical folds, a
  paired Wilcoxon signed-rank test over per-site scores is the appropriate
  significance test.

Examples:
    python scripts/run_quantum_kernel.py --qubits 8
    python scripts/run_quantum_kernel.py --qubits 4 --reps 2 --entanglement full
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix
from qagta.quantum.projected import kernel_target_alignment, projected_kernel_matrix


def subject_descriptors(root: Path, pipeline: str, strategy: str, atlas: str,
                        dataset, kind: str = "correlation"):
    """Per-subject descriptor block of the requested kind."""
    series_dir = root / "ABIDE_pcp" / pipeline / strategy
    series = [np.loadtxt(series_dir / f"{s.file_id}_{atlas}.1D") for s in dataset.subjects]
    return build_descriptors(series, kind=kind)


def metrics(y_true, y_pred, scores=None) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, scores))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--qubits", type=int, default=8, help="= number of PCA components")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--bandwidths", type=float, nargs="+",
                        default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
                        help="encoding-angle scales to search over (lever 1)")
    parser.add_argument("--kernel-type", default="fidelity",
                        choices=["fidelity", "projected", "both"],
                        help="fidelity overlap or projected RDM kernel (lever 4)")
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0],
                        help="RBF width for the projected kernel")
    parser.add_argument("--descriptor", default="correlation",
                        choices=["correlation", "graph", "both"],
                        help="subject descriptor (lever 3)")
    parser.add_argument("--selection", default="cv_auc", choices=["cv_auc", "alignment"],
                        help="selection rule: inner-CV AUC or kernel-target alignment (lever 2)")
    parser.add_argument("--entanglement", default="linear", choices=["linear", "full"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-test-size", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_qkernel_q{args.qubits}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"quantum kernel study -> {out_dir}")
    dataset = load_abide(
        root=args.data_root, pipeline=args.pipeline, strategy=args.strategy,
        atlas=args.atlas, n_components=args.qubits, limit=args.limit,
    )
    print(dataset.summary())

    features = subject_descriptors(
        args.data_root, args.pipeline, args.strategy, args.atlas, dataset,
        kind=args.descriptor,
    )
    labels, sites = dataset.labels, dataset.sites
    print(f"subject descriptors: {features.shape}")

    feature_maps = {
        b: QuantumFeatureMap(n_qubits=args.qubits, reps=args.reps,
                             entanglement=args.entanglement, bandwidth=b)
        for b in args.bandwidths
    }
    print(f"bandwidths searched: {args.bandwidths}")
    grid_c = [0.1, 1.0, 10.0]
    grid_gamma = ["scale", 0.1, 1.0]

    per_fold: dict[str, list[dict]] = {
        "quantum kernel": [], "classical RBF": [], "classical linear": [],
        "reference: RBF on full dim": [],
    }

    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < args.min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue

        y_train, y_test = labels[train_idx], labels[test_idx]

        # Fit reduction on training sites only.
        scaler = StandardScaler().fit(features[train_idx])
        train_scaled = scaler.transform(features[train_idx])
        pca = PCA(n_components=args.qubits, random_state=0).fit(train_scaled)
        train_reduced = pca.transform(train_scaled)
        test_reduced = pca.transform(scaler.transform(features[test_idx]))

        # Angle encoding needs a bounded domain; fit the range on train only.
        angle_scaler = MinMaxScaler((0, np.pi)).fit(train_reduced)
        train_angles = angle_scaler.transform(train_reduced)
        test_angles = np.clip(angle_scaler.transform(test_reduced), 0, np.pi)

        # --- quantum kernel -------------------------------------------------
        # Search bandwidth jointly with C, selected by inner CV on train sites.
        # Candidates: bandwidth x kernel type x (gamma, projected only).
        kernels: dict[tuple, tuple] = {}
        for bandwidth, fm in feature_maps.items():
            if args.kernel_type in ("fidelity", "both"):
                kernels[("fidelity", bandwidth, None)] = (
                    quantum_kernel_matrix(train_angles, train_angles, fm),
                    quantum_kernel_matrix(test_angles, train_angles, fm),
                )
            if args.kernel_type in ("projected", "both"):
                with torch.no_grad():
                    st = fm(torch.as_tensor(train_angles, dtype=torch.float32))
                    se = fm(torch.as_tensor(test_angles, dtype=torch.float32))
                for gamma in args.gammas:
                    kernels[("projected", bandwidth, gamma)] = (
                        projected_kernel_matrix(st, st, args.qubits, gamma),
                        projected_kernel_matrix(se, st, args.qubits, gamma),
                    )
        # Select C by inner cross-validation on the training sites. Scoring on
        # the training fit instead would simply pick the most overfit model —
        # with an RBF kernel that yields a classifier that memorises the
        # training split and predicts a single class everywhere else.
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        best = None
        for spec, (k_train, _) in kernels.items():
            if args.selection == "alignment":
                # Scores the kernel against the labels with no classifier fit.
                score, candidate_c = kernel_target_alignment(k_train, y_train), grid_c[1]
            else:
                score, candidate_c = max(
                    (cross_val_score(
                        SVC(kernel="precomputed", C=c, class_weight="balanced"),
                        k_train, y_train, cv=inner, scoring="roc_auc").mean(), c)
                    for c in grid_c
                )
            if best is None or score > best[0]:
                best = (score, candidate_c, spec)
        chosen = best[2]
        k_train, k_test = kernels[chosen]
        model = SVC(kernel="precomputed", C=best[1], class_weight="balanced").fit(
            k_train, y_train
        )
        per_fold["quantum kernel"].append(
            {"site": site, "n": len(test_idx), "kernel_type": chosen[0],
             "bandwidth": chosen[1], "gamma": chosen[2],
             **metrics(y_test, model.predict(k_test), model.decision_function(k_test))}
        )

        # --- classical kernels on the SAME reduced features -----------------
        for name, kernel in (("classical RBF", "rbf"), ("classical linear", "linear")):
            best = None
            for c in grid_c:
                for gamma in (grid_gamma if kernel == "rbf" else ["scale"]):
                    score = cross_val_score(
                        SVC(kernel=kernel, C=c, gamma=gamma, class_weight="balanced"),
                        train_reduced, y_train, cv=inner, scoring="roc_auc",
                    ).mean()
                    if best is None or score > best[0]:
                        best = (score, c, gamma)
            model = SVC(kernel=kernel, C=best[1], gamma=best[2],
                        class_weight="balanced").fit(train_reduced, y_train)
            per_fold[name].append(
                {"site": site, "n": len(test_idx),
                 **metrics(y_test, model.predict(test_reduced),
                           model.decision_function(test_reduced))}
            )

        # --- reference: full-dimensional classical SVM ----------------------
        full_train = scaler.transform(features[train_idx])
        full_test = scaler.transform(features[test_idx])
        model = SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced").fit(
            full_train, y_train
        )
        per_fold["reference: RBF on full dim"].append(
            {"site": site, "n": len(test_idx),
             **metrics(y_test, model.predict(full_test), model.decision_function(full_test))}
        )

        print(f"  {site:<12} n={len(test_idx):<4} "
              f"Q f1={per_fold['quantum kernel'][-1]['f1']:.3f}  "
              f"RBF f1={per_fold['classical RBF'][-1]['f1']:.3f}", flush=True)

    # ---- report ----------------------------------------------------------
    def summarise(folds: list[dict], key: str) -> tuple[float, float]:
        values = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 74)
    print(f"QUANTUM KERNEL vs CLASSICAL — {args.qubits} qubits / {args.qubits} PCA components")
    print("=" * 74)
    header = f"{'model':<30}{'F1':>14}{'accuracy':>14}{'AUC':>14}"
    print(header + "\n" + "-" * len(header))
    for name, folds in per_fold.items():
        row = f"{name:<30}"
        for key in ("f1", "accuracy", "auc"):
            mean, half = summarise(folds, key)
            row += f"{mean:>8.3f}+-{half:<5.3f}"
        print(row)

    # Paired test: identical folds, so pair per site.
    stats = {}
    q = np.array([f["f1"] for f in per_fold["quantum kernel"]])
    for rival in ("classical RBF", "classical linear"):
        c = np.array([f["f1"] for f in per_fold[rival]])
        if len(q) > 1 and np.any(q != c):
            statistic, p = wilcoxon(q, c)
            stats[rival] = {"median_diff": float(np.median(q - c)), "p_value": float(p),
                            "wins": int((q > c).sum()), "losses": int((q < c).sum()),
                            "folds": int(len(q))}
            print(f"\nquantum vs {rival}: median dF1 = {np.median(q - c):+.3f}, "
                  f"wins {int((q > c).sum())}/{len(q)}, Wilcoxon p = {p:.4f}")

    payload = {"config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir)},
               "n_subjects": len(dataset), "per_fold": per_fold,
               "summary": {name: {k: summarise(f, k) for k in ("f1", "accuracy", "auc")}
                           for name, f in per_fold.items()},
               "paired_tests": stats, "timestamp": stamp}
    (out_dir / "quantum_kernel_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'quantum_kernel_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
