#!/usr/bin/env python3
"""Benchmark pipeline: supervised feature selection, ensembling, quantum vs classical.

The earlier population-graph and kernel studies used unsupervised PCA on the
connectivity vector. PCA maximises variance, not class separation, so it can
discard exactly the directions that distinguish groups — and published ABIDE
pipelines typically use supervised selection instead. This script upgrades the
feature stage and re-runs the quantum-versus-classical comparison on the
stronger baseline, so any quantum claim is tested against a well-tuned
classical pipeline rather than a weak one.

Changes from the earlier studies:

- **Supervised feature selection.** Top-k connections ranked by absolute
  t-statistic, fit on the training sites only. This is the standard approach in
  the ABIDE literature and it substantially outperforms unsupervised PCA here.
- **Phenotypic features.** Age and sex appended when available. On ADHD-200
  phenotypic data alone reaches 62.5% in the original competition, ahead of the
  best imaging entry, so omitting it understates what is achievable.
- **Seed ensembling.** Predictions averaged over several seeds, which reduces
  the fold-to-fold variance that dominates small held-out sites.
- **Site harmonisation.** Optional per-site standardisation of features, the
  cheap form of what ComBat does; reported separately since the literature
  finds harmonisation helps less than expected over a well-tuned baseline.

Models compared on identical folds and identical features: classical SVM
(linear/RBF), MLP, and the quantum fidelity kernel.

Examples:
    python scripts/run_benchmark.py --data-root data/ABIDE-I
    python scripts/run_benchmark.py --data-root data/REST-meta-MDD --harmonise
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix


def select_features(x_train, y_train, k: int) -> np.ndarray:
    """Indices of the top-k features by absolute t-statistic on the training set.

    Fit on training data only; applying it to the held-out site would leak the
    test labels into feature selection, which is a common and serious error in
    this literature.
    """
    group_a = x_train[y_train == 0]
    group_b = x_train[y_train == 1]
    if len(group_a) < 2 or len(group_b) < 2:
        return np.arange(min(k, x_train.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat, _ = stats.ttest_ind(group_a, group_b, axis=0, equal_var=False)
    t_stat = np.nan_to_num(np.abs(t_stat))
    return np.argsort(-t_stat)[: min(k, x_train.shape[1])]


def harmonise_by_site(features: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Centre and scale each site's features independently.

    The cheap form of ComBat: it removes additive and multiplicative site
    effects without modelling them jointly. Applied per site across the whole
    cohort, which is legitimate because it uses no labels.
    """
    out = features.copy()
    for site in np.unique(sites):
        mask = sites == site
        block = out[mask]
        out[mask] = (block - block.mean(0)) / (block.std(0) + 1e-8)
    return out


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
    parser.add_argument("--top-k", type=int, default=2000,
                        help="connections retained by supervised selection")
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--bandwidths", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="ensembled over these seeds")
    parser.add_argument("--harmonise", action="store_true",
                        help="per-site standardisation before selection")
    parser.add_argument("--no-phenotypic", action="store_true")
    parser.add_argument("--min-test-size", type=int, default=10)
    parser.add_argument("--cv", default="leave-site-out",
                        choices=["leave-site-out", "stratified"],
                        help="stratified k-fold for cohorts with too few sites")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"benchmark study -> {out_dir}")

    dataset = load_abide(
        root=args.data_root, pipeline=args.pipeline, strategy=args.strategy,
        atlas=args.atlas, n_components=args.qubits, limit=args.limit,
    )
    print(dataset.summary())

    series_dir = args.data_root / "ABIDE_pcp" / args.pipeline / args.strategy
    series = [
        np.loadtxt(series_dir / f"{s.file_id}_{args.atlas}.1D") for s in dataset.subjects
    ]
    connectivity = build_descriptors(series, kind="correlation")
    labels, sites = dataset.labels, dataset.sites
    print(f"connectivity: {connectivity.shape}")

    if args.harmonise:
        connectivity = harmonise_by_site(connectivity, sites)
        print("per-site harmonisation: on")

    # Phenotypic covariates, appended after selection so they are never
    # competing with 19,900 connections for a top-k slot.
    extras = None
    if not args.no_phenotypic:
        table = pd.read_csv(args.data_root / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv")
        if "FILE_ID" in table.columns:
            table = table.set_index("FILE_ID")
            cols = []
            for names in (("SEX", "sex", "Gender"), ("AGE_AT_SCAN", "AGE", "Age")):
                for name in names:
                    if name in table.columns:
                        values = pd.to_numeric(table[name], errors="coerce")
                        col = np.array([values.get(s.file_id, np.nan)
                                        for s in dataset.subjects], dtype=float)
                        if not np.isnan(col).all():
                            cols.append(np.nan_to_num(col, nan=float(np.nanmean(col))))
                        break
            if cols:
                extras = np.column_stack(cols)
                print(f"phenotypic covariates: {extras.shape[1]}")

    # "matched" models see the same n_qubits features the quantum kernel sees.
    # Without them, a quantum loss cannot be attributed: the quantum register
    # caps the feature count at n_qubits while the full models get top-k, so a
    # gap would measure dimensionality rather than the kernel.
    models = ["quantum kernel", "SVM RBF (matched)", "SVM (RBF)", "SVM (linear)", "MLP"]
    per_fold: dict[str, list[dict]] = {m: [] for m in models}

    if args.cv == "stratified":
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
        split_iter = [(f"fold{i + 1}", tr, te) for i, (tr, te)
                      in enumerate(splitter.split(connectivity, labels))]
        print(f"stratified {args.folds}-fold CV")
    else:
        split_iter = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
                      for s in sorted(set(sites.tolist()))]

    for site, train_idx, test_idx in split_iter:
        if len(test_idx) < args.min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue
        y_train, y_test = labels[train_idx], labels[test_idx]

        chosen = select_features(connectivity[train_idx], y_train, args.top_k)
        scaler = StandardScaler().fit(connectivity[train_idx][:, chosen])
        x_train = scaler.transform(connectivity[train_idx][:, chosen])
        x_test = scaler.transform(connectivity[test_idx][:, chosen])

        if extras is not None:
            extra_scaler = StandardScaler().fit(extras[train_idx])
            x_train = np.hstack([x_train, extra_scaler.transform(extras[train_idx])])
            x_test = np.hstack([x_test, extra_scaler.transform(extras[test_idx])])

        # --- classical models, ensembled over seeds ------------------------
        for name, factory in (
            ("SVM (RBF)", lambda s: SVC(kernel="rbf", C=1.0, gamma="scale",
                                        class_weight="balanced", random_state=s)),
            ("SVM (linear)", lambda s: SVC(kernel="linear", C=1.0,
                                           class_weight="balanced", random_state=s)),
            ("MLP", lambda s: MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0,
                                            max_iter=400, random_state=s)),
        ):
            scores = []
            for seed in args.seeds:
                model = factory(seed).fit(x_train, y_train)
                raw = (model.decision_function(x_test)
                       if hasattr(model, "decision_function")
                       else model.predict_proba(x_test)[:, 1])
                scores.append(raw)
            mean_score = np.mean(scores, axis=0)
            preds = (mean_score > (0 if "SVM" in name else 0.5)).astype(int)
            per_fold[name].append(
                {"site": site, "n": len(test_idx), **metrics(y_test, preds, mean_score)}
            )

        # --- quantum kernel on the same selected features -------------------
        # The register is narrow, so the selected block is reduced to n_qubits
        # angles by a supervised ranking of the already-selected features.
        angle_idx = chosen[: args.qubits]
        angle_scaler = MinMaxScaler((0, np.pi)).fit(connectivity[train_idx][:, angle_idx])
        train_angles = angle_scaler.transform(connectivity[train_idx][:, angle_idx])
        test_angles = np.clip(
            angle_scaler.transform(connectivity[test_idx][:, angle_idx]), 0, np.pi
        )

        y_signed = np.where(y_train > 0, 1.0, -1.0)
        ideal = np.outer(y_signed, y_signed)
        best = None
        for bandwidth in args.bandwidths:
            fm = QuantumFeatureMap(n_qubits=args.qubits, reps=2,
                                   entanglement="linear", bandwidth=bandwidth)
            k_train = quantum_kernel_matrix(train_angles, train_angles, fm)
            denom = np.linalg.norm(k_train) * np.linalg.norm(ideal)
            score = float((k_train * ideal).sum() / denom) if denom > 0 else 0.0
            if best is None or score > best[0]:
                best = (score, bandwidth, fm, k_train)
        _, bandwidth, fm, k_train = best
        k_test = quantum_kernel_matrix(test_angles, train_angles, fm)

        model = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(k_train, y_train)
        raw = model.decision_function(k_test)
        per_fold["quantum kernel"].append(
            {"site": site, "n": len(test_idx), "bandwidth": bandwidth,
             **metrics(y_test, (raw > 0).astype(int), raw)}
        )

        # Matched control: classical RBF on the identical n_qubits features.
        matched_scaler = StandardScaler().fit(connectivity[train_idx][:, angle_idx])
        m_train = matched_scaler.transform(connectivity[train_idx][:, angle_idx])
        m_test = matched_scaler.transform(connectivity[test_idx][:, angle_idx])
        matched = SVC(kernel="rbf", C=1.0, gamma="scale",
                      class_weight="balanced").fit(m_train, y_train)
        m_raw = matched.decision_function(m_test)
        per_fold["SVM RBF (matched)"].append(
            {"site": site, "n": len(test_idx),
             **metrics(y_test, (m_raw > 0).astype(int), m_raw)}
        )

        print(f"  {site:<12} n={len(test_idx):<4} "
              + "  ".join(f"{m.split()[0][:4]} {per_fold[m][-1]['accuracy']:.3f}"
                          for m in models), flush=True)

    def summarise(folds, key):
        values = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 74)
    print(f"BENCHMARK — top-{args.top_k} supervised selection, {len(args.seeds)}-seed ensemble")
    print("=" * 74)
    header = f"{'model':<20}{'accuracy':>15}{'F1':>15}{'AUC':>15}"
    print(header + "\n" + "-" * len(header))
    for name, folds in per_fold.items():
        row = f"{name:<20}"
        for key in ("accuracy", "f1", "auc"):
            mean, half = summarise(folds, key)
            row += f"{mean:>9.3f}+-{half:<5.3f}"
        print(row)

    stats_out = {}
    q = np.array([f["accuracy"] for f in per_fold["quantum kernel"]])
    for rival in ("SVM RBF (matched)", "SVM (RBF)", "SVM (linear)", "MLP"):
        c = np.array([f["accuracy"] for f in per_fold[rival]])
        if len(q) > 1 and np.any(q != c):
            _, p = wilcoxon(q, c)
            stats_out[rival] = {"median_diff": float(np.median(q - c)), "p_value": float(p),
                                "wins": int((q > c).sum()), "folds": int(len(q))}
            print(f"\nquantum vs {rival}: median dAcc = {np.median(q - c):+.3f}, "
                  f"wins {int((q > c).sum())}/{len(q)}, p = {p:.4f}")

    payload = {
        "config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir)},
        "n_subjects": len(dataset),
        "summary": {k: {m: summarise(v, m) for m in ("accuracy", "f1", "auc")}
                    for k, v in per_fold.items()},
        "per_fold": per_fold, "paired_tests": stats_out, "timestamp": stamp,
    }
    (out_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'benchmark_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
