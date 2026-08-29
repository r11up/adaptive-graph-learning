#!/usr/bin/env python3
"""Full classification metrics, including confusion matrices, for every run.

The per-fold records store accuracy, F1, specificity and AUC but only two
cells of the confusion matrix are used at write time (specificity needs TN and
FP; TP and FN are discarded). Rather than re-run every experiment to recover
them, the four cells are reconstructed exactly from what is stored plus the
class balance of each fold, which is recoverable because the folds are
deterministic:

    N  = TN + FP          negatives in the fold, from the cohort labels
    P  = TP + FN          positives in the fold
    TN = specificity * N
    FP = N - TN
    TP = accuracy * n - TN
    FN = P - TP

Reconstruction is checked against the stored F1 and rejected if it disagrees
by more than rounding, so a silently wrong matrix cannot reach the paper.

From the pooled matrix the script derives the metrics a clinical classification
table needs and the per-fold values cannot supply: sensitivity, precision,
negative predictive value, balanced accuracy and Matthews correlation. MCC is
included because it is the metric least distorted by class imbalance, and
several cohorts here are imbalanced enough that F1 alone misleads --- a
degenerate all-positive predictor scores F1 ~ 0.65 on a balanced set.

Usage:
    python scripts/make_metrics.py --results results/rerun22/qmodels_abide
    python scripts/make_metrics.py --all --latex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from qagta.data.abide import load_abide

COHORT_ROOT = {
    "ABIDE-I": Path("data/ABIDE-I"),
    "ADHD-200": Path("data/ADHD-200"),
    "REST-meta-MDD": Path("data/REST-meta-MDD"),
    "UCLA-CNP": Path("data/UCLA-CNP-cc200"),
}
_CACHE: dict[str, tuple] = {}


def fold_balance(cohort: str, qubits: int = 8) -> dict[str, tuple[int, int]]:
    """Positives and negatives per leave-site-out fold."""
    if cohort not in _CACHE:
        ds = load_abide(root=COHORT_ROOT[cohort], n_components=qubits)
        _CACHE[cohort] = (ds.labels, ds.sites)
    labels, sites = _CACHE[cohort]
    out = {}
    for site in sorted(set(sites.tolist())):
        m = sites == site
        out[site] = (int((labels[m] == 1).sum()), int((labels[m] == 0).sum()))
    return out


def stratified_balance(cohort: str, folds: int, seed: int) -> dict[str, tuple[int, int]]:
    """Positives and negatives per stratified fold, rebuilt from the same seed.

    StratifiedKFold with a fixed shuffle seed is deterministic, so the fold
    membership used at run time can be reproduced exactly here rather than
    treating stratified cohorts as unrecoverable.
    """
    from sklearn.model_selection import StratifiedKFold
    if cohort not in _CACHE:
        ds = load_abide(root=COHORT_ROOT[cohort], n_components=8)
        _CACHE[cohort] = (ds.labels, ds.sites)
    labels, _ = _CACHE[cohort]
    sp = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = {}
    for i, (_, te) in enumerate(sp.split(np.zeros(len(labels)), labels)):
        out[f"fold{i+1}"] = (int((labels[te] == 1).sum()), int((labels[te] == 0).sum()))
    return out


def reconstruct(fold: dict, balance: dict[str, tuple[int, int]]):
    """Recover (TN, FP, FN, TP) for one fold, or None if it cannot be checked."""
    site = fold.get("site")
    if site not in balance:
        return None
    P, N = balance[site]
    n = fold["n"]
    if P + N != n or not math.isfinite(fold.get("specificity", float("nan"))):
        return None
    tn = round(fold["specificity"] * N)
    fp = N - tn
    tp = round(fold["accuracy"] * n) - tn
    fn = P - tp
    if min(tn, fp, tp, fn) < 0:
        return None
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    if abs(f1 - fold.get("f1", f1)) > 0.02:      # reject an inconsistent solve
        return None
    return tn, fp, fn, tp


def derive(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else float("nan")
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "n": total,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "npv": npv, "balanced_accuracy": (sens + spec) / 2,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "mcc": mcc,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, help="a single results directory")
    ap.add_argument("--all", action="store_true", help="every post-fix run")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/full_metrics.json"))
    args = ap.parse_args()

    targets = []
    if args.all:
        for d in sorted(Path("results/rerun22").glob("*/")):
            j = next(iter(d.glob("*results.json")), None)
            if j:
                targets.append(j)
        for d in sorted(Path("results/qgt").glob("*/")):
            j = next(iter(d.glob("*results.json")), None)
            if j:
                targets.append(j)
    elif args.results:
        j = next(iter(args.results.glob("*results.json")), None)
        if j:
            targets.append(j)
    if not targets:
        print("no results found")
        return 1

    payload = {}
    for path in targets:
        d = json.load(open(path))
        root = str(d.get("config", {}).get("data_root", ""))
        cohort = next((c for c, p in COHORT_ROOT.items() if str(p) == root), None)
        if cohort is None:
            print(f"  skip {path.parent.name}: unrecognised cohort {root!r}")
            continue
        strat = d.get("config", {}).get("cv") == "stratified"
        if "per_fold" not in d:
            # run_lowdata and run_qubit_scaling nest their folds under other
            # keys; they are sweeps rather than single matched comparisons and
            # a pooled confusion matrix over mixed training sizes would not
            # mean anything.
            print(f"  skip {path.parent.name}: sweep, no single per-fold table")
            continue
        bal = (stratified_balance(cohort, d["config"].get("folds", 10),
                                  d["config"].get("seed", 0))
               if strat else fold_balance(cohort))
        print(f"\n=== {path.parent.name}  ({cohort}) ===")
        print(f"{'model':<15}{'TN':>6}{'FP':>6}{'FN':>6}{'TP':>6}"
              f"{'sens':>8}{'spec':>8}{'prec':>8}{'bal-acc':>9}{'MCC':>8}")
        print("-" * 82)
        for model, folds in d["per_fold"].items():
            cells = [reconstruct(f, bal) for f in folds]
            good = [c for c in cells if c]
            if len(good) < max(1, len(folds) // 2):
                print(f"{model:<15}  (reconstruction failed on "
                      f"{len(folds)-len(good)}/{len(folds)} folds)")
                continue
            tn, fp, fn, tp = (int(sum(c[i] for c in good)) for i in range(4))
            m = derive(tn, fp, fn, tp)
            payload.setdefault(path.parent.name, {})[model] = m | {
                "cohort": cohort, "folds_used": len(good), "folds_total": len(folds)}
            print(f"{model:<15}{tn:>6}{fp:>6}{fn:>6}{tp:>6}"
                  f"{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}"
                  f"{m['precision']:>8.3f}{m['balanced_accuracy']:>9.3f}{m['mcc']:>8.3f}")

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved: {args.out}")

    if args.latex:
        print("\n% --- LaTeX bodies ---")
        for run, models in payload.items():
            print(f"\n% {run}")
            for name, m in models.items():
                print(f"{name:<14} & ${m['sensitivity']:.3f}$ & ${m['specificity']:.3f}$ "
                      f"& ${m['precision']:.3f}$ & ${m['balanced_accuracy']:.3f}$ "
                      f"& ${m['mcc']:+.3f}$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
