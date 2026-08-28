#!/usr/bin/env python3
"""Plan A: learned projection into a quantum head, against matched classical twins.

FINDING 11 located the constraint on the quantum arm: at eight features the
quantum and classical models are indistinguishable, while the classical model
given all 2000 features is better. The register caps how much reaches the
circuit, so this replaces fixed feature selection with a projection learned
jointly with the circuit — the quantum stage sees an optimised summary of every
connection instead of eight raw ones.

The comparison is built so a quantum win cannot be an artefact:

- **Same projection for both arms.** Each quantum head has a classical twin
  with an identical learned projection. A projection feeding a classical
  network is an equally valid architecture; if the quantum arm only wins when
  the classical arm is denied it, that is a flaw in the comparison.
- **Same bottleneck width.** Both compress to n_qubits before the head.
- **Same training.** Identical optimiser, epochs, learning rate, class
  weighting and seeds.
- **A linear-head control.** It measures how much of any gain comes from the
  projection alone rather than from what follows it — without this, a hybrid
  gain could be entirely classical and still look quantum.
- **Fixed-selection baselines** carried through from the earlier suite, so the
  effect of learning the projection is separable from the effect of the head.

Examples:
    python scripts/run_hybrid.py --data-root data/ABIDE-I
    python scripts/run_hybrid.py --data-root data/UCLA-CNP-cc200 --cv stratified
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.hybrid import HybridClassical, HybridQCNN, HybridVQC


def select_features(x_train, y_train, k):
    a, b = x_train[y_train == 0], x_train[y_train == 1]
    if len(a) < 2 or len(b) < 2:
        return np.arange(min(k, x_train.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    return np.argsort(-np.nan_to_num(np.abs(t_stat)))[: min(k, x_train.shape[1])]


def metrics(y_true, y_pred, scores=None):
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


def train_model(model, x_train, y_train, x_test, epochs, lr, seed, batch_size=128):
    """One training routine for every model, so no arm gets a different budget."""
    torch.manual_seed(seed)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    counts = torch.bincount(yt, minlength=2).float()
    weight = (counts.sum() / (2 * counts.clamp_min(1))).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(xt))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xt[idx]), yt[idx], weight=weight)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x_test, dtype=torch.float32))
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return logits.argmax(1).numpy(), probs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2000,
                        help="features the projection sees; the circuit still gets n_qubits")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--cv", default="leave-site-out",
                        choices=["leave-site-out", "stratified"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--min-test-size", type=int, default=10)
    parser.add_argument("--no-warm-start", action="store_true",
                        help="random projection init instead of the fixed-selection solution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"hybrid study (Plan A) -> {out_dir}")

    dataset = load_abide(
        root=args.data_root, pipeline=args.pipeline, strategy=args.strategy,
        atlas=args.atlas, n_components=args.qubits, limit=args.limit,
    )
    print(dataset.summary())

    series_dir = args.data_root / "ABIDE_pcp" / args.pipeline / args.strategy
    connectivity = build_descriptors(
        [np.loadtxt(series_dir / f"{s.file_id}_{args.atlas}.1D") for s in dataset.subjects],
        kind="correlation",
    )
    labels, sites = dataset.labels, dataset.sites
    print(f"connectivity: {connectivity.shape}  ->  projection sees top-{args.top_k}\n")

    # Every arm gets the same warm start, so the comparison still isolates the
    # head rather than rewarding whichever side happened to initialise better.
    builders = {
        "H-QCNN": lambda d, w: HybridQCNN(d, args.qubits, seed=args.seed, warm_start=w),
        "H-VQC": lambda d, w: HybridVQC(d, args.qubits, seed=args.seed, warm_start=w),
        "H-MLP": lambda d, w: HybridClassical(d, args.qubits, "mlp",
                                              seed=args.seed, warm_start=w),
        "H-CNN": lambda d, w: HybridClassical(d, args.qubits, "cnn",
                                              seed=args.seed, warm_start=w),
        "H-Linear": lambda d, w: HybridClassical(d, args.qubits, "linear",
                                                 seed=args.seed, warm_start=w),
    }
    names = list(builders)
    per_fold: dict[str, list[dict]] = {n: [] for n in names}
    timing: dict[str, float] = dict.fromkeys(names, 0.0)

    if args.cv == "stratified":
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = [(f"fold{i+1}", tr, te) for i, (tr, te)
                  in enumerate(splitter.split(connectivity, labels))]
    else:
        splits = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
                  for s in sorted(set(sites.tolist()))]

    for site, train_idx, test_idx in splits:
        if len(test_idx) < args.min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Selection still runs, but only to bound the projection's input width;
        # the circuit's own budget is unchanged at n_qubits.
        chosen = select_features(connectivity[train_idx], y_train, args.top_k)
        scaler = StandardScaler().fit(connectivity[train_idx][:, chosen])
        x_train = scaler.transform(connectivity[train_idx][:, chosen])
        x_test = scaler.transform(connectivity[test_idx][:, chosen])

        # The projection is warm-started at the top-n_qubits selection, which
        # is exactly what the fixed-feature models used, so no arm starts worse
        # than that baseline.
        warm = np.arange(args.qubits) if not args.no_warm_start else None

        for name, build in builders.items():
            t0 = time.perf_counter()
            preds, scores = train_model(
                build(x_train.shape[1], warm), x_train, y_train, x_test,
                epochs=args.epochs, lr=args.lr, seed=args.seed,
            )
            per_fold[name].append(
                {"site": site, "n": len(test_idx), **metrics(y_test, preds, scores)}
            )
            timing[name] += time.perf_counter() - t0

        print(f"  {site:<12} n={len(test_idx):<4} " + "  ".join(
            f"{n}={per_fold[n][-1]['accuracy']:.3f}" for n in names), flush=True)

    def summarise(folds, key):
        values = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 78)
    print(f"PLAN A — learned projection ({args.top_k} -> {args.qubits}), shared by all arms")
    print("=" * 78)
    header = f"{'model':<12}{'accuracy':>15}{'F1':>15}{'AUC':>15}{'sec/fold':>10}"
    print(header + "\n" + "-" * len(header))
    for name in names:
        if not per_fold[name]:
            continue
        row = f"{name:<12}"
        for key in ("accuracy", "f1", "auc"):
            mean, half = summarise(per_fold[name], key)
            row += f"{mean:>9.3f}+-{half:<5.3f}"
        row += f"{timing[name] / max(len(per_fold[name]), 1):>10.1f}"
        print(row)

    pairs = [("H-QCNN", "H-CNN"), ("H-VQC", "H-MLP"),
             ("H-QCNN", "H-Linear"), ("H-VQC", "H-Linear")]
    tests = {}
    print()
    for quantum, classical in pairs:
        if not per_fold[quantum] or not per_fold[classical]:
            continue
        q = np.array([f["accuracy"] for f in per_fold[quantum]])
        c = np.array([f["accuracy"] for f in per_fold[classical]])
        if len(q) > 1 and np.any(q != c):
            _, p = wilcoxon(q, c)
            tests[f"{quantum} vs {classical}"] = {
                "median_diff": float(np.median(q - c)), "p_value": float(p),
                "wins": int((q > c).sum()), "folds": int(len(q)),
            }
            print(f"{quantum} vs {classical}: median dAcc = {np.median(q - c):+.3f}, "
                  f"wins {int((q > c).sum())}/{len(q)}, p = {p:.4f}")

    payload = {
        "config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir)},
        "n_subjects": len(dataset),
        "summary": {n: {m: summarise(per_fold[n], m) for m in ("accuracy", "f1", "auc")}
                    for n in names if per_fold[n]},
        "seconds_per_fold": {n: timing[n] / max(len(per_fold[n]), 1)
                             for n in names if per_fold[n]},
        "per_fold": per_fold, "paired_tests": tests, "timestamp": stamp,
    }
    (out_dir / "hybrid_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'hybrid_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
