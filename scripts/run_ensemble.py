#!/usr/bin/env python3
"""Plan B: quantum ensembles over disjoint feature blocks, against classical twins.

The register caps one circuit at ``n_qubits`` features. This trains ``k``
narrow models on ``k`` disjoint blocks and combines them, so the ensemble sees
``k * n_qubits`` features while every circuit stays narrow.

Comparators, in increasing order of what they control for:

    Q-Ensemble    k quantum models, one per block
    C-Ensemble    k classical models over the same blocks, same combiner
    C-Single      one classical model given all k * n_qubits features at once

C-Ensemble isolates the model: same partition, same combination, quantum
against classical. C-Single is the reference ceiling — it receives the same
information without paying the block partition, so it bounds what the
partitioning costs. A quantum ensemble that beats C-Ensemble but not C-Single
has gained from ensembling, not from being quantum.

Examples:
    python scripts/run_ensemble.py --data-root data/ABIDE-I --blocks 4
    python scripts/run_ensemble.py --data-root data/UCLA-CNP-cc200 --cv stratified
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.ensemble import BlockEnsemble, make_blocks
from qagta.quantum.qcnn import QCNNClassifier


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


def make_classical_member(n_features: int, hidden: int = 16) -> torch.nn.Module:
    """Per-block classical member, deliberately small to match a narrow circuit."""
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden), torch.nn.ReLU(),
        torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 2),
    )


def train_ensemble(model, blocks_train, y_train, blocks_test, epochs, lr, seed,
                   batch_size=128):
    torch.manual_seed(seed)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    counts = torch.bincount(yt, minlength=2).float()
    weight = (counts.sum() / (2 * counts.clamp_min(1))).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(yt)

    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            batch = [b[idx] for b in blocks_train]
            loss = F.cross_entropy(model(batch), yt[idx], weight=weight)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(blocks_test)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return logits.argmax(1).numpy(), probs


def train_single(model, x_train, y_train, x_test, epochs, lr, seed, batch_size=128):
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
            F.cross_entropy(model(xt[idx]), yt[idx], weight=weight).backward()
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
    parser.add_argument("--qubits", type=int, default=8, help="features per block")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--cv", default="leave-site-out",
                        choices=["leave-site-out", "stratified"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--min-test-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_features = args.blocks * args.qubits
    print(f"ensemble study (Plan B) -> {out_dir}")
    print(f"{args.blocks} blocks x {args.qubits} features = {total_features} total\n")

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

    names = ["Q-Ensemble", "C-Ensemble", "C-Single"]
    per_fold: dict[str, list[dict]] = {n: [] for n in names}
    timing: dict[str, float] = dict.fromkeys(names, 0.0)
    weights_log: list[list[float]] = []

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

        ranked = select_features(connectivity[train_idx], y_train, total_features)
        blocks = make_blocks(ranked, args.blocks, args.qubits)
        if len(blocks) < args.blocks:
            print(f"  skipping {site}: only {len(blocks)} full blocks available")
            continue

        # Angles for the quantum members, standardised features for classical.
        q_train, q_test, c_train, c_test = [], [], [], []
        for block in blocks:
            scaler = StandardScaler().fit(connectivity[train_idx][:, block])
            tr = scaler.transform(connectivity[train_idx][:, block])
            te = scaler.transform(connectivity[test_idx][:, block])
            c_train.append(torch.as_tensor(tr, dtype=torch.float32))
            c_test.append(torch.as_tensor(te, dtype=torch.float32))

            # FINDING 22: [0, pi] spans a full phase period in RZ(2x); use [0, pi/2].
            angle = MinMaxScaler((0, np.pi / 2)).fit(tr)
            q_train.append(torch.as_tensor(angle.transform(tr), dtype=torch.float32))
            q_test.append(
                torch.as_tensor(np.clip(angle.transform(te), 0, np.pi / 2),
                                dtype=torch.float32)
            )

        # --- quantum ensemble ---------------------------------------------
        t0 = time.perf_counter()
        model = BlockEnsemble(
            [QCNNClassifier(n_qubits=args.qubits, seed=args.seed + i)
             for i in range(args.blocks)]
        )
        preds, scores = train_ensemble(model, q_train, y_train, q_test,
                                       args.epochs, args.lr, args.seed)
        per_fold["Q-Ensemble"].append(
            {"site": site, "n": len(test_idx), **metrics(y_test, preds, scores)}
        )
        timing["Q-Ensemble"] += time.perf_counter() - t0
        weights_log.append([float(w) for w in model.member_weights])

        # --- classical ensemble over identical blocks ----------------------
        t0 = time.perf_counter()
        model = BlockEnsemble([make_classical_member(args.qubits)
                               for _ in range(args.blocks)])
        preds, scores = train_ensemble(model, c_train, y_train, c_test,
                                       args.epochs, args.lr, args.seed)
        per_fold["C-Ensemble"].append(
            {"site": site, "n": len(test_idx), **metrics(y_test, preds, scores)}
        )
        timing["C-Ensemble"] += time.perf_counter() - t0

        # --- classical single model on all features at once ----------------
        t0 = time.perf_counter()
        flat_train = torch.cat(c_train, dim=1).numpy()
        flat_test = torch.cat(c_test, dim=1).numpy()
        single = torch.nn.Sequential(
            torch.nn.Linear(total_features, 32), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(32, 2),
        )
        preds, scores = train_single(single, flat_train, y_train, flat_test,
                                     args.epochs, args.lr, args.seed)
        per_fold["C-Single"].append(
            {"site": site, "n": len(test_idx), **metrics(y_test, preds, scores)}
        )
        timing["C-Single"] += time.perf_counter() - t0

        print(f"  {site:<12} n={len(test_idx):<4} " + "  ".join(
            f"{n}={per_fold[n][-1]['accuracy']:.3f}" for n in names), flush=True)

    def summarise(folds, key):
        values = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 78)
    print(f"PLAN B — {args.blocks} disjoint blocks of {args.qubits} features")
    print("=" * 78)
    header = f"{'model':<14}{'accuracy':>15}{'F1':>15}{'AUC':>15}{'sec/fold':>10}"
    print(header + "\n" + "-" * len(header))
    for name in names:
        if not per_fold[name]:
            continue
        row = f"{name:<14}"
        for key in ("accuracy", "f1", "auc"):
            mean, half = summarise(per_fold[name], key)
            row += f"{mean:>9.3f}+-{half:<5.3f}"
        row += f"{timing[name] / max(len(per_fold[name]), 1):>10.1f}"
        print(row)

    if weights_log:
        mean_weights = np.mean(weights_log, axis=0)
        print("\nmean learned block weight (strongest features first): "
              + "  ".join(f"{w:.3f}" for w in mean_weights))

    tests = {}
    print()
    for a, b in [("Q-Ensemble", "C-Ensemble"), ("Q-Ensemble", "C-Single")]:
        if not per_fold[a] or not per_fold[b]:
            continue
        qa = np.array([f["accuracy"] for f in per_fold[a]])
        cb = np.array([f["accuracy"] for f in per_fold[b]])
        if len(qa) > 1 and np.any(qa != cb):
            _, p = wilcoxon(qa, cb)
            tests[f"{a} vs {b}"] = {
                "median_diff": float(np.median(qa - cb)), "p_value": float(p),
                "wins": int((qa > cb).sum()), "folds": int(len(qa)),
            }
            print(f"{a} vs {b}: median dAcc = {np.median(qa - cb):+.3f}, "
                  f"wins {int((qa > cb).sum())}/{len(qa)}, p = {p:.4f}")

    payload = {
        "config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir)},
        "n_subjects": len(dataset),
        "summary": {n: {m: summarise(per_fold[n], m) for m in ("accuracy", "f1", "auc")}
                    for n in names if per_fold[n]},
        "seconds_per_fold": {n: timing[n] / max(len(per_fold[n]), 1)
                             for n in names if per_fold[n]},
        "mean_block_weights": (np.mean(weights_log, axis=0).tolist()
                               if weights_log else []),
        "per_fold": per_fold, "paired_tests": tests, "timestamp": stamp,
    }
    (out_dir / "ensemble_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'ensemble_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
