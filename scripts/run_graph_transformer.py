#!/usr/bin/env python3
"""Experiment 4: quantum attention against classical attention, all else equal.

Every quantum construct evaluated earlier scores a region pair by the overlap
of two separately prepared states, which concentrates as the register widens
(FINDING 01) and cannot be trained through a post-encoding ansatz
(FINDING 19). A graph transformer encodes query and key into one register and
lets them interact before measurement, so neither limit applies.

The comparison swaps only the attention module. Projection, value transform,
aggregation, read-out, classifier, graph, folds, optimiser and seeds are
identical, so any difference is attributable to the attention mechanism.

Expectation, recorded before running: FINDING 06 measured node features as the
binding constraint on the graph path, and every read-out and topology variant
tested since has stayed at chance. Quantum attention addresses neither. A
result at chance for both arms is the likely outcome and is still informative,
because it closes the standing objection that only older quantum constructs
were tried.

Examples:
    python scripts/run_graph_transformer.py --data-root data/UCLA-CNP-cc200 --cv stratified
    python scripts/run_graph_transformer.py --data-root data/ABIDE-I
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
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from qagta.data.abide import load_abide
from qagta.quantum.graph_transformer import GraphTransformer


def knn_edges(features: np.ndarray, k: int) -> torch.Tensor:
    """k-nearest-neighbour edges from mean node features, as the baseline uses."""
    mean = features.mean(axis=0)
    d = ((mean[:, None, :] - mean[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1)[:, :k]
    src = nbr.reshape(-1)
    dst = np.repeat(np.arange(len(mean)), k)
    # Self-loops: without them a node's own features reach its representation
    # only after mixing with neighbours', which dilutes any localised signal.
    loops = np.arange(len(mean))
    src = np.concatenate([src, loops])
    dst = np.concatenate([dst, loops])
    return torch.as_tensor(np.stack([src, dst]), dtype=torch.long)


def metrics(y, pred, score=None):
    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"accuracy": float(accuracy_score(y, pred)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan")}
    if score is not None and len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, score))
    return out


def run_fold(kind, feats, labels, tr, te, edges, args):
    torch.manual_seed(args.seed)
    model = GraphTransformer(feats.shape[2], node_qubits=args.node_qubits,
                             attention=kind, seed=args.seed, n_roi=feats.shape[1])
    xt = torch.as_tensor(feats[tr], dtype=torch.float32)
    yt = torch.as_tensor(labels[tr], dtype=torch.long)
    counts = torch.bincount(yt, minlength=2).float()
    w = (counts.sum() / (2 * counts.clamp_min(1))).float()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    model.train()
    for _ in range(args.epochs):
        order = rng.permutation(len(xt))
        for s in range(0, len(order), args.batch):
            idx = order[s : s + args.batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            F.cross_entropy(model(xt[idx], edges), yt[idx], weight=w).backward()
            opt.step()

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        xe = torch.as_tensor(feats[te], dtype=torch.float32)
        for s in range(0, len(xe), args.batch):
            out = model(xe[s : s + args.batch], edges)
            preds.append(out.argmax(1).numpy())
            probs.append(F.softmax(out, dim=1)[:, 1].numpy())
    return np.concatenate(preds), np.concatenate(probs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--node-qubits", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cv", default="leave-site-out",
                    choices=["leave-site-out", "stratified"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_qgt"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"graph transformer study -> {out_dir}", flush=True)

    ds = load_abide(root=args.data_root, atlas=args.atlas,
                    n_components=args.features, limit=args.limit)
    print(ds.summary(), flush=True)
    feats, labels, sites = ds.features, ds.labels, ds.sites
    edges = knn_edges(feats, args.k)
    print(f"graph: {feats.shape[1]} nodes, k={args.k}, "
          f"{edges.shape[1]} edges | {2*args.node_qubits} qubits per pair\n", flush=True)

    names = ["Q-Attention", "C-Attention"]
    per_fold = {n: [] for n in names}
    timing = dict.fromkeys(names, 0.0)

    if args.cv == "stratified":
        sp = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = [(f"fold{i+1}", a, b) for i, (a, b) in enumerate(sp.split(feats, labels))]
    else:
        splits = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
                  for s in sorted(set(sites.tolist()))]

    for site, tr, te in splits:
        if len(te) < args.min_test_size or len(np.unique(labels[te])) < 2:
            continue
        for kind, name in (("quantum", "Q-Attention"), ("classical", "C-Attention")):
            t0 = time.perf_counter()
            preds, probs = run_fold(kind, feats, labels, tr, te, edges, args)
            per_fold[name].append({"site": site, "n": len(te),
                                   **metrics(labels[te], preds, probs)})
            timing[name] += time.perf_counter() - t0
        print(f"  {site:<12} n={len(te):<4} " + "  ".join(
            f"{n}={per_fold[n][-1]['accuracy']:.3f}" for n in names), flush=True)
        (out_dir / "qgt_results.json").write_text(json.dumps(
            {"config": vars(args) | {"data_root": str(args.data_root),
                                     "out": str(out_dir)},
             "per_fold": per_fold, "timestamp": stamp}, indent=2, default=str))

    def summarise(folds, key):
        v = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if not len(v):
            return float("nan"), float("nan")
        return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    print("\n" + "=" * 72)
    print(f"EXPERIMENT 4 — quantum vs classical attention, {2*args.node_qubits} qubits per pair")
    print("=" * 72)
    hdr = f"{'model':<14}{'accuracy':>15}{'F1':>15}{'AUC':>15}{'s/fold':>10}"
    print(hdr + "\n" + "-" * len(hdr))
    for n in names:
        if not per_fold[n]:
            continue
        row = f"{n:<14}"
        for key in ("accuracy", "f1", "auc"):
            m, h = summarise(per_fold[n], key)
            row += f"{m:>9.3f}+-{h:<5.3f}"
        print(row + f"{timing[n]/max(len(per_fold[n]),1):>10.1f}")

    q = np.array([f["accuracy"] for f in per_fold["Q-Attention"]])
    c = np.array([f["accuracy"] for f in per_fold["C-Attention"]])
    if len(q) > 1 and np.any(q != c):
        p = wilcoxon(q, c)[1]
        print(f"\nQ-Attention vs C-Attention: median dAcc = {np.median(q-c):+.3f}, "
              f"wins {int((q>c).sum())}/{len(q)}, p = {p:.4f}")
    print(f"\nsaved: {out_dir / 'qgt_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
