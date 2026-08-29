#!/usr/bin/env python3
"""Does the quantum arm overtake classical when training data becomes scarce?

FINDING 18 measured a quantum advantage on one quantity -- resistance to
overfitting -- whose margin grows monotonically as the cohort shrinks
($0.022$ at n=2428 up to $0.181$ at n=226), and at the smallest cohort the
quantum arms have both the smallest overfit gap and the nominally best
accuracy. That is a falsifiable prediction: if the trend is real rather than a
property of UCLA-CNP specifically, then subsampling a *large* cohort down to a
few hundred subjects should reproduce it, and there should be a crossover
sample size below which quantum wins on accuracy too.

This tests that directly. Leave-site-out folds are kept exactly as elsewhere in
the study, so the test sets are unchanged; only the training pool is
subsampled, to a target size, several times per fold with different draws.

Interpretation is fixed in advance:

- A crossover that appears on BOTH cohorts at a similar n is a real
  low-data effect and the paper's one conditional quantum advantage.
- A crossover on one cohort only is noise, and is reported as such --
  FINDING 08 and FINDING 17 both record single-cohort positives that
  did not replicate.
- No crossover falsifies the FINDING 18 extrapolation, which is equally
  worth reporting: it would mean the UCLA-CNP result is about that cohort
  rather than about sample size.

Examples:
    python scripts/run_lowdata.py --data-root data/ABIDE-I
    python scripts/run_lowdata.py --data-root data/REST-meta-MDD --repeats 3
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.qcnn import QCNNClassifier


def select_features(x_train, y_train, k):
    a, b = x_train[y_train == 0], x_train[y_train == 1]
    if len(a) < 2 or len(b) < 2:
        return np.arange(min(k, x_train.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    return np.argsort(-np.nan_to_num(np.abs(t_stat)))[: min(k, x_train.shape[1])]


def build(name: str, n_features: int) -> torch.nn.Module:
    if name == "Quantum":
        return QCNNClassifier(n_qubits=n_features, seed=0)
    if name == "C-MLP":
        return torch.nn.Sequential(
            torch.nn.Linear(n_features, 32), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(32, 2))
    if name == "C-Linear":
        return torch.nn.Linear(n_features, 2)
    raise ValueError(name)


def fit(model, x_tr, y_tr, x_te, epochs, lr, seed, batch=128):
    torch.manual_seed(seed)
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.long)
    counts = torch.bincount(yt, minlength=2).float()
    w = (counts.sum() / (2 * counts.clamp_min(1))).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(xt))
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            F.cross_entropy(model(xt[idx]), yt[idx], weight=w).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = accuracy_score(y_tr, model(xt).argmax(1).numpy())
        logits = model(torch.as_tensor(x_te, dtype=torch.float32))
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return logits.argmax(1).numpy(), probs, float(tr_acc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent training-set draws per fold and size")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_lowdata"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"low-data crossover -> {out_dir}", flush=True)

    ds = load_abide(root=args.data_root, atlas=args.atlas, n_components=args.qubits)
    print(ds.summary(), flush=True)
    series = args.data_root / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    conn = build_descriptors(
        [np.loadtxt(series / f"{s.file_id}_{args.atlas}.1D") for s in ds.subjects],
        kind="correlation")
    labels, sites = ds.labels, ds.sites
    names = ["Quantum", "C-MLP", "C-Linear"]

    splits = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
              for s in sorted(set(sites.tolist()))]
    rows = []
    header = f"{'n_train':<9}" + "".join(f"{n:>11}" for n in names) + f"{'Q-best':>9}"
    print("\n" + header + "\n" + "-" * len(header), flush=True)

    for size in args.sizes:
        per_model = {n: [] for n in names}
        for site, tr_pool, te_idx in splits:
            if len(te_idx) < args.min_test_size or len(np.unique(labels[te_idx])) < 2:
                continue
            for rep in range(args.repeats):
                rng = np.random.default_rng(args.seed + 1000 * rep + hash(site) % 997)
                if len(tr_pool) < size:
                    continue
                tr_idx = rng.choice(tr_pool, size=size, replace=False)
                y_tr, y_te = labels[tr_idx], labels[te_idx]
                if len(np.unique(y_tr)) < 2:
                    continue
                chosen = select_features(conn[tr_idx], y_tr, args.qubits)
                sc = StandardScaler().fit(conn[tr_idx][:, chosen])
                c_tr, c_te = sc.transform(conn[tr_idx][:, chosen]), sc.transform(conn[te_idx][:, chosen])
                # FINDING 22: [0, pi] spans a full phase period in RZ(2x); use [0, pi/2].
                ang = MinMaxScaler((0, np.pi / 2)).fit(c_tr)
                q_tr = ang.transform(c_tr)
                q_te = np.clip(ang.transform(c_te), 0, np.pi / 2)
                for name in names:
                    xa, xb = (q_tr, q_te) if name == "Quantum" else (c_tr, c_te)
                    preds, probs, tr_acc = fit(build(name, args.qubits), xa, y_tr, xb,
                                               args.epochs, args.lr, args.seed + rep)
                    per_model[name].append({
                        "site": site, "rep": rep, "n_train": size,
                        "accuracy": float(accuracy_score(y_te, preds)),
                        "train_accuracy": tr_acc,
                        "auc": float(roc_auc_score(y_te, probs))
                        if len(np.unique(y_te)) > 1 else float("nan")})
        means = {n: float(np.mean([r["accuracy"] for r in per_model[n]])) for n in names}
        best = max(means, key=means.get)
        print(f"{size:<9}" + "".join(f"{means[n]:>11.3f}" for n in names)
              + f"{('YES' if best == 'Quantum' else 'no'):>9}", flush=True)
        rows.append({"n_train": size, "means": means,
                     "per_model": per_model, "best": best})
        (out_dir / "lowdata_results.json").write_text(json.dumps(
            {"config": vars(args) | {"data_root": str(args.data_root),
                                     "out": str(out_dir)},
             "rows": rows, "timestamp": stamp}, indent=2, default=str))

    print("\npaired tests, quantum against each classical arm, per training size")
    for r in rows:
        line = f"  n={r['n_train']:<5}"
        for c in ("C-MLP", "C-Linear"):
            q = np.array([x["accuracy"] for x in r["per_model"]["Quantum"]])
            k = np.array([x["accuracy"] for x in r["per_model"][c]])
            if len(q) > 1 and np.any(q != k):
                p = wilcoxon(q, k)[1]
                line += f"   vs {c}: {np.median(q-k):+.3f} (p={p:.3f})"
        print(line, flush=True)
    print(f"\nsaved: {out_dir / 'lowdata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
