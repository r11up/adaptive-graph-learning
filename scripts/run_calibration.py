#!/usr/bin/env python3
"""Does threshold calibration convert the quantum arm's ranking into accuracy?

FINDING 24 measured an asymmetry: across 284 matched comparisons the quantum
arm leads on AUC in 34 of 68 but on accuracy in only 24 of 72. AUC scores
ranking; accuracy scores ranking *and* where the decision threshold falls. A
QCNN reads out a bounded Pauli-Z expectation mapped to logits by a
two-parameter affine layer, which is a weaker calibration mechanism than a
classical network's final linear layer, so the quantum arm may rank well and
threshold badly.

This tests that directly. For every fold and every model the threshold is
chosen on the TRAINING scores only -- maximising Youden's J, the point that
best separates the two classes -- and then applied unchanged to the test
scores. No test information reaches the choice.

Two design points make the result interpretable:

- **Both arms are calibrated identically.** Calibrating only the quantum side
  would replace a matched comparison with an unmatched one, which is the
  failure mode Section "How an Advantage Appears" documents.
- **The uncalibrated result is reported alongside.** The question is not which
  arm is better after calibration but how much each arm *gains* from it. If the
  quantum arm gains more, the asymmetry was threshold placement; if both gain
  equally, it was not.

Examples:
    python scripts/run_calibration.py --data-root data/ABIDE-I
    python scripts/run_calibration.py --data-root data/UCLA-CNP-cc200 --cv stratified
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.qcnn import QCNNClassifier


def select_features(x, y, k):
    a, b = x[y == 0], x[y == 1]
    if len(a) < 2 or len(b) < 2:
        return np.arange(min(k, x.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    return np.argsort(-np.nan_to_num(np.abs(t)))[: min(k, x.shape[1])]


def youden_threshold(y_true, scores):
    """Threshold maximising sensitivity + specificity - 1, fit on training data."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def build(name, n):
    if name == "Quantum":
        return QCNNClassifier(n_qubits=n, seed=0)
    if name == "C-CNN":
        return torch.nn.Sequential(
            torch.nn.Conv1d(1, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv1d(8, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(8, 2))
    if name == "C-MLP":
        return torch.nn.Sequential(
            torch.nn.Linear(n, 32), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(32, 2))
    if name == "C-Linear":
        return torch.nn.Linear(n, 2)
    raise ValueError(name)


def fit(model, x_tr, y_tr, x_te, epochs, lr, seed, is_cnn=False, batch=128):
    torch.manual_seed(seed)
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.long)
    cnt = torch.bincount(yt, minlength=2).float()
    w = (cnt.sum() / (2 * cnt.clamp_min(1))).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    shape = (lambda t: t.unsqueeze(1)) if is_cnn else (lambda t: t)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(xt))
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            F.cross_entropy(model(shape(xt[idx])), yt[idx], weight=w).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        tr = F.softmax(model(shape(xt)), dim=1)[:, 1].numpy()
        xe = torch.as_tensor(x_te, dtype=torch.float32)
        te = F.softmax(model(shape(xe)), dim=1)[:, 1].numpy()
    return tr, te


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    ap.add_argument("--atlas", default="rois_cc200")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--cv", default="leave-site-out", choices=["leave-site-out","stratified"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--min-test-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"threshold calibration -> {out_dir}", flush=True)

    ds = load_abide(root=args.data_root, atlas=args.atlas, n_components=args.qubits)
    print(ds.summary(), flush=True)
    series = args.data_root / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    conn = build_descriptors(
        [np.loadtxt(series / f"{s.file_id}_{args.atlas}.1D") for s in ds.subjects],
        kind="correlation")
    labels, sites = ds.labels, ds.sites
    names = ["Quantum", "C-CNN", "C-MLP", "C-Linear"]
    per_fold = {n: [] for n in names}

    if args.cv == "stratified":
        sp = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = [(f"fold{i+1}", a, b) for i,(a,b) in enumerate(sp.split(conn, labels))]
    else:
        splits = [(s, np.where(sites != s)[0], np.where(sites == s)[0])
                  for s in sorted(set(sites.tolist()))]

    for site, tr, te in splits:
        if len(te) < args.min_test_size or len(np.unique(labels[te])) < 2:
            continue
        y_tr, y_te = labels[tr], labels[te]
        ch = select_features(conn[tr], y_tr, args.qubits)
        sc = StandardScaler().fit(conn[tr][:, ch])
        c_tr = sc.transform(conn[tr][:, ch])
        c_te = sc.transform(conn[te][:, ch])
        ang = MinMaxScaler((0, np.pi / 2)).fit(c_tr)
        q_tr = ang.transform(c_tr)
        q_te = np.clip(ang.transform(c_te), 0, np.pi / 2)

        for name in names:
            xa, xb = (q_tr, q_te) if name == "Quantum" else (c_tr, c_te)
            s_tr, s_te = fit(build(name, args.qubits), xa, y_tr, xb,
                             args.epochs, args.lr, args.seed, is_cnn=(name=="C-CNN"))
            thr = youden_threshold(y_tr, s_tr)
            per_fold[name].append({
                "site": site, "n": len(te), "threshold": thr,
                "acc_default":    float(accuracy_score(y_te, (s_te >= 0.5).astype(int))),
                "acc_calibrated": float(accuracy_score(y_te, (s_te >= thr).astype(int))),
                "f1_default":     float(f1_score(y_te, (s_te >= 0.5).astype(int), zero_division=0)),
                "f1_calibrated":  float(f1_score(y_te, (s_te >= thr).astype(int), zero_division=0)),
                "auc": (float(roc_auc_score(y_te, s_te))
                        if len(np.unique(y_te)) > 1 else float("nan")),
            })
        print(f"  {site:<12} n={len(te):<4} " + "  ".join(
            f"{n}={per_fold[n][-1]['acc_default']:.3f}->{per_fold[n][-1]['acc_calibrated']:.3f}"
            for n in names), flush=True)

    print("\n" + "="*78)
    print("THRESHOLD CALIBRATION — default 0.5 against a Youden threshold fit on train")
    print("="*78)
    print(f"{'model':<12}{'AUC':>9}{'acc @0.5':>11}{'acc calib':>11}{'gain':>9}{'p':>9}")
    print("-"*61)
    summary = {}
    for n in names:
        d = np.array([f["acc_default"] for f in per_fold[n]])
        c = np.array([f["acc_calibrated"] for f in per_fold[n]])
        a = np.nanmean([f["auc"] for f in per_fold[n]])
        p = wilcoxon(c, d)[1] if np.any(c != d) else float("nan")
        summary[n] = {"auc": float(a), "acc_default": float(d.mean()),
                      "acc_calibrated": float(c.mean()), "gain": float(c.mean()-d.mean()),
                      "p": float(p)}
        print(f"{n:<12}{a:>9.3f}{d.mean():>11.3f}{c.mean():>11.3f}{c.mean()-d.mean():>+9.3f}{p:>9.4f}")

    print("\nquantum against each classical arm, calibrated accuracy:")
    q = np.array([f["acc_calibrated"] for f in per_fold["Quantum"]])
    for c_name in names[1:]:
        k = np.array([f["acc_calibrated"] for f in per_fold[c_name]])
        if np.any(q != k):
            print(f"  vs {c_name:<10}{np.median(q-k):+.3f}  p={wilcoxon(q,k)[1]:.4f}")

    (out_dir / "calibration_results.json").write_text(json.dumps(
        {"config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir)},
         "summary": summary, "per_fold": per_fold, "timestamp": stamp}, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'calibration_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
