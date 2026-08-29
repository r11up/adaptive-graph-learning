#!/usr/bin/env python3
"""Plan C: data re-uploading in the QCNN, against a plain QCNN and classical twins.

FINDING 16's qubit-scaling diagnostic showed the QCNN *underfitting*: training
accuracy never exceeded 0.65 at any register width, so the model was failing to
fit the training set before generalisation was even in question. Widening the
register did not fix it, and cost 163x the runtime for +0.017 test accuracy.

Data re-uploading (Perez-Salinas et al., Quantum 4:226, 2020) attacks capacity
instead of width. Re-injecting the features between convolution stages raises
the order of Fourier terms the circuit can express in the data, which a single
encoding at the input cannot reach. The register stays at ``n_qubits``.

The comparison is built so the answer is interpretable either way:

- **Q-Plain against Q-Reup.** Identical circuit, identical training, identical
  seeds; the only difference is the re-injection. This isolates re-uploading
  itself, and is the question data-reuploading theory actually makes a claim
  about.
- **Train accuracy is recorded alongside test.** If re-uploading raises train
  accuracy it has added usable capacity, which is the mechanism claimed. If it
  raises train but not test, the added capacity does not generalise. Those are
  different results and reporting only test accuracy cannot distinguish them.
- **Matched classical heads** on the same features, same optimiser, epochs,
  learning rate, class weighting and seeds, so a quantum gain over Q-Plain can
  still be checked against what a classical model of similar size already does.

A note on what is re-injected: the schedule halves the active register at each
pooling stage, so stage 1 re-injects only the features on the surviving qubits.
Re-injecting a *learned mixture* of all features instead would be more
expressive, but that is Plan A's learned projection, which FINDING 14 measured
as significantly worse. This keeps to the literal re-uploading construction.

Examples:
    python scripts/run_reupload.py --data-root data/UCLA-CNP-cc200 --cv stratified
    python scripts/run_reupload.py --data-root data/ABIDE-I
"""

from __future__ import annotations

import argparse
import json
import math
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


class Classical1DCNN(torch.nn.Module):
    """Matched-capacity classical CNN, as used throughout the model suite."""

    def __init__(self, n_features: int, channels: int = 8) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, channels, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv1d(channels, channels, kernel_size=3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
            torch.nn.Linear(channels, 2),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def train_model(model, x_train, y_train, x_test, epochs, lr, seed, batch_size=128):
    """One training routine for every arm, so no model gets a different budget.

    Returns test predictions and scores plus *train* accuracy, because the
    question Plan C asks is whether re-uploading fixes underfitting, and that
    is invisible in test accuracy alone.
    """
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
        train_acc = accuracy_score(y_train, model(xt).argmax(1).numpy())
        logits = model(torch.as_tensor(x_test, dtype=torch.float32))
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return logits.argmax(1).numpy(), probs, float(train_acc)


def effective_parameters(n_qubits: int, reupload: int, seed: int) -> int:
    """Count parameters that actually reach the circuit, not those allocated.

    ``reupload_scale`` is allocated as ``(reupload, n_qubits)``, but the
    register halves at every pooling stage, so a re-upload at stage k touches
    only the qubits still active there. At eight qubits, depth 1 uses four of
    its eight gains and depth 2 uses six of sixteen; the rest never receive
    gradient and stay at their initial value. Reporting the allocated shape
    would overstate the model's size, so this probes which entries are live.
    """
    model = QCNNClassifier(n_qubits=n_qubits, seed=seed, reupload=reupload)
    x = torch.rand(8, n_qubits) * math.pi
    F.cross_entropy(model(x), torch.tensor([0, 1] * 4)).backward()
    return sum(int((p.grad.abs() > 0).sum()) if p.grad is not None else 0
               for p in model.parameters())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2],
                        help="re-upload depths to compare against the plain QCNN")
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
    out_dir = args.out or Path("results") / f"{stamp}_reupload"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"re-uploading study (Plan C) -> {out_dir}", flush=True)

    dataset = load_abide(
        root=args.data_root, pipeline=args.pipeline, strategy=args.strategy,
        atlas=args.atlas, n_components=args.qubits, limit=args.limit,
    )
    print(dataset.summary(), flush=True)

    series_dir = args.data_root / "ABIDE_pcp" / args.pipeline / args.strategy
    connectivity = build_descriptors(
        [np.loadtxt(series_dir / f"{s.file_id}_{args.atlas}.1D") for s in dataset.subjects],
        kind="correlation",
    )
    labels, sites = dataset.labels, dataset.sites

    # Quantum arms differ only in re-upload depth; classical arms are the
    # established matched comparators on the identical feature set.
    quantum_arms = {"Q-Plain": 0} | {f"Q-Reup{d}": d for d in args.depths}
    classical_builders = {
        "C-CNN": lambda n: Classical1DCNN(n),
        "C-MLP": lambda n: torch.nn.Sequential(
            torch.nn.Linear(n, 32), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(32, 2),
        ),
        "C-Linear": lambda n: torch.nn.Linear(n, 2),
    }
    names = list(quantum_arms) + list(classical_builders)
    per_fold: dict[str, list[dict]] = {n: [] for n in names}
    timing: dict[str, float] = dict.fromkeys(names, 0.0)

    counts = {n: effective_parameters(args.qubits, d, args.seed)
              for n, d in quantum_arms.items()}
    print("quantum trainable parameters: "
          + "  ".join(f"{n}={c}" for n, c in counts.items()) + "\n", flush=True)

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

        chosen = select_features(connectivity[train_idx], y_train, args.qubits)
        scaler = StandardScaler().fit(connectivity[train_idx][:, chosen])
        c_train = scaler.transform(connectivity[train_idx][:, chosen])
        c_test = scaler.transform(connectivity[test_idx][:, chosen])

        # FINDING 22: the ZFeatureMap/ZZFeatureMap encode a feature as RZ(2x),
        # so an input range of [0, pi] spans a full 2*pi phase period and the
        # two ends of every feature collapse onto the same state. Scaling to
        # [0, pi/2] keeps the phase injective. The circuit is unchanged.
        angle = MinMaxScaler((0, np.pi / 2)).fit(c_train)
        q_train = angle.transform(c_train)
        q_test = np.clip(angle.transform(c_test), 0, np.pi / 2)

        for name, depth in quantum_arms.items():
            t0 = time.perf_counter()
            preds, scores, train_acc = train_model(
                QCNNClassifier(n_qubits=args.qubits, seed=args.seed, reupload=depth),
                q_train, y_train, q_test, args.epochs, args.lr, args.seed,
            )
            per_fold[name].append({"site": site, "n": len(test_idx),
                                   "train_accuracy": train_acc,
                                   **metrics(y_test, preds, scores)})
            timing[name] += time.perf_counter() - t0

        for name, build in classical_builders.items():
            t0 = time.perf_counter()
            preds, scores, train_acc = train_model(
                build(args.qubits), c_train, y_train, c_test,
                args.epochs, args.lr, args.seed,
            )
            per_fold[name].append({"site": site, "n": len(test_idx),
                                   "train_accuracy": train_acc,
                                   **metrics(y_test, preds, scores)})
            timing[name] += time.perf_counter() - t0

        print(f"  {site:<12} n={len(test_idx):<4} " + "  ".join(
            f"{n}={per_fold[n][-1]['accuracy']:.3f}" for n in names), flush=True)

    def summarise(folds, key):
        values = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 86)
    print(f"PLAN C — data re-uploading at {args.qubits} qubits, depths {args.depths}")
    print("=" * 86)
    header = (f"{'model':<12}{'train':>9}{'accuracy':>15}{'F1':>15}"
              f"{'AUC':>15}{'gap':>9}{'sec/fold':>10}")
    print(header + "\n" + "-" * len(header))
    for name in names:
        if not per_fold[name]:
            continue
        train_mean, _ = summarise(per_fold[name], "train_accuracy")
        test_mean, _ = summarise(per_fold[name], "accuracy")
        row = f"{name:<12}{train_mean:>9.3f}"
        for key in ("accuracy", "f1", "auc"):
            mean, half = summarise(per_fold[name], key)
            row += f"{mean:>9.3f}+-{half:<5.3f}"
        row += f"{train_mean - test_mean:>+9.3f}"
        row += f"{timing[name] / max(len(per_fold[name]), 1):>10.1f}"
        print(row)

    # The within-quantum tests come first: they are the ones re-uploading
    # theory makes a claim about. The classical tests say whether any gain is
    # enough to matter.
    pairs = [(f"Q-Reup{d}", "Q-Plain") for d in args.depths]
    deepest = f"Q-Reup{max(args.depths)}"
    pairs += [(deepest, "C-CNN"), (deepest, "C-MLP"), (deepest, "C-Linear")]

    tests = {}
    print()
    for a, b in pairs:
        if not per_fold.get(a) or not per_fold.get(b):
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
        "quantum_parameter_counts": counts,
        "summary": {n: {m: summarise(per_fold[n], m)
                        for m in ("accuracy", "f1", "auc", "train_accuracy")}
                    for n in names if per_fold[n]},
        "seconds_per_fold": {n: timing[n] / max(len(per_fold[n]), 1)
                             for n in names if per_fold[n]},
        "per_fold": per_fold, "paired_tests": tests, "timestamp": stamp,
    }
    (out_dir / "reupload_results.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_dir / 'reupload_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
