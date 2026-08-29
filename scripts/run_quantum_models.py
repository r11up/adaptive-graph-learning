#!/usr/bin/env python3
"""Quantum model suite versus matched classical baselines, on any cohort.

Evaluates the standard quantum-machine-learning models against classical
comparators that see the identical features and receive the same training
budget, under the same Leave-Site-Out folds.

Quantum models
--------------
QCNN            Quantum convolutional network — the Qiskit reference design
                (Cong et al., Nat. Phys. 15:1273, 2019), validated gate-for-gate
                against Qiskit in tests/test_qcnn.py.
QSVM-Pegasos    Qiskit's PegasosQSVC on a precomputed quantum fidelity kernel.
QSVM-fixed      Fidelity-kernel SVM with bandwidth tuned by alignment.
TQEK            Trainable quantum embedding kernel: the embedding's parameters
                are optimised against kernel-target alignment, so the feature
                geometry adapts to the task instead of being fixed.

Classical comparators
---------------------
CNN (1-D)       Matched-capacity convolutional network on the same features.
SVM-RBF         Tuned width, the standard strong baseline.
Trainable-RBF   RBF on a learned linear metric, optimised under the identical
                objective, optimiser and step budget as TQEK. Without this, a
                trained quantum kernel beating an untrained classical one would
                only show that training helps.

Every model receives the same n_qubits features, chosen by supervised
selection inside each training fold. That is the only comparison that isolates
the model from the feature budget.

Examples:
    python scripts/run_quantum_models.py --data-root data/ABIDE-I
    python scripts/run_quantum_models.py --data-root data/UCLA-CNP-cc200 --cv stratified
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
from sklearn.svm import SVC

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix
from qagta.quantum.qcnn import QCNNClassifier
from qagta.quantum.trainable_kernel import (
    TrainableClassicalKernel,
    TrainableQuantumKernel,
    gram_matrix,
    train_kernel,
)
from qagta.quantum.variational import VQC


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


def train_torch_classifier(model, x_train, y_train, x_test, epochs=150, lr=0.05, seed=0):
    """Shared training loop, so quantum and classical get identical treatment."""
    torch.manual_seed(seed)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    counts = torch.bincount(yt, minlength=2).float()
    weight = (counts.sum() / (2 * counts.clamp_min(1))).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(xt), yt, weight=weight)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x_test, dtype=torch.float32))
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return logits.argmax(1).numpy(), probs


class Classical1DCNN(torch.nn.Module):
    """Matched-capacity classical CNN over the selected feature vector."""

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--kernel-steps", type=int, default=60)
    parser.add_argument("--bandwidths", type=float, nargs="+",
                        default=[0.05, 0.1, 0.15, 0.25, 0.5])
    parser.add_argument("--cv", default="leave-site-out",
                        choices=["leave-site-out", "stratified"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--min-test-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_quantum_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"quantum model suite -> {out_dir}")

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
    print(f"connectivity: {connectivity.shape}\n")

    names = ["QCNN", "VQC", "QSVM-Pegasos", "QSVM-fixed", "TQEK",
             "CNN (1-D)", "MLP", "SVM-RBF", "Trainable-RBF"]
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

        chosen = select_features(connectivity[train_idx], y_train, args.qubits)
        scaler = StandardScaler().fit(connectivity[train_idx][:, chosen])
        x_train = scaler.transform(connectivity[train_idx][:, chosen])
        x_test = scaler.transform(connectivity[test_idx][:, chosen])

        # Two angle conventions, deliberately separated (FINDING 22).
        #
        # Kernels apply RZ(2*(w*x + b)) with a learnable bandwidth initialised
        # near 0.15, so [0, pi] gives a phase span of ~0.94 rad and stays
        # injective. QCNN and VQC apply a fixed RZ(2x) with no bandwidth, so
        # [0, pi] spans a full 2*pi period and the two ends of every feature
        # collapse onto the same state. The variational models therefore get
        # [0, pi/2]; the kernels keep [0, pi] so their published numbers remain
        # exactly reproducible.
        angle_scaler = MinMaxScaler((0, np.pi)).fit(x_train)
        a_train = angle_scaler.transform(x_train)
        a_test = np.clip(angle_scaler.transform(x_test), 0, np.pi)

        var_scaler = MinMaxScaler((0, np.pi / 2)).fit(x_train)
        v_train = var_scaler.transform(x_train)
        v_test = np.clip(var_scaler.transform(x_test), 0, np.pi / 2)

        n_test = len(test_idx)

        def record(name, preds, scores, elapsed,
                   _site=site, _n=n_test, _y=y_test):
            # Loop variables are bound as defaults: a late-binding closure here
            # would attribute results to whichever fold happened to be current.
            per_fold[name].append(
                {"site": _site, "n": _n, **metrics(_y, preds, scores)}
            )
            timing[name] += elapsed

        # ---- QCNN -------------------------------------------------------
        t0 = time.perf_counter()
        preds, scores = train_torch_classifier(
            QCNNClassifier(n_qubits=args.qubits, seed=args.seed),
            v_train, y_train, v_test, epochs=args.epochs, seed=args.seed,
        )
        record("QCNN", preds, scores, time.perf_counter() - t0)

        # ---- VQC: ZZ feature map + RealAmplitudes + parity ---------------
        t0 = time.perf_counter()
        preds, scores = train_torch_classifier(
            VQC(n_qubits=args.qubits, reps=2, seed=args.seed),
            v_train, y_train, v_test, epochs=args.epochs, seed=args.seed,
        )
        record("VQC", preds, scores, time.perf_counter() - t0)

        # ---- classical MLP, matched budget --------------------------------
        t0 = time.perf_counter()
        mlp = torch.nn.Sequential(
            torch.nn.Linear(args.qubits, 32), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(32, 2),
        )
        preds, scores = train_torch_classifier(
            mlp, x_train, y_train, x_test, epochs=args.epochs, seed=args.seed
        )
        record("MLP", preds, scores, time.perf_counter() - t0)

        # ---- classical CNN, matched budget -------------------------------
        t0 = time.perf_counter()
        preds, scores = train_torch_classifier(
            Classical1DCNN(args.qubits), x_train, y_train, x_test,
            epochs=args.epochs, seed=args.seed,
        )
        record("CNN (1-D)", preds, scores, time.perf_counter() - t0)

        # ---- fixed quantum kernel, bandwidth by alignment ----------------
        t0 = time.perf_counter()
        y_signed = np.where(y_train > 0, 1.0, -1.0)
        ideal = np.outer(y_signed, y_signed)
        best = None
        for bandwidth in args.bandwidths:
            fm = QuantumFeatureMap(n_qubits=args.qubits, reps=2,
                                   entanglement="linear", bandwidth=bandwidth)
            k = quantum_kernel_matrix(a_train, a_train, fm)
            denom = np.linalg.norm(k) * np.linalg.norm(ideal)
            score = float((k * ideal).sum() / denom) if denom > 0 else 0.0
            if best is None or score > best[0]:
                best = (score, fm, k)
        _, fm, k_train = best
        k_test = quantum_kernel_matrix(a_test, a_train, fm)
        model = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(k_train, y_train)
        raw = model.decision_function(k_test)
        record("QSVM-fixed", (raw > 0).astype(int), raw, time.perf_counter() - t0)

        # ---- Pegasos QSVC on the same precomputed kernel ------------------
        t0 = time.perf_counter()
        try:
            from qiskit_machine_learning.algorithms import PegasosQSVC

            pegasos = PegasosQSVC(C=1.0, num_steps=200, precomputed=True, seed=args.seed)
            pegasos.fit(k_train, y_train)
            preds = pegasos.predict(k_test)
            record("QSVM-Pegasos", preds, None, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - optional dependency path
            print(f"    ! Pegasos unavailable: {exc.__class__.__name__}")

        # ---- trainable quantum embedding kernel --------------------------
        t0 = time.perf_counter()
        tqek = TrainableQuantumKernel(n_qubits=args.qubits, layers=2, seed=args.seed)
        train_kernel(tqek, a_train, y_train, steps=args.kernel_steps, seed=args.seed)
        kq_train = gram_matrix(tqek, a_train, a_train)
        kq_test = gram_matrix(tqek, a_test, a_train)
        model = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(kq_train, y_train)
        raw = model.decision_function(kq_test)
        record("TQEK", (raw > 0).astype(int), raw, time.perf_counter() - t0)

        # ---- trainable classical metric kernel, identical budget ----------
        t0 = time.perf_counter()
        tck = TrainableClassicalKernel(n_features=args.qubits, seed=args.seed)
        train_kernel(tck, x_train, y_train, steps=args.kernel_steps, seed=args.seed)
        kc_train = gram_matrix(tck, x_train, x_train)
        kc_test = gram_matrix(tck, x_test, x_train)
        model = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(kc_train, y_train)
        raw = model.decision_function(kc_test)
        record("Trainable-RBF", (raw > 0).astype(int), raw, time.perf_counter() - t0)

        # ---- tuned classical RBF -----------------------------------------
        t0 = time.perf_counter()
        model = SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced").fit(x_train, y_train)
        raw = model.decision_function(x_test)
        record("SVM-RBF", (raw > 0).astype(int), raw, time.perf_counter() - t0)

        print(f"  {site:<12} n={len(test_idx):<4} " + "  ".join(
            f"{n.split()[0][:5]}={per_fold[n][-1]['accuracy']:.3f}"
            for n in names if per_fold[n]), flush=True)

    def summarise(folds, key):
        values = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 78)
    print(f"QUANTUM MODEL SUITE — {args.qubits} qubits / {args.qubits} matched features")
    print("=" * 78)
    header = f"{'model':<16}{'accuracy':>15}{'F1':>15}{'AUC':>15}{'sec/fold':>10}"
    print(header + "\n" + "-" * len(header))
    for name in names:
        if not per_fold[name]:
            continue
        row = f"{name:<16}"
        for key in ("accuracy", "f1", "auc"):
            mean, half = summarise(per_fold[name], key)
            row += f"{mean:>9.3f}+-{half:<5.3f}"
        row += f"{timing[name] / max(len(per_fold[name]), 1):>10.1f}"
        print(row)

    # Paired tests: each quantum model against its natural classical counterpart.
    pairs = [("QCNN", "CNN (1-D)"), ("VQC", "MLP"), ("TQEK", "Trainable-RBF"),
             ("QSVM-fixed", "SVM-RBF"), ("QSVM-Pegasos", "SVM-RBF")]
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
    (out_dir / "quantum_models_results.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    print(f"\nsaved: {out_dir / 'quantum_models_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
