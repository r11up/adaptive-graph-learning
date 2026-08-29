#!/usr/bin/env python3
"""Does a wider register help a *variational* quantum model?

FINDING 01 showed fidelity collapsing as the register widens, but that applies
to kernels, which compare states pairwise. A QCNN reads a single Pauli-Z
expectation instead, so concentration need not bite the same way. Whether width
helps a variational model is therefore an open question, and this measures it.

Design notes, after a first attempt that ran 62 minutes on 14 GB and produced
nothing:

- **Subsampled cohort.** A few hundred subjects is enough to see a trend, and
  16-qubit states are 65,536-dimensional, so full-cohort training is what made
  the first attempt intractable.
- **Per-stage output, unbuffered.** Each width reports before the next begins,
  so a run that has to be abandoned still leaves the narrower results behind.
- **Train and test accuracy together.** The interesting question is not only
  whether width helps but whether a wider circuit underfits or overfits, and
  the gap between the two answers that.

Examples:
    python scripts/run_qubit_scaling.py --widths 4 8 12
    python scripts/run_qubit_scaling.py --widths 4 8 16 --subjects 250
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.quantum.qcnn import QCNNClassifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16],
                        help="qubit counts to compare; must be powers of two")
    parser.add_argument("--subjects", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hold-out-site", default="NYU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_qubit_scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"qubit scaling -> {out_dir}", flush=True)

    dataset = load_abide(root=args.data_root, n_components=max(args.widths))
    series_dir = args.data_root / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    connectivity = build_descriptors(
        [np.loadtxt(series_dir / f"{s.file_id}_{args.atlas}.1D") for s in dataset.subjects],
        kind="correlation",
    )
    labels, sites = dataset.labels, dataset.sites

    # One held-out site, matching the real protocol, then subsample the
    # training pool so 16 qubits stays tractable.
    test_idx = np.where(sites == args.hold_out_site)[0]
    train_pool = np.where(sites != args.hold_out_site)[0]
    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(train_pool, size=min(args.subjects, len(train_pool)),
                           replace=False)
    print(f"train {len(train_idx)} subjects | test {len(test_idx)} "
          f"(held-out site {args.hold_out_site})\n", flush=True)

    y_train, y_test = labels[train_idx], labels[test_idx]
    a, b = connectivity[train_idx][y_train == 0], connectivity[train_idx][y_train == 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    ranked = np.argsort(-np.nan_to_num(np.abs(t_stat)))

    header = (f"{'qubits':<8}{'params':<8}{'train':<9}{'test':<9}"
              f"{'gap':<9}{'AUC':<8}{'sec':<8}{'GB':<6}")
    print(header)
    print("-" * len(header), flush=True)

    rows = []
    for width in args.widths:
        chosen = ranked[:width]
        scaler = StandardScaler().fit(connectivity[train_idx][:, chosen])
        xtr = scaler.transform(connectivity[train_idx][:, chosen])
        xte = scaler.transform(connectivity[test_idx][:, chosen])
        # FINDING 22: [0, pi] spans a full phase period in RZ(2x); use [0, pi/2].
        angle = MinMaxScaler((0, np.pi / 2)).fit(xtr)
        xtr = angle.transform(xtr)
        xte = np.clip(angle.transform(xte), 0, np.pi / 2)

        torch.manual_seed(args.seed)
        model = QCNNClassifier(n_qubits=width, seed=args.seed)
        n_params = sum(p.numel() for p in model.parameters())

        xt = torch.as_tensor(xtr, dtype=torch.float32)
        yt = torch.as_tensor(y_train, dtype=torch.long)
        counts = torch.bincount(yt, minlength=2).float()
        weight = (counts.sum() / (2 * counts.clamp_min(1))).float()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        start = time.perf_counter()
        model.train()
        for _ in range(args.epochs):
            optimizer.zero_grad()
            F.cross_entropy(model(xt), yt, weight=weight).backward()
            optimizer.step()
        elapsed = time.perf_counter() - start

        model.eval()
        with torch.no_grad():
            train_logits = model(xt)
            test_logits = model(torch.as_tensor(xte, dtype=torch.float32))
            train_acc = accuracy_score(y_train, train_logits.argmax(1).numpy())
            test_acc = accuracy_score(y_test, test_logits.argmax(1).numpy())
            probs = F.softmax(test_logits, dim=1)[:, 1].numpy()
            auc = (roc_auc_score(y_test, probs)
                   if len(np.unique(y_test)) > 1 else float("nan"))

        # Statevector memory is the practical limit on width.
        gb = len(xt) * (2**width) * 8 / 1e9
        print(f"{width:<8}{n_params:<8}{train_acc:<9.3f}{test_acc:<9.3f}"
              f"{train_acc - test_acc:<+9.3f}{auc:<8.3f}{elapsed:<8.1f}{gb:<6.2f}",
              flush=True)
        rows.append({
            "qubits": width, "params": n_params, "train_acc": float(train_acc),
            "test_acc": float(test_acc), "gap": float(train_acc - test_acc),
            "auc": float(auc), "seconds": elapsed, "statevector_gb": gb,
        })
        (out_dir / "qubit_scaling.json").write_text(
            json.dumps({"config": vars(args) | {"data_root": str(args.data_root),
                                                "out": str(out_dir)},
                        "rows": rows, "timestamp": stamp}, indent=2, default=str)
        )

    print(f"\nsaved: {out_dir / 'qubit_scaling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
