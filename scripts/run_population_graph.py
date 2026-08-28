#!/usr/bin/env python3
"""Quantum Population Graph (QPG) vs classical population graphs.

Tests whether defining population-graph edges by quantum fidelity beats
defining them by a classical similarity, holding everything else fixed: same
node features, same sparsification, same GCN, same folds.

Three edge definitions are compared:

    quantum       fidelity kernel |<psi_i|psi_j>|^2 between subjects
    correlation   Pearson correlation between connectivity profiles
    rbf           Gaussian similarity between connectivity profiles

plus a no-graph MLP control, which isolates how much the graph contributes at
all. Optionally each is gated by phenotypic agreement, as in Parisot et al.

Why this architecture rather than the region-level one: FINDING 06 measures
per-region temporal features at chance (AUC 0.459) while pairwise correlations
reach 0.693. QPG puts the informative representation on the nodes and the
quantum kernel on the edges, where it stays inside the narrow register that
FINDING 01 shows fidelity requires.

Examples:
    python scripts/run_population_graph.py --data-root data/ABIDE-I
    python scripts/run_population_graph.py --data-root data/REST-meta-MDD --qubits 8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qagta.data.abide import load_abide
from qagta.data.descriptors import build_descriptors
from qagta.models.population import (
    PopulationGCN,
    build_population_graph,
    phenotypic_affinity,
)
from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix


def load_phenotypic_extras(root: Path, file_ids: list[str]):
    """Recover SEX and AGE for the loaded subjects, when the table has them."""
    path = root / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"
    frame = pd.read_csv(path)
    if "FILE_ID" not in frame.columns:
        return None, None
    frame = frame.set_index("FILE_ID")

    def column(*names):
        for name in names:
            if name in frame.columns:
                values = pd.to_numeric(frame[name], errors="coerce")
                return np.array([values.get(f, np.nan) for f in file_ids], dtype=float)
        return None

    return column("SEX", "sex", "Gender"), column("AGE_AT_SCAN", "AGE", "Age")


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


def train_population_gcn(
    features: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int = 200,
    hidden: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train transductively; return (predictions, scores) for the held-out nodes."""
    torch.manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)

    model = PopulationGCN(in_features=x.shape[1], hidden=hidden, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    counts = torch.bincount(y[train_idx], minlength=2).float()
    class_weight = (counts.sum() / (2 * counts.clamp_min(1))).to(torch.float32)
    train_mask = torch.as_tensor(train_idx, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_weight)
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weight)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_weight)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return logits.argmax(1).numpy()[test_idx], probs[test_idx]


def train_mlp_control(features, labels, train_idx, test_idx, epochs=200, seed=0):
    """Same features and budget, no graph — isolates the graph's contribution."""
    torch.manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Dropout(0.3), torch.nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    counts = torch.bincount(y[train_idx], minlength=2).float()
    class_weight = (counts.sum() / (2 * counts.clamp_min(1))).to(torch.float32)
    mask = torch.as_tensor(train_idx, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x[mask]), y[mask], weight=class_weight)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    return logits.argmax(1).numpy()[test_idx], probs[test_idx]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--atlas", default="rois_cc200")
    parser.add_argument("--qubits", type=int, default=8,
                        help="register width for the subject fidelity kernel")
    parser.add_argument("--node-dim", type=int, default=128,
                        help="PCA width of node features; unconstrained by qubit count")
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--bandwidths", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--no-phenotypic", action="store_true",
                        help="drop the phenotypic gate, leaving imaging similarity alone")
    parser.add_argument("--min-test-size", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_population_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"population graph study -> {out_dir}")

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
    print(f"connectivity descriptors: {connectivity.shape}")

    sex, age = load_phenotypic_extras(
        args.data_root, [s.file_id for s in dataset.subjects]
    )
    if args.no_phenotypic or (sex is None and age is None):
        affinity = None
        print("phenotypic gate: disabled")
    else:
        if sex is not None and np.isnan(sex).all():
            sex = None
        if age is not None and np.isnan(age).all():
            age = None
        affinity = phenotypic_affinity(sex, age, use_site=False)
        available = [n for n, v in (("sex", sex), ("age", age)) if v is not None]
        print(f"phenotypic gate: {', '.join(available) or 'none available'}")

    feature_maps = {
        b: QuantumFeatureMap(n_qubits=args.qubits, reps=2, entanglement="linear", bandwidth=b)
        for b in args.bandwidths
    }

    edge_kinds = ["quantum", "correlation", "rbf"]
    per_fold: dict[str, list[dict]] = {k: [] for k in edge_kinds}
    per_fold["no-graph MLP"] = []

    for site in sorted(set(sites.tolist())):
        test_idx = np.where(sites == site)[0]
        train_idx = np.where(sites != site)[0]
        if len(test_idx) < args.min_test_size or len(np.unique(labels[test_idx])) < 2:
            continue
        y_test = labels[test_idx]

        # Node features: fit the reduction on training subjects only.
        scaler = StandardScaler().fit(connectivity[train_idx])
        train_scaled = scaler.transform(connectivity[train_idx])
        node_dim = min(args.node_dim, train_scaled.shape[0], train_scaled.shape[1])
        pca = PCA(n_components=node_dim, random_state=args.seed).fit(train_scaled)
        node_features = pca.transform(scaler.transform(connectivity))
        norms = np.linalg.norm(node_features, axis=1, keepdims=True)
        node_features = node_features / (norms + 1e-8)

        # Quantum similarity uses a narrow encoding of the same reduction.
        angle_source = node_features[:, : args.qubits]
        angle_scaler = MinMaxScaler((0, np.pi)).fit(angle_source[train_idx])
        angles = np.clip(angle_scaler.transform(angle_source), 0, np.pi)

        # Bandwidth chosen on training subjects only, by alignment with labels.
        best = None
        y_train_signed = np.where(labels[train_idx] > 0, 1.0, -1.0)
        for bandwidth, fm in feature_maps.items():
            k_train = quantum_kernel_matrix(angles[train_idx], angles[train_idx], fm)
            ideal = np.outer(y_train_signed, y_train_signed)
            denom = np.linalg.norm(k_train) * np.linalg.norm(ideal)
            score = float((k_train * ideal).sum() / denom) if denom > 0 else 0.0
            if best is None or score > best[0]:
                best = (score, bandwidth)
        bandwidth = best[1]

        similarity = {
            "quantum": quantum_kernel_matrix(angles, angles, feature_maps[bandwidth]),
            "correlation": np.corrcoef(node_features),
            "rbf": None,
        }
        distances = ((node_features[:, None, :] - node_features[None, :, :]) ** 2).sum(-1)
        median = np.median(distances[distances > 0]) if (distances > 0).any() else 1.0
        similarity["rbf"] = np.exp(-distances / max(median, 1e-8))

        for kind in edge_kinds:
            sim = np.nan_to_num(similarity[kind].astype(float))
            edge_index, edge_weight = build_population_graph(
                sim, affinity, k_neighbors=args.k_neighbors
            )
            preds, scores = train_population_gcn(
                node_features, edge_index, edge_weight, labels, train_idx, test_idx,
                epochs=args.epochs, hidden=args.hidden, seed=args.seed,
            )
            row = {"site": site, "n": len(test_idx), **metrics(y_test, preds, scores)}
            if kind == "quantum":
                row["bandwidth"] = bandwidth
            per_fold[kind].append(row)

        preds, scores = train_mlp_control(
            node_features, labels, train_idx, test_idx, epochs=args.epochs, seed=args.seed
        )
        per_fold["no-graph MLP"].append(
            {"site": site, "n": len(test_idx), **metrics(y_test, preds, scores)}
        )

        print(f"  {site:<12} n={len(test_idx):<4} "
              + "  ".join(f"{k[:4]} f1={per_fold[k][-1]['f1']:.3f}"
                          for k in [*edge_kinds, "no-graph MLP"]), flush=True)

    def summarise(folds: list[dict], key: str) -> tuple[float, float]:
        values = np.array([f[key] for f in folds if np.isfinite(f.get(key, np.nan))])
        if len(values) == 0:
            return float("nan"), float("nan")
        half = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return float(values.mean()), float(half)

    print("\n" + "=" * 74)
    print(f"QUANTUM POPULATION GRAPH — {args.qubits} qubits, {args.node_dim}-d node features")
    print("=" * 74)
    header = f"{'edge definition':<22}{'F1':>14}{'accuracy':>14}{'AUC':>14}"
    print(header + "\n" + "-" * len(header))
    for name, folds in per_fold.items():
        row = f"{name:<22}"
        for key in ("f1", "accuracy", "auc"):
            mean, half = summarise(folds, key)
            row += f"{mean:>8.3f}+-{half:<5.3f}"
        print(row)

    stats = {}
    q = np.array([f["f1"] for f in per_fold["quantum"]])
    for rival in ("correlation", "rbf", "no-graph MLP"):
        c = np.array([f["f1"] for f in per_fold[rival]])
        if len(q) > 1 and np.any(q != c):
            _, p = wilcoxon(q, c)
            stats[rival] = {"median_diff": float(np.median(q - c)), "p_value": float(p),
                            "wins": int((q > c).sum()), "folds": int(len(q))}
            print(f"\nquantum vs {rival}: median dF1 = {np.median(q - c):+.3f}, "
                  f"wins {int((q > c).sum())}/{len(q)}, Wilcoxon p = {p:.4f}")

    payload = {
        "config": vars(args) | {"data_root": str(args.data_root), "out": str(out_dir)},
        "n_subjects": len(dataset),
        "summary": {k: {m: summarise(v, m) for m in ("f1", "accuracy", "auc")}
                    for k, v in per_fold.items()},
        "per_fold": per_fold, "paired_tests": stats, "timestamp": stamp,
    }
    (out_dir / "population_graph_results.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    print(f"\nsaved: {out_dir / 'population_graph_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
