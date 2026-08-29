#!/usr/bin/env python3
"""Render the paper figures from measured results.

Writes each figure as PDF (for LaTeX) and PNG (for quick viewing) into a
timestamped results directory, and copies the PDFs into the paper folder so
`\\includegraphics{<name>.pdf}` resolves without a path.

Figures:
  fig_fidelity_collapse   fidelity dynamic range vs register width
  fig_lso_comparison      Leave-Site-Out F1/accuracy across models
  fig_quantum_kernel      quantum vs classical kernel, per-site paired
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Print-safe palette: distinguishable in greyscale and colour-blind friendly.
QUANTUM = "#2b6cb0"
CLASSICAL = "#a0aec0"
ACCENT = "#c05621"
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def save(fig, name: str, out_dir: Path, paper_dir: Path | None) -> None:
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    if paper_dir:
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, paper_dir / f"{name}.pdf")
    print(f"  {name}.pdf / .png")


def fig_fidelity_collapse(out_dir: Path, paper_dir: Path | None) -> None:
    """Measured fidelity dynamic range against register width."""
    qubits = np.array([4, 8, 12, 16])
    mean = np.array([0.0624, 0.0039, 0.0003, 0.00001])
    maximum = np.array([0.970, 0.662, 0.144, 0.023])
    usable = np.array([47.5, 7.4, 0.6, 0.0])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))

    axes[0].semilogy(qubits, mean, "o-", color=QUANTUM, label="mean fidelity")
    axes[0].semilogy(qubits, maximum, "s--", color=ACCENT, label="max fidelity")
    axes[0].axhline(0.01, color="grey", ls=":", lw=1)
    axes[0].text(12.2, 0.013, "usability floor", fontsize=7, color="grey")
    axes[0].set_xlabel("qubits per region")
    axes[0].set_ylabel("pairwise fidelity")
    axes[0].set_xticks(qubits)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Fidelity range collapses with width", fontsize=9)

    bars = axes[1].bar(qubits, usable, width=2.2,
                       color=[QUANTUM if u > 5 else CLASSICAL for u in usable])
    axes[1].set_xlabel("qubits per region")
    axes[1].set_ylabel("region pairs with fidelity > 0.01 (%)")
    axes[1].set_xticks(qubits)
    axes[1].set_title("Usable structure remaining", fontsize=9)
    for bar, value in zip(bars, usable, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.4,
                     f"{value:.1f}%", ha="center", fontsize=6)
    axes[1].set_ylim(0, 55)

    # Secondary axis showing how far 2^n outruns the 200 regions.
    top = axes[1].twiny()
    top.set_xlim(axes[1].get_xlim())
    top.set_xticks(qubits)
    top.set_xticklabels([f"$2^{{{q}}}$" for q in qubits], fontsize=7, color="grey")
    top.set_xlabel("Hilbert dimension (200 regions to populate)", fontsize=7, color="grey")
    top.grid(False)

    fig.tight_layout()
    save(fig, "fig_fidelity_collapse", out_dir, paper_dir)


def fig_lso_comparison(results_json: Path, out_dir: Path, paper_dir: Path | None) -> None:
    """Leave-Site-Out comparison across every evaluated model."""
    if not results_json.exists():
        print(f"  (skipped fig_lso_comparison: {results_json} not found)")
        return
    payload = json.loads(results_json.read_text())
    results = payload["results"]

    names = list(results.keys())
    f1 = [results[n]["f1"]["mean"] for n in names]
    f1_ci = [results[n]["f1"]["ci95"] for n in names]
    acc = [results[n]["accuracy"]["mean"] for n in names]
    acc_ci = [results[n]["accuracy"]["ci95"] for n in names]

    short = [n.replace("Proposed ", "").replace("Ablation ", "")
              .replace(" (quantum + GAT)", "quantum+GAT")
              .replace(" (quantum + GCN)", "quantum+GCN") for n in names]
    colours = [QUANTUM if "quantum" in n.lower() else CLASSICAL for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    y = np.arange(len(names))
    for ax, values, errors, label in (
        (axes[0], f1, f1_ci, "F1 (ASD-positive)"),
        (axes[1], acc, acc_ci, "accuracy"),
    ):
        ax.barh(y, values, xerr=errors, color=colours, height=0.62,
                error_kw={"lw": 1, "ecolor": "#4a5568"})
        ax.set_yticks(y)
        ax.set_yticklabels(short, fontsize=7.5)
        ax.set_xlabel(label)
        ax.set_xlim(0, 0.85)
        ax.invert_yaxis()
    axes[1].axvline(0.5, color=ACCENT, ls="--", lw=1)
    axes[1].text(0.505, len(names) - 0.4, "chance", fontsize=7, color=ACCENT)

    fig.suptitle("Leave-Site-Out on ABIDE I (1035 subjects, 20 sites)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig_lso_comparison", out_dir, paper_dir)


def fig_quantum_kernel(kernel_json: Path, out_dir: Path, paper_dir: Path | None) -> None:
    """Quantum vs classical kernel: aggregate bars plus per-site paired scatter."""
    if not kernel_json.exists():
        print(f"  (skipped fig_quantum_kernel: {kernel_json} not found)")
        return
    payload = json.loads(kernel_json.read_text())
    summary, per_fold = payload["summary"], payload["per_fold"]

    order = ["quantum kernel", "classical RBF", "classical linear"]
    order = [n for n in order if n in summary]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))

    x = np.arange(len(order))
    width = 0.38
    for offset, metric, colour in ((-width / 2, "f1", QUANTUM), (width / 2, "auc", ACCENT)):
        values = [summary[n][metric][0] for n in order]
        errors = [summary[n][metric][1] for n in order]
        axes[0].bar(x + offset, values, width, yerr=errors, label=metric.upper(),
                    color=colour, error_kw={"lw": 1, "ecolor": "#4a5568"})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("classical ", "") for n in order], fontsize=8)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Matched features, matched folds", fontsize=9)
    axes[0].axhline(0.5, color="grey", ls=":", lw=1)

    # Paired per-site scatter: every point is one held-out site.
    if "classical RBF" in per_fold:
        q = np.array([f["f1"] for f in per_fold["quantum kernel"]])
        c = np.array([f["f1"] for f in per_fold["classical RBF"]])
        axes[1].scatter(c, q, s=34, color=QUANTUM, zorder=3, edgecolor="white", lw=0.6)
        limits = [0, max(q.max(), c.max()) * 1.1 + 0.02]
        axes[1].plot(limits, limits, "--", color="grey", lw=1, zorder=1)
        axes[1].set_xlim(limits)
        axes[1].set_ylim(limits)
        axes[1].set_xlabel("classical RBF kernel, F1")
        axes[1].set_ylabel("quantum kernel, F1")
        wins = int((q > c).sum())
        axes[1].set_title(f"Per-site paired ({wins}/{len(q)} sites favour quantum)", fontsize=9)
        axes[1].text(0.04, 0.93, "above line:\nquantum better", transform=axes[1].transAxes,
                     fontsize=7, va="top", color=QUANTUM)

    fig.tight_layout()
    save(fig, "fig_quantum_kernel", out_dir, paper_dir)


def fig_topology_ablation(sweep_dir: Path, out_dir: Path, paper_dir: Path | None) -> None:
    """Over-smoothing versus classification across topology configurations.

    The point of this figure is the dissociation: narrowing the register and
    restoring the mixed kernel recovers node distinctiveness (MAD), while
    accuracy stays at chance. Topology was therefore not the binding constraint
    on classification.
    """
    import re

    rows = []
    for run_log in sorted(sweep_dir.glob("*/run.log")):
        text = run_log.read_text()
        mad = re.search(r"MAD, higher = more distinct\): ([\d.]+)", text)
        prop = re.search(r"Proposed \(quantum \+ GAT\)\s+([\d.]+)\+-[\d.]+\s+([\d.]+)", text)
        if mad and prop:
            rows.append({"tag": run_log.parent.name, "mad": float(mad.group(1)),
                         "f1": float(prop.group(1)), "accuracy": float(prop.group(2))})
    if not rows:
        print("  (skipped fig_topology_ablation: no completed runs)")
        return

    labels = [r["tag"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))

    bars = axes[0].bar(x, [r["mad"] for r in rows], color=QUANTUM, width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7.5)
    axes[0].set_ylabel("MAD (node distinctiveness)")
    axes[0].set_title("Over-smoothing is relieved", fontsize=9)
    for bar, r in zip(bars, rows, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, r["mad"] + 0.002,
                     f"{r['mad']:.3f}", ha="center", fontsize=6)

    axes[1].bar(x, [r["accuracy"] for r in rows], color=CLASSICAL, width=0.6)
    axes[1].axhline(0.5, color=ACCENT, ls="--", lw=1.2)
    axes[1].text(len(rows) - 0.6, 0.507, "chance", fontsize=7, color=ACCENT)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 0.75)
    axes[1].set_title("Classification does not follow", fontsize=9)

    fig.tight_layout()
    save(fig, "fig_topology_ablation", out_dir, paper_dir)


def fig_cross_cohort(out_dir: Path, paper_dir: Path | None) -> None:
    """Quantum vs classical kernels across every cohort.

    The point of this figure is the sign flip: the one cohort where the quantum
    kernel beats a classical one is contradicted by the largest cohort, which
    runs the other way with more folds and ten times the subjects.
    """
    runs = [
        ("REST-meta-MDD", "results/MDD_qkernel", 2428, 25),
        ("ADHD-200", "results/adhd200_qkernel", 767, 7),
        ("UCLA-CNP", "results/UCLA_qkernel_cv10", 226, 10),
    ]
    loaded = []
    for label, path, n, folds in runs:
        f = Path(path) / "quantum_kernel_results.json"
        if f.exists():
            loaded.append((label, json.loads(f.read_text()), n, folds))
    if not loaded:
        print("  (skipped fig_cross_cohort: no results)")
        return

    models = ["quantum kernel", "classical RBF", "classical linear"]
    colours = [QUANTUM, CLASSICAL, "#718096"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))

    x = np.arange(len(loaded))
    width = 0.26
    for i, (model, colour) in enumerate(zip(models, colours, strict=True)):
        vals = [d["summary"][model]["f1"][0] for _, d, _, _ in loaded]
        errs = [d["summary"][model]["f1"][1] for _, d, _, _ in loaded]
        axes[0].bar(x + (i - 1) * width, vals, width, yerr=errs, label=model,
                    color=colour, error_kw={"lw": 1, "ecolor": "#4a5568"})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{lab}\nn={n}, {f} folds" for lab, _, n, f in loaded],
                            fontsize=6)
    axes[0].set_ylabel("F1")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Matched features, matched folds", fontsize=9)

    # Signed effect against the linear kernel, the only comparison that reached
    # significance anywhere — and it reaches it in both directions.
    labels, deltas, ps = [], [], []
    for label, data, _, _ in loaded:
        test = data.get("paired_tests", {}).get("classical linear")
        if test:
            labels.append(label)
            deltas.append(test["median_diff"])
            ps.append(test["p_value"])
    bars = axes[1].barh(np.arange(len(labels)), deltas,
                        color=[QUANTUM if d > 0 else ACCENT for d in deltas], height=0.55)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("median $\\Delta$F1  (quantum $-$ classical linear)")
    axes[1].set_title("The effect changes sign", fontsize=9)
    for bar, d, pv in zip(bars, deltas, ps, strict=True):
        mark = "*" if pv < 0.05 else ""
        axes[1].text(d + (0.006 if d > 0 else -0.006), bar.get_y() + bar.get_height() / 2,
                     f"p={pv:.3f}{mark}", va="center", fontsize=7,
                     ha="left" if d > 0 else "right")
    span = max(abs(min(deltas)), abs(max(deltas))) * 2.1
    axes[1].set_xlim(-span, span)

    fig.tight_layout()
    save(fig, "fig_cross_cohort", out_dir, paper_dir)


def fig_benchmark(out_dir: Path, paper_dir: Path | None) -> None:
    """Matched-feature comparison across all four cohorts.

    The left panel shows that the quantum kernel tracks a classical kernel on
    the same inputs; the right shows the per-cohort paired test, none of which
    separates them. The full-dimensional model is drawn separately because it
    is not a matched comparator — it answers a different question.
    """
    runs = [
        ("ABIDE-I", "results/ABIDE_benchmark"),
        ("ADHD-200", "results/ADHD200_benchmark"),
        ("REST-meta-MDD", "results/MDD_benchmark"),
        ("UCLA-CNP", "results/UCLA_benchmark"),
    ]
    loaded = []
    for label, path in runs:
        f = Path(path) / "benchmark_results.json"
        if f.exists():
            loaded.append((label, json.loads(f.read_text())))
    if not loaded:
        print("  (skipped fig_benchmark: no results)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    x = np.arange(len(loaded))
    width = 0.27
    series = [
        ("quantum kernel", QUANTUM, "quantum kernel"),
        ("SVM RBF (matched)", CLASSICAL, "classical, matched features"),
        ("SVM (RBF)", ACCENT, "classical, all features"),
    ]
    for i, (key, colour, label) in enumerate(series):
        vals = [d["summary"][key]["accuracy"][0] for _, d in loaded]
        errs = [d["summary"][key]["accuracy"][1] for _, d in loaded]
        axes[0].bar(x + (i - 1) * width, vals, width, yerr=errs, label=label,
                    color=colour, error_kw={"lw": 1, "ecolor": "#4a5568"})
    axes[0].axhline(0.5, color="grey", ls=":", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lab for lab, _ in loaded], fontsize=7, rotation=12)
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 0.8)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    axes[0].set_title("Quantum tracks classical on matched features", fontsize=9)

    labels, wins, folds, ps = [], [], [], []
    for label, data in loaded:
        test = data.get("paired_tests", {}).get("SVM RBF (matched)")
        if test:
            labels.append(label)
            wins.append(test["wins"] / test["folds"])
            folds.append(test["folds"])
            ps.append(test["p_value"])
    y = np.arange(len(labels))
    axes[1].barh(y, wins, color=QUANTUM, height=0.55)
    axes[1].axvline(0.5, color=ACCENT, ls="--", lw=1.2)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("fraction of folds quantum wins")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("No cohort separates them", fontsize=9)
    for i, (w, n, pv) in enumerate(zip(wins, folds, ps, strict=True)):
        axes[1].text(min(w + 0.03, 0.72), i, f"{int(w * n)}/{n}, p={pv:.2f}",
                     va="center", fontsize=6)
    axes[1].text(0.505, len(labels) - 0.35, "parity", fontsize=7, color=ACCENT)

    fig.tight_layout()
    save(fig, "fig_benchmark", out_dir, paper_dir)


ATLAS_PATH = Path(
    "data/sources/ADHD-200-Athena-release/ADHD200_CC200_TCs_filtfix/"
    "templates/ADHD200_parcellate_200.nii.gz"
)


def fig_parcellation(out_dir: Path, paper_dir: Path | None) -> None:
    """The CC200 parcellation on a template brain.

    Purely descriptive: it shows what enters the model as graph nodes and
    claims no result. Included because every cohort in the study is defined on
    this parcellation, and because the 190-versus-200 discrepancy between
    releases is only intelligible once the atlas is shown.
    """
    if not ATLAS_PATH.exists():
        print("  (skipped fig_parcellation: atlas not found)")
        return
    from nilearn import plotting

    fig = plt.figure(figsize=(7.1, 2.4))
    display = plotting.plot_roi(
        str(ATLAS_PATH), figure=fig, display_mode="z",
        cut_coords=(-20, 0, 20, 40), cmap="tab20", colorbar=False,
        annotate=True, draw_cross=False, black_bg=False,
    )
    fig.suptitle(
        "CC200 functional parcellation — 190 non-empty regions, whole brain",
        fontsize=9, y=0.99,
    )
    save(fig, "fig_parcellation", out_dir, paper_dir)
    display.close()


def fig_connectivity_comparison(out_dir: Path, paper_dir: Path | None) -> None:
    """Pearson correlation against quantum fidelity for one subject.

    This is the picture behind FINDING 01. The correlation matrix is dense and
    structured; the fidelity matrix at 16 qubits is empty, because every pair of
    regional states has become orthogonal. The 4-qubit panel shows the same
    metric inside its usable range.
    """
    import glob

    import torch

    from qagta.graph.connectome import pearson_connectivity
    from qagta.quantum.fidelity import pairwise_fidelity
    from qagta.quantum.fmri_encoder import RingEntangledEncoder

    files = sorted(glob.glob("data/ABIDE-I/ABIDE_pcp/cpac/filt_noglobal/*.1D"))
    if not files:
        print("  (skipped fig_connectivity_comparison: no ABIDE series)")
        return

    series = np.loadtxt(files[0])
    correlation = pearson_connectivity(series)

    from qagta.data.abide import compress_connectivity_profiles

    panels = [("Pearson correlation", correlation, "RdBu_r", (-1, 1))]
    for n_qubits in (4, 16):
        features, _ = compress_connectivity_profiles(series, n_qubits)
        torch.manual_seed(0)
        encoder = RingEntangledEncoder(n_qubits)
        with torch.no_grad():
            _, states = encoder(torch.as_tensor(features), return_state=True)
            fidelity = pairwise_fidelity(states).numpy()
        panels.append(
            (f"Quantum fidelity, {n_qubits} qubits", fidelity, "magma", (0, None))
        )

    # Order regions by correlation clustering so block structure is visible.
    order = np.argsort(correlation.mean(axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    for ax, (title, matrix, cmap, limits) in zip(axes, panels, strict=True):
        shown = matrix[np.ix_(order, order)]
        vmin, vmax = limits
        if vmax is None:
            vmax = float(np.percentile(shown, 99.5)) or 1e-6
        im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
        off = shown[~np.eye(len(shown), dtype=bool)]
        ax.set_xlabel(f"mean |value| = {np.abs(off).mean():.3f}", fontsize=6)

    fig.suptitle("What the quantum metric does to topology (one subject)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_connectivity_comparison", out_dir, paper_dir)


def fig_kernel_spectrum(out_dir: Path, paper_dir: Path | None) -> None:
    """Effective dimension and eigenvalue decay, quantum kernel versus RBF.

    A Gram matrix's participation ratio, (sum lambda)^2 / sum lambda^2, counts
    how many directions the kernel actually uses. Near 1 the kernel is almost
    constant and every subject looks alike; near n it is almost the identity and
    every subject is its own cluster. Useful kernels sit between.

    The measurement shows both kernels sweeping the same range as their width
    parameter varies, and both possessing a usable window. That is the
    mechanism behind the matched-feature tie: at comparable effective dimension
    the quantum fidelity kernel and the classical RBF kernel occupy the same
    function class on these data.
    """
    from scipy import stats as sstats
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from qagta.data.abide import load_abide
    from qagta.data.descriptors import build_descriptors
    from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix

    root = Path("data/ABIDE-I")
    if not (root / "ABIDE_pcp").exists():
        print("  (skipped fig_kernel_spectrum: ABIDE not found)")
        return

    dataset = load_abide(root=root, n_components=8, limit=400)
    series_dir = root / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    connectivity = build_descriptors(
        [np.loadtxt(series_dir / f"{s.file_id}_rois_cc200.1D") for s in dataset.subjects],
        kind="correlation",
    )
    y = dataset.labels
    t_stat, _ = sstats.ttest_ind(connectivity[y == 0], connectivity[y == 1],
                                 axis=0, equal_var=False)
    chosen = np.argsort(-np.nan_to_num(np.abs(t_stat)))[:8]
    angles = MinMaxScaler((0, np.pi)).fit_transform(connectivity[:, chosen])
    scaled = StandardScaler().fit_transform(connectivity[:, chosen])

    def effective_dim(kernel: np.ndarray) -> float:
        ev = np.clip(np.linalg.eigvalsh(kernel), 0, None)
        return float((ev.sum() ** 2) / (ev**2).sum()) if (ev**2).sum() > 0 else 0.0

    def spectrum(kernel: np.ndarray) -> np.ndarray:
        ev = np.clip(np.linalg.eigvalsh(kernel), 0, None)[::-1]
        return ev / ev.sum()

    bandwidths = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
    gammas = [0.01, 0.05, 0.125, 0.5, 2.0]

    q_dims, q_kernels = [], {}
    for b in bandwidths:
        fm = QuantumFeatureMap(n_qubits=8, reps=2, entanglement="linear", bandwidth=b)
        kernel = quantum_kernel_matrix(angles, angles, fm)
        q_dims.append(effective_dim(kernel))
        q_kernels[b] = kernel
    c_dims, c_kernels = [], {}
    for g in gammas:
        kernel = rbf_kernel(scaled, gamma=g)
        c_dims.append(effective_dim(kernel))
        c_kernels[g] = kernel

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))

    axes[0].loglog(bandwidths, q_dims, "o-", color=QUANTUM, label="quantum fidelity")
    axes[0].loglog(gammas, c_dims, "s--", color=ACCENT, label="classical RBF")
    axes[0].axhspan(4, 20, color="grey", alpha=0.16)
    axes[0].text(0.022, 8.5, "usable window", fontsize=7, color="#444")
    axes[0].axhline(len(angles), color="grey", ls=":", lw=1)
    axes[0].text(0.022, len(angles) * 0.62, "identity limit", fontsize=6.5, color="grey")
    axes[0].set_xlabel("width parameter (bandwidth / $\\gamma$)")
    axes[0].set_ylabel("effective dimension")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Both kernels sweep the same range", fontsize=9)

    # Eigenvalue decay at comparable effective dimension.
    best_b = min(bandwidths, key=lambda b: abs(effective_dim(q_kernels[b]) - 8.2))
    axes[1].semilogy(spectrum(q_kernels[best_b])[:40], "o-", color=QUANTUM, ms=3,
                     label=f"quantum (bw={best_b}, eff.dim={effective_dim(q_kernels[best_b]):.1f})")
    axes[1].semilogy(spectrum(c_kernels[0.125])[:40], "s--", color=ACCENT, ms=3,
                     label=f"RBF ($\\gamma$=0.125, eff.dim={effective_dim(c_kernels[0.125]):.1f})")
    axes[1].set_xlabel("eigenvalue index")
    axes[1].set_ylabel("normalised eigenvalue")
    axes[1].legend(fontsize=6.5, frameon=False)
    axes[1].set_title("At matched effective dimension, matched decay", fontsize=9)

    fig.tight_layout()
    save(fig, "fig_kernel_spectrum", out_dir, paper_dir)


def fig_architectures(out_dir: Path, paper_dir: Path | None) -> None:
    """Schematic of the three architectures, drawn to be told apart at a glance.

    RQT, SQK and QPG differ in one structural choice — what occupies a node and
    where the quantum stage acts — and every negative result in the study
    attaches to a specific one of them.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    panels = [
        ("RQT — Region-level Quantum Topology",
         ["200 brain regions", "connectivity profiles", "fidelity between REGIONS",
          "graph attention", "pool over regions"],
         "node = brain region", "chance", ACCENT),
        ("SQK — Subject-level Quantum Kernel",
         ["subjects", "connectivity vector", "fidelity between SUBJECTS",
          "kernel SVM", "(no graph)"],
         "no graph", "ties classical", CLASSICAL),
        ("QPG — Quantum Population Graph",
         ["subjects", "connectivity vector", "fidelity between SUBJECTS",
          "population GCN", "predict held-out node"],
         "node = subject", "ties classical", QUANTUM),
    ]

    for ax, (title, rows, node_label, outcome, colour) in zip(axes, panels, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=8, pad=6)

        y = 0.86
        for i, row in enumerate(rows):
            emphasis = "fidelity" in row
            box = FancyBboxPatch(
                (0.06, y - 0.075), 0.88, 0.115,
                boxstyle="round,pad=0.012",
                facecolor=colour if emphasis else "#f0f2f5",
                edgecolor="#8a94a6", linewidth=0.7,
                alpha=0.95 if emphasis else 1.0,
            )
            ax.add_patch(box)
            ax.text(0.5, y - 0.018, row, ha="center", va="center", fontsize=6.8,
                    color="white" if emphasis else "#1a202c",
                    fontweight="bold" if emphasis else "normal")
            if i < len(rows) - 1:
                ax.add_patch(FancyArrowPatch(
                    (0.5, y - 0.078), (0.5, y - 0.115),
                    arrowstyle="-|>", mutation_scale=7, color="#8a94a6", lw=0.8,
                ))
            y -= 0.175

        ax.text(0.5, 0.045, f"{node_label}   |   {outcome}", ha="center",
                fontsize=6.8, style="italic", color="#4a5568")

    fig.suptitle(
        "Three architectures: what sits on a node, and where the quantum stage acts",
        fontsize=9, y=1.0,
    )
    fig.tight_layout()
    save(fig, "fig_architectures", out_dir, paper_dir)


# Each quantum model and the classical comparator it was matched against.
MODEL_FAMILIES = [
    ("Kernel SVM", ["QSVM-fixed", "QSVM-Pegasos"], "SVM-RBF"),
    ("Convolutional", ["QCNN"], "CNN (1-D)"),
    ("Variational", ["VQC"], "MLP"),
    ("Trainable kernel", ["TQEK"], "Trainable-RBF"),
]
COHORT_ORDER = ["ABIDE-I", "ADHD-200", "REST-meta-MDD", "UCLA-CNP"]


# Cohort directory names in the post-correction runs, mapped to the display
# names the earlier timestamped layout used.
_POSTFIX_SUITE = {
    "qmodels_abide": "ABIDE-I",
    "qmodels_adhd": "ADHD-200",
    "qmodels_mdd": "REST-meta-MDD",
    "qmodels_ucla": "UCLA-CNP",
}


def _load_suite(results_root: Path) -> dict:
    """Model-suite summaries keyed by cohort, preferring the corrected runs.

    The variational models were re-run after the encoding correction, and those
    results live under ``results/rerun22/qmodels_<cohort>/`` rather than in the
    timestamped ``*_qmodels`` layout the earlier runs used. Figures must read
    the corrected numbers, so that location is checked first; the timestamped
    glob remains as a fallback for reproducing an older figure from an
    archived run.
    """
    rerun = results_root / "rerun22"
    if rerun.exists():
        out = {}
        for tag, cohort in _POSTFIX_SUITE.items():
            f = rerun / tag / "quantum_models_results.json"
            if f.exists():
                out[cohort] = json.loads(f.read_text())
        if out:
            return out

    suites = sorted(results_root.glob("*_qmodels"))
    if not suites:
        return {}
    out = {}
    for cohort_dir in sorted(suites[-1].iterdir()):
        f = cohort_dir / "quantum_models_results.json"
        if f.exists():
            out[cohort_dir.name] = json.loads(f.read_text())
    return out


def fig_quantum_vs_classical(out_dir: Path, paper_dir: Path | None,
                             results_root: Path = Path("results")) -> None:
    """Quantum models against their matched classical comparators.

    One panel per model family, so each quantum model is read against the
    classical model it was actually matched to rather than against the best
    classical number anywhere in the study. Bars are grouped by cohort and
    coloured by metric; the classical comparator is hatched.
    """
    suite = _load_suite(results_root)
    if not suite:
        print("  (skipped fig_quantum_vs_classical: no suite results)")
        return

    cohorts = [c for c in COHORT_ORDER if c in suite]
    metrics = [("accuracy", QUANTUM), ("f1", ACCENT), ("auc", "#2f855a")]

    fig, axes = plt.subplots(1, len(MODEL_FAMILIES), figsize=(11.5, 3.4), sharey=True)
    for ax, (title, quantum_models, classical) in zip(axes, MODEL_FAMILIES, strict=True):
        available = [
            m for m in quantum_models
            if any(m in suite[c].get("summary", {}) for c in cohorts)
        ]
        if not available:
            ax.axis("off")
            continue
        entries = available + [classical]

        n_groups = len(cohorts)
        slot = 0.8 / max(len(entries), 1)
        x = np.arange(n_groups)

        for e, model in enumerate(entries):
            is_classical = model == classical
            offset = -0.4 + slot * (e + 0.5)
            for m, (metric, colour) in enumerate(metrics):
                width = slot / len(metrics)
                values, errors = [], []
                for cohort in cohorts:
                    summary = suite[cohort].get("summary", {}).get(model, {})
                    v = summary.get(metric)
                    ok = v and v[0] == v[0]
                    values.append(v[0] if ok else 0.0)
                    errors.append(v[1] if ok else 0.0)
                ax.bar(
                    x + offset + (m - 1) * width, values, width * 0.92, yerr=errors,
                    color=colour, alpha=0.55 if is_classical else 1.0,
                    hatch="//" if is_classical else None,
                    edgecolor="white", linewidth=0.4,
                    error_kw={"lw": 0.7, "ecolor": "#4a5568"},
                )

        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("REST-meta-", "") for c in cohorts],
                           fontsize=6.8, rotation=20, ha="right")
        ax.set_title(f"{title}\n{' / '.join(available)} vs {classical}", fontsize=7.5)
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("score")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in metrics]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="grey", alpha=0.55, hatch="//"))
    fig.legend(
        handles, [m.upper() for m, _ in metrics] + ["classical comparator"],
        loc="upper center", ncol=4, fontsize=7, frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Quantum models against matched classical comparators", fontsize=9.5, y=1.11)
    fig.tight_layout()
    save(fig, "fig_quantum_vs_classical", out_dir, paper_dir)


def fig_quantum_delta(out_dir: Path, paper_dir: Path | None,
                      results_root: Path = Path("results")) -> None:
    """Signed quantum-minus-classical difference, with paired significance.

    Reading the differences directly avoids the eye having to subtract paired
    bars, and makes the sign pattern across cohorts visible at a glance.
    """
    suite = _load_suite(results_root)
    if not suite:
        print("  (skipped fig_quantum_delta: no suite results)")
        return

    cohorts = [c for c in COHORT_ORDER if c in suite]
    rows = []
    for cohort in cohorts:
        tests = suite[cohort].get("paired_tests", {})
        for label, stats in tests.items():
            quantum = label.split(" vs ")[0]
            rows.append((cohort, quantum, stats["median_diff"], stats["p_value"]))
    if not rows:
        print("  (skipped fig_quantum_delta: no paired tests)")
        return

    models = sorted({r[1] for r in rows})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    slot = 0.8 / len(models)

    for i, model in enumerate(models):
        deltas, ps, xs = [], [], []
        for j, cohort in enumerate(cohorts):
            match = [r for r in rows if r[0] == cohort and r[1] == model]
            if match:
                deltas.append(match[0][2])
                ps.append(match[0][3])
                xs.append(j - 0.4 + slot * (i + 0.5))
        if not xs:
            continue
        colours = [QUANTUM if d > 0 else ACCENT for d in deltas]
        ax.bar(xs, deltas, slot * 0.9, color=colours, edgecolor="white", linewidth=0.4)
        for xpos, delta, p in zip(xs, deltas, ps, strict=True):
            if p < 0.05:
                ax.text(xpos, delta + (0.004 if delta > 0 else -0.012), "*",
                        ha="center", fontsize=10, color="#1a202c")

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(np.arange(len(cohorts)))
    ax.set_xticklabels(cohorts, fontsize=8.5)
    ax.set_ylabel("median $\\Delta$accuracy (quantum $-$ classical)")
    ax.set_title("Signed difference per cohort; * marks $p < 0.05$ (paired Wilcoxon)",
                 fontsize=9)

    # Bars are coloured by sign, so a per-model colour legend would be wrong.
    # Each bar is instead labelled with its model name beneath the axis.
    low, high = ax.get_ylim()
    ax.set_ylim(low - (high - low) * 0.18, high)
    for i, model in enumerate(models):
        for j, cohort in enumerate(cohorts):
            if any(r[0] == cohort and r[1] == model for r in rows):
                ax.text(j - 0.4 + slot * (i + 0.5), ax.get_ylim()[0] * 0.97,
                        model, rotation=90, ha="center", va="bottom", fontsize=5.6,
                        color="#4a5568")

    legend = [
        plt.Rectangle((0, 0), 1, 1, color=QUANTUM),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
    ]
    ax.legend(legend, ["quantum better", "classical better"],
              fontsize=7, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_quantum_delta", out_dir, paper_dir)



def fig_overfitting(out_dir: Path, paper_dir: Path | None,
                    results_root: Path = Path("results")) -> None:
    """Overfitting gap by cohort: the one quantity the quantum arm leads on.

    Drawn because it is the study's only reproducible quantum-specific effect
    and a table states it without showing that it holds on every cohort at
    once. Cohorts are ordered by size so the reader can see for themselves that
    the ordering claimed in the text is descriptive, not a trend.
    """
    import json
    Q = ["Q-Plain", "Q-Reup1", "Q-Reup2"]
    cohorts = [("mdd", "REST-meta-MDD", 2428), ("abide", "ABIDE-I", 1035),
               ("adhd", "ADHD-200", 767), ("ucla", "UCLA-CNP", 226)]
    rows = []
    for tag, name, n in cohorts:
        f = results_root / "rerun22" / f"planC_{tag}" / "reupload_results.json"
        if not f.exists():
            return
        pf = json.loads(f.read_text())["per_fold"]
        gap = {k: np.mean([x["train_accuracy"] - x["accuracy"] for x in v])
               for k, v in pf.items()}
        rows.append((name, n, np.mean([gap[q] for q in Q]),
                     gap["C-CNN"], gap["C-MLP"], gap["C-Linear"]))

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(rows))
    w = 0.2
    series = [("Quantum", 2, ACCENT), ("C-CNN", 3, "#8c8c8c"),
              ("C-MLP", 4, "#c0c0c0"), ("C-Linear", 5, "#5b5b5b")]
    for i, (lab, idx, col) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [r[idx] for r in rows], w, label=lab, color=col,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}\n$n$={r[1]}" for r in rows], fontsize=6)
    ax.set_ylabel("overfitting gap (train $-$ test)", fontsize=6)
    ax.set_title("Quantum overfits least of the neural arms,\non every cohort",
                 fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, "fig_overfitting", out_dir, paper_dir)


def fig_lowdata(out_dir: Path, paper_dir: Path | None,
                results_root: Path = Path("results")) -> None:
    """Accuracy and overfitting against training-set size.

    Two panels because the result is a dissociation: the gap separates the arms
    at every size while accuracy does not, and a single panel would let a
    reader take the second for the first.
    """
    import json
    f = results_root / "rerun22" / "lowdata_abide" / "lowdata_results.json"
    if not f.exists():
        return
    rows = json.loads(f.read_text())["rows"]
    ns = [r["n_train"] for r in rows]
    acc = {k: [r["means"][k] for r in rows] for k in ("Quantum", "C-MLP", "C-Linear")}
    gap = {k: [np.mean([x["train_accuracy"] - x["accuracy"] for x in r["per_model"][k]])
               for r in rows] for k in ("Quantum", "C-MLP", "C-Linear")}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharex=True)
    style = {"Quantum": (ACCENT, "o", "-"), "C-MLP": ("#8c8c8c", "s", "--"),
             "C-Linear": ("#3b3b3b", "^", ":")}
    for k, (c, mk, ls) in style.items():
        axes[0].plot(ns, acc[k], marker=mk, ls=ls, color=c, label=k, ms=4, lw=1.3)
        axes[1].plot(ns, gap[k], marker=mk, ls=ls, color=c, label=k, ms=4, lw=1.3)
    axes[0].set_ylabel("test accuracy", fontsize=6)
    axes[0].set_title("Accuracy: no separation at any size", fontsize=7.5)
    axes[1].set_ylabel("overfitting gap", fontsize=6)
    axes[1].set_title("Overfitting: quantum below the MLP at every size",
                      fontsize=7.5)
    axes[1].set_xlabel("training subjects per fold", fontsize=6)
    for a in axes:
        a.set_xscale("log")
        a.set_xticks(ns)
        a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        # Suppress the decade minor labels a log axis adds between the four
        # training sizes, which otherwise collide with the tick labels.
        a.get_xaxis().set_minor_formatter(plt.NullFormatter())
        a.tick_params(axis="x", which="minor", length=0)
        a.tick_params(labelsize=6)
        a.grid(alpha=0.25, linewidth=0.4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    save(fig, "fig_lowdata", out_dir, paper_dir)



def fig_degeneracy(out_dir: Path, paper_dir: Path | None,
                   metrics_json: Path = Path("results/full_metrics.json")) -> None:
    """Accuracy against Matthews correlation, exposing degenerate predictors.

    Raw confusion matrices for nine models on four cohorts would be thirty-six
    panels. This compresses the same information into the comparison that
    matters: a model predicting one class scores near the majority rate on
    accuracy but near zero on MCC, so it sits far to the right of the trend.
    The inset shows the clearest case as an actual confusion matrix.
    """
    import json
    if not metrics_json.exists():
        return
    data = json.loads(metrics_json.read_text())
    runs = {k: v for k, v in data.items() if k.startswith("qmodels_")}
    if not runs:
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    worst = None
    for _run, models in runs.items():
        for name, m in models.items():
            q = name.startswith(("Q", "TQEK"))
            deg = min(m["sensitivity"], m["specificity"]) < 0.30 or abs(m["mcc"]) < 0.02
            ax.scatter(m["accuracy"], m["mcc"], s=26 if deg else 16,
                       c=(ACCENT if q else "#8c8c8c"),
                       marker="X" if deg else "o",
                       edgecolors="black" if deg else "none",
                       linewidths=0.6 if deg else 0, zorder=3 if deg else 2, alpha=0.9)
            if deg and (worst is None or m["mcc"] < worst[1]["mcc"]):
                worst = (f"{name}, {m['cohort']}", m)

    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("accuracy", fontsize=6)
    ax.set_ylabel("Matthews correlation", fontsize=6)
    ax.set_title("Accuracy hides degenerate predictors;\nMCC does not", fontsize=7.5)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, linewidth=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=ACCENT, ms=4, label="quantum"),
        plt.Line2D([], [], marker="o", ls="", color="#8c8c8c", ms=4, label="classical"),
        plt.Line2D([], [], marker="X", ls="", color="white", ms=6,
                   markeredgecolor="black", label="degenerate"),
    ]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left")

    if worst:
        name, m = worst
        txt = (f"{name}\nTN {m['TN']}  FP {m['FP']}\nFN {m['FN']}  TP {m['TP']}\n"
               f"acc {m['accuracy']:.3f}, MCC {m['mcc']:+.3f}")
        ax.annotate(txt, xy=(m["accuracy"], m["mcc"]),
                    xytext=(0.52, 0.13), textcoords="axes fraction",
                    fontsize=5.4, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", lw=0.5),
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
    save(fig, "fig_degeneracy", out_dir, paper_dir)



def fig_kernel_spectrum_cohorts(out_dir: Path, paper_dir: Path | None) -> None:
    """Effective dimension against width parameter, on all four cohorts.

    Section "Why the Two Kernels Tie" argues from a single cohort that a fixed
    fidelity kernel traverses the same one-parameter family of Gram matrices as
    an RBF kernel. The argument is structural rather than cohort-specific, so
    it should hold everywhere; this checks that it does. Each panel sweeps the
    width parameter of both kernels and plots the participation ratio
    $(\\sum\\lambda)^2 / \\sum\\lambda^2$, the number of directions the kernel
    actually uses.
    """
    from scipy import stats as sstats
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from qagta.data.abide import load_abide
    from qagta.data.descriptors import build_descriptors
    from qagta.quantum.kernel import QuantumFeatureMap, quantum_kernel_matrix

    cohorts = [("ABIDE I", Path("data/ABIDE-I"), "rois_cc200"),
               ("ADHD-200", Path("data/ADHD-200"), "rois_cc200"),
               ("REST-meta-MDD", Path("data/REST-meta-MDD"), "rois_cc200"),
               ("UCLA-CNP", Path("data/UCLA-CNP-cc200"), "rois_cc200")]
    bands = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 2.00]
    gammas = [0.01, 0.05, 0.125, 0.5, 2.0]

    def eff_dim(K):
        lam = np.linalg.eigvalsh(K)
        lam = lam[lam > 0]
        return float(lam.sum() ** 2 / (lam ** 2).sum()) if len(lam) else float("nan")

    panels = []
    for name, root, atlas in cohorts:
        if not (root / "ABIDE_pcp").exists():
            continue
        ds = load_abide(root=root, atlas=atlas, n_components=8, limit=400)
        series = root / "ABIDE_pcp" / "cpac" / "filt_noglobal"
        conn = build_descriptors(
            [np.loadtxt(series / f"{s.file_id}_{atlas}.1D") for s in ds.subjects],
            kind="correlation")
        y = ds.labels
        t, _ = sstats.ttest_ind(conn[y == 0], conn[y == 1], axis=0, equal_var=False)
        ch = np.argsort(-np.nan_to_num(np.abs(t)))[:8]
        ang = MinMaxScaler((0, np.pi)).fit_transform(conn[:, ch])
        sc = StandardScaler().fit_transform(conn[:, ch])
        q = [eff_dim(quantum_kernel_matrix(ang, ang, QuantumFeatureMap(8, bandwidth=b)))
             for b in bands]
        c = [eff_dim(rbf_kernel(sc, sc, gamma=g)) for g in gammas]
        panels.append((name, len(ds), q, c))

    if not panels:
        return
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.1), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, n, q, c) in zip(axes, panels, strict=False):
        ax.plot(bands, q, "o-", color=ACCENT, ms=3.2, lw=1.2, label="quantum fidelity")
        ax.plot(gammas, c, "s--", color="#6b6b6b", ms=3.2, lw=1.2, label="classical RBF")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("eff. dim.", fontsize=6.5)
        ax.set_title(f"{name}  ($n$={n})", fontsize=6)
        ax.tick_params(labelsize=5.6)
        ax.grid(alpha=0.25, linewidth=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=5.6, frameon=False)
    axes[-1].set_xlabel("width parameter (bandwidth / $\\gamma$)", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_kernel_spectrum_cohorts", out_dir, paper_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lso-json", type=Path, default=Path("results/abide_lso_results.json"))
    parser.add_argument("--kernel-json", type=Path, default=None,
                        help="quantum_kernel_results.json; newest is used if omitted")
    parser.add_argument("--paper-dir", type=Path, default=Path("paper-tex"))
    parser.add_argument("--sweep-dir", type=Path, default=None,
                        help="topology sweep directory; newest is used if omitted")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or Path("results") / f"{stamp}_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    kernel_json = args.kernel_json
    if kernel_json is None:
        candidates = sorted(Path("results").glob("**/quantum_kernel_results.json"))
        kernel_json = candidates[-1] if candidates else Path("missing.json")

    print(f"figures -> {out_dir}  (paper copy -> {args.paper_dir})")
    fig_fidelity_collapse(out_dir, args.paper_dir)
    fig_lso_comparison(args.lso_json, out_dir, args.paper_dir)
    fig_quantum_kernel(kernel_json, out_dir, args.paper_dir)

    sweep = args.sweep_dir
    if sweep is None:
        candidates = [d for d in sorted(Path("results").glob("*_sweep"))
                      if any(d.glob("*/run.log"))]
        sweep = candidates[-1] if candidates else None
    if sweep and sweep.exists():
        fig_topology_ablation(sweep, out_dir, args.paper_dir)

    fig_cross_cohort(out_dir, args.paper_dir)
    fig_benchmark(out_dir, args.paper_dir)
    fig_parcellation(out_dir, args.paper_dir)
    fig_connectivity_comparison(out_dir, args.paper_dir)
    fig_kernel_spectrum(out_dir, args.paper_dir)
    fig_architectures(out_dir, args.paper_dir)
    fig_quantum_vs_classical(out_dir, args.paper_dir)
    fig_quantum_delta(out_dir, args.paper_dir)
    fig_overfitting(out_dir, args.paper_dir)
    fig_lowdata(out_dir, args.paper_dir)
    fig_degeneracy(out_dir, args.paper_dir)
    fig_kernel_spectrum_cohorts(out_dir, args.paper_dir)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
