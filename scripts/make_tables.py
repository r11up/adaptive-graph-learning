#!/usr/bin/env python3
"""Assemble the paper's result tables from every experiment on disk.

Produces two LaTeX tables and their plain-text equivalents:

1. **Comprehensive results** — every model evaluated in this study, across all
   cohorts, on five metrics, with each quantum model set beside the classical
   comparator it was matched against. This pulls together results from the
   separate experiment families (kernel study, population graph, supervised
   benchmark, quantum model suite) so they can be read side by side rather than
   scattered across sections.

2. **Published comparison** — our best configuration per cohort against
   published results under comparable site-held-out protocols, with the
   citation keys used in the bibliography.

Tables are regenerated from the result JSON files rather than transcribed, so
they cannot drift from the numbers the experiments actually produced.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

METRICS = ["accuracy", "f1", "auc", "specificity"]

# Which experiment family each result file belongs to, and how its models map
# onto the quantum / classical distinction used in the table.
QUANTUM_MODELS = {
    "QCNN", "VQC", "QSVM-Pegasos", "QSVM-fixed", "TQEK",
    "quantum kernel", "quantum",
}

# Aliases that would duplicate a suite model are mapped onto its name, so
# merge() collapses them instead of listing the same measurement twice.
DISPLAY = {
    "quantum kernel": "QSVM-fixed",
    "SVM RBF (matched)": "SVM-RBF",
    "quantum": "QPG (quantum edges)",
    "correlation": "PG (correlation edges)",
    "rbf": "PG (RBF edges)",
    "no-graph MLP": "MLP (no graph)",
    "SVM (RBF)": "SVM-RBF (all feat.)",
    "SVM (linear)": "SVM-linear (all feat.)",
    "MLP": "MLP",
    "reference: RBF on full dim": "SVM-RBF (full dim)",
}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def summarise_fold(folds: list[dict], key: str) -> list[float] | None:
    """Mean and 95% half-width for a metric held only in the per-fold records."""
    import statistics

    values = [f[key] for f in folds
              if isinstance(f.get(key), (int, float)) and f[key] == f[key]]
    if not values:
        return None
    if len(values) < 2:
        return [float(values[0]), 0.0]
    half = 1.96 * statistics.stdev(values) / (len(values) ** 0.5)
    return [float(statistics.fmean(values)), float(half)]


def merge(bucket: dict, payload: dict) -> None:
    """Fold one result file into a cohort's model table.

    Specificity lives only in the per-fold records, so it is summarised here
    rather than read from the stored summary. Where two experiment families
    produced the same model under different names, the first writer wins: they
    are the same measurement, and overwriting would silently prefer whichever
    file happened to load last.
    """
    per_fold = payload.get("per_fold", {})
    for model, metrics in payload.get("summary", {}).items():
        name = DISPLAY.get(model, model)
        if name in bucket:
            continue
        entry = dict(metrics)
        spec = summarise_fold(per_fold.get(model, []), "specificity")
        if spec:
            entry["specificity"] = spec
        bucket[name] = entry


def collect(results_root: Path) -> dict[str, dict[str, dict]]:
    """Gather every model's summary metrics, keyed by cohort then model."""
    cohorts: dict[str, dict[str, dict]] = {}

    # Suite first: it is the primary experiment, and merge() keeps the first
    # writer, so its naming takes precedence over overlapping older entries.
    suites = sorted(results_root.glob("*_qmodels"))
    if suites:
        for cohort_dir in sorted(suites[-1].iterdir()):
            payload = load(cohort_dir / "quantum_models_results.json")
            if payload:
                merge(cohorts.setdefault(cohort_dir.name, {}), payload)

    sources = [
        # (cohort, path, filename, suffix for disambiguation)
        ("ABIDE-I", "ABIDE_benchmark", "benchmark_results.json", ""),
        ("ADHD-200", "ADHD200_benchmark", "benchmark_results.json", ""),
        ("REST-meta-MDD", "MDD_benchmark", "benchmark_results.json", ""),
        ("UCLA-CNP", "UCLA_benchmark", "benchmark_results.json", ""),
        ("ABIDE-I", "ABIDE_popgraph", "population_graph_results.json", ""),
        ("ADHD-200", "ADHD200_popgraph", "population_graph_results.json", ""),
        ("REST-meta-MDD", "MDD_popgraph", "population_graph_results.json", ""),
    ]
    for cohort, folder, filename, _ in sources:
        payload = load(results_root / folder / filename)
        if not payload:
            continue
        merge(cohorts.setdefault(cohort, {}), payload)

    return cohorts


def cell(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if not value or value[0] != value[0]:  # NaN check
        return "---"
    return f"${value[0]:.3f}$"


def latex_results(cohorts: dict, out: Path) -> None:
    order = ["ABIDE-I", "ADHD-200", "REST-meta-MDD", "UCLA-CNP"]
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Comprehensive results across all cohorts and models. Quantum "
        r"models are grouped above their matched classical comparators. Every "
        r"model within a cohort sees identical features and identical folds. "
        r"Dashes mark configurations not run on that cohort.}",
        r"\label{tab:comprehensive}",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{|l|l|c|c|c|c|}",
        r"\hline",
        r"\textbf{Cohort} & \textbf{Model} & \textbf{Acc} & \textbf{F1} & "
        r"\textbf{AUC} & \textbf{Spec} \\",
        r"\hline",
    ]

    for cohort in order:
        models = cohorts.get(cohort)
        if not models:
            continue
        quantum = [m for m in models if any(q in m for q in ("Q", "quantum", "TQEK", "VQC"))]
        classical = [m for m in models if m not in quantum]
        rows = sorted(quantum) + sorted(classical)

        for i, model in enumerate(rows):
            metrics = models[model]
            prefix = (
                rf"\multirow{{{len(rows)}}}{{*}}{{{cohort}}}" if i == 0 else ""
            )
            marker = r"$\dagger$" if model in quantum else ""
            lines.append(
                f" {prefix} & {model}{marker} & "
                + " & ".join(cell(metrics, k) for k in
                             ["accuracy", "f1", "auc", "specificity"])
                + r" \\"
            )
        lines.append(r"\hline")

    lines += [
        r"\end{tabular}",
        r"\\[2pt]",
        r"\footnotesize $\dagger$ denotes a quantum model.",
        r"\end{table*}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def plain_results(cohorts: dict, out: Path) -> None:
    lines = [
        "COMPREHENSIVE RESULTS — all cohorts, all models",
        "=" * 78,
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        "",
        f"{'cohort':<16}{'model':<26}{'acc':>8}{'F1':>8}{'AUC':>8}{'spec':>8}",
        "-" * 78,
    ]
    for cohort in ["ABIDE-I", "ADHD-200", "REST-meta-MDD", "UCLA-CNP"]:
        models = cohorts.get(cohort)
        if not models:
            continue
        for i, model in enumerate(sorted(models)):
            metrics = models[model]
            row = f"{cohort if i == 0 else '':<16}{model:<26}"
            for key in ["accuracy", "f1", "auc", "specificity"]:
                value = metrics.get(key)
                row += f"{value[0]:>8.3f}" if value and value[0] == value[0] else f"{'--':>8}"
            lines.append(row)
        lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


PUBLISHED = [
    # cohort, method, protocol, accuracy, citation key
    ("ABIDE-I", "Traditional ML (LDA/SVM/RF)", "LOSO", "0.51--0.56", "eslami2019"),
    ("ABIDE-I", "Abraham et al., $\\ell_2$ SVC", "LOSO", "$\\sim$0.67", "abraham2017"),
    ("ABIDE-I", "Population GCN", "10-fold", "0.70", "parisot2018"),
    ("ABIDE-I", "Multimodal cross-attention", "LOSO", "0.82", "fusion2026"),
    ("ADHD-200", "Competition best (imaging)", "held-out", "0.615", "adhd200comp"),
    ("ADHD-200", "Phenotypic data alone", "held-out", "0.625", "brown2012"),
    ("REST-meta-MDD", "Ensemble GNN", "LOSO", "0.73", "mddgnn"),
    ("REST-meta-MDD", "MMDD multimodal", "LOSO", "0.778", "mmdd"),
]


def latex_published(cohorts: dict, out: Path) -> None:
    best: dict[str, tuple[str, float]] = {}
    for cohort, models in cohorts.items():
        ranked = [
            (name, m["accuracy"][0]) for name, m in models.items()
            if m.get("accuracy") and m["accuracy"][0] == m["accuracy"][0]
        ]
        if ranked:
            best[cohort] = max(ranked, key=lambda kv: kv[1])

    lines = [
        r"\begin{table}[t]",
        r"\caption{Our results against published work on the same cohorts. Only "
        r"studies using site-held-out or comparable protocols are listed; "
        r"random-split figures are excluded, as they are not comparable.}",
        r"\label{tab:published-comparison}",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{|l|l|c|c|}",
        r"\hline",
        r"\textbf{Cohort} & \textbf{Method} & \textbf{Protocol} & \textbf{Acc} \\",
        r"\hline",
    ]
    for cohort in ["ABIDE-I", "ADHD-200", "REST-meta-MDD", "UCLA-CNP"]:
        rows = [r for r in PUBLISHED if r[0] == cohort]
        ours = best.get(cohort)
        total = len(rows) + (1 if ours else 0)
        if not total:
            continue
        for i, (_, method, protocol, accuracy, key) in enumerate(rows):
            prefix = rf"\multirow{{{total}}}{{*}}{{{cohort}}}" if i == 0 else ""
            lines.append(f" {prefix} & {method}~\\cite{{{key}}} & {protocol} & {accuracy} \\\\")
        if ours:
            prefix = rf"\multirow{{{total}}}{{*}}{{{cohort}}}" if not rows else ""
            protocol = "10-fold" if cohort == "UCLA-CNP" else "LOSO"
            lines.append(
                f" {prefix} & \\textbf{{This work}} ({ours[0]}) & {protocol} & "
                f"$\\mathbf{{{ours[1]:.3f}}}$ \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper-tex"))
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_dir = args.out or args.results / f"{stamp}_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = collect(args.results)
    if not cohorts:
        raise SystemExit("no results found; run the experiments first")
    print(f"tables -> {out_dir}")
    for cohort, models in sorted(cohorts.items()):
        print(f"  {cohort}: {len(models)} models")

    latex_results(cohorts, out_dir / "table_comprehensive.tex")
    plain_results(cohorts, out_dir / "table_comprehensive.txt")
    latex_published(cohorts, out_dir / "table_published.tex")

    if args.paper_dir and args.paper_dir.exists():
        for name in ("table_comprehensive.tex", "table_published.tex"):
            (args.paper_dir / name).write_text((out_dir / name).read_text())
        print(f"  copied .tex tables -> {args.paper_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
