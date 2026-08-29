#!/usr/bin/env python3
"""Pool Plan A / B / C results across cohorts into one comparison.

Each plan wrote one JSON per cohort. This reads them together and reports the
things that decide how a plan should be written up:

- per-cohort means for every arm, so a table can be generated rather than
  transcribed;
- every pairwise paired-Wilcoxon test, counted against the number expected to
  reach p < 0.05 by chance, with the Bonferroni threshold;
- for each nominally significant result, which arm lost. Plan C's six hits all
  had the same losing comparator (C-CNN, which fails to train on the smaller
  cohorts), so "which arm lost" is what separates a real effect from a broken
  baseline.

Usage:
    python scripts/analyse_plans.py --plan A
    python scripts/analyse_plans.py --plan B --latex
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# Cohort display name -> (results subdirectory, fold count) per plan.
# Pre-fix layouts are retained so the corrected numbers can be diffed against
# what the paper reported before FINDING 22, rather than silently replaced.
PREFIX = {
    "A": ("hybrid_results.json", {
        "ABIDE~I": Path("results/superseded-prefix/ABIDE_hybrid"),
        "ADHD-200": Path("results/superseded-prefix/planA/ADHD-200"),
        "REST-meta-MDD": Path("results/superseded-prefix/planA/REST-meta-MDD"),
        "UCLA-CNP": Path("results/superseded-prefix/planA/UCLA-CNP"),
    }),
    "B": ("ensemble_results.json", {
        "ABIDE~I": Path("results/superseded-prefix/ABIDE_ensemble"),
        "ADHD-200": Path("results/superseded-prefix/planB/ADHD-200"),
        "REST-meta-MDD": Path("results/superseded-prefix/planB/REST-meta-MDD"),
        "UCLA-CNP": Path("results/superseded-prefix/planB/UCLA-CNP"),
    }),
    "C": ("reupload_results.json", {
        "ABIDE~I": Path("results/superseded-prefix/planC/abide"),
        "ADHD-200": Path("results/superseded-prefix/planC/adhd"),
        "REST-meta-MDD": Path("results/superseded-prefix/planC/mdd"),
        "UCLA-CNP": Path("results/superseded-prefix/planC/ucla"),
    }),
}

# Post-fix runs, after the encoding range was corrected (FINDING 22).
POSTFIX = {
    "A": ("hybrid_results.json", {
        "ABIDE~I": Path("results/rerun22/planA_abide"),
        "ADHD-200": Path("results/rerun22/planA_adhd"),
        "REST-meta-MDD": Path("results/rerun22/planA_mdd"),
        "UCLA-CNP": Path("results/rerun22/planA_ucla"),
    }),
    "B": ("ensemble_results.json", {
        "ABIDE~I": Path("results/rerun22/planB_abide"),
        "REST-meta-MDD": Path("results/rerun22/planB_mdd"),
    }),
    "C": ("reupload_results.json", {
        "ABIDE~I": Path("results/rerun22/planC_abide"),
        "ADHD-200": Path("results/rerun22/planC_adhd"),
        "REST-meta-MDD": Path("results/rerun22/planC_mdd"),
        "UCLA-CNP": Path("results/rerun22/planC_ucla"),
    }),
}

LAYOUT = POSTFIX


def load(plan):
    filename, dirs = LAYOUT[plan]
    out = {}
    for cohort, directory in dirs.items():
        path = directory / filename
        if path.exists():
            out[cohort] = json.load(open(path))
        else:
            print(f"  (missing: {path})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=["A", "B", "C"], required=True)
    parser.add_argument("--prefix", action="store_true",
                        help="read the pre-FINDING-22 runs instead")
    parser.add_argument("--latex", action="store_true", help="emit a LaTeX table body")
    args = parser.parse_args()

    global LAYOUT
    LAYOUT = PREFIX if args.prefix else POSTFIX
    data = load(args.plan)
    if not data:
        print("no results found")
        return 1

    all_p, losers = [], {}
    print(f"\n{'cohort':<16}{'folds':>6}{'sig':>7}{'min p':>9}   arms")
    print("-" * 78)
    for cohort, d in data.items():
        pf = d["per_fold"]
        names = [n for n in pf if pf[n]]
        acc = {n: np.array([f["accuracy"] for f in pf[n]]) for n in names}
        nf = len(acc[names[0]])
        ps = []
        for a, b in itertools.combinations(names, 2):
            if np.any(acc[a] != acc[b]):
                p = wilcoxon(acc[a], acc[b])[1]
                ps.append(p)
                if p < 0.05:
                    lost = b if np.median(acc[a] - acc[b]) > 0 else a
                    losers[lost] = losers.get(lost, 0) + 1
        all_p += ps
        print(f"{cohort:<16}{nf:>6}{sum(p < 0.05 for p in ps):>4}/{len(ps):<2}"
              f"{min(ps) if ps else float('nan'):>9.3f}   {', '.join(names)}")

    print("-" * 78)
    n = len(all_p)
    print(f"TOTAL {sum(p < 0.05 for p in all_p)}/{n} significant uncorrected "
          f"(expected {0.05 * n:.1f} by chance)")
    print(f"      Bonferroni {0.05 / n:.5f}; smallest p {min(all_p):.4f}; "
          f"survivors {sum(p < 0.05 / n for p in all_p)}")
    if losers:
        print("      losing arm in significant tests: "
              + ", ".join(f"{k} x{v}" for k, v in sorted(losers.items(), key=lambda x: -x[1])))

    print(f"\n{'cohort':<16}" + "".join(f"{n:>14}" for n in
                                        list(data.values())[0]["per_fold"]))
    for cohort, d in data.items():
        pf = d["per_fold"]
        row = f"{cohort:<16}"
        for name in pf:
            row += (f"{np.mean([f['accuracy'] for f in pf[name]]):>14.3f}"
                    if pf[name] else f"{'--':>14}")
        print(row)

    if args.latex:
        print("\n% --- LaTeX table body ---")
        for cohort, d in data.items():
            pf = d["per_fold"]
            names = [n for n in pf if pf[n]]
            nf = len(pf[names[0]])
            accs = {n: np.mean([f["accuracy"] for f in pf[n]]) for n in names}
            best = max(accs, key=accs.get)
            print(f"\\multicolumn{{5}}{{|l|}}{{\\textit{{{cohort}, {nf} folds}}}} \\\\")
            print("\\hline")
            for name in names:
                a = np.mean([f["accuracy"] for f in pf[name]])
                f1 = np.mean([f["f1"] for f in pf[name]])
                au = np.mean([f["auc"] for f in pf[name]
                              if np.isfinite(f.get("auc", np.nan))])
                cell = f"\\mathbf{{{a:.3f}}}" if name == best else f"{a:.3f}"
                print(f"{name:<12} & ${cell}$ & ${f1:.3f}$ & ${au:.3f}$ & "
                      f"${d['seconds_per_fold'][name]:.1f}$ \\\\")
            print("\\hline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
