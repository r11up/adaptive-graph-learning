#!/usr/bin/env bash
# Sweep the configurations that the fidelity measurement implicates.
#
# Every run writes to its own timestamped directory under results/, so
# nothing overwrites an earlier result:
#
#   results/<UTC timestamp>_<tag>/
#       run.log                  full console output
#       abide_lso_results.json   per-fold metrics
#       config.txt               exact command that produced it
#
# Usage:  bash scripts/run_sweep.sh [epochs]
set -uo pipefail

EPOCHS="${1:-30}"
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"
ROOT="results/${STAMP}_sweep"
PY="./.venv/bin/python"
mkdir -p "$ROOT"

echo "sweep -> $ROOT  (epochs=$EPOCHS)"

# tag | qubits | topology | extra flags
#
# q04_mixed carries the classical baselines; the others skip them, since the
# SVM/GCN comparators are re-run per PCA width and one pass is enough to
# anchor the comparison.
run () {
  local tag="$1" qubits="$2" topo="$3"; shift 3
  local dir="$ROOT/$tag"
  mkdir -p "$dir"
  echo "[$(date -u +%H:%M:%SZ)] starting $tag (qubits=$qubits topology=$topo)"
  {
    echo "tag=$tag qubits=$qubits topology=$topo epochs=$EPOCHS extra=$*"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$dir/config.txt"

  $PY scripts/run_abide_study.py \
      --n-qubits "$qubits" \
      --topology "$topo" \
      --epochs "$EPOCHS" \
      --cache "$dir/encoded.pt" \
      --refresh-cache \
      --out "$dir" \
      "$@" > "$dir/run.log" 2>&1

  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$?" >> "$dir/config.txt"
  echo "[$(date -u +%H:%M:%SZ)] finished $tag"
  tail -12 "$dir/run.log" | sed "s/^/    /"
}

# 4 qubits is where fidelity retains usable spread (47% of region pairs
# above 0.01, against 0% at 16) and is the width the original notebook used.
run q04_mixed 4 mixed
run q04_fidelity 4 fidelity --skip-baselines
run q08_mixed 8 mixed --skip-baselines
run q16_mixed 16 mixed --skip-baselines

echo
echo "=== sweep complete: $ROOT ==="
grep -H "Proposed (quantum" "$ROOT"/*/run.log 2>/dev/null | sed 's|results/||'
