#!/usr/bin/env bash
# Run the quantum model suite on every cohort, one after another.
#
# Sequential by design: these runs are CPU-bound and near-linear in cores, so
# running them concurrently on one machine makes each slower without finishing
# the set any sooner, and leaves the machine unusable meanwhile.
set -uo pipefail

EPOCHS="${1:-150}"
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"
ROOT="results/${STAMP}_qmodels"
PY="./.venv/bin/python"
mkdir -p "$ROOT"

echo "quantum model suite -> $ROOT  (epochs=$EPOCHS)"
echo "started $(date -u +%H:%M:%SZ)"

run () {
  local tag="$1" root="$2"; shift 2
  local dir="$ROOT/$tag"
  echo
  echo "[$(date -u +%H:%M:%SZ)] === $tag ==="
  $PY scripts/run_quantum_models.py \
      --data-root "$root" --epochs "$EPOCHS" --out "$dir" "$@" \
      > "$ROOT/$tag.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] finished $tag (rc=$?)"
  sed -n '/QUANTUM MODEL SUITE/,$p' "$ROOT/$tag.log" | head -16
}

# Smallest first, so a configuration error surfaces in minutes not hours.
run UCLA-CNP      data/UCLA-CNP-cc200 --cv stratified --folds 10
run ADHD-200      data/ADHD-200
run ABIDE-I       data/ABIDE-I
run REST-meta-MDD data/REST-meta-MDD

echo
echo "=== ALL COHORTS COMPLETE -> $ROOT ==="
