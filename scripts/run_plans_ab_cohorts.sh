#!/usr/bin/env bash
# Extend Plans A and B from ABIDE-I to the remaining three cohorts.
#
# Sequential by design: these runs are CPU-bound and near-linear in cores, so
# running them concurrently on one machine makes each slower without finishing
# the set any sooner.
#
# ABIDE-I is already done at results/ABIDE_hybrid and results/ABIDE_ensemble
# under these same defaults, so it is not repeated.
#
# Smallest cohort first, so a configuration error surfaces in a minute rather
# than after the half-hour REST-meta-MDD run.
set -uo pipefail
PY="./.venv/bin/python"

run () {
  local plan="$1" script="$2" tag="$3" root="$4"; shift 4
  local dir="results/plan${plan}/${tag}"
  mkdir -p "$dir"
  echo "[$(date -u +%H:%M:%SZ)] === Plan $plan / $tag ==="
  $PY -u "scripts/$script" --data-root "$root" --out "$dir" "$@" \
      > "results/plan${plan}/${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] finished Plan $plan / $tag (rc=$?)"
  sed -n '/^PLAN /,$p' "results/plan${plan}/${tag}.log" | head -14
  echo
}

run A run_hybrid.py   UCLA-CNP data/UCLA-CNP-cc200 --cv stratified --folds 10
run B run_ensemble.py UCLA-CNP data/UCLA-CNP-cc200 --cv stratified --folds 10
run A run_hybrid.py   ADHD-200 data/ADHD-200
run B run_ensemble.py ADHD-200 data/ADHD-200
run A run_hybrid.py   REST-meta-MDD data/REST-meta-MDD
run B run_ensemble.py REST-meta-MDD data/REST-meta-MDD

echo "=== PLANS A AND B COMPLETE ON ALL COHORTS ==="
