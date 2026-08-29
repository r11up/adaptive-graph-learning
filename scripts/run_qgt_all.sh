#!/usr/bin/env bash
# Quantum Graph Transformer on all four cohorts, at a reduced but UNIFORM cost.
#
# QGT evaluates a 2*node_qubits circuit for every edge pair, per subject, per
# epoch, so cost scales with subjects x edges x epochs. At k=8 and 20 epochs a
# REST-meta-MDD run needs roughly eight hours. Halving the neighbourhood and
# cutting epochs to 12 is ~3.3x cheaper.
#
# The settings are identical across cohorts, which is what the matched
# comparison requires: a difference between cohorts must not be a difference in
# budget. Both arms of every comparison see the same k and the same epochs.
set -uo pipefail
PY="./.venv/bin/python"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 OPENBLAS_NUM_THREADS=3
K=4; EPOCHS=12

job () { local tag="$1"; shift
  ( echo "[$(date -u +%H:%M:%SZ)] start $tag"; "$@" > "results/qgt/$tag.log" 2>&1
    echo "[$(date -u +%H:%M:%SZ)] done  $tag (rc=$?)" ) & }

mkdir -p results/qgt
echo "--- small cohorts first, so a config error surfaces in minutes ---"
job ucla  $PY -u scripts/run_graph_transformer.py --data-root data/UCLA-CNP-cc200 \
      --cv stratified --folds 10 --k $K --epochs $EPOCHS --out results/qgt/ucla
job adhd  $PY -u scripts/run_graph_transformer.py --data-root data/ADHD-200 \
      --k $K --epochs $EPOCHS --out results/qgt/adhd
wait
echo "--- large cohorts ---"
job abide $PY -u scripts/run_graph_transformer.py --data-root data/ABIDE-I \
      --k $K --epochs $EPOCHS --out results/qgt/abide
job mdd   $PY -u scripts/run_graph_transformer.py --data-root data/REST-meta-MDD \
      --k $K --epochs $EPOCHS --out results/qgt/mdd
wait
echo "=== QGT ALL FOUR COHORTS COMPLETE ==="
