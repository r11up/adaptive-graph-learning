#!/usr/bin/env bash
set -uo pipefail
PY="./.venv/bin/python"; R="results/rerun22"; mkdir -p "$R"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 OPENBLAS_NUM_THREADS=3
job () { local tag="$1"; shift
  ( echo "[$(date -u +%H:%M:%SZ)] start $tag"; "$@" > "$R/$tag.log" 2>&1
    echo "[$(date -u +%H:%M:%SZ)] done  $tag (rc=$?)" ) & }
echo "--- resume wave 2 (the two that died) ---"
job planA-MDD $PY -u scripts/run_hybrid.py   --data-root data/REST-meta-MDD --out "$R/planA_mdd"
job planC-MDD $PY -u scripts/run_reupload.py --data-root data/REST-meta-MDD --out "$R/planC_mdd"
job planC-ADHD $PY -u scripts/run_reupload.py --data-root data/ADHD-200 --out "$R/planC_adhd"
job planC-UCLA $PY -u scripts/run_reupload.py --data-root data/UCLA-CNP-cc200 --cv stratified --folds 10 --out "$R/planC_ucla"
wait
echo "--- wave 3 ---"
job planB-ABIDE $PY -u scripts/run_ensemble.py --data-root data/ABIDE-I --out "$R/planB_abide"
job planA-ADHD  $PY -u scripts/run_hybrid.py --data-root data/ADHD-200 --out "$R/planA_adhd"
job planA-UCLA  $PY -u scripts/run_hybrid.py --data-root data/UCLA-CNP-cc200 --cv stratified --folds 10 --out "$R/planA_ucla"
job qubit-scaling $PY -u scripts/run_qubit_scaling.py --out "$R/qubit_scaling"
wait
echo "--- wave 4 ---"
job planB-MDD     $PY -u scripts/run_ensemble.py --data-root data/REST-meta-MDD --out "$R/planB_mdd"
job lowdata-ABIDE $PY -u scripts/run_lowdata.py --data-root data/ABIDE-I --out "$R/lowdata_abide"
job qgt-ABIDE     $PY -u scripts/run_graph_transformer.py --data-root data/ABIDE-I --epochs 20 --out results/qgt/abide
job qgt-MDD       $PY -u scripts/run_graph_transformer.py --data-root data/REST-meta-MDD --epochs 20 --out results/qgt/mdd
wait
echo "=== RESUME COMPLETE ==="
