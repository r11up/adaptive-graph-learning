#!/usr/bin/env bash
# Re-run everything affected by FINDING 22, using the whole machine.
#
# Each PyTorch job saturates only ~3 of 14 cores, so the earlier sequential
# policy left the machine two-thirds idle. Four concurrent jobs at 3 threads
# each use 12 cores and leave 2 for the system. Thread counts are pinned so
# the jobs do not oversubscribe and thrash.
#
# Memory: peak RSS was ~1.5 GB per ABIDE job and ~4 GB per REST-meta-MDD job,
# so the two MDD jobs are placed in different waves to stay well inside 24 GB.
set -uo pipefail
PY="./.venv/bin/python"
R="results/rerun22"
mkdir -p "$R"

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export VECLIB_MAXIMUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export TORCH_NUM_THREADS=3

job () {
  local tag="$1"; shift
  ( echo "[$(date -u +%H:%M:%SZ)] start $tag"
    "$@" > "$R/$tag.log" 2>&1
    echo "[$(date -u +%H:%M:%SZ)] done  $tag (rc=$?)" ) &
}

wave () { echo "--- wave $1 ---"; }

# Wave 1: the decisive comparisons. One MDD job only.
wave 1
job qmodels-ABIDE $PY -u scripts/run_quantum_models.py --data-root data/ABIDE-I --out "$R/qmodels_abide"
job qmodels-MDD   $PY -u scripts/run_quantum_models.py --data-root data/REST-meta-MDD --out "$R/qmodels_mdd"
job planC-ABIDE   $PY -u scripts/run_reupload.py --data-root data/ABIDE-I --out "$R/planC_abide"
job planA-ABIDE   $PY -u scripts/run_hybrid.py   --data-root data/ABIDE-I --out "$R/planA_abide"
wait

# Wave 2: second MDD pair plus the small cohorts.
wave 2
job planC-MDD    $PY -u scripts/run_reupload.py --data-root data/REST-meta-MDD --out "$R/planC_mdd"
job planA-MDD    $PY -u scripts/run_hybrid.py   --data-root data/REST-meta-MDD --out "$R/planA_mdd"
job qmodels-ADHD $PY -u scripts/run_quantum_models.py --data-root data/ADHD-200 --out "$R/qmodels_adhd"
job qmodels-UCLA $PY -u scripts/run_quantum_models.py --data-root data/UCLA-CNP-cc200 --cv stratified --folds 10 --out "$R/qmodels_ucla"
wait

# Wave 3: remaining studies.
wave 3
job planC-ADHD    $PY -u scripts/run_reupload.py --data-root data/ADHD-200 --out "$R/planC_adhd"
job planC-UCLA    $PY -u scripts/run_reupload.py --data-root data/UCLA-CNP-cc200 --cv stratified --folds 10 --out "$R/planC_ucla"
job planB-ABIDE   $PY -u scripts/run_ensemble.py --data-root data/ABIDE-I --out "$R/planB_abide"
job qubit-scaling $PY -u scripts/run_qubit_scaling.py --out "$R/qubit_scaling"
wait

# Wave 4: the long tail.
wave 4
job planB-MDD     $PY -u scripts/run_ensemble.py --data-root data/REST-meta-MDD --out "$R/planB_mdd"
job lowdata-ABIDE $PY -u scripts/run_lowdata.py --data-root data/ABIDE-I --out "$R/lowdata_abide"
job planA-ADHD    $PY -u scripts/run_hybrid.py --data-root data/ADHD-200 --out "$R/planA_adhd"
job planA-UCLA    $PY -u scripts/run_hybrid.py --data-root data/UCLA-CNP-cc200 --cv stratified --folds 10 --out "$R/planA_ucla"
wait

echo "=== ALL RE-RUNS COMPLETE ==="
