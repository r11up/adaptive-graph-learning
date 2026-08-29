#!/usr/bin/env bash
# Re-run every experiment whose quantum arm was affected by FINDING 22.
#
# The QCNN and VQC feature maps encode features as RZ(2x) phases. Inputs were
# scaled to [0, pi], spanning a full 2*pi period, so the two ends of every
# feature collapsed onto the same state. Fixed to [0, pi/2].
#
# Kernel results (QSVM, TQEK) are NOT re-run: their learnable bandwidth kept
# the phase injective, so those numbers are unaffected and stay reproducible.
#
# Ordered by decisiveness, so the comparisons that matter most land first.
# Sequential: these are CPU-bound and near-linear in cores.
set -uo pipefail
PY="./.venv/bin/python"
R="results/rerun22"
mkdir -p "$R"

run () {
  local tag="$1"; shift
  echo "[$(date -u +%H:%M:%SZ)] === $tag ==="
  "$@" > "$R/$tag.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] finished $tag (rc=$?)"
  sed -n '/^PLAN \|^QUANTUM MODEL SUITE\|^EXPERIMENT /,$p' "$R/$tag.log" | head -16
  echo
}

# 1. The core matched comparison, both well-powered cohorts first.
run qmodels-ABIDE $PY -u scripts/run_quantum_models.py --data-root data/ABIDE-I      --out "$R/qmodels_abide"
run qmodels-MDD   $PY -u scripts/run_quantum_models.py --data-root data/REST-meta-MDD --out "$R/qmodels_mdd"

# 2. Plan C (re-uploading) — the cleanest quantum-vs-quantum test.
run planC-ABIDE $PY -u scripts/run_reupload.py --data-root data/ABIDE-I       --out "$R/planC_abide"
run planC-MDD   $PY -u scripts/run_reupload.py --data-root data/REST-meta-MDD --out "$R/planC_mdd"

# 3. Plan A — carried the strongest anti-quantum result, so most at stake.
run planA-ABIDE $PY -u scripts/run_hybrid.py --data-root data/ABIDE-I       --out "$R/planA_abide"
run planA-MDD   $PY -u scripts/run_hybrid.py --data-root data/REST-meta-MDD --out "$R/planA_mdd"

# 4. Remaining cohorts and the smaller studies.
run qmodels-ADHD $PY -u scripts/run_quantum_models.py --data-root data/ADHD-200 --out "$R/qmodels_adhd"
run qmodels-UCLA $PY -u scripts/run_quantum_models.py --data-root data/UCLA-CNP-cc200 --cv stratified --folds 10 --out "$R/qmodels_ucla"
run planC-ADHD   $PY -u scripts/run_reupload.py --data-root data/ADHD-200 --out "$R/planC_adhd"
run planC-UCLA   $PY -u scripts/run_reupload.py --data-root data/UCLA-CNP-cc200 --cv stratified --folds 10 --out "$R/planC_ucla"
run planB-ABIDE  $PY -u scripts/run_ensemble.py --data-root data/ABIDE-I --out "$R/planB_abide"
run planB-MDD    $PY -u scripts/run_ensemble.py --data-root data/REST-meta-MDD --out "$R/planB_mdd"
run qubit-scaling $PY -u scripts/run_qubit_scaling.py --out "$R/qubit_scaling"
run lowdata-ABIDE $PY -u scripts/run_lowdata.py --data-root data/ABIDE-I --out "$R/lowdata_abide"

echo "=== RE-RUN AFTER FINDING 22 COMPLETE ==="
