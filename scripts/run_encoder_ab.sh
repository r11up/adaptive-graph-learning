#!/usr/bin/env bash
# Experiment 1: does restoring trainable capacity to the fidelity graph help?
#
# FINDING 19 showed the graph is learnable through 8 parameters, because the
# ansatz sits after the encoding and cancels in the fidelity. InterleavedEncoder
# places trainable blocks between data layers so all 64 reach the kernel.
#
# The comparison is the same graph pipeline, same folds, same everything, with
# only the encoder swapped. Separate caches: the encoders produce different
# embeddings and sharing a cache would silently compare one against itself.
set -uo pipefail
PY="./.venv/bin/python"
Q="${1:-8}"
mkdir -p results/encoder_ab

for backend in torch interleaved; do
  echo "[$(date -u +%H:%M:%SZ)] === graph pipeline, backend=$backend, ${Q} qubits ==="
  $PY -u scripts/run_abide_study.py \
      --backend "$backend" --n-qubits "$Q" \
      --cache "results/encoder_ab/encoded_${backend}_q${Q}.pt" \
      --out "results/encoder_ab/${backend}_q${Q}" \
      > "results/encoder_ab/${backend}_q${Q}.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] finished $backend (rc=$?)"
  tail -20 "results/encoder_ab/${backend}_q${Q}.log"
  echo
done
echo "=== ENCODER A/B COMPLETE ==="
