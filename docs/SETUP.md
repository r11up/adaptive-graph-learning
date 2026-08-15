# Local environment

Built and verified on Apple M4 Pro (14 cores, 24 GB), macOS 26.5.2, Python 3.13.7,
all packages installed as native `arm64` wheels — no Rosetta, no source builds.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz,quantum,neuro]"
```

Verify:

```bash
python -c "import torch, pennylane as qml, platform; print(platform.machine(), torch.__version__, qml.version())"
pytest -q
```

Expected: `arm64 2.13.0 0.45.1`, all tests passing.

## Which quantum framework

**PennyLane, with the batched PyTorch simulator as the default execution path.**

The paper specifies PennyLane with adjoint differentiation, and that is what
`PennyLaneRingEncoder` provides via `lightning.qubit`. But the workload here is
unusual: the *same* 16-qubit circuit is executed once per brain region, 200
times per subject. PennyLane executes those region by region; the built-in
simulator evolves all 200 as one batched complex tensor.

Measured on this machine, 200 regions, forward + backward:

| backend | time per subject |
|---|---|
| `RingEntangledEncoder` (batched PyTorch) | **1.9 s** |
| `PennyLaneRingEncoder` (`lightning.qubit`, adjoint) | ~12.6 s |

Both are exercised in CI and pinned to agree — `tests/test_fmri_encoder.py`
checks expectation values to 1e-5, statevector overlap above 0.99999, and
parameter gradients to 1e-4. The speedup is therefore a pure execution-strategy
win, not an approximation.

Select the backend with `--backend torch` (default) or `--backend pennylane`.

**Qiskit** is supported for the generic patent pipeline
(`qagta.quantum.qiskit_backend`, install with `pip install -e ".[qiskit]"`) but
is not the right tool for this study: its `EstimatorQNN` does not expose
statevectors, and the fidelity metric this framework is built on needs them.

## Compute profile

The dominant cost is quantum encoding, and it depends only on the circuit
parameters — so it is computed once for the whole cohort and cached to
`results/abide_encoded.pt`.

| stage | cost on this machine |
|---|---|
| ABIDE I download (1035 subjects) | ~15 min, 406 MB |
| Load + per-region PCA | ~2 min |
| Quantum encoding, 1035 × 200 regions | ~10 min (cached afterwards) |
| Leave-Site-Out, 20 folds × 30 epochs | ~20 min per model configuration |

Encoding uses ~6 GB peak RSS when gradients are enabled, so keep to one subject
at a time on 24 GB if you unfreeze the quantum layer.

### On end-to-end quantum training

The paper trains the whole pipeline end to end on A100 GPUs, with gradients
flowing from the classification loss back into the circuit. That is a different
compute class from this laptop: backpropagating through the circuit costs 1.9 s
per subject, so a single 30-epoch pass over 1035 subjects is ~16 hours, before
multiplying by 20 LSO folds.

The default configuration therefore encodes with the circuit held fixed and
trains the adaptive-edge kernel and GAT on cached latents, which makes the full
20-fold protocol tractable. The circuit remains fully differentiable and the
gradient path is tested; end-to-end fine-tuning is available and correct, just
not the default at full cohort scale.

## Reproducing the study

```bash
python scripts/download_abide.py                    # ~15 min, 406 MB
python scripts/run_abide_study.py --epochs 30       # full 20-fold LSO
python scripts/run_abide_study.py --limit 120 --epochs 10   # quick check
```

Results land in `results/abide_lso_results.json` with per-fold detail.

Adding `--permutations 100` runs the label-permutation null. It re-runs the
entire LSO evaluation once per shuffle, so budget roughly 100× the base runtime
and start it overnight.

## Troubleshooting

**Downloads failing with `URLError`.** S3 throttles bursts of anonymous
requests. The script retries with exponential backoff and is resumable — just
run it again, and lower `--workers` (default 8) if failures persist.

**`no usable subjects found`.** The derivative files did not download. Check
`data/abide/ABIDE_pcp/cpac/filt_noglobal/` contains `.1D` files.

**Cache/cohort size mismatch.** The cache is keyed to the cohort it was built
from; pass `--refresh-cache` after changing `--limit`, `--n-qubits`, or the
atlas.
