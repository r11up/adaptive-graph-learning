# Results

Per-fold metrics for every experiment in the paper. Published so the reported
numbers can be checked without re-running anything.

## What is and is not here

**Published:** per-fold metrics (accuracy, F1, AUC, specificity), paired-test
outputs, run configurations, figures. These contain no subject identifiers —
each fold record carries a site label, a fold size, and metrics.

**Not published:** encoded-cohort caches (`*.pt`, ~535 MB, regenerable in
minutes) and the cohort data itself. ABIDE, ADHD-200, REST-meta-MDD and
UCLA-CNP are all governed by data use agreements and are obtained from their
respective sources, not from this repository. `scripts/` contains the download
and ingest tooling.

## Layout

| directory | contents |
|---|---|
| `rerun22/` | **Current results.** Every variational experiment after the encoding correction described in `findings/22`. These are the numbers in the paper. |
| `qgt/` | Quantum graph transformer runs. Reported in the paper as confirmation of `findings/06`, not as a comparison arm — see `findings/23`. |
| `superseded-prefix/` | **Superseded runs, retained deliberately.** Produced before the encoding correction. Kept so the correction is auditable rather than invisible; do not quote these numbers. |
| everything else | Kernel, population-graph, benchmark and figure runs. Unaffected by the encoding correction — the kernels' learnable bandwidth kept their encoding injective — so these remain current. |

## Why `superseded-prefix/` exists

The QCNN and VQC feature maps encode a feature as a phase, `RZ(2x)`. Inputs
were scaled to `[0, pi]`, so the encoded phase spanned a full `2*pi` period and
the two ends of every feature collapsed onto the same quantum state. Every
variational experiment was re-run after the fix.

Kernel results were never affected and are unchanged. `findings/22` documents
the defect, its scope, and why three existing checks did not catch it.

## Reproducing a number

    python scripts/analyse_plans.py --plan A            # post-fix
    python scripts/analyse_plans.py --plan A --prefix   # the superseded runs
    python scripts/make_metrics.py --all                # confusion matrices
