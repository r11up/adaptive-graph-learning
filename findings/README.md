# Findings

A chronological record, one file per measured result. Written as work
progressed, so later findings sometimes revise earlier ones — that history is
kept rather than tidied away, because several of the revisions are the most
useful material here.

## Reading order

**01–06 — locating the constraint.** Fidelity collapses as the register widens
(01); the graph pipeline sits at chance (02); topology repair does not help
(05); the node features are the root cause (06).

**07–13 — does it replicate?** Cross-cohort validation across four cohorts.
Positive results on small cohorts do not survive on large ones (08). Matched
feature budgets make quantum and classical indistinguishable (11), and the
kernel spectrum explains why (12).

**14–17 — attempts to lift the feature budget.** Learned projection (14),
block ensembles (15), register width (16), data re-uploading (17).

**18–21 — where quantum does and does not lead.** Overfitting resistance (18);
the fidelity graph is trainable only through its encoding (19); external
architecture suggestions assessed (20); the low-data crossover tested and
falsified (21).

**22–24 — correction and ledger.** An encoding defect found in our own quantum
arm and fixed (22); the quantum graph transformer, excluded from the paper with
reasons (23); an exhaustive ledger of every quantum advantage across 284
matched comparisons (24).

## Findings revised by later work

| finding | revised by | what changed |
|---|---|---|
| 14 (Plan A) | 22 | Its headline result — that a learned projection significantly harms the quantum arm — did not survive the encoding correction. The addendum records both sets of numbers. |
| 15 (Plan B) | extended | Read as a null with a negative trend once all four cohorts were run, rather than a demonstration that the quantum ensemble is worse. |
| 18 (regularisation) | 21 | Its low-data extrapolation was tested directly and falsified. The regularisation effect itself holds. |
| 13, 16, 17 | 22 | Variational numbers re-run after the encoding fix. |

Superseded result files are kept in `results/superseded-prefix/`.

## Convention

Each finding states what was measured, on what data, under what protocol, and
what it does and does not support. Where a result rests on one cohort or fails
to replicate, the finding says so.
