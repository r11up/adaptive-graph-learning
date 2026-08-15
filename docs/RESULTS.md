# Results and a blocking finding

Run on the full ABIDE I cohort: 1035 subjects, 20 sites, CC200 / C-PAC
`filt_noglobal`, 30 epochs per fold, Leave-Site-Out over every site.

## What came out

| model | F1 | accuracy | specificity |
|---|---|---|---|
| SVM (linear) | **0.653 ± 0.039** | 0.671 ± 0.036 | 0.689 ± 0.067 |
| SVM (RBF) | 0.627 ± 0.054 | 0.642 ± 0.042 | 0.634 ± 0.088 |
| GCN + Pearson | 0.252 ± 0.118 | 0.494 ± 0.029 | 0.664 ± 0.185 |
| GCN + RBF | 0.145 ± 0.119 | 0.508 ± 0.029 | 0.783 ± 0.179 |
| Proposed (quantum + GAT) | 0.230 ± 0.141 | 0.506 ± 0.028 | 0.650 ± 0.214 |
| Ablation (quantum + GCN) | 0.164 ± 0.128 | 0.508 ± 0.028 | 0.750 ± 0.195 |

Over-smoothing diagnostic: **MAD = 0.003**.

Two things to read here, and they point in opposite directions.

**The classical half reproduces.** SVM on flattened correlation matrices reaches
F1 0.653 (linear) and 0.627 (RBF) under Leave-Site-Out, against 0.58 and 0.62 in
the manuscript. That is close, slightly better, and it validates the parts that
could have been silently wrong: subject loading, the ASD/control label mapping,
site assignment, and the LSO fold construction. The data carries the signal the
paper says it does.

**Every graph configuration sits at chance.** Accuracy 0.49–0.51 across all four
GNN variants, with per-fold behaviour that gives the game away: folds either
predict all-control (F1 = 0.000, specificity 1.000) or all-ASD (F1 ≈ 0.65,
specificity 0.000). The models are not learning a decision boundary; they are
collapsing to a single class per fold. This does **not** reproduce the
manuscript's reported F1 of 0.72, and the honest statement is that the graph
pipeline as specified does not currently work at this scale.

## Diagnosis: fidelity collapses in a 2^16-dimensional Hilbert space

Measured directly on the encoded cohort:

```
pairwise fidelity between regions:  mean 0.00001   max 0.0225
fraction of region pairs > 0.01:    0.000
edge weights after k-NN:            mean 0.0016
node-feature MAD after propagation: 0.003
```

With 16 qubits the state space has 65,536 dimensions, and **every pair of
regional states is essentially orthogonal** — not just unrelated ones. Mean
fidelity is 1e-5 with a maximum of 0.02, so `|⟨ψi|ψj⟩|²` has no usable dynamic
range. The k-NN step then selects the top 20 of what is effectively noise, the
resulting topology is close to random, and the GAT propagates over it until node
features are indistinguishable (MAD 0.003, versus 0.45 reported in the paper).

This matters for the manuscript's argument specifically. Section III-C treats
near-orthogonality as a *benefit* — "unrelated vectors tend to be close to
orthogonal … genuine structural relationships stand out, giving a naturally
sparser and cleaner adjacency matrix". At 16 qubits with this angle encoding
that mechanism does not hold: everything is orthogonal, so nothing stands out.
The claimed average degree of k ≈ 25 with visible DMN/ECN block structure
requires fidelity to retain real spread, which it does not here.

A contributing factor, though not the main one: PCA component variance decays
steeply (per-component σ from 4.8 down to 1.4), so the later qubits receive
near-constant rotations and contribute little. Standardising the components per
dimension before encoding raises mean fidelity only from 1e-5 to 4e-5 — a
measurable improvement, nowhere near a fix.

## What this does not mean

It is not evidence that quantum-derived topology fails. It is evidence that
**this configuration** destroys the metric it depends on. The distinction
matters: the failure is in the fidelity scale, upstream of the graph and the
classifier, and every downstream number inherits it.

## Directions worth testing

Ranked by how directly they attack the measured cause:

1. **Fewer qubits.** 8 qubits gives a 256-dimensional space where states can
   actually overlap. This is the most direct lever on measure concentration and
   the cheapest to test.
2. **Reduced-density-matrix fidelity.** Compare per-qubit or few-qubit marginals
   rather than the full state, so similarity is not diluted across 65,536
   amplitudes.
3. **Bounded angle ranges.** Compressing encoding angles keeps states in a
   neighbourhood of one another, preserving overlap; the current `x · θ · π`
   scaling spreads them across the whole Bloch sphere.
4. **Revisit global mean pooling.** Averaging 200 region features discards
   *which* region carries what, while the SVM baseline sees all 19,900 pairwise
   correlations. Some of the gap between 0.65 and 0.23 is likely read-out, not
   topology.

Items 1–3 are the ones the measurement points at. Item 4 is a separate concern
that would matter even with healthy fidelity.

## ADHD-200

Run for completeness on the 104-subject S3 subset; not interpretable. Only 3 of
6 sites can form a fold (OHSU and Pittsburgh contain zero ADHD cases), and the
proposed model returns F1 = 0.111 ± 0.218. See
[DATASET_CANDIDATES.md](DATASET_CANDIDATES.md) — the 947-subject NITRC release
is the only version of this cohort that can carry a result.

## Reproducing

```bash
python scripts/download_abide.py
python scripts/run_abide_study.py --epochs 30
```

Raw output in `results/abide_lso_run.log`, per-fold metrics in
`results/abide_lso_results.json`.
