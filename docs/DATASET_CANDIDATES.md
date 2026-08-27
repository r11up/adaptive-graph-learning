# Additional datasets for the connectivity study

The paper's title claims *neuropsychiatric disorders*, plural, but the evidence
is one disorder (ASD) on one cohort family (ABIDE). The most direct way to
strengthen it is to re-run the identical protocol on a **second disorder with a
different multi-site cohort**. If quantum-fidelity topology beats correlation
topology on ASD *and* on ADHD or schizophrenia, under Leave-Site-Out both times,
that is a claim about the representation rather than about one dataset.

Every dataset below is published in a peer-reviewed journal and publicly
available. None come from Kaggle or aggregator re-uploads. They are ordered by
how much work it takes to get them into this pipeline.

---

## Tier 1 — drop-in, same parcellation, no registration

### ADHD-200

**Best single addition.** Same CC200 parcellation, same 0.01–0.1 Hz bandpass,
same C-PAC family, multi-site — so the Leave-Site-Out protocol transfers
unchanged and the only thing that varies is the disorder.

| | |
|---|---|
| Disorder | ADHD vs typically developing |
| Access | INDI S3, anonymous — **no registration** |
| Format | `roi_CC200.1D`, `(T, 200)` — identical to ABIDE |
| Sites | 8 (KKI, NYU, NeuroIMAGE, OHSU, Peking, Pittsburgh, Brown, WashU) |
| Fetch | `python scripts/download_adhd200.py` |

Two release routes, and the difference matters:

- **C-PAC derivatives on S3** (what the script fetches): 104 subjects
  (26 ADHD / 78 control) across 6 sites. Anonymous, scriptable, lands in the
  ABIDE layout so `load_abide(root='data/ADHD-200')` reads it directly.

  **This subset is not viable for a Leave-Site-Out study, and it was measured,
  not assumed.** The per-site class breakdown:

  | site | control | ADHD | total | LSO usable |
  |---|---|---|---|---|
  | KKI | 16 | 8 | 24 | yes |
  | NeuroIMAGE | 7 | 7 | 14 | yes |
  | Peking_1 | 20 | 7 | 27 | yes |
  | OHSU | 14 | 0 | 14 | no — single class |
  | Pittsburgh | 19 | 0 | 19 | no — single class |
  | NYU | 2 | 4 | 6 | no — too small |

  Only 3 of 6 sites survive, carrying 22 ADHD cases between them. Running the
  full protocol on it returns F1 = 0.11 +- 0.22 for the proposed model, with two
  of the three folds collapsing to all-negative predictions — a null result
  driven by cohort size, not by the method. It proves the code path works on a
  second disorder and nothing more.
- **Neuro Bureau "Athena" release**: the full 947 subjects (362 ADHD / 585
  control) with CC200 time series, distributed through NITRC as per-site tar
  archives. Requires a free NITRC account, no data use agreement. Given the
  numbers above this is **not optional** — it is the only route to an ADHD arm
  that can carry a claim. 947 subjects across 8 sites is comparable in weight to
  ABIDE I itself.

> ADHD-200 Consortium. "The ADHD-200 Consortium: a model to advance the
> translational potential of neuroimaging in clinical neuroscience."
> *Frontiers in Systems Neuroscience* 6:62, 2012.
>
> Bellec P. et al. "The Neuro Bureau ADHD-200 Preprocessed Repository."
> *NeuroImage* 144:275–286, 2017.

---

## Tier 2 — open access, needs parcellation first

These ship preprocessed volumes rather than ROI time series, so each needs one
extra step: `nilearn.maskers.NiftiLabelsMasker` with the CC200 atlas to produce
the same `(T, 200)` matrices. That is a standard, well-trodden operation, but it
is compute and disk, not a download.

### UCLA Consortium for Neuropsychiatric Phenomics (ds000030)

Unusually valuable because **three disorders sit in one cohort** with shared
acquisition. That separates two things the ABIDE result cannot: whether the
learned topology captures disorder-specific structure, or just
"patient vs control". A confusion matrix across SZ/BD/ADHD would be a genuinely
new result for this architecture.

| | |
|---|---|
| Disorders | Schizophrenia (50), bipolar (49), ADHD (43), control (130) |
| Access | OpenNeuro `ds000030` — no registration |
| Caveat | Single site, so LSO does not apply; use stratified CV |

> Poldrack R.A. et al. "A phenome-wide examination of neural and cognitive
> function." *Scientific Data* 3:160110, 2016.

### SRPBS Multi-disorder MRI Dataset

**The strongest Leave-Site-Out stress test available.** 11 scanners, multiple
disorders, and a companion *traveling-subject* dataset where the same people
were scanned at different sites — which lets you measure site effects directly
rather than only controlling for them. If the claim is that quantum topology is
robust to scanner variability, this is where to prove it.

| | |
|---|---|
| Disorders | ASD, MDD, schizophrenia, OCD, bipolar, dysthymia, pain, stroke |
| Subjects | 993 patients, 1,421 controls |
| Sites | 11 |
| Access | Open version via BICR/Synapse (`syn22317076`) |
| Caveat | ~75 GB; budget the download |

> Tanaka S.C. et al. "A multi-site, multi-disorder resting-state magnetic
> resonance image database." *Scientific Data* 8:227, 2021.

### COBRE

Small and single-site, but cheap to add and gives a schizophrenia data point.
The NIAK-preprocessed release on figshare is openly downloadable.

| | |
|---|---|
| Disorder | Schizophrenia (72) vs control (74) |
| Access | figshare `10.6084/m9.figshare.4197885` — open |
| Caveat | Single site; 150 volumes, shorter than ABIDE scans |

> Aine C.J. et al. "Multimodal neuroimaging in schizophrenia: description and
> dissemination." *Neuroinformatics* 15:343–364, 2017.

---

## Tier 3 — application or agreement required

Not scriptable here, but worth applying for if the work goes to a journal.

### REST-meta-MDD

The largest depression connectivity cohort, and already distributed as ROI time
series across several atlases — so once access is granted it is a Tier 1
dataset in practice.

| | |
|---|---|
| Disorder | Major depressive disorder |
| Subjects | 1,300 MDD / 1,128 control |
| Sites | 25 |
| Access | Application to the REST-meta-MDD consortium |

> Yan C.-G. et al. "Reduced default mode network functional connectivity in
> patients with recurrent major depressive disorder." *PNAS*
> 116(18):9078–9083, 2019.

### ABIDE II

Needed for the manuscript's pooled 2,214-subject claim. See
[DATASETS.md](DATASETS.md) — requires NITRC registration and a data use
agreement, and no CC200 C-PAC derivative exists on the public bucket.

---

## Recommended plan

1. **ADHD-200 via NITRC (947 subjects).** Highest value per unit effort. Turns
   "neuropsychiatric disorders" from an overclaim into a two-disorder,
   two-cohort result under one protocol. The 104-subject S3 subset already
   downloaded validates the code path end to end while the NITRC transfer is
   arranged.
2. **SRPBS.** The scanner-robustness argument is the weakest link in any
   multi-site fMRI claim, and the traveling-subject companion dataset addresses
   it head-on.
3. **UCLA CNP.** Cheap, and enables the multi-class result — a different kind of
   contribution rather than more of the same.

Anything beyond that is diminishing returns relative to strengthening the
methods: the two ablations the paper itself flags as incomplete (quantum latents
with no graph, and quantum topology with an isotropic GCN) are now implemented
here and cost nothing extra to report.

## A note on comparability

Pooling cohorts preprocessed by *different* pipelines introduces a preprocessing
confound on top of the site confound. ADHD-200's C-PAC derivatives share ABIDE's
processing family, which is why they are Tier 1. For Tier 2 datasets, parcellate
them yourself with a single consistent pipeline rather than mixing vendors'
preprocessed outputs — and report each cohort as its own LSO experiment rather
than merging them into one pool, unless the preprocessing genuinely matches.
