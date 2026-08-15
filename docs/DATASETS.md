# Datasets

## What the study needs

Per subject, the pipeline consumes a matrix of regional BOLD time series:

```
X_raw ∈ R^{200 × T}     200 CC200 regions of interest, T ≈ 78–316 timepoints
```

compressed by per-region PCA to `X ∈ R^{200 × 16}` — one node per brain region,
16 temporal features per node, one qubit per feature. Each subject also needs a
diagnostic label (ASD / control) and an **acquisition site**, since the
evaluation protocol is Leave-Site-Out.

## ABIDE I — automated, no registration

Fetched from the Preprocessed Connectomes Project's anonymous S3 bucket:

```bash
python scripts/download_abide.py
```

| | |
|---|---|
| Subjects downloaded | 1035 |
| Class balance | 505 ASD / 530 control |
| Sites | 20, all with ≥ 10 subjects |
| Parcellation | CC200 (200 ROIs) |
| Pipeline | C-PAC, `filt_noglobal` |
| Size on disk | ~406 MB |
| Timepoints | 78–316, varies by site |

The C-PAC `filt_noglobal` derivative applies slice-timing correction, motion
realignment, nuisance regression (white matter, CSF, 24 motion parameters), and
a 0.01–0.1 Hz bandpass, without global signal regression.

The phenotypic table lists 1112 subjects; 77 carry the sentinel `no_filename`
because preprocessing failed for them, and no derivative file exists to
download. 1035 is therefore the full usable ABIDE I cohort for this
parcellation, not a subsample. The loader reports any further exclusions rather
than silently shrinking the sample.

Other parcellations and pipelines are available through the same script:

```bash
python scripts/download_abide.py --atlas rois_aal        # or rois_ho, rois_cc400
python scripts/download_abide.py --pipeline dparsf
python scripts/download_abide.py --global-signal         # filt_global variant
```

## ABIDE II — not automatable

**ABIDE II is not available in the form this pipeline needs, and cannot be
downloaded without registration.** This is worth stating plainly, because the
manuscript reports a pooled ABIDE I + II cohort of 2,214 subjects.

The INDI S3 bucket does host an `ABIDE2` project, but it carries only:

```
data/Projects/ABIDE2/Outputs/{fmriprep, denoise, mriqc, mindboggle_swf}/
data/Projects/ABIDE2/Derivatives/DCAN/
data/Projects/ABIDE2/RawData/
```

There is no C-PAC + CC200 ROI time-series derivative equivalent to ABIDE I's.
Two routes exist, neither cheap:

1. **NITRC-IR registration.** ABIDE II requires a NITRC account and a signed
   data use agreement. Access is per-user and cannot be scripted here.
2. **Derive CC200 features from `fmriprep` outputs.** The preprocessed BOLD
   volumes are on S3 without registration, but they are full 4-D NIfTI files
   (tens of GB), and each would need parcellating with the CC200 atlas via
   `nilearn.maskers.NiftiLabelsMasker`. That is a days-long download plus a
   substantial compute job — well outside "lightweight", and it would use a
   *different* preprocessing pipeline than ABIDE I, so pooling the two would
   introduce a preprocessing confound on top of the site confound the LSO
   protocol is designed to test.

Everything in this repository therefore runs on ABIDE I. ABIDE I alone supports
the full protocol — 20 sites is ample for Leave-Site-Out — but any claim about a
2,214-subject pooled cohort is **not** reproducible from this repo as it stands.

### If you obtain ABIDE II

Drop per-subject ROI time-series files into the same layout and the loader picks
them up unchanged:

```
data/abide/ABIDE_pcp/<pipeline>/<strategy>/<FILE_ID>_rois_cc200.1D
data/abide/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv   # needs FILE_ID, DX_GROUP, SITE_ID
```

`load_abide()` keys off those three phenotypic columns only, so extending to a
pooled cohort means appending rows to the phenotypic table and files to the
series directory. Use distinct `SITE_ID` values for ABIDE II sites so LSO treats
them as separate folds.

## The `data/Abide-2-eeg-only/` folder is not ABIDE II

That directory contains **EEG recordings, not fMRI**:

```
subject{1..42}/EEG/subject{N}_eeg.mat        seg: (10000, 30, 200)
subject{1..42}/EEG/subject{N}_eeg_label.mat  label: (10, 200)
```

42 subjects, 30 EEG channels, 10 classes — the structure of the EAV
(EEG-Audio-Video) emotion dataset, ~18 GB. It has no brain-region parcellation,
no BOLD signal, no ASD/control labels, and no ABIDE site metadata, so it cannot
serve the ABIDE II arm of this study or any part of the fMRI pipeline.

It could support a *separate* study — the architecture is modality-agnostic, and
EEG channels can be treated as graph nodes exactly as ROIs are here — but that
is a different experiment with different labels, not the pooled ABIDE cohort the
manuscript describes.

## Synthetic data

For tests, CI, and offline development:

```bash
python scripts/generate_data.py --samples 300 --features 10 --out data/synthetic.csv
```

This exercises the generic pipeline, not the connectome study. See the note in
the README about what it does and does not demonstrate.
