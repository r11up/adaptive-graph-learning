# Data request — shortlist for the acquisition team

Four datasets, ranked. For each: what it adds to the argument, and **the exact
derivative to obtain**. The derivative format is the part that usually goes
wrong, so it is stated explicitly — the wrong one means weeks of reprocessing.

**What the pipeline consumes.** Per subject, one file of regional BOLD time
series, `(T timepoints × 200 regions)`, whitespace- or comma-delimited, plus a
table giving each subject's diagnosis and acquisition site. Anything already in
that form is zero-effort to ingest. Anything that arrives as 4-D NIfTI volumes
needs a parcellation pass first — doable, but it is compute and disk, and it
must use the *same* pipeline across cohorts or it introduces a preprocessing
confound on top of the site confound the protocol exists to measure.

---

## 1. ADHD-200 — highest priority, no application needed

**Adds:** a second disorder. This is what turns "neuropsychiatric disorders"
in the title from an overclaim into a two-disorder result under one protocol.
Nothing else on this list is as cheap.

| | |
|---|---|
| Obtain | "CC200 Time Courses (Corrected Filtering)", ~274 MB, and "Test Release CC200 Time Courses", ~131 MB |
| From | https://www.nitrc.org/frs/?group_id=383 |
| Access | Free NITRC account. **No data use agreement.** |
| Yields | 947 subjects (362 ADHD / 585 control), 8 sites |
| Ingest | `python scripts/ingest_adhd200_athena.py <archive>` — already written |

Already in the right format. Same CC200 parcellation as ABIDE, same bandpass,
so it drops straight into the existing Leave-Site-Out protocol with no new code.

> ADHD-200 Consortium, *Front. Syst. Neurosci.* 6:62 (2012);
> Bellec et al., *NeuroImage* 144:275–286 (2017).

---

## 2. REST-meta-MDD — best value once an application is possible

**Adds:** a third disorder and, critically, **25 sites** — more than ABIDE I's
20. It is also already distributed as ROI time series across standard atlases,
so with authorised access it becomes zero-effort like ADHD-200.

| | |
|---|---|
| Obtain | ROI time-series derivatives, **CC200 / Craddock 200 parcellation** if offered; otherwise AAL or Dosenbach and tell me which |
| From | REST-meta-MDD consortium (DIRECT initiative) |
| Access | Institutional application |
| Yields | 1,300 MDD / 1,128 control, 25 sites |

Ask specifically for the **ROI signals** package, not the preprocessed volumes.
The consortium distributes both, and the volumes are orders of magnitude larger
for no benefit here.

> Yan et al., *PNAS* 116(18):9078–9083 (2019).

---

## 3. SRPBS Multi-disorder — the scanner-robustness argument

**Adds:** the one thing no other dataset offers — a companion **traveling-subject**
set where the *same individuals* were scanned on different scanners. That
converts "we control for site effects" from an assumption into a measurement:
you can show directly how much of the representation is scanner and how much is
biology.

| | |
|---|---|
| Obtain | Both the **Multi-disorder** set and the **Traveling Subject** set |
| From | https://bicr-resource.atr.jp/srpbsopen/ or Synapse `syn22317076` |
| Access | Open version needs no application; full version does |
| Yields | 993 patients / 1,421 controls, 11 scanners, 8 disorders |
| Caveat | ~75 GB; ships as volumes, so needs a CC200 parcellation pass |

If the team can only fetch one part, take the **traveling-subject** set — it is
smaller and carries the argument that is hardest to make any other way.

> Tanaka et al., *Scientific Data* 8:227 (2021).

---

## 4. ABIDE II — only to preserve the manuscript's existing claim

**This is the one to think hardest about.** It adds no new disorder and no new
method evidence. Its only role is that **the manuscript currently claims a
pooled ABIDE I + II cohort of 2,214 subjects**, and that number is not
reproducible from public data as it stands.

There are two honest resolutions, and they are both fine:

- **Get it and reprocess.** Obtain ABIDE II rs-fMRI and have the team run
  **C-PAC with the CC200 atlas, 0.01–0.1 Hz bandpass, no global signal
  regression** — matching ABIDE I's `filt_noglobal` exactly. Anything less and
  the pooled cohort mixes preprocessing pipelines, which is worse than not
  pooling. There is **no public CC200 C-PAC derivative** for ABIDE II; the S3
  bucket carries only `fmriprep`, `denoise`, `mriqc`, and `DCAN` outputs.
- **Or restate the manuscript** around ABIDE I (1,035 subjects, 20 sites) plus
  ADHD-200 (947 subjects, 8 sites). Two disorders across 28 sites is a stronger
  claim than one disorder across a larger pool, and it is fully reproducible.

My recommendation is the second, with ABIDE II as a later addition. Twenty sites
is already ample for Leave-Site-Out; breadth across disorders is the binding
constraint, not subject count within ASD.

| | |
|---|---|
| Obtain | rs-fMRI + phenotypic; then C-PAC / CC200 / filt_noglobal to match ABIDE I |
| From | https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html |
| Access | NITRC account + data use agreement |

---

## Optional fifth — UCLA CNP (ds000030)

Cheap, no application, and enables a *different kind* of result: schizophrenia,
bipolar, ADHD and controls in one cohort with shared acquisition, so a
multi-class confusion matrix would show whether the learned topology is
disorder-specific or merely separates patients from controls. Single site, so
Leave-Site-Out does not apply — use stratified cross-validation.

From OpenNeuro `ds000030`, no registration.
> Poldrack et al., *Scientific Data* 3:160110 (2016).

---

## Where to put the files

Drop each cohort in its own directory under `data/`. If the files are already
per-subject ROI time series, this layout is read with no code changes:

```
data/<cohort>/ABIDE_pcp/cpac/filt_noglobal/<SUBJECT_ID>_rois_cc200.1D
data/<cohort>/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv
```

where the phenotypic CSV needs only three columns: `FILE_ID` (matching the
filename stem), `SITE_ID`, and `DX_GROUP` (1 = case, 2 = control).

If the data arrives in any other shape, leave it as delivered and tell me the
layout — writing an adapter is a small job, and it is safer than having the team
restructure files by hand.
