#!/usr/bin/env python3
"""Universal cohort ingester — normalise any delivered dataset into the study layout.

Each cohort arrives in a different shape. This converts all of them into the one
layout the pipeline reads, so `load_abide(root=...)` and the whole Leave-Site-Out
stack work unchanged:

    data/<cohort>/ABIDE_pcp/cpac/filt_noglobal/<SUBJECT_ID>_rois_cc200.1D
    data/<cohort>/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv   FILE_ID,SITE_ID,DX_GROUP

Supported input formats
-----------------------
roi_text   Per-subject text/CSV of ROI time series, shape (T, n_roi).
           ABIDE .1D, ADHD-200 Athena, and most PCP-style derivatives.

roi_mat    Per-subject MATLAB .mat holding an ROI-signals matrix (T, n_roi).
           REST-meta-MDD ships `ROISignals_<subject>.mat`; the variable is
           found by shape rather than by an assumed name, since the key varies
           across releases (ROISignals, signals, ROISignal, ...).

nifti      Per-subject 4-D BOLD volumes, parcellated here with an atlas via
           nilearn's NiftiLabelsMasker. Needed for SRPBS, UCLA CNP (ds000030)
           and COBRE, which ship volumes rather than ROI series. Slow and
           disk-hungry, but it is the only way to put those cohorts on the same
           CC200 footing as ABIDE — mixing vendors' preprocessed ROI outputs
           would confound preprocessing with site.

Labels and sites come from a phenotypic table you point at with --phenotypic;
--label-column / --site-column / --id-column name the relevant columns, and
--positive-values lists the diagnostic codes that count as the case class.

Examples
--------
  # REST-meta-MDD (ROI signals package)
  python scripts/ingest_cohort.py --cohort mdd --format roi_mat \\
      --source ~/data/REST-meta-MDD/ROISignals_FunImgARCWF \\
      --phenotypic ~/data/REST-meta-MDD/phenotypic.csv \\
      --id-column ID --site-column Site --label-column Dx --positive-values 1

  # UCLA CNP / COBRE / SRPBS (volumes -> CC200)
  python scripts/ingest_cohort.py --cohort cnp --format nifti \\
      --source ~/data/ds000030/derivatives --pattern '*task-rest*preproc*.nii.gz' \\
      --phenotypic ~/data/ds000030/participants.tsv \\
      --id-column participant_id --site-column site --label-column diagnosis \\
      --positive-values SCHZ BIPOLAR ADHD

  # Anything already in ROI-text form
  python scripts/ingest_cohort.py --cohort srpbs --format roi_text \\
      --source ~/data/srpbs/roi --phenotypic ~/data/srpbs/participants.tsv \\
      --id-column participant_id --site-column site --label-column dx \\
      --positive-values 1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_IN_NAME = re.compile(r"([A-Za-z0-9]+[-_]?\d{3,})")


def read_phenotypic(path: Path) -> pd.DataFrame:
    """Load a phenotypic table from CSV or TSV."""
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def normalise_id(value: str) -> str:
    """Strip common prefixes so ``sub-0050002`` and ``0050002`` match."""
    text = str(value).strip()
    text = re.sub(r"^(sub|subject|s)[-_]?", "", text, flags=re.IGNORECASE)
    return text.lstrip("0") or "0"


def load_roi_text(path: Path) -> np.ndarray | None:
    try:
        frame = pd.read_csv(path, sep=r"\s+|,", comment="#", header=None, engine="python")
    except (ValueError, OSError, pd.errors.ParserError):
        return None
    array = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    array = array[~np.all(np.isnan(array), axis=1)]
    array = array[:, ~np.all(np.isnan(array), axis=0)]
    return np.nan_to_num(array) if array.ndim == 2 else None


def slice_columns(array: np.ndarray, spec: str | None) -> np.ndarray:
    """Select an atlas block from a multi-atlas ROI-signals matrix.

    REST-meta-MDD ships DPABI output in which several parcellations are
    concatenated column-wise into a single 1833-column matrix (AAL,
    Harvard-Oxford, CC200, Zalesky, Dosenbach, Power). Only one block should
    reach the pipeline, and it must be the CC200 block for parity with ABIDE.
    """
    if not spec:
        return array
    start, _, end = spec.partition(":")
    return array[:, int(start) : int(end)]


def load_roi_mat(path: Path) -> np.ndarray | None:
    """Extract the ROI-signals matrix from a .mat, identified by shape."""
    try:
        import scipy.io as sio

        blob = sio.loadmat(path)
        candidates = [
            v for k, v in blob.items()
            if not k.startswith("__") and hasattr(v, "ndim") and v.ndim == 2
            and min(v.shape) >= 10
        ]
    except (NotImplementedError, ValueError, OSError):
        try:  # MATLAB v7.3 is HDF5
            import h5py

            with h5py.File(path, "r") as handle:
                candidates = [
                    np.array(handle[k]) for k in handle
                    if getattr(handle[k], "ndim", 0) == 2 and min(handle[k].shape) >= 10
                ]
        except (OSError, KeyError, ImportError):
            return None
    if not candidates:
        return None
    # Widest matrix is the ROI-signals one; orient as (T, n_roi).
    array = max(candidates, key=lambda a: a.size).astype(float)
    if array.shape[0] < array.shape[1] and array.shape[0] > 400:
        array = array.T
    return np.nan_to_num(array)


def load_nifti(path: Path, atlas_img, masker_cache: dict):
    """Parcellate a 4-D BOLD volume into ROI time series, aligned to the atlas.

    NiftiLabelsMasker returns only the regions that actually intersect a given
    subject's coverage, so subjects with slightly different field-of-view come
    back with different column counts — measured here as 21 distinct widths
    between 180 and 190 across one cohort. Column *i* would then denote a
    different brain region for different subjects, which silently destroys any
    cross-subject comparison.

    This maps whatever the masker returns back onto the atlas's full, fixed
    label list, so column *i* is always the same region and regions missing for
    a subject are zero-filled.
    """
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker

    if "labels" not in masker_cache:
        atlas_data = nib.load(atlas_img).get_fdata().astype(int)
        canonical = [int(v) for v in np.unique(atlas_data) if v > 0]
        masker_cache["labels"] = canonical
        masker_cache["index"] = {label: i for i, label in enumerate(canonical)}
        masker_cache["masker"] = NiftiLabelsMasker(
            labels_img=atlas_img, standardize="zscore_sample", memory=None, verbose=0,
        )

    masker = masker_cache["masker"]
    try:
        series = np.nan_to_num(masker.fit_transform(str(path)))
    except Exception as exc:  # noqa: BLE001 - nilearn raises many types
        print(f"    ! {path.name}: {exc.__class__.__name__}", file=sys.stderr)
        return None

    canonical = masker_cache["labels"]
    aligned = np.zeros((series.shape[0], len(canonical)), dtype=float)

    # `labels_` lists the region ids actually returned, in column order.
    returned = [int(v) for v in getattr(masker, "labels_", canonical)]
    index = masker_cache["index"]
    for column, label in enumerate(returned):
        if column < series.shape[1] and label in index:
            aligned[:, index[label]] = series[:, column]
    return aligned


def resolve_atlas(name: str, local_path: Path | None = None):
    """Resolve a parcellation image, preferring a local file.

    A local atlas is the safer default: nilearn's Craddock fetcher pulls from a
    NITRC host that is frequently unavailable, and the ADHD-200 release ships
    the exact CC200 template it was parcellated with
    (``templates/ADHD200_parcellate_200.nii.gz``). Reusing that file keeps a
    newly parcellated cohort on the same footing as ADHD-200 rather than a
    near-miss variant of it.

    Note the template resolves to 190 non-empty parcels despite labels running
    to 200, which is why ADHD-200 series are 190 columns wide.
    """
    if local_path is not None:
        if not local_path.exists():
            raise SystemExit(f"atlas not found: {local_path}")
        return str(local_path), "local file"

    from nilearn import datasets

    if name in {"cc200", "craddock"}:
        atlas = datasets.fetch_atlas_craddock_2012()
        # 'scorr_mean' is the CC200-family map; index to the 200-cluster level.
        return atlas["scorr_mean"], "index the 200-cluster volume if 4-D"
    if name == "aal":
        return datasets.fetch_atlas_aal()["maps"], None
    if name == "harvard_oxford":
        return datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")["maps"], None
    raise ValueError(f"unknown atlas {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cohort", required=True, help="output name under data/")
    parser.add_argument("--format", required=True, choices=["roi_text", "roi_mat", "nifti"])
    parser.add_argument("--source", type=Path, required=True, help="directory to scan")
    parser.add_argument("--pattern", default=None, help="glob for subject files")
    parser.add_argument("--phenotypic", type=Path, required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--site-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--positive-values", nargs="+", required=True,
                        help="label values counting as the case class")
    parser.add_argument("--atlas", default="cc200", help="nifti format only")
    parser.add_argument("--atlas-path", type=Path, default=None,
                        help="local parcellation NIfTI; preferred over the "
                             "nilearn fetcher, whose host is often unavailable")
    parser.add_argument("--columns", default=None, metavar="START:END",
                        help="0-indexed column slice to keep from multi-atlas "
                             "ROI matrices, e.g. 228:428 for CC200 in REST-meta-MDD")
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--id-regex", default=None, metavar="PATTERN",
                        help="regex with one capture group extracting the subject "
                             "id from a filename, e.g. '(S\\d+-\\d+-\\d+)' for "
                             "REST-meta-MDD's ROISignals_S1-1-0001.mat. Overrides "
                             "the built-in heuristic, which assumes a plain "
                             "numeric id.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    default_pattern = {"roi_text": "**/*.1D", "roi_mat": "**/*.mat",
                       "nifti": "**/*.nii.gz"}[args.format]
    pattern = args.pattern or default_pattern

    phenotypic = read_phenotypic(args.phenotypic)
    for column in (args.id_column, args.site_column, args.label_column):
        if column not in phenotypic.columns:
            raise SystemExit(
                f"column {column!r} not in phenotypic table. Available: "
                f"{list(phenotypic.columns)}"
            )
    lookup = {normalise_id(r[args.id_column]): r for _, r in phenotypic.iterrows()}
    positives = {str(v).strip() for v in args.positive_values}
    print(f"phenotypic: {len(phenotypic)} rows | positive labels: {sorted(positives)}")

    files = sorted(args.source.glob(pattern))
    print(f"found {len(files)} candidate files under {args.source} ({pattern})")
    if not files:
        raise SystemExit("no files matched; check --source and --pattern")

    atlas_img, note = (None, None)
    masker_cache: dict = {}
    if args.format == "nifti":
        atlas_img, note = resolve_atlas(args.atlas, args.atlas_path)
        print(f"atlas: {args.atlas}" + (f"  ({note})" if note else ""))

    series_dir = args.out_root / args.cohort / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    if not args.dry_run:
        series_dir.mkdir(parents=True, exist_ok=True)

    rows, unmatched, unreadable = [], 0, 0
    id_pattern = re.compile(args.id_regex) if args.id_regex else ID_IN_NAME
    for path in files:
        match = id_pattern.search(path.stem)
        key = normalise_id(match.group(1)) if match else normalise_id(path.stem)
        record = lookup.get(key)
        if record is None:
            unmatched += 1
            continue

        if args.dry_run:
            rows.append(key)
            continue

        if args.format == "roi_text":
            array = load_roi_text(path)
        elif args.format == "roi_mat":
            array = load_roi_mat(path)
            if array is not None:
                array = slice_columns(array, args.columns)
        else:
            array = load_nifti(path, atlas_img, masker_cache)

        if array is None or array.ndim != 2 or array.shape[0] < 20:
            unreadable += 1
            continue

        subject_id = re.sub(r"[^A-Za-z0-9]", "", key)
        np.savetxt(series_dir / f"{subject_id}_rois_cc200.1D", array, fmt="%.6f")
        rows.append({
            "FILE_ID": subject_id,
            "SITE_ID": str(record[args.site_column]).strip(),
            "DX_GROUP": 1 if str(record[args.label_column]).strip() in positives else 2,
            "n_timepoints": int(array.shape[0]),
            "n_roi": int(array.shape[1]),
        })
        if len(rows) % 50 == 0:
            print(f"  ingested {len(rows)}", flush=True)

    if args.dry_run:
        print(f"\nDRY RUN: {len(rows)} files would match, {unmatched} unmatched")
        return 0
    if not rows:
        raise SystemExit(
            "nothing ingested. Most likely the subject IDs in the filenames do "
            "not match --id-column; re-run with --dry-run to check matching."
        )

    table = pd.DataFrame(rows)
    widths = table["n_roi"].value_counts()
    if len(widths) > 1:
        print(f"\nWARNING: inconsistent ROI counts {dict(widths)}. The pipeline "
              "requires one parcellation; keeping the majority width.")
        table = table[table["n_roi"] == widths.index[0]]

    out_table = args.out_root / args.cohort / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"
    table[["FILE_ID", "SITE_ID", "DX_GROUP"]].to_csv(out_table, index=False)

    print(f"\n{len(table)} subjects "
          f"({int((table['DX_GROUP'] == 1).sum())} case / "
          f"{int((table['DX_GROUP'] == 2).sum())} control), "
          f"{table['SITE_ID'].nunique()} sites, {table['n_roi'].iloc[0]} ROIs")
    if unmatched or unreadable:
        print(f"skipped {unmatched} unmatched to phenotypic, {unreadable} unreadable")

    print("\nper-site class balance (both classes needed to form an LSO fold):")
    balance = table.groupby("SITE_ID")["DX_GROUP"].value_counts().unstack(fill_value=0)
    balance.columns = ["case" if c == 1 else "control" for c in balance.columns]
    print(balance.to_string())

    print(f"\nrun with:\n  python scripts/run_abide_study.py --data-root "
          f"{args.out_root / args.cohort} --n-qubits 4 --epochs 30 "
          f"--cache results/{args.cohort}_encoded.pt --refresh-cache "
          f"--out results/{args.cohort}")
    print(f"  python scripts/run_quantum_kernel.py --data-root "
          f"{args.out_root / args.cohort} --qubits 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
