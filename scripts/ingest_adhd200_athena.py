#!/usr/bin/env python3
"""Ingest the ADHD-200 Athena CC200 time-course release into the study layout.

The full ADHD-200 cohort (947 subjects, 8 sites) is distributed by NITRC as a
single archive of CC200 ROI time courses. NITRC requires a free account to
download it — no data use agreement — so the archive has to be fetched by hand;
this script takes over from there.

Get the archive
---------------
1. Create a free account at https://www.nitrc.org/account/register.php
2. Open https://www.nitrc.org/frs/?group_id=383
3. Download **"CC200 Time Courses (Corrected Filtering)"** (~274 MB).
   Optionally also **"Test Release CC200 Time Courses"** (~131 MB) for the
   competition's held-out subjects.
4. Run this script against the download (archive or extracted directory):

       python scripts/ingest_adhd200_athena.py ~/Downloads/adhd200_cc200_tcs.tgz

Output
------
Writes the ABIDE-compatible layout, so ``load_abide(root='data/adhd200')``
reads the result with no code changes:

    data/adhd200/ABIDE_pcp/cpac/filt_noglobal/<ScanDirID>_rois_cc200.1D
    data/adhd200/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv

Diagnosis and site come from the per-site phenotypic tables on the public INDI
S3 bucket (no login needed for those), joined on the numeric scan ID parsed out
of each time-course filename. Athena file names vary across releases
(``sfnwmrda…``/``snwmrda…``, ``_TCs``/``_TCs.1D``), so files are discovered by
pattern rather than by an assumed exact name.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from download_adhd200 import load_phenotypic  # noqa: E402

# Athena names embed the numeric scan id, e.g. sfnwmrda0010001_session_1_rest_1_cc200_TCs.1D
SUBJECT_ID = re.compile(r"(\d{7})")
CC200_FILE = re.compile(r"cc200.*(?:tcs|timecourse|time_course)", re.IGNORECASE)


def extract_archive(archive: Path, workdir: Path) -> Path:
    """Unpack a .tgz/.tar.gz/.zip into ``workdir`` and return the root."""
    print(f"extracting {archive.name} ...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(workdir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            handle.extractall(workdir, filter="data")
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")
    return workdir


def find_timecourse_files(root: Path) -> dict[int, Path]:
    """Map scan id -> CC200 time-course file, preferring filtered variants."""
    found: dict[int, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or not CC200_FILE.search(path.name):
            continue
        match = SUBJECT_ID.search(path.name)
        if not match:
            continue
        scan_id = int(match.group(1))
        # 'sfnwmrda' is the band-pass filtered Athena product; prefer it over
        # the unfiltered 'snwmrda' when a subject has both.
        preferred = path.name.startswith("sfnwmrda")
        if scan_id not in found or (preferred and not found[scan_id].name.startswith("sfnwmrda")):
            found[scan_id] = path
    return found


def normalise_series(path: Path) -> np.ndarray | None:
    """Read an Athena time course as a ``(T, n_roi)`` float array.

    Athena files carry a header row of ROI labels and, in some releases,
    leading index columns; both are stripped here.
    """
    try:
        frame = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    except (ValueError, OSError, pd.errors.ParserError):
        return None

    # Drop any non-numeric header row that survived the comment filter.
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
    array = frame.to_numpy(dtype=float)
    array = array[:, ~np.all(np.isnan(array), axis=0)]
    if array.ndim != 2 or array.shape[0] < 20 or array.shape[1] < 100:
        return None
    return np.nan_to_num(array)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path,
                        help="downloaded archive (.tgz/.tar.gz/.zip) or extracted directory")
    parser.add_argument("--out", type=Path, default=Path("data/adhd200"))
    parser.add_argument("--keep-existing", action="store_true",
                        help="merge with any subjects already present")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"not found: {args.source}")

    series_dir = args.out / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    series_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for stale in series_dir.glob("*_rois_cc200.1D"):
            stale.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        root = args.source if args.source.is_dir() else extract_archive(args.source, Path(tmp))

        print("locating CC200 time courses ...")
        files = find_timecourse_files(root)
        print(f"found {len(files)} subjects with CC200 time courses")
        if not files:
            raise SystemExit(
                "no CC200 time-course files found. Expected names containing "
                "'cc200' and 'TCs' — check the archive is the CC200 release."
            )

        print("fetching phenotypic tables from INDI S3 ...")
        phenotypic = load_phenotypic()
        labelled = set(phenotypic["ScanDir ID"])

        written, skipped_unlabelled, skipped_bad = 0, 0, 0
        for scan_id, path in sorted(files.items()):
            if scan_id not in labelled:
                skipped_unlabelled += 1
                continue
            array = normalise_series(path)
            if array is None:
                skipped_bad += 1
                continue
            np.savetxt(series_dir / f"{scan_id}_rois_cc200.1D", array, fmt="%.6f")
            written += 1
            if written % 100 == 0:
                print(f"  wrote {written} subjects", flush=True)

    available = {int(p.name.split("_")[0]) for p in series_dir.glob("*_rois_cc200.1D")}
    kept = phenotypic[phenotypic["ScanDir ID"].isin(available)].copy()
    kept["FILE_ID"] = kept["ScanDir ID"].astype(str)
    kept["SITE_ID"] = kept["SITE_NAME"]
    kept["DX_GROUP"] = kept["DX"].apply(lambda d: 1 if d > 0 else 2)
    table = args.out / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"
    kept[["FILE_ID", "SITE_ID", "DX_GROUP"]].to_csv(table, index=False)

    if skipped_unlabelled or skipped_bad:
        print(f"skipped {skipped_unlabelled} without a phenotypic record, "
              f"{skipped_bad} unreadable")

    print(f"\n{len(kept)} subjects "
          f"({int((kept['DX_GROUP'] == 1).sum())} ADHD / "
          f"{int((kept['DX_GROUP'] == 2).sum())} control) "
          f"across {kept['SITE_ID'].nunique()} sites")
    print("\nper-site class balance (a site needs both classes to form an LSO fold):")
    balance = kept.groupby("SITE_ID")["DX_GROUP"].value_counts().unstack(fill_value=0)
    balance.columns = ["ADHD" if c == 1 else "control" for c in balance.columns]
    print(balance.to_string())

    print(f"\nrun the study with:\n  python scripts/run_abide_study.py "
          f"--data-root {args.out} --cache results/adhd200_encoded.pt "
          f"--refresh-cache --epochs 30 --out results/adhd200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
