#!/usr/bin/env python3
"""Download ADHD-200 CC200 ROI time series from the INDI S3 bucket.

ADHD-200 is the natural companion cohort to ABIDE for this framework: a
different neuropsychiatric disorder, but the *same* CC200 parcellation, the
same 0.01-0.1 Hz bandpass, and the same multi-site structure that the
Leave-Site-Out protocol needs. No registration or data use agreement is
required — the C-PAC derivatives are served anonymously.

Files are written in the ABIDE-compatible layout, so ``load_abide()`` reads the
resulting cohort without modification:

    data/adhd200/ABIDE_pcp/cpac/filt_noglobal/<ScanDirID>_rois_cc200.1D
    data/adhd200/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv

The phenotypic table is rewritten to ABIDE's column convention
(``FILE_ID`` / ``SITE_ID`` / ``DX_GROUP``, where 1 = case and 2 = control), with
ADHD subtypes 1-3 collapsed to a single positive class.

The derivative path inside each subject directory encodes C-PAC's nuisance and
filtering settings and is not identical across subjects, so it is discovered by
listing rather than assumed.

Examples:
    python scripts/download_adhd200.py
    python scripts/download_adhd200.py --limit 50 --workers 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BUCKET = "https://s3.amazonaws.com/fcp-indi"
PIPELINE_ROOT = (
    "data/Projects/ADHD200/Outputs/cpac/raw_outputs/"
    "pipeline_adhd200-benchmark__freq-filter"
)
PHENOTYPIC_ROOT = "data/Projects/ADHD200/RawDataBIDS"
SITES = [
    "KKI", "NYU", "NeuroIMAGE", "OHSU", "Peking_1", "Pittsburgh",
    "Brown_TestRelease", "OHSU_TestRelease", "Peking_1_TestRelease",
]
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _get(url: str, timeout: int = 60, retries: int = 5) -> bytes | None:
    """GET with backoff; None if the object is absent or repeatedly fails."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if attempt < retries - 1:
            time.sleep(min(2**attempt, 16) * (0.5 + random.random()))
    return None


def list_keys(prefix: str, max_keys: int = 1000) -> list[str]:
    """List object keys under a prefix (single page)."""
    query = urllib.parse.urlencode(
        {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
    )
    body = _get(f"{BUCKET}?{query}")
    if body is None:
        return []
    return [e.text for e in ET.fromstring(body).iter(f"{S3_NS}Key") if e.text]


def load_phenotypic() -> pd.DataFrame:
    """Fetch and merge every per-site phenotypic table."""
    frames = []
    for site in SITES:
        body = _get(f"{BUCKET}/{PHENOTYPIC_ROOT}/{site}_phenotypic.csv")
        if body is None:
            print(f"  ! no phenotypic table for {site}", file=sys.stderr)
            continue
        path = Path("/tmp") / f"adhd200_{site}.csv"
        path.write_bytes(body)
        frame = pd.read_csv(path)
        frame.columns = [c.strip() for c in frame.columns]
        if "ScanDir ID" not in frame.columns or "DX" not in frame.columns:
            continue
        # TestRelease cohorts are the competition's held-out sets; keep the
        # base site name so LSO treats them as one acquisition site.
        frame["SITE_NAME"] = site.replace("_TestRelease", "")
        frames.append(frame[["ScanDir ID", "DX", "SITE_NAME"]])
        print(f"  {site}: {len(frame)} subjects")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["ScanDir ID", "DX"])
    merged["ScanDir ID"] = merged["ScanDir ID"].astype(int)
    # DX: 0 = control, 1-3 = ADHD subtypes. Some rows carry 'pending' sentinels.
    merged = merged[merged["DX"].isin([0, 1, 2, 3])]
    return merged.drop_duplicates(subset=["ScanDir ID"])


def fetch_subject(scan_id: int, dest_dir: Path) -> tuple[str, int]:
    """Locate and download one subject's CC200 series. Returns (status, id)."""
    dest = dest_dir / f"{scan_id}_rois_cc200.1D"
    if dest.exists() and dest.stat().st_size > 0:
        return "cached", scan_id

    # S3 zero-pads scan IDs to 7 digits; the phenotypic tables store them as ints.
    prefix = f"{PIPELINE_ROOT}/{scan_id:07d}_session_1/roi_timeseries/"
    keys = [k for k in list_keys(prefix) if k.endswith("_mask_CC200/roi_CC200.1D")]
    if not keys:
        return "absent", scan_id

    # Prefer the first resting scan when several are present.
    keys.sort()
    body = _get(f"{BUCKET}/{urllib.parse.quote(keys[0])}")
    if body is None:
        return "failed", scan_id

    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.rename(dest)
    return "ok", scan_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/adhd200"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    series_dir = args.out / "ABIDE_pcp" / "cpac" / "filt_noglobal"
    series_dir.mkdir(parents=True, exist_ok=True)

    print("fetching ADHD-200 phenotypic tables")
    phenotypic = load_phenotypic()
    if args.limit:
        phenotypic = phenotypic.head(args.limit)
    print(f"{len(phenotypic)} subjects across {phenotypic['SITE_NAME'].nunique()} sites")

    counts: dict[str, int] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_subject, int(sid), series_dir)
            for sid in phenotypic["ScanDir ID"]
        ]
        for future in as_completed(futures):
            status, _ = future.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}  " +
                      "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)

    # Keep only subjects whose series actually landed, then write the
    # phenotypic table in ABIDE's column convention.
    available = {int(p.name.split("_")[0]) for p in series_dir.glob("*_rois_cc200.1D")}
    kept = phenotypic[phenotypic["ScanDir ID"].isin(available)].copy()
    kept["FILE_ID"] = kept["ScanDir ID"].astype(str)
    kept["SITE_ID"] = kept["SITE_NAME"]
    kept["DX_GROUP"] = kept["DX"].apply(lambda d: 1 if d > 0 else 2)

    table = args.out / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"
    kept[["FILE_ID", "SITE_ID", "DX_GROUP"]].to_csv(table, index=False)

    size_mb = sum(f.stat().st_size for f in series_dir.glob("*.1D")) / 1e6
    print(f"\n{len(kept)} subjects usable "
          f"({int((kept['DX_GROUP'] == 1).sum())} ADHD / "
          f"{int((kept['DX_GROUP'] == 2).sum())} control), "
          f"{kept['SITE_ID'].nunique()} sites, {size_mb:.1f} MB")
    print(f"phenotypic table -> {table}")
    print(f"\nload with:  load_abide(root='{args.out}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
