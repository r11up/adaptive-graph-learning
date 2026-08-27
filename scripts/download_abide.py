#!/usr/bin/env python3
"""Download ABIDE I preprocessed ROI time series from the Preprocessed Connectomes Project.

Pulls per-subject ROI time-series derivatives (default: CC200 parcellation,
C-PAC pipeline, band-pass filtered without global signal regression) plus the
phenotypic table that carries diagnosis and acquisition site.

The files are small (~400 KB each), but there are ~1100 of them, so they are
fetched in parallel. Downloads are resumable: existing, non-empty files are
skipped, so re-running after an interruption only fetches what is missing.

No registration is required for ABIDE I — the data are served anonymously from
the INDI S3 bucket. ABIDE II preprocessed derivatives are *not* distributed
this way; see docs/DATASETS.md.

Examples:
    python scripts/download_abide.py                      # CC200, C-PAC, filt_noglobal
    python scripts/download_abide.py --atlas rois_aal     # a different parcellation
    python scripts/download_abide.py --workers 24         # more parallelism
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

S3_ROOT = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"
PHENOTYPIC_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Phenotypic_V1_0b_preprocessed1.csv"
)


def strategy(band_pass: bool, global_signal: bool) -> str:
    """C-PAC nuisance-regression variant directory name."""
    return f"{'filt' if band_pass else 'nofilt'}_{'global' if global_signal else 'noglobal'}"


def fetch_phenotypic(dest: Path) -> pd.DataFrame:
    """Download (once) and load the phenotypic table."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"phenotypic table -> {dest}")
        urllib.request.urlretrieve(PHENOTYPIC_URL, dest)
    return pd.read_csv(dest)


def download_one(
    url: str, dest: Path, timeout: int = 60, retries: int = 5
) -> tuple[str, str]:
    """Fetch a single derivative, retrying with backoff. Returns (status, name).

    S3 throttles bursts of parallel anonymous requests, which surfaces as a
    URLError rather than an HTTP 503, so transient failures are retried with
    exponential backoff and jitter instead of being treated as fatal.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return "cached", dest.name

    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            tmp.rename(dest)  # atomic: an interrupted run never leaves a truncated file
            return "ok", dest.name
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # derivative genuinely absent for this subject
                return "missing (404)", dest.name
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if tmp.exists():
            tmp.unlink()
        if attempt < retries - 1:
            time.sleep(min(2**attempt, 16) * (0.5 + random.random()))

    return f"failed ({last.__class__.__name__})", dest.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/ABIDE-I"))
    parser.add_argument("--pipeline", default="cpac", choices=["cpac", "ccs", "dparsf", "niak"])
    parser.add_argument("--atlas", default="rois_cc200",
                        help="derivative name, e.g. rois_cc200, rois_cc400, rois_aal, rois_ho")
    parser.add_argument("--no-band-pass", action="store_true")
    parser.add_argument("--global-signal", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel downloads; S3 throttles anonymous bursts above ~10")
    parser.add_argument("--limit", type=int, help="only fetch the first N subjects (smoke test)")
    args = parser.parse_args()

    variant = strategy(not args.no_band_pass, args.global_signal)
    series_dir = args.out / "ABIDE_pcp" / args.pipeline / variant
    series_dir.mkdir(parents=True, exist_ok=True)

    phenotypic = fetch_phenotypic(args.out / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv")
    # Subjects that failed preprocessing carry the sentinel 'no_filename'.
    subjects = phenotypic[phenotypic["FILE_ID"] != "no_filename"]
    if args.limit:
        subjects = subjects.head(args.limit)

    print(f"{len(subjects)} subjects | pipeline={args.pipeline} "
          f"strategy={variant} atlas={args.atlas}")
    print(f"destination: {series_dir}")

    jobs = []
    for file_id in subjects["FILE_ID"]:
        name = f"{file_id}_{args.atlas}.1D"
        jobs.append((f"{S3_ROOT}/{args.pipeline}/{variant}/{args.atlas}/{name}", series_dir / name))

    counts: dict[str, int] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, url, dest): dest for url, dest in jobs}
        for future in as_completed(futures):
            status, name = future.result()
            key = status if status.startswith("failed") else status
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  " +
                      "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
            if status.startswith("failed"):
                print(f"    ! {name}: {status}", file=sys.stderr)

    total_mb = sum(f.stat().st_size for f in series_dir.glob(f"*{args.atlas}.1D")) / 1e6
    n_files = len(list(series_dir.glob(f"*{args.atlas}.1D")))
    print(f"\n{n_files} files on disk, {total_mb:.1f} MB in {series_dir}")
    return 0 if not any(k.startswith("failed") for k in counts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
