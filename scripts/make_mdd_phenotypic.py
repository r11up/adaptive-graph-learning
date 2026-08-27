#!/usr/bin/env python3
"""Build a phenotypic table for REST-meta-MDD from its filename convention.

REST-meta-MDD encodes cohort membership in the subject identifier itself:

    ROISignals_S<site>-<group>-<id>.mat        group 1 = MDD, 2 = control

so site and diagnosis can be recovered without a separate phenotypic file.
The 25 sites become the Leave-Site-Out folds.

    python scripts/make_mdd_phenotypic.py data/mdd_raw --out data/mdd_phenotypic.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SUBJECT = re.compile(r"(S\d+)-(\d)-(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/mdd_phenotypic.csv"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.source.glob("*.mat")):
        match = SUBJECT.search(path.stem)
        if not match:
            continue
        site, group, number = match.groups()
        rows.append({
            "subject_id": f"{site}-{group}-{number}",
            "site": site,
            "group": int(group),  # 1 = MDD, 2 = control
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no subjects matched under {args.source}")
    frame.to_csv(args.out, index=False)

    counts = frame.groupby(["site", "group"]).size().unstack(fill_value=0)
    counts.columns = ["MDD" if c == 1 else "control" for c in counts.columns]
    print(f"{len(frame)} subjects across {frame['site'].nunique()} sites -> {args.out}")
    print(counts.to_string())
    usable = int(((counts > 0).sum(axis=1) == 2).sum())
    print(f"\nsites with both classes (usable LSO folds): {usable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
