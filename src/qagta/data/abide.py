"""ABIDE rs-fMRI loading and per-region PCA compression.

Turns the Preprocessed Connectomes Project ROI time-series derivatives into
the node-feature representation the framework consumes:

    X_raw in R^{N_roi x T}   (BOLD time series per region, T varies by site)
        -> per-region PCA -> X in R^{N_roi x d}

Each brain region becomes one graph node carrying ``d`` temporal features.
The compression exists because encoding T ~ 150-300 timepoints directly would
need prohibitively deep circuits; ``d`` defaults to 16, one feature per qubit.

Subject labels come from the phenotypic table's ``DX_GROUP`` (1 = ASD,
2 = control, remapped here to 1 / 0), and ``SITE_ID`` supplies the grouping
for Leave-Site-Out cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class SubjectRecord:
    """One subject: node features, diagnosis and acquisition site."""

    file_id: str
    site: str
    label: int  # 1 = ASD, 0 = control
    features: np.ndarray  # (n_roi, n_components)
    n_timepoints: int
    explained_variance: float


@dataclass
class AbideDataset:
    """A loaded ABIDE cohort ready for graph construction."""

    subjects: list[SubjectRecord]

    def __len__(self) -> int:
        return len(self.subjects)

    @property
    def features(self) -> np.ndarray:
        """Stacked node features, shape ``(n_subjects, n_roi, n_components)``."""
        return np.stack([s.features for s in self.subjects])

    @property
    def labels(self) -> np.ndarray:
        return np.array([s.label for s in self.subjects], dtype=np.int64)

    @property
    def sites(self) -> np.ndarray:
        return np.array([s.site for s in self.subjects])

    @property
    def n_roi(self) -> int:
        return self.subjects[0].features.shape[0]

    @property
    def n_components(self) -> int:
        return self.subjects[0].features.shape[1]

    def summary(self) -> str:
        labels = self.labels
        sites = self.sites
        timepoints = [s.n_timepoints for s in self.subjects]
        variance = float(np.mean([s.explained_variance for s in self.subjects]))
        return (
            f"{len(self)} subjects | {int((labels == 1).sum())} ASD / "
            f"{int((labels == 0).sum())} control | {len(set(sites))} sites\n"
            f"nodes={self.n_roi}  features={self.n_components}  "
            f"T range={min(timepoints)}-{max(timepoints)}  "
            f"mean explained variance={variance:.1%}"
        )


def compress_connectivity_profiles(
    timeseries: np.ndarray, n_components: int = 16
) -> tuple[np.ndarray, float]:
    """Compress each region's *connectivity profile* to node features.

    Region ``i``'s profile is row ``i`` of the subject's correlation matrix —
    how that region relates to every other region. This is what carries the
    diagnostic signal in rs-fMRI: FINDING 06 measures per-region temporal
    features at AUC 0.459 (chance) against 0.693 for pairwise correlations.

    Compressing profiles rather than raw time courses also suits a fidelity
    kernel better. Two regions with similar connectivity fingerprints should
    produce overlapping quantum states; two regions with similar raw time
    courses need not be functionally related at all.
    """
    from qagta.graph.connectome import pearson_connectivity

    profiles = pearson_connectivity(timeseries)  # (n_roi, n_roi), row = profile
    n_roi = profiles.shape[0]

    usable = min(n_components, n_roi)
    pca = PCA(n_components=usable, random_state=0)
    features = pca.fit_transform(profiles)
    retained = float(pca.explained_variance_ratio_.sum())

    if usable < n_components:
        features = np.pad(features, ((0, 0), (0, n_components - usable)))
    return features.astype(np.float32), retained


def compress_region_timeseries(
    timeseries: np.ndarray, n_components: int = 16
) -> tuple[np.ndarray, float]:
    """PCA-compress a ``(T, n_roi)`` BOLD matrix to ``(n_roi, n_components)``.

    The time series is transposed so each *region* is a sample and each
    timepoint a raw dimension; PCA over that matrix yields, for every region, a
    compact temporal signature in a shared component basis. A shared basis is
    what makes the resulting node features comparable across regions — which
    matters here, because downstream every pair of regions is compared through
    a similarity kernel.

    Returns the features and the fraction of variance retained.
    """
    if timeseries.ndim != 2:
        raise ValueError(f"expected a 2-D (T, n_roi) array, got shape {timeseries.shape}")

    regions = timeseries.T  # (n_roi, T)
    n_roi, n_time = regions.shape
    # Standardise each region's series so PCA is driven by temporal shape
    # rather than by regional differences in BOLD amplitude.
    regions = regions - regions.mean(axis=1, keepdims=True)
    scale = regions.std(axis=1, keepdims=True)
    regions = regions / np.where(scale > 0, scale, 1.0)

    usable = min(n_components, n_roi, n_time)
    pca = PCA(n_components=usable, random_state=0)
    features = pca.fit_transform(regions)
    retained = float(pca.explained_variance_ratio_.sum())

    if usable < n_components:  # short scan: pad so every subject has the same width
        features = np.pad(features, ((0, 0), (0, n_components - usable)))
    return features.astype(np.float32), retained


def load_abide(
    root: str | Path = "data/abide",
    pipeline: str = "cpac",
    strategy: str = "filt_noglobal",
    atlas: str = "rois_cc200",
    n_components: int = 16,
    min_timepoints: int = 60,
    limit: int | None = None,
    feature_mode: str = "connectivity",
) -> AbideDataset:
    """Load every downloaded subject and compress it to node features.

    Subjects whose derivative file is missing, unreadable, or shorter than
    ``min_timepoints`` are skipped; a summary of exclusions is printed so the
    effective sample size is never silently different from the nominal one.
    """
    root = Path(root)
    phenotypic_path = root / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"
    if not phenotypic_path.exists():
        raise FileNotFoundError(
            f"phenotypic table not found at {phenotypic_path}. "
            "Run: python scripts/download_abide.py"
        )
    series_dir = root / "ABIDE_pcp" / pipeline / strategy

    phenotypic = pd.read_csv(phenotypic_path)
    phenotypic = phenotypic[phenotypic["FILE_ID"] != "no_filename"]
    if limit:
        phenotypic = phenotypic.head(limit)

    subjects: list[SubjectRecord] = []
    skipped = {"absent": 0, "unreadable": 0, "too_short": 0}

    for row in phenotypic.itertuples():
        path = series_dir / f"{row.FILE_ID}_{atlas}.1D"
        if not path.exists():
            skipped["absent"] += 1
            continue
        try:
            timeseries = np.loadtxt(path)
        except (ValueError, OSError):
            skipped["unreadable"] += 1
            continue
        if timeseries.ndim != 2 or timeseries.shape[0] < min_timepoints:
            skipped["too_short"] += 1
            continue

        if feature_mode == "connectivity":
            features, retained = compress_connectivity_profiles(timeseries, n_components)
        elif feature_mode == "temporal":
            features, retained = compress_region_timeseries(timeseries, n_components)
        else:
            raise ValueError(f"unknown feature_mode {feature_mode!r}")
        subjects.append(
            SubjectRecord(
                file_id=row.FILE_ID,
                site=str(row.SITE_ID),
                # DX_GROUP: 1 = autism, 2 = control -> 1 / 0
                label=1 if int(row.DX_GROUP) == 1 else 0,
                features=features,
                n_timepoints=int(timeseries.shape[0]),
                explained_variance=retained,
            )
        )

    if not subjects:
        raise RuntimeError(
            f"no usable subjects found in {series_dir}. "
            "Run: python scripts/download_abide.py"
        )
    if any(skipped.values()):
        print(f"skipped {sum(skipped.values())} subjects: " +
              ", ".join(f"{k}={v}" for k, v in skipped.items() if v))

    n_roi = {s.features.shape[0] for s in subjects}
    if len(n_roi) > 1:
        raise RuntimeError(f"inconsistent ROI counts across subjects: {sorted(n_roi)}")

    return AbideDataset(subjects=subjects)
