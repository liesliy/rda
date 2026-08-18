"""Reference profile data classes for behavioral scoring.

A :class:`ReferenceProfile` captures the distribution of portable metrics
across a calibration population of "normal" episodes. It is used by
:class:`~rda.calibration.scorer.BehavioralScorer` to compute robust
deviation scores for incoming episodes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MetricStats:
    """Robust distribution statistics for a single portable metric.

    Attributes:
        median: Population median (50th percentile).
        mad: Median Absolute Deviation.
        p05: 5th percentile.
        p25: 25th percentile.
        p75: 75th percentile.
        p95: 95th percentile.
    """

    median: float
    mad: float
    p05: float
    p25: float
    p75: float
    p95: float

    @property
    def iqr(self) -> float:
        """Inter-quartile range (p75 - p25)."""
        return self.p75 - self.p25

    def robust_z(self, value: float) -> float:
        """Compute the robust (MAD-based) z-score of *value*.

        Uses the standard 0.6745 scaling factor for comparability with
        standard z-scores under normality.
        """
        if self.mad == 0.0:
            return 0.0
        return 0.6745 * (value - self.median) / self.mad

    def bad_z(self, value: float, direction: str) -> float:
        """Compute the direction-aware bad z-score of *value*.

        Positive result always means "more anomalous" regardless of
        the original metric direction.

        Args:
            value: The raw metric value.
            direction: ``"higher_is_worse"`` or ``"lower_is_worse"``.

        Returns:
            Bad z-score. Positive = more anomalous.
        """
        raw_z = self.robust_z(value)
        if direction == "lower_is_worse":
            return -raw_z
        return raw_z

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a plain dict."""
        return {
            "median": self.median,
            "mad": self.mad,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "MetricStats":
        """Deserialize from a dict."""
        return cls(
            median=data["median"],
            mad=data["mad"],
            p05=data["p05"],
            p25=data["p25"],
            p75=data["p75"],
            p95=data["p95"],
        )

    def percentile_rank(self, value: float) -> float:
        """Approximate percentile rank of *value* via piecewise linear
        interpolation across stored percentiles. Returns value in [0, 1]."""
        anchors = [
            (0.05, self.p05),
            (0.25, self.p25),
            (0.50, self.median),
            (0.75, self.p75),
            (0.95, self.p95),
        ]
        if value <= anchors[0][1]:
            return 0.05
        if value >= anchors[-1][1]:
            return 0.95
        for i in range(len(anchors) - 1):
            p_lo, v_lo = anchors[i]
            p_hi, v_hi = anchors[i + 1]
            if v_lo <= value <= v_hi:
                if v_hi == v_lo:
                    return p_lo
                frac = (value - v_lo) / (v_hi - v_lo)
                return p_lo + frac * (p_hi - p_lo)
        return 0.5


@dataclass
class ReferenceProfile:
    """Reference distribution for behavioral scoring.

    Built by :func:`rda.calibration.scorer.calibrate` from a set of
    calibration episodes.
    """

    dataset_name: str
    platform: str
    n_calibration: int
    metrics: Dict[str, MetricStats] = field(default_factory=dict)
    task_scope: Optional[str] = None
    # New multi-component PCA fields (P0-2)
    pca_components: List[Dict[str, float]] = field(default_factory=list)
    pca_explained_variance_ratios: List[float] = field(default_factory=list)
    pca_n_components: int = 0
    # Backward compatibility: legacy fields kept for old code
    pca_loadings: Optional[Dict[str, float]] = None
    pca_explained_variance: float = 0.0

    def __post_init__(self) -> None:
        """Sync legacy fields with new multi-component fields."""
        # If new fields are populated but legacy fields are not, sync them
        if self.pca_components and self.pca_loadings is None:
            self.pca_loadings = self.pca_components[0] if self.pca_components else None
        if self.pca_explained_variance_ratios and self.pca_explained_variance == 0.0:
            self.pca_explained_variance = self.pca_explained_variance_ratios[0]
        # If legacy fields are populated but new fields are not, sync them
        if self.pca_loadings and not self.pca_components:
            self.pca_components = [self.pca_loadings]
            self.pca_n_components = 1
            if self.pca_explained_variance > 0.0 and not self.pca_explained_variance_ratios:
                self.pca_explained_variance_ratios = [self.pca_explained_variance]
        if self.pca_components and self.pca_n_components == 0:
            self.pca_n_components = len(self.pca_components)

    @property
    def metric_names(self) -> list[str]:
        """Sorted list of metric names in the profile."""
        return sorted(self.metrics.keys())

    def has_pca(self) -> bool:
        """Return True if PCA loadings are available."""
        return bool(self.pca_components) or (
            self.pca_loadings is not None and len(self.pca_loadings) > 0
        )

    def to_dict(self) -> Dict:
        """Serialize the profile to a JSON-compatible dict."""
        return {
            "dataset_name": self.dataset_name,
            "platform": self.platform,
            "n_calibration": self.n_calibration,
            "task_scope": self.task_scope,
            "metrics": {
                name: stats.to_dict() for name, stats in self.metrics.items()
            },
            "pca_components": self.pca_components,
            "pca_explained_variance_ratios": self.pca_explained_variance_ratios,
            "pca_n_components": self.pca_n_components,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReferenceProfile":
        """Deserialize a profile from a dict (e.g. loaded from JSON)."""
        metrics = {
            name: MetricStats.from_dict(stats_data)
            for name, stats_data in data.get("metrics", {}).items()
        }
        return cls(
            dataset_name=data["dataset_name"],
            platform=data["platform"],
            n_calibration=data["n_calibration"],
            metrics=metrics,
            task_scope=data.get("task_scope"),
            pca_components=data.get("pca_components", []),
            pca_explained_variance_ratios=data.get("pca_explained_variance_ratios", []),
            pca_n_components=data.get("pca_n_components", 0),
        )

    def save(self, path: str | Path) -> None:
        """Save the profile to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceProfile":
        """Load a profile from a JSON file.

        Args:
            path: Source file path.

        Returns:
            A :class:`ReferenceProfile` instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def save_reference(profile: ReferenceProfile, path: str | Path) -> None:
    """Save a reference profile to a JSON file.

    Convenience function equivalent to ``profile.save(path)``.

    Args:
        profile: The profile to save.
        path: Destination file path.
    """
    profile.save(path)


def load_reference(path: str | Path) -> ReferenceProfile:
    """Load a reference profile from a JSON file.

    Convenience function equivalent to ``ReferenceProfile.load(path)``.

    Args:
        path: Source file path.

    Returns:
        A :class:`ReferenceProfile` instance.
    """
    return ReferenceProfile.load(path)
