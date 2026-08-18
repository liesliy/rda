"""Reference calibration and PCA-based behavioral scoring.

This is the core of the Reference Calibration Engine. It provides:

- :func:`calibrate` — build a :class:`ReferenceProfile` from a set of
  "normal" calibration episodes.
- :class:`BehavioralScorer` — compute robust deviation scores for new
  episodes against a reference profile.

The scoring pipeline works as follows:

1. Extract metrics (portable and/or platform-specific).
2. Compute direction-aware bad z-scores per metric using MAD. Positive
   bad_z always means "more anomalous" regardless of metric direction.
3. Apply PCA loadings (first 2 principal components) to get a composite
   deviation score: score = sqrt(sum(pc_i_score^2)).
4. Fall back to equal-weight RMS of bad_z if PCA is unavailable.

Portable vs Platform-specific metrics
-------------------------------------
Per P2.5 experimentation, the three portable-only metrics (duration,
spike_count, effective_motion_ratio) correlate at ρ=0.960 with the
full metric set while being directly comparable across platforms.
Platform-specific metrics (velocity, path_length, etc.) have been
shown (P2) to be highly platform-dependent (JSD≈1.0) and must NOT be
mixed into a universal "cross-platform" score.

Therefore the scorer now tracks three distinct scores:
- ``portable_score`` — based only on portable Tier-1 metrics (universal)
- ``platform_score`` — based only on platform-specific metrics
- ``combined_score``  — all metrics together (for same-platform deep dive)

The legacy ``deviation_score`` field is kept for backward compatibility
and mirrors the ``combined_score`` (or ``portable_score`` if the
profile was calibrated with ``include_platform=False``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from rda.calibration.normalization import (
    bad_z_scores,
    compute_pca_loadings,
    mad,
    percentile_stats,
    robust_z_scores,
)
from rda.calibration.portable import (
    ALL_SCORE_METRICS,
    METRIC_DIRECTIONS,
    PLATFORM_METRICS,
    PORTABLE_METRICS,
    extract_all_score_metrics,
    extract_platform_metrics,
    extract_portable_metrics,
)
from rda.calibration.reference import MetricStats, ReferenceProfile
from rda.io.schema import EpisodeData


# ---------------------------------------------------------------------------
# BehavioralScore
# ---------------------------------------------------------------------------


@dataclass
class BehavioralScore:
    """Result of scoring a single episode against a reference profile.

    Attributes:
        deviation_score: Legacy / primary deviation score. Defaults to
            the combined score if platform metrics are calibrated,
            otherwise the portable score. Higher = more anomalous.
            Kept for backward compatibility.
        portable_score: Deviation score using only portable Tier-1
            metrics (cross-platform comparable).
        platform_score: Deviation score using only platform-specific
            metrics (meaningful only within the same platform).
        combined_score: Deviation score using all available metrics
            (portable + platform-specific, for same-platform deep dive).
        metric_scores: Per-metric robust z-scores (raw, not direction-flipped).
        percentile: Approximate percentile rank of the deviation score
            relative to the reference population, in ``[0.0, 1.0]``.
            ``0.5`` is median (typical), values near 1.0 are highly
            anomalous.
        method: How the composite score was computed
            (``"pca"`` or ``"rms"``).
        score_type: Which score ``deviation_score`` mirrors
            (``"portable"``, ``"platform"``, or ``"combined"``).
        has_platform_metrics: Whether the reference profile contains
            platform-specific metric statistics.
    """

    deviation_score: float
    metric_scores: Dict[str, float] = field(default_factory=dict)
    percentile: float = 0.5
    method: str = "rms"
    portable_score: float = 0.0
    platform_score: float = 0.0
    combined_score: float = 0.0
    score_type: str = "portable"
    has_platform_metrics: bool = False


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------


def calibrate(
    episodes: Iterable[EpisodeData],
    platform: str,
    dataset_name: str = "calibration",
    task_scope: Optional[str] = None,
    min_episodes: int = 30,
    include_platform: bool = False,
) -> ReferenceProfile:
    """Build a :class:`ReferenceProfile` from calibration episodes.

    The function extracts score metrics from each episode, computes
    robust distribution statistics (median, MAD, percentiles) for each
    metric, and optionally computes PCA loadings for the composite
    deviation score.

    PCA is computed on direction-aware bad z-scores so that all metrics
    are aligned: positive = more anomalous. This prevents opposing
    anomaly directions from cancelling out in the linear projection.

    Args:
        episodes: Iterable of :class:`EpisodeData` objects to use as the
            reference population.
        platform: Robot platform identifier (e.g. ``"unitree_h1"``).
        dataset_name: Name of the calibration dataset.
        task_scope: Optional task scope label (``None`` = all tasks).
        min_episodes: Minimum number of valid episodes required. If fewer
            valid episodes are found, the profile is still returned but
            PCA is skipped (not enough samples for meaningful PCA).
        include_platform: If True, also calibrate platform-specific
            metrics (velocity, path_length, etc.). Default False — only
            portable Tier-1 metrics are calibrated, producing a
            cross-platform comparable profile.

    Returns:
        A :class:`ReferenceProfile` populated with per-metric statistics
        and (when possible) PCA loadings.
    """
    # Determine which metrics to calibrate
    if include_platform:
        metric_names: tuple[str, ...] = ALL_SCORE_METRICS
    else:
        metric_names = PORTABLE_METRICS

    # Collect metric values for all episodes
    all_values: Dict[str, List[float]] = {name: [] for name in metric_names}

    for ep in episodes:
        if include_platform:
            metrics = extract_all_score_metrics(ep)
        else:
            metrics = extract_portable_metrics(ep)
        for name in metric_names:
            all_values[name].append(metrics[name])

    n_calibration = len(all_values[metric_names[0]]) if metric_names else 0

    # Build per-metric stats
    metric_stats: Dict[str, MetricStats] = {}
    for name in metric_names:
        values = np.array(all_values[name], dtype=np.float64)
        stats = percentile_stats(values)
        metric_stats[name] = MetricStats(
            median=stats["median"],
            mad=mad(values),
            p05=stats["p05"],
            p25=stats["p25"],
            p75=stats["p75"],
            p95=stats["p95"],
        )

    # Compute PCA loadings if we have enough data
    pca_components: List[Dict[str, float]] = []
    pca_explained_variance_ratios: List[float] = []

    if n_calibration >= min_episodes and len(metric_names) >= 2:
        # Build direction-aware bad z-score matrix
        z_columns = []
        metric_order = list(metric_names)
        for name in metric_order:
            values = np.array(all_values[name], dtype=np.float64)
            direction = METRIC_DIRECTIONS.get(name, "higher_is_worse")
            z_columns.append(bad_z_scores(values, direction))
        z_matrix = np.column_stack(z_columns)

        components, evrs = compute_pca_loadings(z_matrix, metric_order)
        if components:
            pca_components = components
            pca_explained_variance_ratios = evrs

    return ReferenceProfile(
        dataset_name=dataset_name,
        platform=platform,
        n_calibration=n_calibration,
        metrics=metric_stats,
        task_scope=task_scope,
        pca_components=pca_components,
        pca_explained_variance_ratios=pca_explained_variance_ratios,
        pca_n_components=len(pca_components),
    )


# ---------------------------------------------------------------------------
# BehavioralScorer
# ---------------------------------------------------------------------------


class BehavioralScorer:
    """Compute deviation scores for episodes against a reference profile.

    Uses direction-aware bad z-scores (MAD-based) for per-metric
    normalization and PCA loadings (from the first 2 principal components
    of the calibration population) for the composite score. Falls back to
    equal-weight RMS when PCA is unavailable.

    Three distinct scores are always computed:

    - **portable_score** — uses only portable Tier-1 metrics
      (duration, spike_count, effective_motion_ratio). Cross-platform
      comparable. This is the recommended "universal" score.
    - **platform_score** — uses only platform-specific metrics
      (velocity_p95, path_length, velocity_max). Only meaningful when
      both the reference and the test episode come from the same
      robot platform.
    - **combined_score** — uses all available metrics together. Useful
      for same-platform deep-dive analysis.

    The legacy ``deviation_score`` field is populated with one of these
    (controlled by ``score_type``) for backward compatibility.

    Args:
        reference: The :class:`ReferenceProfile` to score against.
    """

    def __init__(self, reference: ReferenceProfile) -> None:
        self.reference = reference

    @property
    def has_platform_metrics(self) -> bool:
        """True if the reference profile has platform-specific stats."""
        return any(name in self.reference.metrics for name in PLATFORM_METRICS)

    def score_episode(
        self,
        episode: EpisodeData,
        score_type: str = "combined",
    ) -> BehavioralScore:
        """Compute a behavioral deviation score for *episode*.

        Per-metric scores use raw robust z-scores (for display/reporting).
        The composite deviation score uses direction-aware bad z-scores
        so that all metrics contribute positively to anomaly detection.

        Args:
            episode: The episode to score.
            score_type: Which score to expose as ``deviation_score``.
                One of ``"portable"``, ``"platform"``, or ``"combined"``.
                Default ``"combined"`` for backward compatibility — when
                only portable metrics are calibrated, combined == portable.

        Returns:
            A :class:`BehavioralScore` with per-metric z-scores, three
            sub-scores (portable/platform/combined), and a primary
            ``deviation_score`` mirroring the requested ``score_type``.
        """
        if score_type not in ("portable", "platform", "combined"):
            raise ValueError(
                f"score_type must be 'portable', 'platform', or 'combined'; "
                f"got {score_type!r}"
            )

        # Extract all score metrics we can get (portable always, platform when available)
        portable_metrics = extract_portable_metrics(episode)
        has_platform = self.has_platform_metrics
        if has_platform:
            platform_metrics = extract_platform_metrics(episode)
        else:
            platform_metrics = {name: 0.0 for name in PLATFORM_METRICS}

        all_metrics: Dict[str, float] = {}
        all_metrics.update(portable_metrics)
        all_metrics.update(platform_metrics)

        # Per-metric robust z-scores (raw, for reporting)
        # Only include metrics that are actually in the reference profile
        # so that a portable-only profile doesn't leak platform metric keys.
        metric_scores: Dict[str, float] = {}
        for name in self.reference.metric_names:
            value = all_metrics.get(name, 0.0)
            if name in self.reference.metrics:
                stats = self.reference.metrics[name]
                metric_scores[name] = stats.robust_z(value)

        # Compute bad_z for each metric present in the reference
        bad_z_dict: Dict[str, float] = {}
        for name in self.reference.metric_names:
            value = all_metrics.get(name, 0.0)
            stats = self.reference.metrics[name]
            direction = METRIC_DIRECTIONS.get(name, "higher_is_worse")
            bad_z_dict[name] = stats.bad_z(value, direction)

        # Split bad_z into portable and platform subsets
        portable_bad_z: Dict[str, float] = {
            name: bad_z_dict[name]
            for name in PORTABLE_METRICS
            if name in bad_z_dict
        }
        platform_bad_z: Dict[str, float] = {
            name: bad_z_dict[name]
            for name in PLATFORM_METRICS
            if name in bad_z_dict
        }

        # --- Portable score ---
        portable_score = self._sub_score(portable_bad_z, subset="portable")

        # --- Platform score ---
        if has_platform and platform_bad_z:
            platform_score = self._sub_score(platform_bad_z, subset="platform")
        else:
            platform_score = 0.0

        # --- Combined score ---
        # If platform metrics are calibrated, combined uses all metrics.
        # Otherwise combined == portable (backward compatible).
        if has_platform:
            combined_score = self._sub_score(bad_z_dict, subset="combined")
        else:
            combined_score = portable_score

        # Determine primary deviation_score (legacy field)
        if score_type == "portable":
            deviation_score = portable_score
            method = self._score_method(portable_bad_z, subset="portable")
        elif score_type == "platform":
            deviation_score = platform_score
            method = self._score_method(platform_bad_z, subset="platform")
        else:  # combined
            deviation_score = combined_score
            method = self._score_method(bad_z_dict, subset="combined")

        # Percentile rank (based on portable metrics as the canonical reference)
        percentile = self._approx_percentile(portable_metrics)

        return BehavioralScore(
            deviation_score=deviation_score,
            metric_scores=metric_scores,
            percentile=percentile,
            method=method,
            portable_score=portable_score,
            platform_score=platform_score,
            combined_score=combined_score,
            score_type=score_type,
            has_platform_metrics=has_platform,
        )

    # --- Internal helpers ---

    def _sub_score(self, bad_z_dict: Dict[str, float], subset: str) -> float:
        """Compute a composite score for a subset of metrics.

        Uses PCA if PCA components for this subset are available in the
        reference; otherwise falls back to equal-weight RMS.

        Args:
            bad_z_dict: Direction-aware bad z-scores for the subset.
            subset: One of ``"portable"``, ``"platform"``, ``"combined"``.
                Determines which PCA components to prefer.
        """
        if not bad_z_dict:
            return 0.0
        # Try PCA if available and all PCA metrics are present in this subset
        if self.reference.has_pca() and self._all_pca_metrics_present(bad_z_dict):
            return self._pca_score(bad_z_dict)
        return self._rms_score(bad_z_dict)

    def _score_method(self, bad_z_dict: Dict[str, float], subset: str) -> str:
        """Return the method name (pca/rms) used for a given subset."""
        if (
            self.reference.has_pca()
            and self._all_pca_metrics_present(bad_z_dict)
        ):
            return "pca"
        return "rms"

    def _all_pca_metrics_present(self, metric_scores: Dict[str, float]) -> bool:
        """Check that all metrics required by PCA components are present."""
        if not self.reference.pca_components:
            return False
        # Check first component's metric keys
        if not self.reference.pca_components[0]:
            return False
        return all(
            name in metric_scores
            for name in self.reference.pca_components[0]
        )

    def _pca_score(self, bad_z_dict: Dict[str, float]) -> float:
        """Compute multi-component PCA composite score.

        score = sqrt(sum(pc_i_score^2 for each component))
        where pc_i_score = sum(bad_z[name] * loading[name] for each metric)
        """
        total_sq = 0.0
        for component in self.reference.pca_components:
            pc_score = 0.0
            for name, loading in component.items():
                pc_score += bad_z_dict.get(name, 0.0) * loading
            total_sq += pc_score ** 2
        return math.sqrt(total_sq)

    def _rms_score(self, bad_z_dict: Dict[str, float]) -> float:
        """Compute RMS of bad z-scores (equal-weight fallback)."""
        if not bad_z_dict:
            return 0.0
        sq_sum = sum(z * z for z in bad_z_dict.values())
        return float(math.sqrt(sq_sum / len(bad_z_dict)))

    def _approx_percentile(self, metrics: Dict[str, float]) -> float:
        """Approximate overall percentile from per-metric percentiles.

        Takes the maximum percentile across all portable metrics as a
        conservative "how anomalous is this episode" estimate.
        """
        if not metrics:
            return 0.5
        max_pct = 0.0
        for name, value in metrics.items():
            if name in self.reference.metrics:
                stats = self.reference.metrics[name]
                # Use two-tailed: distance from median in either direction
                pct = stats.percentile_rank(value)
                # Convert to "deviation percentile"
                dev_pct = max(pct, 1.0 - pct)
                max_pct = max(max_pct, dev_pct)
        return max_pct
