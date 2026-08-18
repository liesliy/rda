"""Portable and platform-specific metric extraction for calibration and scoring.

Portable (Tier-1, cross-platform) metrics form the core of behavioral
scoring.  Per P2.5 experimentation they correlate at ρ=0.960 with the
full metric set while being directly comparable across robot platforms:

- ``duration_sec``          — episode duration in seconds
- ``spike_count``           — number of action-discontinuity spikes
- ``effective_motion_ratio``— fraction of frames with non-idle motion

Platform-specific metrics are only meaningful when calibrated on the
same robot platform and must NOT be mixed into a universal score:

- ``velocity_p95``          — 95th percentile of joint velocity magnitude
- ``path_length``           — total action trajectory path length
- ``acceleration_p95``      — 95th percentile of joint acceleration

This module extracts them directly from an :class:`EpisodeData` without
going through the full metric pipeline, for efficiency during calibration.
It reuses the same underlying helpers (``_mad``, ``_primary_action_array``)
from :mod:`rda.metrics.motion` and :mod:`rda.metrics.distribution` so
that the values are numerically identical to what the metrics report.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.motion import (
    IdleRatioMetric,
    ActionDiscontinuityMetric,
    VelocityMetric,
    _primary_action_array,
)
from rda.metrics.distribution import DistributionMetric

# Canonical list of portable metric names (Tier 1 — Universal / cross-platform).
PORTABLE_METRICS: tuple[str, ...] = (
    "duration_sec",
    "spike_count",
    "effective_motion_ratio",
)

# Platform-specific metric names (Tier 2/3 — only meaningful on the
# same robot platform; must NOT be mixed into a universal score).
PLATFORM_METRICS: tuple[str, ...] = (
    "velocity_p95",
    "path_length",
    "velocity_max",
)

# All metrics that can participate in calibration/scoring.
ALL_SCORE_METRICS: tuple[str, ...] = PORTABLE_METRICS + PLATFORM_METRICS

# Direction of "badness" for each score metric.
# "higher_is_worse": higher values = more anomalous (e.g. long duration, many spikes)
# "lower_is_worse": lower values = more anomalous (e.g. low effective motion)
METRIC_DIRECTIONS: Dict[str, str] = {
    # Portable
    "duration_sec": "higher_is_worse",
    "spike_count": "higher_is_worse",
    "effective_motion_ratio": "lower_is_worse",
    # Platform-specific
    "velocity_p95": "higher_is_worse",
    "path_length": "higher_is_worse",
    "velocity_max": "higher_is_worse",
}


def extract_portable_metrics(episode: EpisodeData) -> Dict[str, float]:
    """Extract the three portable Tier-1 metrics from an episode.

    Values are computed with the same logic used by the corresponding
    full metrics (DistributionMetric, ActionDiscontinuityMetric,
    IdleRatioMetric), so the numbers match what the audit pipeline
    reports.

    Args:
        episode: The episode to extract from.

    Returns:
        Dict with keys ``duration_sec``, ``spike_count``, and
        ``effective_motion_ratio``. Missing or non-computable metrics
        default to 0.0 (callers should validate before use).
    """
    result: Dict[str, float] = {
        "duration_sec": 0.0,
        "spike_count": 0.0,
        "effective_motion_ratio": 0.0,
    }

    # --- duration_sec ---
    dist = DistributionMetric()
    dist_result = dist.compute(episode)
    result["duration_sec"] = float(
        dist_result.measurement.get("duration_sec", 0.0)
    )

    # --- spike_count ---
    action = _primary_action_array(episode)
    if action is not None and action.size > 0 and action.shape[0] >= 3:
        ad = ActionDiscontinuityMetric()
        ad_result = ad.compute(episode)
        result["spike_count"] = float(
            ad_result.measurement.get("spike_count", 0.0)
        )
    else:
        result["spike_count"] = 0.0

    # --- effective_motion_ratio ---
    if action is not None and action.size > 0 and action.shape[0] >= 2:
        idle = IdleRatioMetric()
        idle_result = idle.compute(episode)
        result["effective_motion_ratio"] = float(
            idle_result.measurement.get("effective_motion_ratio", 0.0)
        )
    else:
        result["effective_motion_ratio"] = 0.0

    return result


def extract_platform_metrics(episode: EpisodeData) -> Dict[str, float]:
    """Extract platform-specific metrics from an episode.

    These are only meaningful within the same robot platform and must
    NOT be used for cross-platform ranking.

    Values are computed with the same logic used by the corresponding
    full metrics so the numbers match what the audit pipeline reports.

    Args:
        episode: The episode to extract from.

    Returns:
        Dict with keys ``velocity_p95``, ``path_length``, and
        ``velocity_max``.  Missing or non-computable metrics
        default to 0.0.
    """
    result: Dict[str, float] = {
        "velocity_p95": 0.0,
        "path_length": 0.0,
        "velocity_max": 0.0,
    }

    action = _primary_action_array(episode)

    # --- path_length (from DistributionMetric) ---
    dist = DistributionMetric()
    dist_result = dist.compute(episode)
    result["path_length"] = float(
        dist_result.measurement.get("path_length", 0.0)
    )

    # --- velocity_p95 and velocity_max (from VelocityMetric) ---
    if action is not None and action.size > 0 and action.shape[0] >= 3:
        vel = VelocityMetric()
        vel_result = vel.compute(episode)
        meas = vel_result.measurement
        result["velocity_p95"] = float(meas.get("velocity_p95", 0.0))
        result["velocity_max"] = float(meas.get("velocity_max", 0.0))
    else:
        result["velocity_p95"] = 0.0
        result["velocity_max"] = 0.0

    return result


def extract_all_score_metrics(episode: EpisodeData) -> Dict[str, float]:
    """Extract both portable and platform-specific metrics.

    Convenience helper that combines :func:`extract_portable_metrics`
    and :func:`extract_platform_metrics` into a single dict.

    Args:
        episode: The episode to extract from.

    Returns:
        Dict with all portable and platform metric keys.
    """
    result = extract_portable_metrics(episode)
    result.update(extract_platform_metrics(episode))
    return result