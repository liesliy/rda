"""Top Observations ranking for dataset audit results.

Extracts the most noteworthy findings from all episode-level metric
results, ranked by significance. Observations are neutral — they
describe what was found, not what is "wrong."

Replaces the old "Top Issues" terminology with "Top Observations".
N/A metrics (e.g. sensor_sync without stream_timestamps) are excluded
from ranking.
"""
from __future__ import annotations

from typing import Any, Dict, List

from rda.audit.dataset_audit import DatasetAuditResult
from rda.audit.rules import CRITICAL_METRICS, REVIEW_METRICS
from rda.metrics.base import MetricAvailability


# Hero metrics — get elevated visibility when data is available
HERO_METRICS: List[str] = [
    "sensor_synchronization",
    "action_discontinuity",
    "coverage",
]


# ---------------------------------------------------------------------------
# Per-metric observation descriptions
# ---------------------------------------------------------------------------

def _describe_integrity(metric: str, fail_count: int, total: int) -> str:
    ratio = fail_count / total if total > 0 else 0.0
    descs = {
        "missing_dropout": f"Missing frames or sensor dropout in {fail_count} episodes ({ratio:.0%})",
        "invalid_values": f"NaN/Inf values found in {fail_count} episode(s) ({ratio:.1%})",
        "schema_consistency": f"Schema/shape mismatch in {fail_count} episode(s) ({ratio:.1%})",
        "timestamp_validity": f"Invalid timestamps in {fail_count} episode(s) ({ratio:.1%})",
        "joint_limit": f"Joint-limit violations in {fail_count} episode(s) ({ratio:.1%})",
    }
    return descs.get(metric, f"{metric}: {fail_count} episode(s) flagged ({ratio:.1%})")


def _describe_sensor_sync(dataset_metrics: Dict[str, Any], total: int) -> str:
    temporal = dataset_metrics.get("temporal_motion", {})
    sync = temporal.get("sensor_synchronization", {})
    na_count = sync.get("na_episodes", 0)
    if na_count == total:
        return "Sensor sync: N/A (no per-stream timestamps in any episode)"
    p95 = sync.get("worst_p95_offset_ms", {})
    median_p95 = p95.get("median", 0.0)
    available = sync.get("available_episodes", 0)
    return f"Sensor sync: median p95 offset = {median_p95:.1f}ms ({available}/{total} episodes)"


def _describe_action_disc(dataset_metrics: Dict[str, Any], total: int) -> str:
    temporal = dataset_metrics.get("temporal_motion", {})
    disc = temporal.get("action_discontinuity", {})
    total_spikes = disc.get("total_spikes", 0)
    affected = disc.get("episodes_with_spikes", 0)
    ratio = affected / total if total > 0 else 0.0
    return f"Action discontinuity: {total_spikes} spike(s) across {affected} episode(s) ({ratio:.0%})"


def _describe_coverage(dataset_metrics: Dict[str, Any], total: int) -> str:
    utility = dataset_metrics.get("dataset_utility", {})
    cov = utility.get("coverage", {})
    occ = cov.get("state_space_occupancy", {})
    median = occ.get("median", 0.0)
    return f"State-space occupancy: median = {median:.1%} across episodes"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_top_observations(
    result: DatasetAuditResult,
    dataset_metrics: Dict[str, Any] | None = None,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Identify and rank the top observations across a dataset.

    N/A metrics are excluded from ranking. Only metrics with available
    data participate.

    Args:
        result: Dataset audit result with per-episode metric results.
        dataset_metrics: Pre-computed dataset-level aggregates.
        top_n: Maximum number of observations to return.

    Returns:
        A list of observation dicts with keys:
        rank, metric, description, significance, layer, details.
    """
    if dataset_metrics is None:
        from rda.report.aggregation import aggregate_dataset_metrics
        dataset_metrics = aggregate_dataset_metrics(result)

    total_episodes = result.num_episodes
    if total_episodes == 0:
        return []

    observations: List[Dict[str, Any]] = []

    # Layer 1: Integrity failures (high significance)
    integrity = dataset_metrics.get("integrity", {})
    for metric_name, stats in integrity.items():
        fail_count = stats.get("failed", 0)
        na_count = stats.get("na", 0)
        if fail_count == 0:
            continue
        observations.append({
            "metric": metric_name,
            "description": _describe_integrity(metric_name, fail_count, total_episodes),
            "significance": "high",
            "layer": "integrity",
            "affected_episodes": fail_count,
            "affected_ratio": fail_count / total_episodes,
            "na_episodes": na_count,
        })

    # Layer 2: Temporal & Motion observations
    temporal = dataset_metrics.get("temporal_motion", {})

    # Sensor sync — only if data is available
    sync = temporal.get("sensor_synchronization", {})
    sync_available = sync.get("available_episodes", 0)
    sync_na = sync.get("na_episodes", 0)
    if sync_available > 0:
        observations.append({
            "metric": "sensor_synchronization",
            "description": _describe_sensor_sync(dataset_metrics, total_episodes),
            "significance": "medium",
            "layer": "temporal_motion",
            "affected_episodes": sync_available,
            "affected_ratio": sync_available / total_episodes,
            "na_episodes": sync_na,
            "hero": True,
        })
    elif sync_na == total_episodes:
        observations.append({
            "metric": "sensor_synchronization",
            "description": _describe_sensor_sync(dataset_metrics, total_episodes),
            "significance": "info",
            "layer": "temporal_motion",
            "affected_episodes": 0,
            "affected_ratio": 0.0,
            "na_episodes": sync_na,
            "hero": True,
        })

    # Action discontinuity
    disc = temporal.get("action_discontinuity", {})
    disc_spikes = disc.get("total_spikes", 0)
    disc_affected = disc.get("episodes_with_spikes", 0)
    if disc_spikes > 0:
        observations.append({
            "metric": "action_discontinuity",
            "description": _describe_action_disc(dataset_metrics, total_episodes),
            "significance": "medium" if disc_affected / total_episodes < 0.1 else "high",
            "layer": "temporal_motion",
            "affected_episodes": disc_affected,
            "affected_ratio": disc_affected / total_episodes,
            "hero": True,
        })

    # Velocity/Acceleration
    vel = temporal.get("velocity_acceleration", {})
    vel_spikes_total = sum(
        ep.metrics.get("velocity_acceleration", type('', (), {'measurement': {}})).measurement.get("acceleration_spikes", 0)
        for ep in result.episodes.values()
        if ep.metrics.get("velocity_acceleration") is not None
        and ep.metrics["velocity_acceleration"].availability == MetricAvailability.AVAILABLE
    )
    if vel_spikes_total > 0:
        vel_affected = sum(
            1 for ep in result.episodes.values()
            if ep.metrics.get("velocity_acceleration") is not None
            and ep.metrics["velocity_acceleration"].availability == MetricAvailability.AVAILABLE
            and ep.metrics["velocity_acceleration"].measurement.get("acceleration_spikes", 0) > 0
        )
        observations.append({
            "metric": "velocity_acceleration",
            "description": f"Extreme acceleration spikes: {vel_spikes_total} across {vel_affected} episode(s)",
            "significance": "low",
            "layer": "temporal_motion",
            "affected_episodes": vel_affected,
            "affected_ratio": vel_affected / total_episodes,
        })

    # Layer 3: Dataset Utility
    utility = dataset_metrics.get("dataset_utility", {})

    # State-space occupancy
    sso = utility.get("state_space_occupancy", {})
    occ = sso.get("state_space_occupancy", {})
    if occ:
        median_occ = occ.get("median", 0.0)
        observations.append({
            "metric": "state_space_occupancy",
            "description": _describe_coverage(dataset_metrics, total_episodes),
            "significance": "medium" if median_occ < 0.1 else "low",
            "layer": "dataset_utility",
            "affected_episodes": sso.get("available_episodes", 0),
            "affected_ratio": sso.get("available_episodes", 0) / total_episodes,
            "hero": True,
        })

    # Idle ratio / effective motion
    idle = utility.get("idle_ratio", {})
    if idle:
        idle_stats = idle.get("idle_ratio", {})
        effective_stats = idle.get("effective_motion_ratio", {})
        median_idle = idle_stats.get("median", 0.0)
        median_effective = effective_stats.get("median", 0.0)
        if median_idle > 0.3:
            observations.append({
                "metric": "idle_ratio",
                "description": f"High idle ratio: median {median_idle:.1%} idle, {median_effective:.1%} effective motion",
                "significance": "medium",
                "layer": "dataset_utility",
                "affected_episodes": idle.get("available_episodes", 0),
                "affected_ratio": idle.get("available_episodes", 0) / total_episodes,
            })

    # Sort: significance (high > medium > low > info), then affected_ratio desc
    sig_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    observations.sort(key=lambda x: (sig_order.get(x["significance"], 99), -x.get("affected_ratio", 0)))

    # Assign ranks
    for i, obs in enumerate(observations):
        obs["rank"] = i + 1

    return observations[:top_n]


# Backward compatibility alias
compute_top_issues = compute_top_observations


# ---------------------------------------------------------------------------
# Hero metrics summary
# ---------------------------------------------------------------------------

def compute_hero_metrics(dataset_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract hero-metric summary values for the report.

    Handles N/A cases: if sensor_sync has no available data, returns
    interpretation="na" instead of a numeric value.
    """
    temporal = dataset_metrics.get("temporal_motion", {})
    utility = dataset_metrics.get("dataset_utility", {})

    # Sensor sync — handle N/A
    sync = temporal.get("sensor_synchronization", {})
    sync_available = sync.get("available_episodes", 0)
    sync_na = sync.get("na_episodes", 0)

    if sync_available == 0:
        sync_interp = "na"
        sync_p95 = {"median": 0.0, "p95": 0.0, "max": 0.0}
    else:
        p95_data = sync.get("worst_p95_offset_ms", {})
        median_p95 = p95_data.get("median", 0.0)
        if median_p95 < 5:
            sync_interp = "excellent"
        elif median_p95 < 20:
            sync_interp = "acceptable"
        elif median_p95 < 50:
            sync_interp = "needs_check"
        else:
            sync_interp = "severe"
        sync_p95 = p95_data

    # Action discontinuity
    disc = temporal.get("action_discontinuity", {})

    # State-space occupancy
    cov = utility.get("coverage", {})
    occ = cov.get("state_space_occupancy", {})
    median_cov = occ.get("median", 0.0)
    min_cov = occ.get("p5", 0.0)  # Use p5 as proxy for min
    max_cov = occ.get("p95", 0.0)  # Use p95 as proxy for max

    if median_cov > 0.5:
        cov_interp = "good"
    elif median_cov > 0.2:
        cov_interp = "moderate"
    else:
        cov_interp = "low"

    return {
        "sensor_synchronization": {
            "median_p95_offset_ms": sync_p95.get("median", 0.0),
            "p95_p95_offset_ms": sync_p95.get("p95", 0.0),
            "max_p95_offset_ms": sync_p95.get("p99", 0.0),
            "interpretation": sync_interp,
            "available_episodes": sync_available,
            "na_episodes": sync_na,
        },
        "action_discontinuity": {
            "total_spikes": disc.get("total_spikes", 0),
            "affected_episodes": disc.get("episodes_with_spikes", 0),
        },
        "state_space_occupancy": {
            "median_occupancy": float(median_cov),
            "min_occupancy": float(min_cov),
            "max_occupancy": float(max_cov),
            "range": [float(min_cov), float(max_cov)],
            "interpretation": cov_interp,
        },
    }
