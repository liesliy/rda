"""Dataset-level aggregation of per-episode metric results.

Aggregates metric measurements across all episodes to produce dataset-level
statistics for the JSON report and text summary.

Three-layer output:
- Layer 1 (Integrity): counts of pass/exclude per integrity metric
- Layer 2 (Temporal & Motion): distribution of observational measurements
- Layer 3 (Dataset Utility): coverage and efficiency statistics
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from rda.audit.dataset_audit import DatasetAuditResult
from rda.metrics.base import MetricAvailability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_dict(
    values: np.ndarray,
    ps: tuple = (5.0, 50.0, 95.0),
) -> Dict[str, float]:
    if values.size == 0:
        out: Dict[str, float] = {}
        for p in ps:
            key = "median" if p == 50.0 else f"p{int(p) if p == int(p) else p}"
            out[key] = 0.0
        return out
    result: Dict[str, float] = {}
    for p in ps:
        key = "median" if p == 50.0 else f"p{int(p) if p == int(p) else p}"
        result[key] = float(np.percentile(values, p))
    return result


def _collect_measurement_values(
    result: DatasetAuditResult,
    metric_name: str,
    field: str,
) -> np.ndarray:
    """Collect a specific measurement field from all available episodes."""
    values: List[float] = []
    for ep in result.episodes.values():
        m = ep.metrics.get(metric_name)
        if m is None:
            continue
        if m.availability != MetricAvailability.AVAILABLE:
            continue
        val = m.measurement.get(field)
        if val is not None:
            values.append(float(val))
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-layer aggregators
# ---------------------------------------------------------------------------

def _aggregate_integrity(result: DatasetAuditResult) -> Dict[str, Any]:
    """Layer 1: Data Integrity — pass/fail counts per metric."""
    integrity_metrics = [
        "missing_dropout", "invalid_values", "schema_consistency",
        "timestamp_validity", "joint_limit",
    ]
    out: Dict[str, Any] = {}
    total = result.num_episodes

    for metric_name in integrity_metrics:
        available_count = 0
        pass_count = 0
        fail_count = 0
        na_count = 0

        for ep in result.episodes.values():
            m = ep.metrics.get(metric_name)
            if m is None:
                continue
            if m.availability != MetricAvailability.AVAILABLE:
                na_count += 1
                continue
            available_count += 1
            if m.passed:
                pass_count += 1
            else:
                fail_count += 1

        out[metric_name] = {
            "total_episodes": total,
            "available": available_count,
            "passed": pass_count,
            "failed": fail_count,
            "na": na_count,
            "pass_rate": pass_count / available_count if available_count > 0 else None,
        }

    return out


def _aggregate_temporal_motion(result: DatasetAuditResult) -> Dict[str, Any]:
    """Layer 2: Temporal & Motion — distribution of observational measurements."""
    out: Dict[str, Any] = {}

    # timestamp_validity: dt statistics
    dt_medians = _collect_measurement_values(result, "timestamp_validity", "median_dt_ms")
    if dt_medians.size > 0:
        out["timestamp_validity"] = {
            "median_dt_ms": _percentile_dict(dt_medians, (50.0, 95.0, 99.0)),
            "available_episodes": int(dt_medians.size),
        }

    # sensor_synchronization: worst p95 offset per episode
    sync_p95s: List[float] = []
    sync_available = 0
    sync_na = 0
    for ep in result.episodes.values():
        m = ep.metrics.get("sensor_synchronization")
        if m is None:
            continue
        if m.availability != MetricAvailability.AVAILABLE:
            sync_na += 1
            continue
        sync_available += 1
        p95 = m.measurement.get("worst_p95_offset_ms", 0.0)
        sync_p95s.append(float(p95))

    sync_arr = np.array(sync_p95s, dtype=np.float64)
    out["sensor_synchronization"] = {
        "available_episodes": sync_available,
        "na_episodes": sync_na,
        "worst_p95_offset_ms": _percentile_dict(sync_arr, (50.0, 95.0, 99.0)) if sync_arr.size else {"median": 0.0, "p95": 0.0, "p99": 0.0},
    }

    # velocity_acceleration: velocity percentiles
    vel_p95s = _collect_measurement_values(result, "velocity_acceleration", "velocity_p95")
    if vel_p95s.size > 0:
        out["velocity_acceleration"] = {
            "velocity_p95": _percentile_dict(vel_p95s, (50.0, 95.0, 99.0)),
            "available_episodes": int(vel_p95s.size),
        }

    # action_discontinuity: spike counts
    spike_counts = _collect_measurement_values(result, "action_discontinuity", "spike_count")
    if spike_counts.size > 0:
        affected = int(np.sum(spike_counts > 0))
        out["action_discontinuity"] = {
            "total_spikes": int(np.sum(spike_counts)),
            "episodes_with_spikes": affected,
            "spike_count_distribution": _percentile_dict(spike_counts, (50.0, 95.0, 99.0)),
            "available_episodes": int(spike_counts.size),
        }

    # temporal_sufficiency: idle structure and valid window ratios
    ts_fields = [
        "idle_total_ratio", "idle_prefix_ratio",
        "active_run_p50", "active_run_p90", "active_run_max",
        "transition_count",
        "valid_window_ratio_5", "valid_window_ratio_10", "valid_window_ratio_20",
    ]
    ts_available = 0
    ts_values: Dict[str, np.ndarray] = {}
    for field in ts_fields:
        arr = _collect_measurement_values(result, "temporal_sufficiency", field)
        ts_values[field] = arr
        if arr.size > ts_available:
            ts_available = int(arr.size)

    if ts_available > 0:
        ts_out: Dict[str, Any] = {"available_episodes": ts_available}
        for field in ts_fields:
            arr = ts_values[field]
            if arr.size > 0:
                ts_out[field] = _percentile_dict(arr, (5.0, 50.0, 95.0))
            else:
                ts_out[field] = {"p5": 0.0, "median": 0.0, "p95": 0.0}
        out["temporal_sufficiency"] = ts_out

    return out


def _aggregate_dataset_utility(result: DatasetAuditResult) -> Dict[str, Any]:
    """Layer 3: Dataset Utility — training data efficiency."""
    out: Dict[str, Any] = {}

    # idle_ratio / effective_motion_ratio
    idle_ratios = _collect_measurement_values(result, "idle_ratio", "idle_ratio")
    effective_ratios = _collect_measurement_values(result, "idle_ratio", "effective_motion_ratio")
    if idle_ratios.size > 0:
        out["idle_ratio"] = {
            "idle_ratio": _percentile_dict(idle_ratios, (5.0, 50.0, 95.0)),
            "effective_motion_ratio": _percentile_dict(effective_ratios, (5.0, 50.0, 95.0)),
            "available_episodes": int(idle_ratios.size),
        }

    # coverage (state_space_occupancy)
    occupancies = _collect_measurement_values(result, "coverage", "occupancy_rate")
    if occupancies.size > 0:
        out["coverage"] = {
            "state_space_occupancy": _percentile_dict(occupancies, (5.0, 50.0, 95.0)),
            "available_episodes": int(occupancies.size),
        }

    # distribution: trajectory duration and path length
    durations = _collect_measurement_values(result, "distribution", "duration_sec")
    path_lengths = _collect_measurement_values(result, "distribution", "path_length")
    if durations.size > 0:
        out["distribution"] = {
            "duration_sec": _percentile_dict(durations, (5.0, 50.0, 95.0)),
            "path_length": _percentile_dict(path_lengths, (5.0, 50.0, 95.0)),
            "available_episodes": int(durations.size),
        }

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def aggregate_dataset_metrics(result: DatasetAuditResult) -> Dict[str, Any]:
    """Compute dataset-level aggregate statistics from episode results.

    Returns a three-layer structure plus backward-compatible legacy keys:
        integrity:          Layer 1 pass/fail counts
        temporal_motion:    Layer 2 observational distributions
        dataset_utility:    Layer 3 training efficiency metrics
        score_distribution: Legacy compatibility — per-metric score stats
        
        Legacy keys (for backward compatibility):
        temporal: old-style temporal aggregation
        motion:   old-style motion aggregation
        distribution: old-style distribution aggregation
    """
    layer1 = _aggregate_integrity(result)
    layer2 = _aggregate_temporal_motion(result)
    layer3 = _aggregate_dataset_utility(result)
    
    return {
        # New three-layer structure
        "integrity": layer1,
        "temporal_motion": layer2,
        "dataset_utility": layer3,
        "score_distribution": _aggregate_score_distribution_legacy(result),
        # Legacy backward-compatible structure
        "temporal": _legacy_temporal(result),
        "motion": _legacy_motion(result),
        "distribution": _legacy_distribution(result),
    }


def _legacy_temporal(result: DatasetAuditResult) -> Dict[str, Any]:
    """Legacy temporal aggregation for backward compatibility."""
    dt_medians: List[float] = []
    sync_worst_p95s: List[float] = []
    sync_worst_maxs: List[float] = []

    for ep in result.episodes.values():
        m = ep.metrics.get("timestamp_validity")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            dt_ms = m.details.get("dt_ms", {})
            if isinstance(dt_ms, dict):
                if "median" in dt_ms and dt_ms["median"] is not None:
                    dt_medians.append(float(dt_ms["median"]))

        m = ep.metrics.get("sensor_synchronization")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            pairs = m.details.get("pairs", [])
            ep_worst_p95 = 0.0
            ep_worst_max = 0.0
            for pair in pairs:
                offset = pair.get("offset_ms", {})
                p95 = offset.get("p95", 0.0)
                mx = offset.get("max", 0.0)
                if p95 > ep_worst_p95:
                    ep_worst_p95 = p95
                if mx > ep_worst_max:
                    ep_worst_max = mx
            sync_worst_p95s.append(ep_worst_p95)
            sync_worst_maxs.append(ep_worst_max)

    dt_arr = np.array(dt_medians, dtype=np.float64)
    sync_p95_arr = np.array(sync_worst_p95s, dtype=np.float64)
    sync_max_arr = np.array(sync_worst_maxs, dtype=np.float64)

    return {
        "dt_median_ms": _percentile_dict(dt_arr, (50.0, 95.0, 99.0)) if dt_arr.size else {"median": 0.0, "p95": 0.0, "p99": 0.0},
        "sensor_sync_worst_p95_ms": {
            "median": float(np.median(sync_p95_arr)) if sync_p95_arr.size else 0.0,
            "p95": float(np.percentile(sync_p95_arr, 95)) if sync_p95_arr.size else 0.0,
            "max": float(np.max(sync_p95_arr)) if sync_p95_arr.size else 0.0,
        },
    }


def _legacy_motion(result: DatasetAuditResult) -> Dict[str, Any]:
    """Legacy motion aggregation for backward compatibility."""
    total_joint_violations = 0
    total_discontinuity_spikes = 0
    discontinuity_affected_episodes = 0
    idle_ratios: List[float] = []

    for ep in result.episodes.values():
        m = ep.metrics.get("joint_limit")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            total_joint_violations += int(m.details.get("violations", 0))

        m = ep.metrics.get("action_discontinuity")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            spikes = int(m.details.get("spike_count", 0))
            total_discontinuity_spikes += spikes
            if spikes > 0:
                discontinuity_affected_episodes += 1

        m = ep.metrics.get("idle_ratio")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            idle_ratio = m.details.get("idle_ratio")
            if idle_ratio is not None:
                idle_ratios.append(float(idle_ratio))

    idle_arr = np.array(idle_ratios, dtype=np.float64)

    return {
        "total_joint_violations": total_joint_violations,
        "total_discontinuity_spikes": total_discontinuity_spikes,
        "discontinuity_affected_episodes": discontinuity_affected_episodes,
        "idle_ratio": _percentile_dict(idle_arr, (5.0, 50.0, 95.0)) if idle_arr.size else {"p5": 0.0, "median": 0.0, "p95": 0.0},
    }


def _legacy_distribution(result: DatasetAuditResult) -> Dict[str, Any]:
    """Legacy distribution aggregation for backward compatibility."""
    coverages: List[float] = []

    for ep in result.episodes.values():
        m = ep.metrics.get("state_space_occupancy")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            grid = m.details.get("grid", {})
            cov = grid.get("occupancy_rate")
            if cov is not None:
                coverages.append(float(cov))

    cov_arr = np.array(coverages, dtype=np.float64)

    coverage_stats: Dict[str, float] = {}
    if cov_arr.size > 0:
        coverage_stats = {
            "median": float(np.median(cov_arr)),
            "min": float(np.min(cov_arr)),
            "max": float(np.max(cov_arr)),
            "p5": float(np.percentile(cov_arr, 5)),
            "p95": float(np.percentile(cov_arr, 95)),
        }
    else:
        coverage_stats = {
            "median": 0.0, "min": 0.0, "max": 0.0,
            "p5": 0.0, "p95": 0.0,
        }

    return {"coverage": coverage_stats}


def _aggregate_score_distribution_legacy(result: DatasetAuditResult) -> Dict[str, Any]:
    """Legacy: per-episode score percentiles for backward compatibility."""
    out: Dict[str, Dict[str, float]] = {}
    all_metric_names: set = set()
    for ep in result.episodes.values():
        all_metric_names.update(ep.metrics.keys())

    for name in sorted(all_metric_names):
        scores: List[float] = []
        for ep in result.episodes.values():
            m = ep.metrics.get(name)
            if m is not None:
                scores.append(m.score)
        arr = np.array(scores, dtype=np.float64)
        if arr.size == 0:
            continue
        out[name] = {
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return out
