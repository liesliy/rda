"""Audit history tracking for RDA.

Records audit snapshots to ``.rda/audit_history.json`` under the dataset
directory, enabling trend analysis across multiple audit runs.

Data is stored per-dataset so that different datasets maintain independent
histories.  Each snapshot captures the key metrics from a single audit run:
verdict distribution, integrity pass rate, deviation scores, top patterns,
and hero metrics.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rda.audit.dataset_audit import DatasetAuditResult
from rda.audit.rules import AuditVerdict, CRITICAL_METRICS, REVIEW_METRICS
from rda.metrics.base import MetricAvailability


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HISTORY_FILENAME = "audit_history.json"
_RDA_DIR = ".rda"


def _history_path(dataset_path: str) -> Path:
    """Return the path to the audit history JSON file."""
    return Path(dataset_path) / _RDA_DIR / _HISTORY_FILENAME


def _detect_pattern_type(ep_result) -> Optional[str]:
    """Detect pattern type for an episode (mirrors UI logic)."""
    m_idle = ep_result.metrics.get("idle_ratio")
    m_disc = ep_result.metrics.get("action_discontinuity")
    m_dist = ep_result.metrics.get("distribution")

    idle_low = False
    spikes_high = False
    duration_long = False

    if m_idle and m_idle.availability == MetricAvailability.AVAILABLE:
        eff = m_idle.measurement.get("effective_motion_ratio", 1.0)
        idle_low = eff < 0.3

    if m_disc and m_disc.availability == MetricAvailability.AVAILABLE:
        spikes = m_disc.measurement.get("spike_count", 0)
        spikes_high = spikes > 20

    if m_dist and m_dist.availability == MetricAvailability.AVAILABLE:
        if not m_dist.passed:
            duration_long = True

    if idle_low and spikes_high:
        return "Stuck"
    if idle_low and not spikes_high:
        return "Frozen"
    if spikes_high and not idle_low:
        return "Jittery"
    if duration_long and not idle_low:
        return "Inefficient"

    for m_name, m in ep_result.metrics.items():
        if m_name in REVIEW_METRICS and m.availability == MetricAvailability.AVAILABLE and not m.passed:
            return "Unusual"

    return None


def _count_patterns(result: DatasetAuditResult) -> List[Dict[str, Any]]:
    """Count pattern type occurrences across all episodes."""
    pattern_counts: Dict[str, int] = {}
    for ep in result.episodes.values():
        pt = _detect_pattern_type(ep)
        if pt is not None:
            pattern_counts[pt] = pattern_counts.get(pt, 0) + 1
    # Sort by count descending
    return [
        {"pattern": p, "count": c}
        for p, c in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    ]


def _compute_integrity_pass_rate(result: DatasetAuditResult) -> float:
    """Fraction of episodes with no critical metric failures."""
    total = result.num_episodes
    if total == 0:
        return 0.0
    pass_count = 0
    for ep in result.episodes.values():
        has_fail = False
        for m_name in CRITICAL_METRICS:
            m = ep.metrics.get(m_name)
            if m is None:
                continue
            if m.availability != MetricAvailability.AVAILABLE:
                continue
            if not m.passed:
                has_fail = True
                break
        if not has_fail:
            pass_count += 1
    return round(pass_count / total, 4)


def _compute_deviation_stats(result: DatasetAuditResult) -> Dict[str, float]:
    """Compute mean and median deviation scores across episodes."""
    scores: List[float] = []
    for ep in result.episodes.values():
        ep_scores: List[float] = []
        for m_name, m in ep.metrics.items():
            if m.availability == MetricAvailability.AVAILABLE:
                ep_scores.append(m.score)
        if ep_scores:
            avg = float(np.mean(ep_scores))
            scores.append(round((1.0 - avg) * 100, 4))
    if not scores:
        return {"mean_deviation_score": 0.0, "median_deviation_score": 0.0}
    arr = np.array(scores, dtype=np.float64)
    return {
        "mean_deviation_score": round(float(np.mean(arr)), 4),
        "median_deviation_score": round(float(np.median(arr)), 4),
    }


def _build_hero_metrics_summary(
    result: DatasetAuditResult,
) -> Dict[str, Any]:
    """Build a compact hero-metrics summary for the snapshot."""
    from rda.report.top_issues import compute_hero_metrics
    from rda.report.aggregation import aggregate_dataset_metrics

    dataset_metrics = aggregate_dataset_metrics(result)
    hero = compute_hero_metrics(dataset_metrics)

    # Simplify for snapshot storage
    summary: Dict[str, Any] = {}

    sync = hero.get("sensor_synchronization", {})
    summary["sensor_sync_median_p95_ms"] = sync.get("median_p95_offset_ms", 0.0)
    summary["sensor_sync_interpretation"] = sync.get("interpretation", "na")

    disc = hero.get("action_discontinuity", {})
    summary["action_disc_total_spikes"] = disc.get("total_spikes", 0)
    summary["action_disc_affected_episodes"] = disc.get("affected_episodes", 0)

    cov = hero.get("state_space_occupancy", {})
    summary["state_space_median_occupancy"] = cov.get("median_occupancy", 0.0)
    summary["state_space_interpretation"] = cov.get("interpretation", "low")

    return summary


def _get_rda_version() -> str:
    """Get the current RDA version string."""
    try:
        from rda import __version__
        return __version__
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_audit_snapshot(
    audit_result: DatasetAuditResult,
    dataset_path: str,
) -> Path:
    """Save an audit snapshot to the history file.

    Appends a new snapshot entry to ``{dataset_path}/.rda/audit_history.json``.
    Creates the file and directory if they do not exist.

    Args:
        audit_result: The completed audit result.
        dataset_path: Path to the dataset directory.

    Returns:
        Path to the history JSON file.
    """
    # Build verdict counts dict with string keys
    verdict_counts = {v.value: c for v, c in audit_result.verdict_counts.items()}

    deviation_stats = _compute_deviation_stats(audit_result)
    top_patterns = _count_patterns(audit_result)
    hero_summary = _build_hero_metrics_summary(audit_result)

    # Derive a human-readable dataset name from the path
    dataset_name = Path(dataset_path).name

    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "total_episodes": audit_result.num_episodes,
        "verdict_counts": verdict_counts,
        "integrity_pass_rate": _compute_integrity_pass_rate(audit_result),
        "mean_deviation_score": deviation_stats["mean_deviation_score"],
        "median_deviation_score": deviation_stats["median_deviation_score"],
        "top_patterns": top_patterns,
        "hero_metrics_summary": hero_summary,
        "rda_version": _get_rda_version(),
    }

    # Read existing history or start fresh
    history_file = _history_path(dataset_path)
    history = _load_raw(history_file)

    history["audits"].append(snapshot)

    # Write back
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return history_file


def load_audit_history(
    dataset_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load audit history, optionally filtered by dataset path.

    When ``dataset_path`` is ``None``, attempts to discover and merge
    all audit history files found under common dataset directories.
    However, the typical usage is to pass a specific dataset path.

    Args:
        dataset_path: If provided, only return audits for this dataset.
            If None, returns an empty list (multi-dataset discovery is
            not supported in V1).

    Returns:
        List of audit snapshot dicts, sorted by timestamp ascending.
    """
    if dataset_path is None:
        return []

    history_file = _history_path(dataset_path)
    history = _load_raw(history_file)
    return history.get("audits", [])


def compute_trend(
    history: List[Dict[str, Any]],
    metric_name: str,
) -> List[Dict[str, Any]]:
    """Compute trend data for a specific metric across audit history.

    Supported metric names:
    - ``"pass_rate"``: PASS verdict ratio over time
    - ``"review_rate"``: REVIEW verdict ratio over time
    - ``"exclude_rate"``: EXCLUDE verdict ratio over time
    - ``"mean_deviation_score"``: Mean deviation score over time
    - ``"median_deviation_score"``: Median deviation score over time
    - ``"integrity_pass_rate"``: Integrity pass rate over time
    - ``"exclude_count"``: Raw EXCLUDE count over time
    - ``"total_episodes"``: Total episodes audited over time

    Args:
        history: List of audit snapshot dicts (from ``load_audit_history``).
        metric_name: The metric to extract.

    Returns:
        List of ``{"timestamp": str, "value": float}`` dicts sorted by
        timestamp ascending.
    """
    trend: List[Dict[str, Any]] = []

    for snapshot in sorted(history, key=lambda s: s.get("timestamp", "")):
        ts = snapshot.get("timestamp", "")
        total = snapshot.get("total_episodes", 0)
        vc = snapshot.get("verdict_counts", {})

        if metric_name == "pass_rate":
            val = vc.get("PASS", 0) / total if total > 0 else 0.0
        elif metric_name == "review_rate":
            val = vc.get("REVIEW", 0) / total if total > 0 else 0.0
        elif metric_name == "exclude_rate":
            val = vc.get("EXCLUDE", 0) / total if total > 0 else 0.0
        elif metric_name == "exclude_count":
            val = float(vc.get("EXCLUDE", 0))
        elif metric_name == "total_episodes":
            val = float(total)
        elif metric_name == "mean_deviation_score":
            val = snapshot.get("mean_deviation_score", 0.0)
        elif metric_name == "median_deviation_score":
            val = snapshot.get("median_deviation_score", 0.0)
        elif metric_name == "integrity_pass_rate":
            val = snapshot.get("integrity_pass_rate", 0.0)
        else:
            # Try direct key lookup
            val = snapshot.get(metric_name, 0.0)

        trend.append({"timestamp": ts, "value": round(val, 4)})

    return trend


# ---------------------------------------------------------------------------
# Internal I/O
# ---------------------------------------------------------------------------

def _load_raw(history_file: Path) -> Dict[str, Any]:
    """Load raw history dict from file, returning empty structure if missing."""
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "audits" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"audits": []}


def discover_datasets_with_history(base_dirs: List[str]) -> List[str]:
    """Discover dataset directories that have audit history files.

    Args:
        base_dirs: List of base directories to scan.

    Returns:
        List of dataset paths that contain audit history files.
    """
    found: List[str] = []
    for base in base_dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for history_file in base_path.rglob(_HISTORY_FILENAME):
            # The dataset path is the grandparent of .rda/
            dataset_dir = history_file.parent.parent
            found.append(str(dataset_dir))
    return sorted(set(found))
