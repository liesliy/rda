"""JSON report generation with three-layer output.

Produces structured JSON reports from dataset audit results. Two output
formats are provided:

**Engine format** (:func:`generate_json_report`):
  Three-layer metric aggregates with full per-metric detail — used for
  programmatic consumption and as the source of truth.

**MVP Spec v0.2.0 product format** (:func:`generate_episode_report`,
:func:`generate_dataset_report`):
  User-facing JSON schema defined in MVP Product Spec §7.3 / §7.4, with
  fields like ``integrity_check``, ``behavior_verdict``,
  ``deviation_score``, ``reason_vector``, ``pattern_type``, and
  ``user_verdict``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rda.audit.dataset_audit import DatasetAuditResult
from rda.audit.rules import AuditVerdict, compute_behavior_severity
from rda.metrics.base import MetricAvailability
from rda.report.aggregation import aggregate_dataset_metrics
from rda.report.summary import build_summary
from rda.report.top_issues import compute_top_observations, compute_hero_metrics


# ---------------------------------------------------------------------------
# Engine-format: three-layer JSON report (backward compatible)
# ---------------------------------------------------------------------------

def _episode_result_to_dict(ep_result) -> Dict[str, Any]:
    """Convert an EpisodeAuditResult to a JSON-serializable dict.

    Args:
        ep_result: The :class:`EpisodeAuditResult` to serialize.

    Returns:
        Dict with keys ``episode_index``, ``num_frames``, ``verdict``,
        and ``metrics`` — each metric has the full MetricResult schema.
    """
    metrics_dict: Dict[str, Any] = {}
    for name, m in ep_result.metrics.items():
        metrics_dict[name] = {
            "availability": m.availability.value,
            "measurement": m.measurement,
            "assessment": m.assessment,
            "details": m.details,
            "message": m.message,
            "evidence_level": _evidence_level_for_metric(name),
            # Backward compat
            "passed": m.passed,
            "score": m.score,
            "has_finding": m.has_finding,
        }

    return {
        "episode_index": ep_result.episode_index,
        "num_frames": ep_result.num_frames,
        "verdict": ep_result.verdict.value,
        "metrics": metrics_dict,
    }


def generate_json_report(result: DatasetAuditResult) -> Dict[str, Any]:
    """Generate the full JSON report structure (engine format).

    This is the detailed three-layer format used for programmatic
    consumption. For the user-facing MVP Spec v0.2.0 format, use
    :func:`generate_dataset_report` and :func:`generate_episode_report`
    instead.

    Args:
        result: The dataset audit result.

    Returns:
        A dict ready for JSON serialization with keys:
        ``version``, ``dataset``, ``summary``,
        ``three_layer_aggregates``, ``hero_metrics``,
        ``top_observations``, ``episodes``.
    """
    dataset_metrics = aggregate_dataset_metrics(result)
    top_obs = compute_top_observations(result, dataset_metrics=dataset_metrics)
    hero_metrics = compute_hero_metrics(dataset_metrics)
    compact = build_summary(result)

    # Per-episode results
    episodes = [_episode_result_to_dict(ep) for ep in result.episodes.values()]

    # Import tool version
    from rda import __version__ as _rda_version

    report = {
        "version": "0.2.0",
        "tool_version": _rda_version,
        "dataset": {
            "path": result.dataset_info.path,
            "num_episodes": result.dataset_info.num_episodes,
            "total_frames": result.dataset_info.total_frames,
            "modalities": result.dataset_info.modalities,
        },
        "summary": {
            "total_episodes": compact.total_episodes,
            "verdict_counts": compact.verdict_counts,
            "pass_rate": compact.pass_rate,
            "exclude_episodes": compact.exclude_episodes,
            "review_episodes": compact.review_episodes,
        },
        "three_layer_aggregates": {
            "layer1_integrity": dataset_metrics.get("integrity", {}),
            "layer2_temporal_motion": dataset_metrics.get("temporal_motion", {}),
            "layer3_dataset_utility": dataset_metrics.get("dataset_utility", {}),
        },
        "hero_metrics": hero_metrics,
        "top_observations": top_obs,
        "episodes": episodes,
    }

    return report


def format_json_report(result: DatasetAuditResult, indent: int = 2) -> str:
    """Generate the JSON report as a formatted string.

    Args:
        result: The dataset audit result.
        indent: JSON indentation level.

    Returns:
        JSON string.
    """
    report = generate_json_report(result)
    return json.dumps(report, indent=indent, ensure_ascii=False, default=str)


def save_json_report(result: DatasetAuditResult, path: str, indent: int = 2) -> None:
    """Save the JSON report to a file.

    Args:
        result: The dataset audit result.
        path: Output file path.
        indent: JSON indentation level.
    """
    json_str = format_json_report(result, indent=indent)
    Path(path).write_text(json_str, encoding="utf-8")


# ---------------------------------------------------------------------------
# MVP Spec v0.2.0 — Episode-level JSON schema (§7.3)
# ---------------------------------------------------------------------------

# Mapping from internal AuditVerdict to MVP Spec behavior_verdict values
_BEHAVIOR_VERDICT_MAP: Dict[AuditVerdict, str] = {
    AuditVerdict.PASS: "LIKELY_USABLE",
    AuditVerdict.REVIEW: "NEEDS_REVIEW",
    AuditVerdict.EXCLUDE: "RECOMMENDED_EXCLUDE",
}

# Evidence labels are deliberately explicit: statistical observations must not
# be serialized as if they were deterministic corruption.
_EVIDENCE_LEVEL_BY_METRIC: Dict[str, str] = {
    "missing_dropout": "HARD_FAIL",
    "invalid_values": "HARD_FAIL",
    "schema_consistency": "HARD_FAIL",
    "timestamp_validity": "HARD_FAIL",
    "joint_limit": "HARD_FAIL",
    "video_frame_integrity": "HARD_FAIL",
    "sensor_synchronization": "UNVERIFIABLE",
    "sampling_jitter": "RISK_SIGNAL",
    "velocity_acceleration": "RISK_SIGNAL",
    "action_discontinuity": "RISK_SIGNAL",
    "idle_ratio": "RISK_SIGNAL",
    "distribution": "RISK_SIGNAL",
    "coverage": "RISK_SIGNAL",
}


def _evidence_level_for_metric(metric_name: str) -> str:
    """Return the user-facing evidence class for a metric."""
    return _EVIDENCE_LEVEL_BY_METRIC.get(metric_name, "RISK_SIGNAL")


def _build_integrity_check(ep_result) -> Dict[str, Any]:
    """Build the ``integrity_check`` block for an episode (Layer 1A).

    Collects all Layer 1 (integrity) metric failures.

    Args:
        ep_result: An :class:`EpisodeAuditResult`.

    Returns:
        Dict with ``passed`` (bool) and ``issues`` (list of issue codes
        and descriptions).
    """
    from rda.metrics import LAYER1_INTEGRITY

    integrity_metric_names = {cls.name for cls in LAYER1_INTEGRITY}
    issues: List[Dict[str, str]] = []

    for m_name, m_result in ep_result.metrics.items():
        if m_name not in integrity_metric_names:
            continue
        if m_result.availability != MetricAvailability.AVAILABLE:
            continue
        if not m_result.passed:
            reason = m_result.assessment.get("reason", m_name)
            issues.append({
                "code": _issue_code_for_metric(m_name),
                "metric": m_name,
                "reason": reason or m_name,
            })

    return {
        "passed": len(issues) == 0,
        "issues": issues,
    }


def _issue_code_for_metric(metric_name: str) -> str:
    """Map a metric name to its MVP Spec issue code (§5.1).

    Args:
        metric_name: Internal metric name (e.g. ``"missing_dropout"``).

    Returns:
        Issue code string (e.g. ``"INT-01"``). Returns ``"UNK-00"``
        for unrecognized metrics.
    """
    mapping: Dict[str, str] = {
        "missing_dropout": "INT-01",
        "invalid_values": "INT-02",
        "schema_consistency": "INT-03",
        "timestamp_validity": "INT-04",
        "joint_limit": "INT-05",
        "video_frame_integrity": "INT-06",
        "sensor_synchronization": "TMP-03",
        "sampling_jitter": "TMP-03",
        "action_discontinuity": "MOT-01",
        "velocity_acceleration": "MOT-02",
        "idle_ratio": "MOT-03",
        "distribution": "DIS-01",
        "coverage": "DIS-03",
    }
    return mapping.get(metric_name, "UNK-00")


def _build_reason_vector(ep_result) -> Dict[str, float]:
    """Build the per-metric reason vector for an episode.

    Maps Tier 1 + key Tier 2 metric anomaly magnitudes to a 0–∞ scale,
    where 0 = baseline and higher values = more anomalous.

    Args:
        ep_result: An :class:`EpisodeAuditResult`.

    Returns:
        Dict mapping anomaly name to float score.
    """
    vector: Dict[str, float] = {}

    # Tier 1: duration, spikes, motion ratio
    dist_m = ep_result.metrics.get("distribution")
    if dist_m and dist_m.availability == MetricAvailability.AVAILABLE:
        dur = dist_m.measurement.get("duration_sec", 0.0)
        vector["duration_anomaly"] = float(dur)  # raw value; scaled by reference

    disc_m = ep_result.metrics.get("action_discontinuity")
    if disc_m and disc_m.availability == MetricAvailability.AVAILABLE:
        spikes = disc_m.measurement.get("spike_count", 0)
        vector["spike_anomaly"] = float(spikes)

    idle_m = ep_result.metrics.get("idle_ratio")
    if idle_m and idle_m.availability == MetricAvailability.AVAILABLE:
        motion_ratio = idle_m.measurement.get("effective_motion_ratio", 1.0)
        # anomaly = 1 - ratio (lower motion = more anomalous)
        vector["motion_ratio_anomaly"] = float(1.0 - motion_ratio)

    # Tier 2: velocity, path length (optional diagnostic signals)
    vel_m = ep_result.metrics.get("velocity_acceleration")
    if vel_m and vel_m.availability == MetricAvailability.AVAILABLE:
        vel_p95 = vel_m.measurement.get("velocity_p95", 0.0)
        vector["velocity_anomaly"] = float(vel_p95)

    if dist_m and dist_m.availability == MetricAvailability.AVAILABLE:
        plen = dist_m.measurement.get("path_length", 0.0)
        vector["path_length_anomaly"] = float(plen)

    return vector


def _compute_deviation_score(reason_vector: Dict[str, float]) -> float:
    """Compute a scalar deviation score from the reason vector.

    This is a simplified heuristic for the MVP — the sum of normalized
    Tier 1 anomaly components. The full PCA-based deviation score
    will be implemented in a later phase.

    Args:
        reason_vector: Output of :func:`_build_reason_vector`.

    Returns:
        A float deviation score (higher = more anomalous).
    """
    # Sum of Tier 1 components as a rough proxy
    tier1_keys = ["duration_anomaly", "spike_anomaly", "motion_ratio_anomaly"]
    score = sum(reason_vector.get(k, 0.0) for k in tier1_keys)
    return round(score, 2)


def _determine_pattern_type(
    ep_result,
    reason_vector: Dict[str, float],
) -> Dict[str, Optional[str]]:
    """Determine the primary and secondary pattern types (§4.3).

    Based on the relative magnitudes of components in the reason vector.

    Args:
        ep_result: An :class:`EpisodeAuditResult`.
        reason_vector: Output of :func:`_build_reason_vector`.

    Returns:
        Dict with ``primary`` and ``secondary`` pattern type strings,
        or ``None`` if no clear pattern is detected.
    """
    dur = reason_vector.get("duration_anomaly", 0.0)
    spikes = reason_vector.get("spike_anomaly", 0.0)
    motion_low = reason_vector.get("motion_ratio_anomaly", 0.0)
    path_len = reason_vector.get("path_length_anomaly", 0.0)

    patterns: List[Tuple[str, float]] = []

    # Stuck: long duration + very low motion + normal spikes
    if dur > 0 and motion_low > 0.5 and spikes < 5:
        score = dur * motion_low
        patterns.append(("STUCK_LIKE", score))

    # Frozen: moderate duration + extremely low motion + very low spikes
    if motion_low > 0.8 and spikes < 2:
        patterns.append(("FROZEN_LIKE", motion_low * 10))

    # Jittery: high spike count
    if spikes > 10:
        patterns.append(("JITTERY_LIKE", spikes))

    # Inefficient: long duration + long path + normal motion
    if dur > 0 and path_len > 0 and motion_low < 0.3:
        patterns.append(("INEFFICIENT", dur + path_len / 100.0))

    # Unusual: trajectory distribution anomaly but others normal
    if not patterns and dur > 0:
        patterns.append(("UNUSUAL", dur * 0.1))

    if not patterns:
        return {"primary": None, "secondary": None}

    patterns.sort(key=lambda x: x[1], reverse=True)
    primary = patterns[0][0]
    secondary = patterns[1][0] if len(patterns) > 1 else None

    return {"primary": primary, "secondary": secondary}


def _build_diagnosis(
    ep_result,
    pattern_type: Dict[str, Optional[str]],
    reason_vector: Dict[str, float],
) -> Dict[str, str]:
    """Build the WHAT / WHY / NEXT diagnosis strings (§6.1).

    Args:
        ep_result: An :class:`EpisodeAuditResult`.
        pattern_type: Output of :func:`_determine_pattern_type`.
        reason_vector: Output of :func:`_build_reason_vector`.

    Returns:
        Dict with ``what``, ``why``, and ``next`` strings.
    """
    primary = pattern_type.get("primary")
    dur = reason_vector.get("duration_anomaly", 0.0)
    spikes = reason_vector.get("spike_anomaly", 0.0)
    motion_ratio = 1.0 - reason_vector.get("motion_ratio_anomaly", 0.0)

    # WHAT — brief factual description
    what_parts: List[str] = []
    if dur > 0:
        what_parts.append(f"duration = {dur:.2f}s")
    if spikes > 0:
        what_parts.append(f"{spikes:.0f} action spikes")
    what_parts.append(f"effective motion = {motion_ratio:.0%}")
    what = "Episode " + ", ".join(what_parts) + "."

    # WHY — keep statistical patterns explicitly observational.
    why_map: Dict[str, str] = {
        "STUCK_LIKE": "Observed a stuck-like low-motion pattern; this is a statistical signal, not proof of obstruction or failure.",
        "FROZEN_LIKE": "Observed a frozen-like near-zero-motion pattern; valid waiting behavior or a failed episode cannot be distinguished from metrics alone.",
        "JITTERY_LIKE": "Observed elevated action discontinuity; controller behavior, sensor noise, or a valid sharp maneuver may explain it.",
        "INEFFICIENT": "Observed a long or unusual execution path; this may be valid task variation and requires context.",
        "UNUSUAL": "Observed an unusual trajectory pattern; it may represent valid but rare behavior.",
    }
    why = why_map.get(primary or "", "No clear anomaly pattern detected.")

    # NEXT — recommendations remain human-review actions, never automatic labels.
    next_map: Dict[str, str] = {
        "STUCK_LIKE": "Review the episode and trajectory; confirm whether the low-motion segment is expected before re-collecting.",
        "FROZEN_LIKE": "Review manually; do not exclude from the dataset based on this signal alone.",
        "JITTERY_LIKE": "Review action smoothness and context; check controller tuning or sensor noise if the pattern is unexpected.",
        "INEFFICIENT": "Review the execution path and task context before deciding whether optimization or re-demonstration is needed.",
        "UNUSUAL": "Review the trajectory visualization and confirm whether this rare pattern is valid.",
    }
    next_action = next_map.get(primary or "", "Review the available evidence before deciding how to handle the episode.")

    return {"what": what, "why": why, "next": next_action}


def _build_metrics_block(ep_result) -> Dict[str, float]:
    """Extract core metric values for the episode JSON (§7.3 metrics field).

    Args:
        ep_result: An :class:`EpisodeAuditResult`.

    Returns:
        Dict of metric name → scalar value, including Tier 1 (universal)
        and key Tier 2 (normalizable) metrics.
    """
    metrics: Dict[str, float] = {}

    # Tier 1 — Universal
    dist_m = ep_result.metrics.get("distribution")
    if dist_m and dist_m.availability == MetricAvailability.AVAILABLE:
        metrics["duration_sec"] = float(dist_m.measurement.get("duration_sec", 0.0))
        metrics["path_length"] = float(dist_m.measurement.get("path_length", 0.0))

    disc_m = ep_result.metrics.get("action_discontinuity")
    if disc_m and disc_m.availability == MetricAvailability.AVAILABLE:
        metrics["spike_count"] = float(disc_m.measurement.get("spike_count", 0))

    idle_m = ep_result.metrics.get("idle_ratio")
    if idle_m and idle_m.availability == MetricAvailability.AVAILABLE:
        metrics["effective_motion_ratio"] = float(
            idle_m.measurement.get("effective_motion_ratio", 1.0)
        )

    # Tier 2 — Normalizable (optional diagnostic signals)
    vel_m = ep_result.metrics.get("velocity_acceleration")
    if vel_m and vel_m.availability == MetricAvailability.AVAILABLE:
        metrics["velocity_p95"] = float(vel_m.measurement.get("velocity_p95", 0.0))

    return metrics


def _build_reference_block(result: DatasetAuditResult) -> Dict[str, float]:
    """Build the reference baseline block from dataset aggregates.

    Args:
        result: The full dataset audit result.

    Returns:
        Dict of reference median values for Tier 1 metrics.
    """
    dataset_metrics = aggregate_dataset_metrics(result)
    utility = dataset_metrics.get("dataset_utility", {})
    temporal = dataset_metrics.get("temporal_motion", {})

    reference: Dict[str, float] = {}

    # Duration
    dist = utility.get("distribution", {})
    if dist:
        dur_stats = dist.get("duration_sec", {})
        if dur_stats:
            reference["duration_median"] = float(dur_stats.get("median", 0.0))

    # Spikes
    disc = temporal.get("action_discontinuity", {})
    if disc:
        spike_dist = disc.get("spike_count_distribution", {})
        if spike_dist:
            reference["spike_median"] = float(spike_dist.get("median", 0.0))

    # Velocity
    vel = temporal.get("velocity_acceleration", {})
    if vel:
        vel_stats = vel.get("velocity_p95", {})
        if vel_stats:
            reference["velocity_median"] = float(vel_stats.get("median", 0.0))

    return reference


def generate_episode_report(
    result: DatasetAuditResult,
    episode_index: int,
) -> Dict[str, Any]:
    """Generate the MVP Spec v0.2.0 episode-level JSON report (§7.3).

    Produces a user-facing episode report with fields:
    ``integrity_check``, ``behavior_verdict``, ``deviation_score``,
    ``pattern_type``, ``reason_vector``, ``metrics``, ``reference``,
    ``issues``, ``diagnosis``, and ``user_verdict``.

    Args:
        result: The full dataset audit result (used for reference
            distribution computation).
        episode_index: Index of the episode to generate the report for.

    Returns:
        Dict matching the MVP Spec v0.2.0 Episode JSON Schema.

    Raises:
        KeyError: If ``episode_index`` is not found in the result.
    """
    ep_result = result.episodes[episode_index]

    integrity_check = _build_integrity_check(ep_result)
    behavior_verdict = _BEHAVIOR_VERDICT_MAP.get(
        ep_result.verdict, "NEEDS_REVIEW"
    )
    reason_vector = _build_reason_vector(ep_result)
    deviation_score = _compute_deviation_score(reason_vector)
    pattern_type = _determine_pattern_type(ep_result, reason_vector)
    metrics_block = _build_metrics_block(ep_result)
    reference = _build_reference_block(result)
    diagnosis = _build_diagnosis(ep_result, pattern_type, reason_vector)

    # Separate deterministic failures, statistical observations, and metrics
    # that could not be evaluated. Keep the legacy ``issues`` list intact.
    issues: List[str] = []
    evidence_by_metric: Dict[str, str] = {}
    risk_signals: List[str] = []
    unverifiable: List[str] = []
    for m_name, m_result in ep_result.metrics.items():
        if m_result.availability == MetricAvailability.NOT_AVAILABLE:
            unverifiable.append(m_name)
            evidence_by_metric[m_name] = "UNVERIFIABLE"
            continue
        if m_result.availability != MetricAvailability.AVAILABLE:
            continue
        evidence = _evidence_level_for_metric(m_name)
        evidence_by_metric[m_name] = evidence
        if not m_result.passed:
            code = _issue_code_for_metric(m_name)
            if code not in issues:
                issues.append(code)

    # Observational metrics intentionally remain PASS-by-rules. Recompute the
    # derived behavior evidence so it is visible in the user-facing schema.
    _, behavior_findings = compute_behavior_severity(list(ep_result.metrics.values()))
    for finding in behavior_findings:
        metric = finding["metric"]
        evidence_by_metric[metric] = "RISK_SIGNAL"
        if metric not in risk_signals:
            risk_signals.append(metric)
        code = _issue_code_for_metric(metric)
        if code not in issues:
            issues.append(code)

    hard_fail_issues = [
        _issue_code_for_metric(name)
        for name, level in evidence_by_metric.items()
        if level == "HARD_FAIL" and _issue_code_for_metric(name) in issues
    ]

    return {
        "episode_id": episode_index,
        "integrity_check": integrity_check,
        "behavior_verdict": behavior_verdict,
        "evidence_level": "HARD_FAIL" if not integrity_check["passed"] else (
            "RISK_SIGNAL" if risk_signals else "UNVERIFIABLE" if unverifiable else "NO_FINDING"
        ),
        "evidence_by_metric": evidence_by_metric,
        "hard_fail_issues": hard_fail_issues,
        "risk_signals": risk_signals,
        "unverifiable_metrics": unverifiable,
        "deviation_score": deviation_score,
        "pattern_type": pattern_type,
        "reason_vector": reason_vector,
        "metrics": metrics_block,
        "reference": reference,
        "issues": issues,
        "diagnosis": diagnosis,
        "user_verdict": None,  # Reserved for V0.2 human feedback loop
    }


# ---------------------------------------------------------------------------
# MVP Spec v0.2.0 — Dataset-level JSON schema (§7.4)
# ---------------------------------------------------------------------------

def _dhi_from_dimensions(dimensions: Dict[str, float]) -> float:
    """Compute Dataset Health Index from four quality dimensions (§3.2).

    DHI = 0.4 × Integrity + 0.2 × Temporal + 0.2 × Motion + 0.2 × Consistency

    Args:
        dimensions: Dict with keys ``integrity``, ``temporal``,
            ``motion``, ``consistency`` — each 0–100.

    Returns:
        DHI score as integer 0–100.
    """
    dhi = (
        0.4 * dimensions.get("integrity", 0.0)
        + 0.2 * dimensions.get("temporal", 0.0)
        + 0.2 * dimensions.get("motion", 0.0)
        + 0.2 * dimensions.get("consistency", 0.0)
    )
    return int(round(dhi))


def _dhi_grade(dhi: int) -> str:
    """Map DHI score to grade label (§3.2).

    Args:
        dhi: Dataset Health Index (0–100).

    Returns:
        Grade string: "Excellent", "Good", "Fair", or "Poor".
    """
    if dhi >= 90:
        return "Excellent"
    elif dhi >= 75:
        return "Good"
    elif dhi >= 60:
        return "Fair"
    else:
        return "Poor"


def _training_readiness(
    exclude_count: int,
    review_count: int,
    total: int,
) -> Tuple[str, str]:
    """Compute Training Readiness level and detail text (§3.3).

    Args:
        exclude_count: Number of EXCLUDE episodes.
        review_count: Number of REVIEW episodes.
        total: Total number of episodes.

    Returns:
        Tuple of ``(level, detail_text)`` where level is one of
        "Ready", "Conditionally Ready", or "Not Ready".
    """
    if total == 0:
        return "Ready", "No episodes to evaluate."

    exclude_pct = exclude_count / total
    review_pct = review_count / total

    if exclude_pct < 0.05 and review_pct < 0.15:
        level = "Ready"
        detail = "Dataset is ready for training."
    elif exclude_pct < 0.10 and review_pct < 0.30:
        level = "Conditionally Ready"
        detail = f"Usable after reviewing {review_count} episodes."
    else:
        level = "Not Ready"
        detail = "Significant data quality issues detected."

    return level, detail


def _compute_quality_dimensions(result: DatasetAuditResult) -> Dict[str, float]:
    """Compute four Dataset Score quality dimensions (§3.1).

    Each dimension is normalized to 0–100.

    Args:
        result: The dataset audit result.

    Returns:
        Dict with keys ``integrity``, ``temporal``, ``motion``,
        ``consistency`` — each a float 0–100.
    """
    from rda.metrics import LAYER1_INTEGRITY

    total = max(result.num_episodes, 1)
    dataset_metrics = aggregate_dataset_metrics(result)
    integrity_layer = dataset_metrics.get("integrity", {})

    # Integrity (Layer 1A): pass rate across all integrity metrics
    total_integrity_available = 0
    total_integrity_passed = 0
    for metric_name in [cls.name for cls in LAYER1_INTEGRITY]:
        stats = integrity_layer.get(metric_name, {})
        available = stats.get("available", 0)
        passed = stats.get("passed", 0)
        total_integrity_available += available
        total_integrity_passed += passed

    if total_integrity_available > 0:
        integrity_score = 100.0 * total_integrity_passed / total_integrity_available
    else:
        integrity_score = 100.0

    # Temporal Quality: based on timestamp validity + jitter
    temporal_stats = dataset_metrics.get("temporal_motion", {})
    ts_validity = integrity_layer.get("timestamp_validity", {})
    ts_passed = ts_validity.get("passed", total)
    ts_available = ts_validity.get("available", total)
    ts_pass_rate = ts_passed / ts_available if ts_available > 0 else 1.0
    temporal_score = 50.0 + 50.0 * ts_pass_rate  # base 50, bonus for valid timestamps

    # Motion Quality: based on effective motion ratio distribution
    utility = dataset_metrics.get("dataset_utility", {})
    idle = utility.get("idle_ratio", {})
    if idle:
        eff_motion = idle.get("effective_motion_ratio", {})
        median_eff = eff_motion.get("median", 0.5)
        motion_score = 100.0 * median_eff
    else:
        motion_score = 75.0

    # Behavioral Consistency: placeholder based on distribution spread
    dist = utility.get("distribution", {})
    if dist:
        dur_stats = dist.get("duration_sec", {})
        median_dur = dur_stats.get("median", 0.0)
        p95_dur = dur_stats.get("p95", 0.0)
        if p95_dur > 0 and median_dur > 0:
            # Lower ratio = more consistent
            ratio = p95_dur / median_dur
            consistency_score = max(0.0, 100.0 * (1.0 - min(ratio - 1.0, 1.0)))
        else:
            consistency_score = 70.0
    else:
        consistency_score = 70.0

    return {
        "integrity": round(integrity_score, 1),
        "temporal": round(temporal_score, 1),
        "motion": round(motion_score, 1),
        "consistency": round(consistency_score, 1),
    }


def _build_evidence_summary(result: DatasetAuditResult) -> Dict[str, Any]:
    """Summarize evidence classes without turning risk signals into failures."""
    from rda.metrics import LAYER1_INTEGRITY

    integrity_names = {cls.name for cls in LAYER1_INTEGRITY}
    hard_fail_episodes = 0
    risk_signal_episodes = 0
    unverifiable_counts: Dict[str, int] = {}

    for ep_result in result.episodes.values():
        has_hard_fail = False
        has_risk_signal = False
        for name, metric in ep_result.metrics.items():
            if metric.availability == MetricAvailability.NOT_AVAILABLE:
                unverifiable_counts[name] = unverifiable_counts.get(name, 0) + 1
            elif name in integrity_names and not metric.passed:
                has_hard_fail = True

        _, findings = compute_behavior_severity(list(ep_result.metrics.values()))
        if findings:
            has_risk_signal = True
        if has_hard_fail:
            hard_fail_episodes += 1
        if has_risk_signal:
            risk_signal_episodes += 1

    return {
        "hard_fail_episodes": hard_fail_episodes,
        "risk_signal_episodes": risk_signal_episodes,
        "unverifiable_metric_episodes": unverifiable_counts,
        "interpretation": (
            "Hard Fail is deterministic evidence; Risk Signal is an observational "
            "pattern requiring confirmation; Unverifiable means the input was insufficient."
        ),
    }


def _build_pattern_distribution(result: DatasetAuditResult) -> Dict[str, int]:
    """Count episodes per pattern type across the dataset.

    Args:
        result: The dataset audit result.

    Returns:
        Dict mapping pattern type name to episode count.
    """
    counts: Dict[str, int] = {
        "stuck_like": 0,
        "jittery_like": 0,
        "inefficient": 0,
        "frozen_like": 0,
        "unusual": 0,
    }

    for ep_result in result.episodes.values():
        reason_vector = _build_reason_vector(ep_result)
        pattern = _determine_pattern_type(ep_result, reason_vector)
        primary = (pattern.get("primary") or "").lower()
        if primary in counts:
            counts[primary] += 1

    return counts


def generate_dataset_report(result: DatasetAuditResult) -> Dict[str, Any]:
    """Generate the MVP Spec v0.2.0 dataset-level JSON report (§7.4).

    Produces a user-facing dataset summary with fields:
    ``dataset_id``, ``profile``, ``integrity``, ``quality``,
    ``behavior_summary``, ``pattern_distribution``, and
    ``estimated_post_cleanup_quality``.

    Args:
        result: The dataset audit result.

    Returns:
        Dict matching the MVP Spec v0.2.0 Dataset JSON Schema.
    """
    compact = build_summary(result)
    dimensions = _compute_quality_dimensions(result)
    dhi = _dhi_from_dimensions(dimensions)
    grade = _dhi_grade(dhi)

    exclude_count = result.verdict_counts.get(AuditVerdict.EXCLUDE, 0)
    review_count = result.verdict_counts.get(AuditVerdict.REVIEW, 0)
    pass_count = result.verdict_counts.get(AuditVerdict.PASS, 0)
    total = max(result.num_episodes, 1)

    readiness_level, readiness_detail = _training_readiness(
        exclude_count, review_count, total
    )

    # Integrity stats
    dataset_metrics = aggregate_dataset_metrics(result)
    integrity_layer = dataset_metrics.get("integrity", {})
    integrity_passed = sum(
        s.get("passed", 0) for s in integrity_layer.values() if isinstance(s, dict)
    )
    # Normalize: each metric has per-episode counts, so divide by metric count
    n_integrity_metrics = max(len(integrity_layer), 1)
    integrity_passed_eps = integrity_passed // n_integrity_metrics
    integrity_failed_eps = total - integrity_passed_eps

    # Top integrity issues
    top_issues: Dict[str, int] = {}
    issue_codes = {
        "missing_dropout": "INT-01",
        "invalid_values": "INT-02",
        "schema_consistency": "INT-03",
        "timestamp_validity": "INT-04",
        "joint_limit": "INT-05",
    }
    for metric_name, code in issue_codes.items():
        stats = integrity_layer.get(metric_name, {})
        failed = stats.get("failed", 0)
        if failed > 0:
            top_issues[code] = failed

    # Pattern distribution
    pattern_dist = _build_pattern_distribution(result)
    evidence_summary = _build_evidence_summary(result)

    # Estimated post-cleanup quality (simplified)
    # Assume cleaning removes EXCLUDE episodes and 50% of REVIEW issues
    remaining = pass_count + review_count
    post_cleanup_dhi = dhi
    if total > 0 and exclude_count > 0:
        improvement = min(20, exclude_count / total * 30)
        post_cleanup_dhi = int(min(100, dhi + improvement))

    from rda import __version__ as _rda_version

    return {
        "dataset_id": result.dataset_info.path,
        "tool_version": _rda_version,
        "profile": {
            "total_episodes": result.dataset_info.num_episodes,
            "robot": result.dataset_info.meta.get("robot", "unknown"),
            "dof": result.dataset_info.meta.get("dof", 0),
            "fps": result.dataset_info.meta.get("fps", 0),
            "tasks": result.dataset_info.meta.get("num_tasks", 0),
        },
        "integrity": {
            "passed": integrity_passed_eps,
            "failed": integrity_failed_eps,
            "top_issues": top_issues,
        },
        "quality": {
            "dhi": dhi,
            "dhi_note": (
                "Relative assessment based on structural integrity "
                "and behavioral consistency"
            ),
            "grade": grade,
            "training_readiness": readiness_level,
            "training_readiness_detail": readiness_detail,
            "dimensions": dimensions,
        },
        "behavior_summary": {
            "pass": pass_count,
            "review": review_count,
            "exclude": exclude_count,
        },
        "evidence_summary": evidence_summary,
        "pattern_distribution": pattern_dist,
        "estimated_post_cleanup_quality": post_cleanup_dhi,
    }


# ---------------------------------------------------------------------------
# Blind / anonymized report — for external sharing
# ---------------------------------------------------------------------------

def anonymize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Strip all identifying information from a JSON report for blind sharing.

    Replaces filesystem paths with a stable SHA-256 hash (first 16 chars),
    redacts the robot name, and marks the report as ``blind_audit``.

    The hash allows the receiver to deduplicate reports from the same
    dataset without knowing the original path.

    What gets redacted:
        - ``dataset.path`` → ``"redacted:<hash>"``
        - ``dataset_id`` → ``"redacted:<hash>"``
        - ``profile.robot`` → ``"redacted"``

    What is **not** redacted (share-safe by design):
        - All metric values (idle ratio, spike count, etc.) — just numbers
        - Verdicts (PASS / REVIEW / EXCLUDE)
        - Episode counts, frame counts
        - Tool version

    No raw trajectory data, images, or action arrays are present in the
    report to begin with — RDA's audit only stores aggregated metrics.

    Args:
        report: A report dict from :func:`generate_json_report` or
            :func:`generate_dataset_report`.

    Returns:
        A new dict with identifying info removed.
    """
    import copy
    import hashlib

    anon = copy.deepcopy(report)

    # --- Determine original path for hashing ---
    original_path = ""
    if isinstance(anon.get("dataset"), dict):
        original_path = anon["dataset"].get("path", "")
    elif anon.get("dataset_id"):
        original_path = str(anon["dataset_id"])

    path_hash = (
        hashlib.sha256(str(original_path).encode()).hexdigest()[:16]
        if original_path
        else "unknown"
    )

    # --- Redact engine-format fields ---
    if isinstance(anon.get("dataset"), dict):
        anon["dataset"]["path"] = f"redacted:{path_hash}"

    # --- Redact MVP-spec-format fields ---
    if "dataset_id" in anon:
        anon["dataset_id"] = f"redacted:{path_hash}"

    if isinstance(anon.get("profile"), dict):
        anon["profile"]["robot"] = "redacted"

    # --- Mark as blind ---
    anon["blind_audit"] = True
    anon["report_hash"] = path_hash

    return anon
