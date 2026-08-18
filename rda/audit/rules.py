"""Three-tier verdict classification rules.

Each episode is classified into one of three tiers based on metric results:

- **PASS**: All critical metrics pass; the episode is ready for training.
- **REVIEW**: Some non-critical metrics failed; human review is recommended.
- **EXCLUDE**: One or more critical metrics failed; the episode should be excluded.

Metric names follow the V0 Technical Specification.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Sequence

from rda.metrics.base import MetricResult, MetricAvailability


class AuditVerdict(str, Enum):
    """Three-tier audit classification."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    EXCLUDE = "EXCLUDE"

    def __str__(self) -> str:  # noqa: D401
        return self.value


# Metric names that trigger EXCLUDE on failure.
# These correspond to deterministic, unambiguous data corruption.
# - missing_dropout:  critical frame loss
# - invalid_values:   NaN/Inf (data is structurally broken)
# - schema_consistency: shape/dtype mismatch across features
# - timestamp_validity: non-monotonic or negative deltas (time broken)
# - joint_limit:      joints outside mechanical limits (physical impossibility)
CRITICAL_METRICS: List[str] = [
    "missing_dropout",
    "invalid_values",
    "schema_consistency",
    "timestamp_validity",
    "joint_limit",
]

# Metric names that trigger REVIEW on failure (but not EXCLUDE).
# These are statistical anomalies — unusual but not definitively broken.
REVIEW_METRICS: List[str] = [
    "sensor_synchronization",
    "sampling_jitter",
    "velocity_acceleration",
    "action_discontinuity",
    "idle_ratio",
    "distribution",
    "coverage",
]


def classify_episode(
    metric_results: Sequence[MetricResult],
    critical_metrics: Sequence[str] | None = None,
    review_metrics: Sequence[str] | None = None,
) -> AuditVerdict:
    """Classify an episode based on its metric results.

    Args:
        metric_results: List of MetricResult from all computed metrics.
        critical_metrics: Metric names whose failure means EXCLUDE.
            Defaults to :data:`CRITICAL_METRICS`.
        review_metrics: Metric names whose failure means REVIEW.
            Defaults to :data:`REVIEW_METRICS`.

    Returns:
        AuditVerdict: PASS, REVIEW, or EXCLUDE.
    """
    critical = set(critical_metrics) if critical_metrics is not None else set(CRITICAL_METRICS)
    review = set(review_metrics) if review_metrics is not None else set(REVIEW_METRICS)

    # Skip metrics that are N/A or have no finding — they don't affect verdict
    failed = set()
    for r in metric_results:
        if r.availability != MetricAvailability.AVAILABLE:
            continue  # N/A / error metrics don't affect verdict
        if not r.has_finding:
            continue  # metric found no issue
        failed.add(r.name)

    if failed & critical:
        return AuditVerdict.EXCLUDE

    if failed & review:
        return AuditVerdict.REVIEW

    return AuditVerdict.PASS


def compute_behavior_severity(metric_results: Sequence[MetricResult]) -> tuple[float, list[dict]]:
    """Compute behavioral severity score (0-100) from metric measurements.

    Even though behavioral metrics are observational (always pass), their
    raw measurements contain meaningful signals about data quality.
    This function aggregates those signals into a severity score AND
    generates findings for explainability.

    Returns:
        Tuple of (severity_score, findings_list) where findings_list contains
        dicts with keys: metric, severity, reason, measurement_value

    Severity factors (each 0-40 points, capped at 100 total):
    - Low effective_motion_ratio (stuck/frozen indicator)
    - High action discontinuity spike count (jitter indicator)
    - Low state-space coverage / abnormal distribution
    """
    severity = 0.0
    findings = []

    for m in metric_results:
        if m.availability.value != "available":
            continue
        meas = m.measurement

        if m.name == "idle_ratio":
            eff = meas.get("effective_motion_ratio", 1.0)
            if eff < 0.1:
                severity += 40  # Frozen / severely stuck
                findings.append({
                    "metric": "idle_ratio",
                    "severity": 40,
                    "reason": f"Effective motion ratio {eff:.3f} < 0.1: frozen/severely stuck behavior detected",
                    "measurement_value": eff,
                })
            elif eff < 0.2:
                severity += 30  # Very low motion
                findings.append({
                    "metric": "idle_ratio",
                    "severity": 30,
                    "reason": f"Effective motion ratio {eff:.3f} < 0.2: very low motion, possible stuck pattern",
                    "measurement_value": eff,
                })
            elif eff < 0.3:
                severity += 20  # Low motion (Stuck-like)
                findings.append({
                    "metric": "idle_ratio",
                    "severity": 20,
                    "reason": f"Effective motion ratio {eff:.3f} < 0.3: low motion, stuck-like behavior",
                    "measurement_value": eff,
                })
            elif eff < 0.5:
                severity += 10  # Somewhat low
                findings.append({
                    "metric": "idle_ratio",
                    "severity": 10,
                    "reason": f"Effective motion ratio {eff:.3f} < 0.5: somewhat low motion",
                    "measurement_value": eff,
                })

        elif m.name == "action_discontinuity":
            spikes = meas.get("spike_count", 0)
            if spikes > 100:
                severity += 30  # Extreme jitter
                findings.append({
                    "metric": "action_discontinuity",
                    "severity": 30,
                    "reason": f"Spike count {spikes} > 100: extreme action discontinuity/jitter detected",
                    "measurement_value": spikes,
                })
            elif spikes > 50:
                severity += 20  # High jitter
                findings.append({
                    "metric": "action_discontinuity",
                    "severity": 20,
                    "reason": f"Spike count {spikes} > 50: high action discontinuity/jitter",
                    "measurement_value": spikes,
                })
            elif spikes > 20:
                severity += 10  # Moderate jitter
                findings.append({
                    "metric": "action_discontinuity",
                    "severity": 10,
                    "reason": f"Spike count {spikes} > 20: moderate action discontinuity",
                    "measurement_value": spikes,
                })

        elif m.name == "distribution":
            occ = meas.get("occupancy_rate", 1.0)
            if occ < 0.05:
                severity += 30  # Extremely low coverage
                findings.append({
                    "metric": "distribution",
                    "severity": 30,
                    "reason": f"Occupancy rate {occ:.3f} < 0.05: extremely low state-space coverage",
                    "measurement_value": occ,
                })
            elif occ < 0.1:
                severity += 20  # Very low coverage
                findings.append({
                    "metric": "distribution",
                    "severity": 20,
                    "reason": f"Occupancy rate {occ:.3f} < 0.1: very low state-space coverage",
                    "measurement_value": occ,
                })
            elif occ < 0.2:
                severity += 10  # Low coverage
                findings.append({
                    "metric": "distribution",
                    "severity": 10,
                    "reason": f"Occupancy rate {occ:.3f} < 0.2: low state-space coverage",
                    "measurement_value": occ,
                })

    return min(severity, 100.0), findings


def upgrade_verdict_by_behavior(
    verdict: AuditVerdict,
    metric_results: Sequence[MetricResult],
) -> AuditVerdict:
    """Upgrade verdict if behavioral anomalies are detected.

    If the rule-based verdict is PASS but behavior severity >= 20,
    upgrade to REVIEW. Higher severity stays REVIEW (not EXCLUDE)
    because behavioral anomalies are soft signals, not hard corruption.
    """
    if verdict != AuditVerdict.PASS:
        return verdict  # Already REVIEW or EXCLUDE, don't downgrade

    severity, _ = compute_behavior_severity(metric_results)
    if severity >= 20:
        return AuditVerdict.REVIEW

    return verdict
