"""Four-tier verdict classification rules (v0.9).

v0.9 changes (Phase 1):
  - CRITICAL_METRICS: add ``video_frame_integrity`` (was computed but not in the list).
  - REVIEW_METRICS renamed to DIAGNOSTIC_METRICS: these produce diagnostic
    findings but do NOT affect the episode verdict (PASS/EXCLUDE only).
  - Removed ``distribution`` and ``coverage`` from DIAGNOSTIC_METRICS — they
    are Dataset Profile metrics (dataset-level, not per-episode).
  - Added DATASET_PROFILE_METRICS constant.
  - classify_episode(): simplified — only CRITICAL metrics affect verdict.
    REVIEW verdict is now triggered only by conditional critical findings
    (e.g., joint_limit with uncertain limit_source).
  - compute_behavior_severity(): fixed distribution/coverage field mapping bug.
  - upgrade_verdict_by_behavior(): now opt-in (enabled=False by default).

Each episode is classified into one of three tiers based on metric results:

- **PASS**: All critical metrics pass; the episode is ready for training.
- **REVIEW**: Conditional critical findings detected (e.g., joint_limit with
  uncertain limits); human review is recommended.
- **EXCLUDE**: One or more critical metrics failed hard; the episode should be
  excluded from training.

Metric names follow the V0.9 Technical Specification.
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


# ---------------------------------------------------------------------------
# Integrity Gate — CRITICAL metrics
# ---------------------------------------------------------------------------
# These correspond to deterministic, unambiguous data corruption.
# A failure in any of these triggers EXCLUDE (or REVIEW for conditional
# cases like joint_limit with uncertain limit_source).
#
# v0.9 changes:
#   - Added: video_frame_integrity (was computed in codebase but missing
#     from this list — preflight.py noted this gap).
#   - Renamed: video_stream_sync → video_stream_presence (presence-only check)

CRITICAL_METRICS: List[str] = [
    "missing_dropout",          # Critical frame loss
    "invalid_values",           # NaN/Inf (data structurally broken)
    "schema_consistency",       # Shape/dtype mismatch across features
    "timestamp_validity",       # Non-monotonic or negative time deltas
    "joint_limit",              # Joints outside mechanical limits
    "video_freeze",             # Camera drop-out (identical frames while arm moves)
    "video_timestamp_alignment",# Video/parquet timeline span divergence
    "video_stream_presence",    # Multi-camera presence check (v0.9: split from video_stream_sync)
    "video_frame_integrity",    # MP4 frame count vs parquet mismatch (NEW in v0.9)
]


# ---------------------------------------------------------------------------
# Trajectory Diagnostics — DIAGNOSTIC metrics
# ---------------------------------------------------------------------------
# These are statistical anomalies — unusual but not definitively broken.
# In v0.9 they produce diagnostic findings (measurement + finding text)
# but do NOT affect the episode verdict by default.
#
# v0.9 changes:
#   - Renamed from REVIEW_METRICS to DIAGNOSTIC_METRICS
#   - Removed distribution, coverage (moved to Dataset Profile — dataset-level)

DIAGNOSTIC_METRICS: List[str] = [
    "sensor_synchronization",   # Multi-sensor temporal offset
    "sampling_jitter",          # Irregular sampling intervals
    "velocity_acceleration",    # Kinematic anomalies (MAD z-score spikes)
    "action_discontinuity",     # Action trajectory discontinuities
    "idle_ratio",               # Low motion density
    "visual_quality",           # Blur/exposure/contrast issues
    "video_stream_span_consistency",    # Cross-camera span consistency (v0.9: split from video_stream_sync)
    "video_stream_temporal_offset",     # Frame-level pairwise temporal offset (v0.9: split from video_stream_sync)
    "video_stream_temporal_drift",      # Clock drift rate between cameras (v0.9: split from video_stream_sync)
]

# Backward compatibility alias — deprecated, will be removed in v1.0
REVIEW_METRICS: List[str] = DIAGNOSTIC_METRICS


# ---------------------------------------------------------------------------
# Dataset Profile metrics (dataset-level, not per-episode verdict)
# ---------------------------------------------------------------------------
# These are computed at the dataset level and do NOT participate in
# per-episode verdict classification.

DATASET_PROFILE_METRICS: List[str] = [
    "distribution",             # Action/trajectory statistical distribution
    "coverage",                 # State-space occupancy analysis
    "temporal_sufficiency",     # Temporal sufficiency / idle structure
]


def classify_episode(
    metric_results: Sequence[MetricResult],
    critical_metrics: Sequence[str] | None = None,
    diagnostic_metrics: Sequence[str] | None = None,
) -> AuditVerdict:
    """Classify an episode based on its metric results.

    v0.9: Diagnostic metrics no longer trigger REVIEW verdict.
    Only CRITICAL metrics can trigger EXCLUDE. REVIEW is now only
    triggered by conditional findings within critical metrics
    (e.g., joint_limit with uncertain limit_source).

    Args:
        metric_results: List of MetricResult from all computed metrics.
        critical_metrics: Metric names whose failure means EXCLUDE (or
            REVIEW if the metric's assessment status is "review").
            Defaults to :data:`CRITICAL_METRICS`.
        diagnostic_metrics: Kept for API backward compatibility. In v0.9,
            these metrics produce findings but do NOT affect the verdict.

    Returns:
        AuditVerdict: PASS, REVIEW, or EXCLUDE.
    """
    critical = set(critical_metrics) if critical_metrics is not None else set(CRITICAL_METRICS)

    # Collect critical metrics that have findings
    failed_critical = set()
    review_only_metrics = set()
    for r in metric_results:
        if r.availability != MetricAvailability.AVAILABLE:
            continue  # N/A / error metrics don't affect verdict
        if not r.has_finding:
            continue  # metric found no issue
        if r.name not in critical:
            continue  # diagnostic/dataset metrics don't affect verdict
        failed_critical.add(r.name)
        # Some critical metrics can produce REVIEW instead of EXCLUDE
        # based on their assessment status (e.g., joint_limit with
        # uncertain limit_source)
        if r.assessment.get("status") == "review":
            review_only_metrics.add(r.name)

    if not failed_critical:
        return AuditVerdict.PASS

    # If ALL failed critical metrics are REVIEW-level (not hard EXCLUDE), return REVIEW
    if failed_critical == review_only_metrics:
        return AuditVerdict.REVIEW

    # Otherwise, at least one metric has a hard EXCLUDE finding
    return AuditVerdict.EXCLUDE


def compute_behavior_severity(
    metric_results: Sequence[MetricResult],
) -> tuple[float, list[dict]]:
    """Compute behavioral severity score (0-100) from metric measurements.

    Even though behavioral/diagnostic metrics are observational (always pass),
    their raw measurements contain meaningful signals about data quality.
    This function aggregates those signals into a severity score AND
    generates findings for explainability.

    v0.9 fix: The ``distribution`` branch previously read ``occupancy_rate``
    from the ``distribution`` metric, but that field actually lives on the
    ``coverage`` metric. Now correctly reads from ``coverage``.

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

        elif m.name == "sampling_jitter":
            jr = meas.get("jitter_ratio", 0.0)
            if jr > 0.3:
                severity += 30
                findings.append({
                    "metric": "sampling_jitter",
                    "severity": 30,
                    "reason": f"Jitter ratio {jr:.2f} > 0.3: severe sampling irregularity",
                    "measurement_value": jr,
                })
            elif jr > 0.1:
                severity += 10
                findings.append({
                    "metric": "sampling_jitter",
                    "severity": 10,
                    "reason": f"Jitter ratio {jr:.2f} > 0.1: moderate sampling irregularity",
                    "measurement_value": jr,
                })

        # v0.9 fix: was reading occupancy_rate from 'distribution' metric,
        # but occupancy_rate is a field of the 'coverage' metric.
        elif m.name == "coverage":
            occ = meas.get("occupancy_rate", 1.0)
            if occ < 0.05:
                severity += 30  # Extremely low coverage
                findings.append({
                    "metric": "coverage",
                    "severity": 30,
                    "reason": f"Occupancy rate {occ:.3f} < 0.05: extremely low state-space coverage",
                    "measurement_value": occ,
                })
            elif occ < 0.1:
                severity += 20  # Very low coverage
                findings.append({
                    "metric": "coverage",
                    "severity": 20,
                    "reason": f"Occupancy rate {occ:.3f} < 0.1: very low state-space coverage",
                    "measurement_value": occ,
                })
            elif occ < 0.2:
                severity += 10  # Low coverage
                findings.append({
                    "metric": "coverage",
                    "severity": 10,
                    "reason": f"Occupancy rate {occ:.3f} < 0.2: low state-space coverage",
                    "measurement_value": occ,
                })

    return min(severity, 100.0), findings


def upgrade_verdict_by_behavior(
    verdict: AuditVerdict,
    metric_results: Sequence[MetricResult],
    enabled: bool = False,
) -> AuditVerdict:
    """Upgrade verdict if behavioral anomalies are detected.

    v0.9: This is now opt-in (enabled=False by default). In v0.8 it was
    always active, which caused confusion — diagnostic metrics were
    silently upgrading PASS to REVIEW. Now it's explicit.

    If the rule-based verdict is PASS but behavior severity >= 20,
    upgrade to REVIEW. Higher severity stays REVIEW (not EXCLUDE)
    because behavioral anomalies are soft signals, not hard corruption.

    Args:
        verdict: The current rule-based verdict.
        metric_results: All metric results for the episode.
        enabled: Whether to enable behavior-based verdict upgrade.
            Default False (v0.9 behavior).

    Returns:
        AuditVerdict: Possibly upgraded verdict.
    """
    if not enabled:
        return verdict

    if verdict != AuditVerdict.PASS:
        return verdict  # Already REVIEW or EXCLUDE, don't downgrade

    severity, _ = compute_behavior_severity(metric_results)
    if severity >= 20:
        return AuditVerdict.REVIEW

    return verdict
