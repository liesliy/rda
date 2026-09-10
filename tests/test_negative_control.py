"""Negative-control tests: the audit gate must consume its own evidence.

Historical bug (fixed in v0.4.9–0.4.11 era): the behavior layer correctly
computed action-discontinuity spikes and idle ratios, but the verdict
aggregator ignored those signals entirely — anomalous episodes walked away
with a PASS badge. These tests exist so that exact failure mode can never
silently return.

v0.9 update: DIAGNOSTIC_METRICS (formerly REVIEW_METRICS) are now diagnostic
findings by default and do NOT affect the verdict. The behavior-severity
upgrade is opt-in via upgrade_verdict_by_behavior(enabled=True).
Tests are updated to reflect this new design.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.audit.rules import (
    AuditVerdict,
    classify_episode,
    compute_behavior_severity,
    upgrade_verdict_by_behavior,
)
from rda.metrics.base import MetricAvailability, MetricResult


def _observation(name: str, measurement: dict) -> MetricResult:
    """Build a Layer-2 observational result: computed, PASS-by-rules, no finding.

    This mirrors how behavioral metrics (idle_ratio, action_discontinuity,
    etc.) report in production: they measure, they don't fail.
    The historical bug hid here — "computed but never consumed".
    """
    return MetricResult(
        name=name,
        availability=MetricAvailability.AVAILABLE,
        measurement=measurement,
        assessment={"status": "pass", "severity": None, "reason": None},
        details={},
        message="",
        has_finding=False,
    )


def _full_pipeline(metric_results: list[MetricResult], behavior_upgrade: bool = False) -> AuditVerdict:
    """Replica of the production verdict path in EpisodeAuditor.audit().

    v0.9: behavior upgrade is opt-in. Pass behavior_upgrade=True to enable it.
    """
    verdict = classify_episode(metric_results)
    return upgrade_verdict_by_behavior(verdict, metric_results, enabled=behavior_upgrade)


# --- v0.9: Diagnostic metrics are findings-only by default ----------------


def test_diagnostic_metrics_do_not_change_verdict_by_default():
    """v0.9: DIAGNOSTIC_METRICS (action_discontinuity, idle_ratio) are
    diagnostic-only by default. Even severe anomalies stay PASS unless
    the behavior upgrade is explicitly enabled.
    """
    results = [
        _observation("action_discontinuity", {"spike_count": 150}),
        _observation("idle_ratio", {"effective_motion_ratio": 0.9}),
    ]
    # Default: diagnostic metrics don't change the verdict
    assert _full_pipeline(results) == AuditVerdict.PASS


def test_behavior_upgrade_catches_spikes():
    """When behavior upgrade is enabled, spikes must escalate verdict."""
    results = [
        _observation("action_discontinuity", {"spike_count": 150}),
        _observation("idle_ratio", {"effective_motion_ratio": 0.9}),
    ]
    # With behavior upgrade enabled, anomalies are caught
    assert _full_pipeline(results, behavior_upgrade=True) != AuditVerdict.PASS
    assert _full_pipeline(results, behavior_upgrade=True) == AuditVerdict.REVIEW


def test_frozen_episode_with_upgrade():
    """Effective motion ratio 0.05 (arm stationary 95% of frames).
    v0.9: only caught when behavior upgrade is enabled.
    """
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.05})]
    # Default: stays PASS (diagnostic only)
    assert _full_pipeline(results) == AuditVerdict.PASS
    # With upgrade: escalates to REVIEW
    assert _full_pipeline(results, behavior_upgrade=True) == AuditVerdict.REVIEW


def test_combined_anomalies_escalate_severity():
    """v0.9: distribution moved to Dataset Profile, replaced by sampling_jitter."""
    spikes = _observation("action_discontinuity", {"spike_count": 200})  # +30
    frozen = _observation("idle_ratio", {"effective_motion_ratio": 0.05})  # +40
    jitter = _observation("sampling_jitter", {"jitter_ratio": 0.5})  # +30
    severity, findings = compute_behavior_severity([spikes, frozen, jitter])
    assert severity >= 20
    assert len(findings) == 3
    assert all("reason" in f and f["reason"] for f in findings)


# --- The gate must not over-fire on clean data ----------------------------


def test_clean_episode_stays_pass():
    """Negative control's negative control: good data keeps its PASS."""
    results = [
        _observation("action_discontinuity", {"spike_count": 2}),
        _observation("idle_ratio", {"effective_motion_ratio": 0.75}),
        _observation("sampling_jitter", {"jitter_ratio": 0.02}),
    ]
    assert _full_pipeline(results) == AuditVerdict.PASS
    # Even with behavior upgrade, clean data stays PASS
    assert _full_pipeline(results, behavior_upgrade=True) == AuditVerdict.PASS


def test_severity_threshold_boundary():
    """Boundary: exactly 20 (10 spikes>20 + 10 eff<0.5) upgrades to REVIEW
    when behavior upgrade is enabled."""
    results = [
        _observation("action_discontinuity", {"spike_count": 25}),  # +10
        _observation("idle_ratio", {"effective_motion_ratio": 0.45}),  # +10
    ]
    severity, _ = compute_behavior_severity(results)
    assert severity == 20
    # Default: stays PASS (upgrade disabled)
    assert _full_pipeline(results) == AuditVerdict.PASS
    # With upgrade: escalates to REVIEW
    assert _full_pipeline(results, behavior_upgrade=True) == AuditVerdict.REVIEW


# --- The gate must respect hard corruption --------------------------------


def test_critical_failure_stays_exclude():
    """Behavior upgrade must never soften a hard EXCLUDE (no downgrade)."""
    clean_behavior = [
        _observation("action_discontinuity", {"spike_count": 0}),
        _observation("idle_ratio", {"effective_motion_ratio": 0.9}),
    ]
    nan_finding = MetricResult(
        name="invalid_values",
        availability=MetricAvailability.AVAILABLE,
        measurement={"nan_count": 12, "inf_count": 0},
        assessment={"status": "exclude", "severity": "high",
                    "reason": "NaN values in observation.state"},
        details={},
        message="EXCLUDE: NaN values detected.",
        has_finding=True,
    )
    verdict = _full_pipeline(clean_behavior + [nan_finding], behavior_upgrade=True)
    assert verdict == AuditVerdict.EXCLUDE


def test_unavailable_metrics_do_not_block_pass():
    """N/A metrics (no timestamps, no state) are ignored, not failures."""
    na = MetricResult(
        name="sensor_synchronization",
        availability=MetricAvailability.NOT_AVAILABLE,
        measurement={},
        assessment={"status": "na", "severity": None, "reason": None},
        details={},
        message="",
        has_finding=False,
    )
    assert _full_pipeline([na]) == AuditVerdict.PASS


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
