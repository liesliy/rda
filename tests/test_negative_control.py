"""Negative-control tests: the audit gate must consume its own evidence.

Historical bug (fixed in v0.4.9–0.4.11 era): the behavior layer correctly
computed action-discontinuity spikes and idle ratios, but the verdict
aggregator ignored those signals entirely — anomalous episodes walked away
with a PASS badge. These tests exist so that exact failure mode can never
silently return.

They test the *wiring*, not the metrics: given metric results that a
rule-based pass would classify as PASS but whose raw measurements contain
known anomalies, the final verdict must NOT be PASS.
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
    distribution) report in production: they measure, they don't fail.
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


def _full_pipeline(metric_results: list[MetricResult]) -> AuditVerdict:
    """Replica of the production verdict path in EpisodeAuditor.audit()."""
    verdict = classify_episode(metric_results)
    return upgrade_verdict_by_behavior(verdict, metric_results)


# --- The original failure mode: spikes + frozen arm must not PASS ---------


def test_spikes_alone_never_pass():
    """ALOHA-style run: ~30 spikes/episode across 50 eps ⇒ ~1500 total.

    Rules say PASS (observational metric). The gate must still say REVIEW.
    This is the exact scenario the historical bug green-badged.
    """
    results = [
        _observation("action_discontinuity", {"spike_count": 150}),
        _observation("idle_ratio", {"effective_motion_ratio": 0.9}),
    ]
    assert _full_pipeline(results) != AuditVerdict.PASS
    assert _full_pipeline(results) == AuditVerdict.REVIEW


def test_frozen_episode_never_pass():
    """Effective motion ratio 0.05 (arm stationary 95% of frames)."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.05})]
    assert _full_pipeline(results) == AuditVerdict.REVIEW


def test_combined_anomalies_escalate_severity():
    spikes = _observation("action_discontinuity", {"spike_count": 200})  # +30
    frozen = _observation("idle_ratio", {"effective_motion_ratio": 0.05})  # +40
    low_cov = _observation("distribution", {"occupancy_rate": 0.03})  # +30
    severity, findings = compute_behavior_severity([spikes, frozen, low_cov])
    assert severity >= 20
    assert len(findings) == 3
    assert all("reason" in f and f["reason"] for f in findings)


# --- The gate must not over-fire on clean data ----------------------------


def test_clean_episode_stays_pass():
    """Negative control's negative control: good data keeps its PASS."""
    results = [
        _observation("action_discontinuity", {"spike_count": 2}),
        _observation("idle_ratio", {"effective_motion_ratio": 0.75}),
        _observation("distribution", {"occupancy_rate": 0.42}),
    ]
    assert _full_pipeline(results) == AuditVerdict.PASS


def test_severity_threshold_boundary():
    """Boundary: exactly 20 (10 spikes>20 + 10 eff<0.5) upgrades to REVIEW."""
    results = [
        _observation("action_discontinuity", {"spike_count": 25}),  # +10
        _observation("idle_ratio", {"effective_motion_ratio": 0.45}),  # +10
    ]
    severity, _ = compute_behavior_severity(results)
    assert severity == 20
    assert _full_pipeline(results) == AuditVerdict.REVIEW


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
    verdict = _full_pipeline(clean_behavior + [nan_finding])
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
