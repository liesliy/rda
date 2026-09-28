"""Tests for v0.9.17 changes: frozen episode detection and calibrated thresholds.

v0.9.17 introduces:
  1. check_frozen_episode() — restores frozen episode detection (regression fix)
  2. Data-driven severity brackets for action_discontinuity (ArmnetBench calibration)
  3. FROZEN_EPISODE_EMR_THRESHOLD = 0.02
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.audit.rules import (  # noqa: E402
    AuditVerdict,
    FROZEN_EPISODE_EMR_THRESHOLD,
    check_frozen_episode,
    classify_episode,
    compute_behavior_severity,
    upgrade_verdict_by_behavior,
)
from rda.metrics.base import MetricAvailability, MetricResult  # noqa: E402


def _observation(name: str, measurement: dict) -> MetricResult:
    """Build a Layer-2 observational result (PASS-by-rules)."""
    return MetricResult(
        name=name,
        availability=MetricAvailability.AVAILABLE,
        measurement=measurement,
        assessment={"status": "pass", "severity": None, "reason": None},
        details={},
        message="",
        has_finding=False,
    )


def _na_result(name: str, reason: str = "not_available") -> MetricResult:
    """Build a N/A result."""
    return MetricResult(
        name=name,
        availability=MetricAvailability.NOT_AVAILABLE,
        measurement={},
        assessment={"status": "na", "severity": None, "reason": reason},
        details={},
        message="",
        has_finding=False,
    )


# --- FROZEN_EPISODE_EMR_THRESHOLD constant ---------------------------------


def test_frozen_threshold_value():
    """Threshold must be 0.02 (calibrated on ArmnetBench)."""
    assert FROZEN_EPISODE_EMR_THRESHOLD == 0.02


# --- check_frozen_episode() direct tests -----------------------------------


def test_frozen_episode_detected_at_zero_motion():
    """EMR = 0.0 (frozen) must upgrade PASS → REVIEW."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.0})]
    verdict = check_frozen_episode(AuditVerdict.PASS, results)
    assert verdict == AuditVerdict.REVIEW


def test_frozen_episode_detected_below_threshold():
    """EMR = 0.01 (below threshold) must upgrade PASS → REVIEW."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.01})]
    verdict = check_frozen_episode(AuditVerdict.PASS, results)
    assert verdict == AuditVerdict.REVIEW


def test_frozen_episode_not_triggered_at_threshold():
    """EMR = 0.02 (exactly at threshold) must NOT trigger frozen detection."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.02})]
    verdict = check_frozen_episode(AuditVerdict.PASS, results)
    assert verdict == AuditVerdict.PASS


def test_frozen_episode_not_triggered_above_threshold():
    """EMR = 0.05 (above threshold) must NOT trigger frozen detection."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.05})]
    verdict = check_frozen_episode(AuditVerdict.PASS, results)
    assert verdict == AuditVerdict.PASS


def test_frozen_episode_normal_data_not_triggered():
    """Normal episode (EMR = 0.75) must NOT trigger frozen detection."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.75})]
    verdict = check_frozen_episode(AuditVerdict.PASS, results)
    assert verdict == AuditVerdict.PASS


def test_frozen_episode_does_not_downgrade_review():
    """Frozen check must NOT downgrade an existing REVIEW."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.0})]
    verdict = check_frozen_episode(AuditVerdict.REVIEW, results)
    assert verdict == AuditVerdict.REVIEW


def test_frozen_episode_does_not_downgrade_exclude():
    """Frozen check must NOT downgrade an existing EXCLUDE."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.0})]
    verdict = check_frozen_episode(AuditVerdict.EXCLUDE, results)
    assert verdict == AuditVerdict.EXCLUDE


def test_frozen_episode_na_idle_ratio_not_triggered():
    """If idle_ratio is N/A, frozen check must not trigger."""
    results = [_na_result("idle_ratio")]
    verdict = check_frozen_episode(AuditVerdict.PASS, results)
    assert verdict == AuditVerdict.PASS


def test_frozen_episode_no_idle_ratio_metric():
    """If idle_ratio metric is not present, frozen check must not trigger."""
    results = [_observation("action_discontinuity", {"spike_count": 5})]
    verdict = check_frozen_episode(AuditVerdict.PASS, results)
    assert verdict == AuditVerdict.PASS


def test_frozen_episode_custom_threshold():
    """Custom threshold must override default."""
    results = [_observation("idle_ratio", {"effective_motion_ratio": 0.10})]
    # Default threshold (0.02): not triggered
    assert check_frozen_episode(AuditVerdict.PASS, results) == AuditVerdict.PASS
    # Custom threshold (0.15): triggered
    assert check_frozen_episode(AuditVerdict.PASS, results, emr_threshold=0.15) == AuditVerdict.REVIEW


# --- Calibrated severity brackets for action_discontinuity ------------------


def test_severity_bracket_extreme_spikes():
    """spike_count > 37 (failure P75) → severity 30."""
    result = _observation("action_discontinuity", {"spike_count": 40})
    severity, findings = compute_behavior_severity([result])
    assert severity == 30
    assert len(findings) == 1
    assert "37" in findings[0]["reason"]


def test_severity_bracket_high_spikes():
    """spike_count > 27 (failure median) but ≤ 37 → severity 20."""
    result = _observation("action_discontinuity", {"spike_count": 30})
    severity, findings = compute_behavior_severity([result])
    assert severity == 20
    assert len(findings) == 1
    assert "27" in findings[0]["reason"]


def test_severity_bracket_moderate_spikes():
    """spike_count > 16 (Youden optimal) but ≤ 27 → severity 10."""
    result = _observation("action_discontinuity", {"spike_count": 20})
    severity, findings = compute_behavior_severity([result])
    assert severity == 10
    assert len(findings) == 1
    assert "16" in findings[0]["reason"]


def test_severity_bracket_normal_spikes():
    """spike_count ≤ 16 → severity 0 (within normal range)."""
    result = _observation("action_discontinuity", {"spike_count": 10})
    severity, findings = compute_behavior_severity([result])
    assert severity == 0
    assert len(findings) == 0


def test_severity_bracket_boundary_16():
    """spike_count = 16 → severity 0 (NOT above threshold, boundary is exclusive)."""
    result = _observation("action_discontinuity", {"spike_count": 16})
    severity, _ = compute_behavior_severity([result])
    assert severity == 0


def test_severity_bracket_boundary_17():
    """spike_count = 17 → severity 10 (just above Youden threshold)."""
    result = _observation("action_discontinuity", {"spike_count": 17})
    severity, _ = compute_behavior_severity([result])
    assert severity == 10


# --- Integration: frozen episode in full pipeline ---------------------------


def test_frozen_episode_pipeline_integration():
    """Full pipeline must catch frozen episodes via check_frozen_episode."""
    results = [
        _observation("action_discontinuity", {"spike_count": 0}),
        _observation("idle_ratio", {"effective_motion_ratio": 0.0}),
    ]
    # classify_episode: all metrics are PASS/diagnostic → PASS
    verdict = classify_episode(results)
    assert verdict == AuditVerdict.PASS
    # check_frozen_episode: EMR=0.0 < 0.02 → REVIEW
    verdict = check_frozen_episode(verdict, results)
    assert verdict == AuditVerdict.REVIEW


def test_frozen_episode_with_existing_exclude():
    """Frozen check must not override hard EXCLUDE from critical metrics."""
    nan_finding = MetricResult(
        name="invalid_values",
        availability=MetricAvailability.AVAILABLE,
        measurement={"nan_count": 5},
        assessment={"status": "exclude", "severity": "high", "reason": "NaN"},
        details={},
        message="EXCLUDE",
        has_finding=True,
    )
    idle = _observation("idle_ratio", {"effective_motion_ratio": 0.0})
    verdict = classify_episode([nan_finding, idle])
    assert verdict == AuditVerdict.EXCLUDE
    # Frozen check should not downgrade EXCLUDE
    verdict = check_frozen_episode(verdict, [nan_finding, idle])
    assert verdict == AuditVerdict.EXCLUDE
