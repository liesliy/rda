"""Tests for NumericSanityMetric (L1 WARNING-level check).

This metric checks numeric features for physically implausible values:
  - Joint angles: |value| > angle_limit (default ~π)
  - Force/torque: |value| > force/torque_limit
  - Velocity: |value| > velocity_limit

The metric always returns PASS but attaches warnings in details.
These tests verify:
  1. Normal data produces no warnings
  2. Out-of-range data produces correct warnings
  3. Custom thresholds work
  4. Feature name pattern matching works
  5. The metric does NOT affect episode verdict
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.io.schema import EpisodeData  # noqa: E402
from rda.metrics.base import MetricAvailability, AssessmentStatus  # noqa: E402
from rda.metrics.integrity import NumericSanityMetric  # noqa: E402


def _episode(
    observation: dict | None = None,
    action: dict | None = None,
    n_frames: int = 100,
) -> EpisodeData:
    """Create a minimal EpisodeData for testing."""
    obs = observation or {}
    act = action or {}
    return EpisodeData(
        episode_index=0,
        num_frames=n_frames,
        timestamps=np.arange(n_frames, dtype=np.float64) / 10.0,
        observation=obs,
        action=act,
    )


# --- Test 1: Normal data produces no warnings ---


def test_normal_joint_angles_no_warning():
    """Joint angles within ±π should produce no warnings."""
    obs = {"state": np.random.uniform(-3.0, 3.0, (100, 6))}
    ep = _episode(observation=obs)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert result.availability == MetricAvailability.AVAILABLE
    assert result.assessment["status"] == AssessmentStatus.PASS
    assert len(result.details["warnings"]) == 0
    assert result.measurement["warning_count"] == 0


def test_normal_forces_no_warning():
    """Force values within reasonable bounds should produce no warnings."""
    obs = {"wrench": np.random.uniform(-100, 100, (100, 6))}
    ep = _episode(observation=obs)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert len(result.details["warnings"]) == 0


# --- Test 2: Out-of-range data produces correct warnings ---


def test_angle_exceeds_limit_produces_warning():
    """Joint angles > angle_limit should produce a warning."""
    # Create angles that exceed the default limit (3.2)
    obs = {"joint_pos": np.array([[5.0, -4.0, 2.0]] * 100)}
    ep = _episode(observation=obs)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert len(result.details["warnings"]) == 1
    warning = result.details["warnings"][0]
    assert warning["feature"] == "observation.joint_pos"
    assert warning["check"] == "angle_range"
    assert warning["max_abs"] == 5.0
    assert warning["count"] == 200  # 100 frames * 2 columns exceed limit


def test_force_exceeds_limit_produces_warning():
    """Force values > force_limit should produce a warning."""
    obs = {"force_sensor": np.array([[1e5, -5e4]] * 50)}
    ep = _episode(observation=obs)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert len(result.details["warnings"]) == 1
    warning = result.details["warnings"][0]
    assert warning["check"] == "force_range"
    assert warning["max_abs"] == 1e5


def test_velocity_exceeds_limit_produces_warning():
    """Velocity values > velocity_limit should produce a warning."""
    obs = {"joint_vel": np.array([[200.0, -150.0]] * 30)}
    ep = _episode(observation=obs)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert len(result.details["warnings"]) == 1
    warning = result.details["warnings"][0]
    assert warning["check"] == "velocity_range"
    assert warning["max_abs"] == 200.0


def test_multiple_features_with_warnings():
    """Multiple features exceeding limits should produce multiple warnings."""
    obs = {
        "joint_pos": np.array([[10.0]] * 50),  # angle violation
        "joint_vel": np.array([[500.0]] * 50),  # velocity violation
        "force_sensor": np.array([[500.0]] * 50),  # within bounds (500 < 1e4)
    }
    ep = _episode(observation=obs)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert len(result.details["warnings"]) == 2
    warned_features = {w["feature"] for w in result.details["warnings"]}
    assert "observation.joint_pos" in warned_features
    assert "observation.joint_vel" in warned_features


# --- Test 3: Custom thresholds work ---


def test_custom_angle_limit():
    """Custom angle_limit should override the default."""
    obs = {"joint_pos": np.array([[2.0]] * 10)}
    ep = _episode(observation=obs)

    # Default limit (3.2) — no warning
    result_default = NumericSanityMetric().compute(ep)
    assert len(result_default.details["warnings"]) == 0

    # Custom limit (1.5) — warning
    result_strict = NumericSanityMetric(angle_limit=1.5).compute(ep)
    assert len(result_strict.details["warnings"]) == 1


def test_custom_force_limit():
    """Custom force_limit should override the default."""
    obs = {"force_sensor": np.array([[500.0]] * 10)}
    ep = _episode(observation=obs)

    # Default limit (1e4) — no warning
    result_default = NumericSanityMetric().compute(ep)
    assert len(result_default.details["warnings"]) == 0

    # Custom limit (100) — warning
    result_strict = NumericSanityMetric(force_limit=100.0).compute(ep)
    assert len(result_strict.details["warnings"]) == 1


# --- Test 4: Feature name pattern matching ---


def test_feature_name_patterns():
    """Different feature name patterns should trigger correct checks."""
    # Test angle patterns
    for pattern in ["joint_pos", "joint_angle", "angle", "position"]:
        obs = {pattern: np.array([[10.0]] * 10)}
        ep = _episode(observation=obs)
        result = NumericSanityMetric().compute(ep)
        assert len(result.details["warnings"]) == 1, f"Pattern '{pattern}' should trigger angle check"
        assert result.details["warnings"][0]["check"] == "angle_range"

    # Test force patterns
    for pattern in ["force", "wrench_force"]:
        obs = {pattern: np.array([[5e4]] * 10)}
        ep = _episode(observation=obs)
        result = NumericSanityMetric().compute(ep)
        assert len(result.details["warnings"]) == 1, f"Pattern '{pattern}' should trigger force check"
        assert result.details["warnings"][0]["check"] == "force_range"

    # Test velocity patterns
    for pattern in ["velocity", "vel", "joint_vel"]:
        obs = {pattern: np.array([[500.0]] * 10)}
        ep = _episode(observation=obs)
        result = NumericSanityMetric().compute(ep)
        assert len(result.details["warnings"]) == 1, f"Pattern '{pattern}' should trigger velocity check"
        assert result.details["warnings"][0]["check"] == "velocity_range"


# --- Test 5: Metric does NOT affect episode verdict ---


def test_plausibility_always_returns_pass():
    """Even with extreme values, the metric should return PASS."""
    obs = {
        "joint_pos": np.array([[1e6, -1e6]] * 50),
        "force_sensor": np.array([[1e10]] * 50),
        "joint_vel": np.array([[1e8]] * 50),
    }
    ep = _episode(observation=obs)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert result.assessment["status"] == AssessmentStatus.PASS
    assert len(result.details["warnings"]) == 3


# --- Test 6: Edge cases ---


def test_empty_episode_no_crash():
    """Empty episode should not crash."""
    ep = _episode(n_frames=0)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert result.availability == MetricAvailability.AVAILABLE
    assert result.assessment["status"] == AssessmentStatus.PASS


def test_no_numeric_features():
    """Episode with no numeric features should return PASS."""
    ep = _episode(observation={}, action={})
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert len(result.details["warnings"]) == 0
    assert result.measurement["warning_count"] == 0


def test_action_features_also_checked():
    """Action features should also be checked for plausibility."""
    act = {"joint_pos": np.array([[10.0]] * 50)}
    ep = _episode(action=act)
    metric = NumericSanityMetric()
    result = metric.compute(ep)

    assert len(result.details["warnings"]) == 1
    assert result.details["warnings"][0]["feature"] == "action.joint_pos"


def test_thresholds_in_details():
    """Thresholds should be included in details for reproducibility."""
    ep = _episode()
    metric = NumericSanityMetric(
        angle_limit=2.5,
        force_limit=500.0,
        torque_limit=100.0,
        velocity_limit=50.0,
    )
    result = metric.compute(ep)

    thresholds = result.details["thresholds"]
    assert thresholds["angle_limit"] == 2.5
    assert thresholds["force_limit"] == 500.0
    assert thresholds["torque_limit"] == 100.0
    assert thresholds["velocity_limit"] == 50.0
