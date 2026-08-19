"""Boundary tests for the two core behavioral metrics.

``idle_ratio`` and ``action_discontinuity`` are the signals that feed
the verdict gate (see test_negative_control.py for the wiring side).
These tests pin the *measurement* side: synthetic episodes with known
ground truth must produce the expected numbers.

All episodes below are hand-constructed, so every expected value is
exact — no statistical hand-waving.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.io.schema import EpisodeData  # noqa: E402
from rda.metrics.base import MetricAvailability  # noqa: E402
from rda.metrics.motion import (  # noqa: E402
    ActionDiscontinuityMetric,
    IdleRatioMetric,
)


def _episode(action: np.ndarray) -> EpisodeData:
    """Wrap a (T, D) action array into a minimal EpisodeData."""
    t = action.shape[0]
    return EpisodeData(
        episode_index=0,
        num_frames=t,
        timestamps=np.arange(t, dtype=np.float64) / 10.0,
        observation={},
        action={"joint_pos": action.astype(np.float64)},
    )


def _bimodal_episode(n_idle: int = 50, n_move: int = 50) -> np.ndarray:
    """First half stationary, second half moving at constant speed."""
    moving = np.linspace(0.0, 1.0, n_move)
    flat = np.zeros(n_idle)
    pos = np.concatenate([flat, moving])
    return np.stack([pos, pos * 0.5], axis=1)


# --- idle_ratio -----------------------------------------------------------


def test_idle_bimodal_episode_measures_half():
    """50% idle + 50% moving must measure effective_motion_ratio ~= 0.5."""
    ep = _episode(_bimodal_episode(50, 50))
    result = IdleRatioMetric().compute(ep)
    m = result.measurement
    assert 0.4 <= m["effective_motion_ratio"] <= 0.6
    assert abs(m["idle_ratio"] + m["effective_motion_ratio"] - 1.0) < 1e-9


def test_idle_frozen_episode_is_fully_idle():
    """Constant actions => zero motion => idle ratio 1.0."""
    ep = _episode(np.zeros((60, 2)))
    m = IdleRatioMetric().compute(ep).measurement
    assert m["idle_ratio"] > 0.99
    assert m["effective_motion_ratio"] < 0.01


def test_idle_smooth_motion_is_not_idle():
    """Constant-velocity ramp => all frames moving => idle ratio ~0."""
    pos = np.linspace(0.0, 5.0, 100)
    ep = _episode(np.stack([pos, pos], axis=1))
    m = IdleRatioMetric().compute(ep).measurement
    assert m["idle_ratio"] < 0.1


def test_idle_ratio_bounded_unit_interval():
    """Whatever the input, the ratio must stay within [0, 1]."""
    rng = np.random.default_rng(42)
    for scale in (1e-9, 1e-3, 1.0, 1e3):
        action = rng.normal(0.0, scale, size=(80, 3)).cumsum(axis=0)
        m = IdleRatioMetric().compute(_episode(action)).measurement
        assert 0.0 <= m["idle_ratio"] <= 1.0, f"idle_ratio out of bounds at scale {scale}"


# --- action_discontinuity --------------------------------------------------


def test_spikes_smooth_ramp_zero():
    """Constant-velocity motion has no jerk spikes."""
    pos = np.linspace(0.0, 5.0, 100)
    ep = _episode(np.stack([pos, pos], axis=1))
    m = ActionDiscontinuityMetric().compute(ep).measurement
    assert m["spike_count"] == 0


def test_spikes_injected_jumps_detected():
    """Inject 5 violent jumps into a noisy ramp => >= 5 spikes flagged.

    Note on realism: the ramp carries sensor-scale noise (sigma=0.01),
    like every real recording. On a *perfectly* smooth synthetic ramp
    the MAD of the second difference is zero, and the z-score
    degenerates to all-zeros — a known limit of MAD-based detectors on
    noiseless inputs, not a bug this test should paper over.
    """
    rng = np.random.default_rng(7)
    pos = np.linspace(0.0, 5.0, 200) + rng.normal(0.0, 0.01, 200)
    action = np.stack([pos, pos], axis=1)
    for i in range(5):
        action[40 * i + 20, 0] += 25.0  # violent jump
    ep = _episode(action)
    m = ActionDiscontinuityMetric().compute(ep).measurement
    assert m["spike_count"] >= 5


def test_spikes_nan_action_is_exclude():
    """NaN in actions is hard corruption: EXCLUDE, not a spike count."""
    action = np.zeros((50, 2))
    action[10, 0] = np.nan
    result = ActionDiscontinuityMetric().compute(_episode(action))
    assert result.availability == MetricAvailability.AVAILABLE
    assert result.assessment["status"] == "exclude"


def test_idle_nan_action_is_exclude():
    """The idle metric must treat NaN the same way: hard EXCLUDE."""
    action = np.zeros((50, 2))
    action[10, 0] = np.nan
    result = IdleRatioMetric().compute(_episode(action))
    assert result.assessment["status"] == "exclude"


def test_missing_action_is_na_not_failure():
    """No action array => N/A for both metrics, never a crash or FAIL."""
    ep = EpisodeData(episode_index=0, num_frames=10,
                     timestamps=np.arange(10.0), observation={}, action={})
    for metric in (IdleRatioMetric(), ActionDiscontinuityMetric()):
        result = metric.compute(ep)
        assert result.availability == MetricAvailability.NOT_AVAILABLE


def test_too_few_frames_is_na():
    """Single-frame episodes cannot yield diffs: N/A, not garbage numbers."""
    for arr in (np.zeros((1, 2)), np.zeros((2, 2))):
        ep = _episode(arr)
        result = ActionDiscontinuityMetric().compute(ep)
        assert result.availability == MetricAvailability.NOT_AVAILABLE


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
