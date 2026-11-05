"""Idle-threshold selection — regression tests for the v0.12.3 fallback fix.

The defect these tests pin
-------------------------
``idle_ratio`` picks its threshold in two tiers: a bimodal-gap valley when
the trajectory has a real *static mode*, otherwise a fallback. Up to
v0.12.2 that fallback was ``3 × MAD``. MAD is a **spread** statistic
(≈0.54σ on a smooth unimodal distribution), so ``3 × MAD`` lands near
1.6σ — *above* the median — and therefore labels the bulk of a
continuously moving trajectory as "idle" **by construction**.

Measured on real data before the fix (median idle ratio over all episodes):

    lerobot/aloha_sim_transfer_cube_human   71.2 %   (43/50 eps used the fallback)
    lerobot/droid_100                       70.7 %   (89/100 eps used the fallback)
    OpenLET / Kuavo5W box_depalletizing     74.5 %

i.e. healthy continuous motion reported as a stuck arm. After the fix the
same episodes measure 0.6 %, 8.6 % and 3.5 %, while the two published
benchmark numbers barely move (PushT 81.7 → 81.1, xArm 20.8 → 20.8 —
those trajectories genuinely *are* bimodal, so they keep the gap path).

The fix: when no static mode is found there is nothing to separate, so cut
at ``IDLE_FALLBACK_RATIO × median step motion`` instead — a frame counts as
idle only if it moves slower than a tenth of the episode's typical step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.io.schema import EpisodeData  # noqa: E402
from rda.metrics.motion import (  # noqa: E402
    IDLE_FALLBACK_RATIO,
    IdleRatioMetric,
    _mad,
    select_idle_threshold,
)
from rda.recommend.temporal_metrics import compute_temporal_sufficiency  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Per-step motion (L2 norm across 14 joints, degrees/frame) of one real
#: episode of ``OpenLET/LET-Base-Dataset``
#: (``Kuavo5W/rosbag/real/Labelled/box_depalletizing_and_transfer-5W-leju_claw``,
#: 142 frames @ 10 Hz). Extracted from the converted parquet, not simulated:
#: this is the distribution that produced the 74.5 % reading.
LET_STEP_MOTION = np.array([
    0.000, 0.000, 0.000, 0.000, 0.000, 8.514, 10.996, 16.781, 16.759, 8.752, 6.878, 6.167,
    6.036, 5.915, 6.101, 5.133, 3.840, 3.155, 3.121, 2.880, 2.710, 5.078, 16.134, 14.785,
    4.593, 1.510, 1.172, 2.128, 3.596, 9.042, 7.741, 4.846, 4.211, 0.894, 8.365, 4.045,
    2.363, 3.922, 2.816, 2.993, 2.634, 2.135, 1.893, 2.427, 2.154, 2.064, 1.613, 1.386,
    1.464, 2.760, 4.810, 4.591, 6.600, 1.073, 1.304, 1.723, 1.995, 2.230, 2.331, 3.263,
    3.414, 4.177, 5.280, 6.206, 5.337, 4.167, 3.153, 2.699, 3.165, 3.078, 2.629, 1.671,
    1.383, 1.345, 1.853, 2.888, 3.166, 2.256, 1.347, 1.671, 1.779, 1.460, 1.682, 2.881,
    3.485, 3.595, 4.647, 4.024, 3.409, 2.483, 1.264, 0.947, 0.804, 0.876, 1.147, 0.968,
    1.318, 1.804, 3.109, 3.873, 3.905, 5.176, 6.581, 6.826, 6.555, 5.359, 5.102, 5.737,
    7.578, 9.114, 9.529, 10.511, 11.652, 12.240, 11.670, 11.187, 11.897, 14.256, 13.010,
    9.318, 8.659, 8.252, 6.437, 5.891, 5.753, 6.314, 7.070, 7.216, 7.921, 6.438, 4.828,
    4.523, 3.588, 2.668, 5.344, 5.754, 4.411, 7.037, 6.224, 6.630, 4.401,
])


def _episode_from_steps(steps: np.ndarray) -> EpisodeData:
    """Build a 1-joint episode whose per-step motion equals ``steps``.

    With a single joint the L2 norm across joints reduces to ``|Δ|``, so
    diffing the cumulative sum reproduces the motion vector exactly.
    """
    pos = np.concatenate([[0.0], np.cumsum(steps)]).reshape(-1, 1)
    n = pos.shape[0]
    return EpisodeData(
        episode_index=0,
        num_frames=n,
        timestamps=np.arange(n, dtype=np.float64) / 10.0,
        observation={"state": pos.copy()},
        action={"joint_pos": pos},
        meta={"fps": 10},
    )


def _bimodal_steps(n_idle: int = 50, n_move: int = 50) -> np.ndarray:
    """Half perfectly still, half constant-speed motion — a real static mode."""
    steps = np.concatenate([np.zeros(n_idle), np.full(n_move, 0.02)])
    return steps


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------

def test_continuous_motion_is_not_reported_as_mostly_idle():
    """Real LET episode: continuous motion must not read as 74.5 % idle."""
    ep = _episode_from_steps(LET_STEP_MOTION)
    m = IdleRatioMetric().compute(ep).measurement

    # Only the five genuinely frozen leading steps are idle.
    assert m["idle_ratio"] < 0.15, (
        f"idle_ratio={m['idle_ratio']:.3f} — a continuously moving trajectory "
        "was classified as mostly idle"
    )
    assert m["effective_motion_ratio"] > 0.85


def test_legacy_mad_fallback_would_have_flagged_this_episode():
    """Pin the mechanism: 3×MAD overshoots the median on this distribution.

    Guards against a future "simplification" that quietly restores the old
    fallback — the bug is that the threshold lands above the median, so the
    majority of frames are counted idle no matter how the data moves.
    """
    motion = LET_STEP_MOTION
    legacy_threshold = 3.0 * _mad(motion)

    assert legacy_threshold > np.median(motion), (
        "3×MAD is expected to exceed the median on a smooth unimodal "
        "distribution; if this fails the fixture no longer demonstrates the bug"
    )
    legacy_idle = float((motion < legacy_threshold).mean())
    assert legacy_idle > 0.6, (
        f"legacy fallback would report {legacy_idle:.1%} idle — fixture drifted"
    )


def test_no_static_mode_falls_back_to_median_ratio():
    """No bimodal gap → threshold is a fixed fraction of the median."""
    threshold, method = select_idle_threshold(LET_STEP_MOTION)
    assert method == "median_ratio"
    assert threshold == pytest.approx(IDLE_FALLBACK_RATIO * np.median(LET_STEP_MOTION))


def test_static_mode_still_uses_the_bimodal_gap():
    """A genuine idle/motion split keeps the calibrated gap behaviour."""
    pos = _episode_from_steps(_bimodal_steps()).action["joint_pos"]
    step_motion = np.abs(np.diff(pos, axis=0)[:, 0])
    threshold, method = select_idle_threshold(step_motion)
    assert method == "bimodal_gap"
    assert 0.0 < threshold < 0.02


def test_frozen_episode_hits_the_floor_and_is_fully_idle():
    """All-zero motion → median 0 → threshold clamps to the floor → 100 % idle."""
    motion = np.zeros(60)
    threshold, method = select_idle_threshold(motion)
    assert method == "floor"
    assert threshold == pytest.approx(1e-6)

    ep = _episode_from_steps(motion)
    m = IdleRatioMetric().compute(ep).measurement
    assert m["idle_ratio"] > 0.99
    assert m["effective_motion_ratio"] < 0.01


# ---------------------------------------------------------------------------
# The two consumers must never diverge again
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("steps", [LET_STEP_MOTION, _bimodal_steps(), np.zeros(40)],
                         ids=["continuous", "bimodal", "frozen"])
def test_audit_and_recommend_agree_on_idle_ratio(steps):
    """``rda audit`` and ``rda recommend`` share one threshold selector."""
    ep = _episode_from_steps(steps)
    audit_value = IdleRatioMetric().compute(ep).measurement["idle_ratio"]
    ts = compute_temporal_sufficiency(ep)
    assert ts is not None
    assert ts.idle_total_ratio == pytest.approx(audit_value, abs=1e-9)


# ---------------------------------------------------------------------------
# Contracts kept from v0.12.2
# ---------------------------------------------------------------------------

def test_threshold_is_scale_invariant():
    """Scaling the action unit must not move the idle ratio (INV-011)."""
    base = IdleRatioMetric().compute(_episode_from_steps(LET_STEP_MOTION)).measurement
    for scale in (1e-3, 1e2):
        scaled = IdleRatioMetric().compute(
            _episode_from_steps(LET_STEP_MOTION * scale)
        ).measurement
        assert scaled["idle_ratio"] == pytest.approx(base["idle_ratio"], abs=1e-9)


def test_mad_multiplier_is_deprecated_but_still_accepted():
    """Old call sites keep working; the parameter is simply inert now."""
    with pytest.warns(DeprecationWarning, match="mad_multiplier"):
        metric = IdleRatioMetric(mad_multiplier=3.0)  # type: ignore[arg-type]
    assert metric.idle_fallback_ratio == IDLE_FALLBACK_RATIO

    with pytest.warns(DeprecationWarning, match="mad_multiplier"):
        IdleRatioMetric(mad_multiplier=5.0).compute(_episode_from_steps(LET_STEP_MOTION))


def test_relative_threshold_edges_stay_finite():
    """Degenerate inputs must not produce NaN / negative / infinite thresholds."""
    for motion in (np.zeros(3), np.full(20, 1e-12), np.array([0.0, 1e-7] * 10)):
        threshold, method = select_idle_threshold(motion)
        assert np.isfinite(threshold) and threshold >= 1e-6
        assert method in {"bimodal_gap", "median_ratio", "floor"}
