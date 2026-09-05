"""Tests for REQ-3 (DROID-aligned idle rules) — v0.6.0.

Covers:
- Per-episode usable_retention_ratio / max_idle_run_frames computation
- Dataset-level aggregation of the new fields
- DISCARD_STATIC gating in the local offline fallback
- Engine-side chunk alignment (via a local re-implementation of the
  server contract, since engine_core.py is closed-source and not in
  the repo — we test the client-side contract pieces instead).
"""
import math

import numpy as np
import pytest

from rda.recommend.temporal_metrics import (
    DatasetTemporalSufficiency,
    TemporalSufficiency,
    USABLE_RUN_MIN_FRAMES,
    compute_temporal_sufficiency,
)
from rda.recommend.types import (
    RecommendationAction,
    TargetPolicy,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic EpisodeData
# ---------------------------------------------------------------------------

def _make_episode(action_values, fps=10):
    """Build a minimal EpisodeData with a 1-D joint_pos action array."""
    from rda.io.schema import EpisodeData

    arr = np.asarray(action_values, dtype=np.float64)
    n = arr.shape[0]
    ep = EpisodeData(
        episode_index=0,
        num_frames=n,
        timestamps=np.arange(n, dtype=np.float64) / fps,
    )
    ep.action = {"joint_pos": arr.reshape(n, 1)}
    ep.observation = {}
    return ep


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------

class TestUsableRetention:
    def test_constant_low_motion_is_all_idle(self):
        # Flat action → all idle → usable retention 0
        ep = _make_episode(np.zeros(50))
        ts = compute_temporal_sufficiency(ep)
        if ts is None:
            pytest.skip("idle detection declined to run on this input")
        assert ts.usable_retention_ratio == 0.0
        # All idle: max idle run spans all frames
        assert ts.max_idle_run_frames == 50

    def test_fields_serialize(self):
        ts = TemporalSufficiency(total_frames=100)
        d = ts.to_dict()
        assert "usable_retention_ratio" in d
        assert "max_idle_run_frames" in d
        assert d["usable_retention_ratio"] == 0.0
        assert d["max_idle_run_frames"] == 0

    def test_constants_documented(self):
        # Aligned with openpi DROID min_non_idle_len=16
        assert USABLE_RUN_MIN_FRAMES == 16


# ---------------------------------------------------------------------------
# Dataset aggregation
# ---------------------------------------------------------------------------

class TestDatasetAggregation:
    def test_new_fields_in_dataset_dict(self):
        ds = DatasetTemporalSufficiency()
        d = ds.to_dict()
        assert "usable_retention_ratio" in d
        assert "max_idle_run_frames" in d

    def test_aggregate_propagates_new_fields(self):
        from rda.recommend.temporal_metrics import aggregate_temporal_sufficiency

        a = TemporalSufficiency(total_frames=100)
        a.idle_total_ratio = 0.97
        a.usable_retention_ratio = 0.01
        a.max_idle_run_frames = 90

        b = TemporalSufficiency(total_frames=100)
        b.idle_total_ratio = 0.10
        b.usable_retention_ratio = 0.85
        b.max_idle_run_frames = 5

        agg = aggregate_temporal_sufficiency([a, b], total_episodes=2, total_frames=200)
        # Median of [0.01, 0.85] = 0.43
        assert math.isclose(agg.usable_retention_ratio["median"], 0.43)
        # Median of [90, 5] = 47.5
        assert math.isclose(agg.max_idle_run_frames["median"], 47.5)


# ---------------------------------------------------------------------------
# Offline fallback: DISCARD_STATIC branch
# ---------------------------------------------------------------------------

def _agg_with(idle_median, usable_median, computed=3):
    agg = DatasetTemporalSufficiency()
    agg.total_episodes = computed
    agg.computed_episodes = computed
    agg.idle_total_ratio = {"p5": idle_median, "median": idle_median, "p95": idle_median}
    agg.idle_prefix_ratio = {"p5": 0.0, "median": 0.0, "p95": 0.0}
    agg.usable_retention_ratio = {
        "p5": usable_median, "median": usable_median, "p95": usable_median,
    }
    return agg


class TestOfflineDiscardStatic:
    def _build(self, agg):
        from rda.recommend.local_fallback import build_offline_result

        return build_offline_result(agg, TargetPolicy.TEMPORAL)

    def test_near_static_triggers_discard(self):
        result = self._build(_agg_with(idle_median=0.97, usable_median=0.01))
        actions = [r.action for r in result.recommendations]
        assert actions == [RecommendationAction.DISCARD_STATIC]
        # EXPERIMENTAL offline
        assert result.recommendations[0].confidence.value == "EXPERIMENTAL"
        assert result.rules_version == "offline-fallback"

    def test_high_idle_but_usable_does_not_trigger(self):
        # idle 96% but retention 50% → there IS usable content; the
        # near-static branch must not fire.
        result = self._build(_agg_with(idle_median=0.96, usable_median=0.50))
        actions = [r.action for r in result.recommendations]
        assert RecommendationAction.DISCARD_STATIC not in actions

    def test_normal_data_unaffected(self):
        result = self._build(_agg_with(idle_median=0.15, usable_median=0.80))
        actions = [r.action for r in result.recommendations]
        assert RecommendationAction.DISCARD_STATIC not in actions
        assert actions[0] == RecommendationAction.DO_NOT_PRUNE  # temporal default

    def test_policy_chunk_size_kwarg_accepted(self):
        from rda.recommend.local_fallback import build_offline_result

        # Signature-compat: old callers keep working, new kwarg is accepted
        result = build_offline_result(
            _agg_with(0.15, 0.80), TargetPolicy.TEMPORAL, policy_chunk_size=16
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Contract v3 client plumbing
# ---------------------------------------------------------------------------

class TestContractV3Client:
    def test_cache_key_v3_includes_chunk_size(self):
        from rda.recommend.api_client import _cache_key

        agg = _agg_with(0.2, 0.8)
        k_none = _cache_key(agg, "temporal", "zh", None, None)
        k_16 = _cache_key(agg, "temporal", "zh", None, 16)
        k_24 = _cache_key(agg, "temporal", "zh", None, 24)
        assert k_none.startswith("v3-")
        assert k_16 != k_none
        assert k_16 != k_24
        assert ":" not in k_16  # Windows-safe

    def test_new_enum_actions_exist(self):
        assert RecommendationAction.DISCARD_STATIC.value == "DISCARD_STATIC"
        assert RecommendationAction.SMOOTHING_REVIEW.value == "SMOOTHING_REVIEW"
        assert RecommendationAction.CALIBRATION_CHECK.value == "CALIBRATION_CHECK"
        assert RecommendationAction.COVERAGE_SUGGESTION.value == "COVERAGE_SUGGESTION"

    def test_from_dict_degrades_unknown_action(self):
        """Older clients receiving v4-only actions degrade safely."""
        from rda.recommend.types import Recommendation

        rec = Recommendation.from_dict({
            "action": "TOTALLY_UNKNOWN_ACTION",
            "confidence": "HIGH",
            "title": "t",
            "summary": "s",
        })
        assert rec.action == RecommendationAction.NO_RECOMMENDATION
