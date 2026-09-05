"""Tests for REQ-1: verdict-gated recommendations (v0.5.9).

Pins the REQ-1 guarantee: a dataset whose episodes fail the
deterministic CRITICAL checks must NEVER receive a TRIM_* suggestion —
regardless of whether the result came from the server, the cache, or
the offline fallback.

All episodes are hand-constructed; every expected verdict is exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.audit.rules import AuditVerdict  # noqa: E402
from rda.io.schema import EpisodeData  # noqa: E402
from rda.recommend.preflight import (  # noqa: E402
    PreflightAuditor,
    PreflightSummary,
    aggregate_preflight,
    gate_result_by_verdict,
)
from rda.recommend.local_fallback import build_offline_result  # noqa: E402
from rda.recommend.temporal_metrics import DatasetTemporalSufficiency  # noqa: E402
from rda.recommend.types import (  # noqa: E402
    RecommendationAction,
    TargetPolicy,
)


def _healthy_episode(idx: int = 0, n: int = 60) -> EpisodeData:
    """Smooth constant-velocity motion — passes all CRITICAL checks."""
    pos = np.linspace(0.0, 1.0, n)
    return EpisodeData(
        episode_index=idx,
        num_frames=n,
        timestamps=np.arange(n, dtype=np.float64) / 10.0,
        observation={"state": np.stack([pos, pos * 0.5], axis=1)},
        action={"joint_pos": np.stack([pos, pos * 0.5], axis=1)},
    )


def _nan_episode(idx: int = 0, n: int = 60) -> EpisodeData:
    """Same motion but with NaN contamination in actions (INVALID)."""
    ep = _healthy_episode(idx, n)
    act = ep.action["joint_pos"].copy()
    act[10, 0] = np.nan
    act[20, 1] = np.inf
    ep.action["joint_pos"] = act
    return ep


def _broken_timestamp_episode(idx: int = 0, n: int = 60) -> EpisodeData:
    """Non-monotonic timestamps (INVALID)."""
    ep = _healthy_episode(idx, n)
    ts = ep.timestamps.copy()
    ts[30:] = ts[:30]  # time goes backwards
    ep.timestamps = ts
    return ep


def _dropout_episode(idx: int = 0, n: int = 60) -> EpisodeData:
    """State and action have different frame counts (REPAIRABLE)."""
    ep = _healthy_episode(idx, n)
    ep.observation["state"] = ep.observation["state"][: n - 5]  # 5 frames short
    return ep


def _zero_frame_episode(idx: int = 0) -> EpisodeData:
    return EpisodeData(
        episode_index=idx,
        num_frames=0,
        timestamps=np.zeros(0),
        observation={},
        action={},
    )


def _agg(idle_total: float = 0.05, idle_prefix: float = 0.05,
         computed: int = 2) -> DatasetTemporalSufficiency:
    d = lambda v: {"p5": v, "median": v, "p95": v}
    return DatasetTemporalSufficiency(
        total_episodes=computed,
        total_frames=100_000,
        computed_episodes=computed,
        idle_total_ratio=d(idle_total),
        idle_prefix_ratio=d(idle_prefix),
        active_run_p50=d(10.0),
        active_run_p90=d(20.0),
        active_run_max=d(40.0),
        transition_count=d(5.0),
        valid_window_ratio_5=d(0.5),
        valid_window_ratio_10=d(0.3),
        valid_window_ratio_20=d(0.1),
    )


# --- per-episode evaluation -------------------------------------------------


def test_healthy_episode_passes_preflight():
    v = PreflightAuditor().evaluate(_healthy_episode())
    assert v.verdict == AuditVerdict.PASS.value
    assert v.reason_code is None
    assert v.failed_metrics == []


def test_nan_episode_is_exclude_invalid():
    v = PreflightAuditor().evaluate(_nan_episode())
    assert v.verdict == AuditVerdict.EXCLUDE.value
    assert v.reason_code == "INVALID"
    assert "invalid_values" in v.failed_metrics


def test_broken_timestamps_are_exclude_invalid():
    v = PreflightAuditor().evaluate(_broken_timestamp_episode())
    assert v.verdict == AuditVerdict.EXCLUDE.value
    assert v.reason_code == "INVALID"
    assert "timestamp_validity" in v.failed_metrics


def test_dropout_episode_is_exclude_repairable():
    v = PreflightAuditor().evaluate(_dropout_episode())
    assert v.verdict == AuditVerdict.EXCLUDE.value
    assert v.reason_code == "REPAIRABLE"
    assert "missing_dropout" in v.failed_metrics


def test_zero_frame_episode_is_exclude_repairable():
    v = PreflightAuditor().evaluate(_zero_frame_episode())
    assert v.verdict == AuditVerdict.EXCLUDE.value
    assert v.reason_code == "REPAIRABLE"


# --- aggregation -------------------------------------------------------------


def test_aggregate_counts_and_dominant_code():
    verdicts = [
        PreflightAuditor().evaluate(e)
        for e in [_healthy_episode(0), _healthy_episode(1),
                  _nan_episode(2), _dropout_episode(3),
                  _nan_episode(4)]
    ]
    s = aggregate_preflight(verdicts)
    assert s.episodes_total == 5
    assert s.pass_count == 2
    assert s.exclude_count == 3
    assert s.review_count == 0
    # INVALID dominates REPAIRABLE
    assert s.dominant_reason_code == "INVALID"
    assert {e.episode_index for e in s.excluded_episodes} == {2, 3, 4}


def test_aggregate_all_healthy_has_no_exclusions():
    verdicts = [PreflightAuditor().evaluate(e) for e in
                [_healthy_episode(0), _healthy_episode(1)]]
    s = aggregate_preflight(verdicts)
    assert s.exclude_count == 0
    assert s.dominant_reason_code is None
    assert s.to_dict().get("excluded_episodes") is None


# --- payload contract ---------------------------------------------------------


def test_summary_dict_roundtrip():
    verdicts = [PreflightAuditor().evaluate(e) for e in
                [_healthy_episode(0), _nan_episode(1), _dropout_episode(2)]]
    s = aggregate_preflight(verdicts)
    d = s.to_dict()
    assert d["exclude_count"] == 2
    assert d["dominant_reason_code"] == "INVALID"
    ex = d["excluded_episodes"][0]
    assert ex["episode_index"] == 1
    assert ex["verdict"] == "EXCLUDE"
    assert ex["reason_code"] == "INVALID"
    assert "invalid_values" in ex["failed_metrics"]
    # Roundtrip
    s2 = PreflightSummary.from_dict(d)
    assert s2.exclude_count == s.exclude_count
    assert s2.dominant_reason_code == s.dominant_reason_code
    assert s2.excluded_episodes[0].failed_metrics == \
        s.excluded_episodes[0].failed_metrics


def test_summary_dict_size_budget():
    """The verdict payload must stay small (privacy: aggregates only)."""
    verdicts = [PreflightAuditor().evaluate(_healthy_episode(i))
                for i in range(200)]
    verdicts.append(PreflightAuditor().evaluate(_nan_episode(999)))
    s = aggregate_preflight(verdicts)
    import json
    size = len(json.dumps(s.to_dict()))
    assert size < 1024, f"verdict payload too large: {size} bytes"


# --- the gate -------------------------------------------------------------------


def _trim_result():
    """A TRIM_INITIAL result as a server/cache might return it."""
    return build_offline_result(_agg(0.06, 0.05), TargetPolicy.FRAME_WISE)


def test_gate_blocks_trim_on_exclude_dataset():
    """THE REQ-1 guarantee: TRIM on a broken dataset becomes REPAIR_FIRST."""
    result = _trim_result()
    assert result.recommendations[0].action == RecommendationAction.TRIM_INITIAL

    verdicts = [PreflightAuditor().evaluate(e) for e in
                [_healthy_episode(0), _nan_episode(1)]]
    s = aggregate_preflight(verdicts)
    gated = gate_result_by_verdict(result, s)

    assert len(gated.recommendations) == 1
    gate = gated.recommendations[0]
    assert gate.action == RecommendationAction.REPAIR_FIRST
    assert gate.title.startswith("先修复数据")  # zh default


def test_gate_passthrough_on_healthy_dataset():
    result = _trim_result()
    verdicts = [PreflightAuditor().evaluate(_healthy_episode(0))]
    s = aggregate_preflight(verdicts)
    gated = gate_result_by_verdict(result, s)
    # No exclusions -> untouched
    assert gated.recommendations[0].action == RecommendationAction.TRIM_INITIAL


def test_gate_none_summary_passthrough():
    result = _trim_result()
    gated = gate_result_by_verdict(result, None)
    assert gated.recommendations[0].action == RecommendationAction.TRIM_INITIAL


def test_gate_english_copy():
    result = _trim_result()
    verdicts = [PreflightAuditor().evaluate(_nan_episode(0))]
    s = aggregate_preflight(verdicts)
    gated = gate_result_by_verdict(result, s, lang="en")
    gate = gated.recommendations[0]
    assert gate.action == RecommendationAction.REPAIR_FIRST
    assert gate.title == "REPAIR_FIRST"
    assert "NaN/Inf" in " ".join(gate.details) or "irrecoverable" in gate.summary


def test_gate_repairable_damage_copy():
    """REPAIRABLE-dominant damage uses the repair wording."""
    result = _trim_result()
    verdicts = [PreflightAuditor().evaluate(_dropout_episode(0))]
    s = aggregate_preflight(verdicts)
    gated = gate_result_by_verdict(result, s, lang="en")
    gate = gated.recommendations[0]
    assert "repairable" in gate.summary.lower()


def test_gate_attaches_verdict_summary_for_json():
    """JSON consumers get the evidence attached on the result object."""
    result = _trim_result()
    verdicts = [PreflightAuditor().evaluate(_nan_episode(0))]
    s = aggregate_preflight(verdicts)
    gated = gate_result_by_verdict(result, s)
    assert getattr(gated, "verdict_summary", None) is not None
    assert gated.verdict_summary["exclude_count"] == 1


# --- single-pass integration ------------------------------------------------------


def test_compute_local_metrics_returns_summary():
    """compute_local_metrics must collect preflight verdicts in one pass.

    Note: the NaN episode is counted in preflight (EXCLUDE) but its
    temporal metrics come back all-zero — compute_temporal_sufficiency
    defensively treats NaN-contaminated actions as non-computable, so
    computed_episodes is 2, not 3. That is expected existing behavior.
    """
    from rda.recommend.api_client import compute_local_metrics

    episodes = [_healthy_episode(0), _nan_episode(1), _dropout_episode(2)]
    agg, per_ep, summary = compute_local_metrics(
        iter(episodes),
        total_episodes=3,
        total_frames=sum(e.num_frames for e in episodes),
        preflight=PreflightAuditor(),
    )
    assert agg.computed_episodes == 2  # NaN episode not computable (existing behavior)
    assert len(per_ep) == 3
    assert summary is not None
    assert summary.exclude_count == 2
    assert summary.pass_count == 1


def test_cache_key_v3_namespacing():
    """v3 cache keys are namespaced and sensitive to verdict + chunk size.

    The prefix must be colon-free: "v3:<hash>" is an invalid filename
    on Windows (colon = drive separator) and cache writes would fail
    silently there. "v3-<hash>" is safe on all platforms. REQ-3 added
    policy_chunk_size as a key ingredient (it changes rule outcomes).
    """
    from rda.recommend.api_client import _cache_key
    agg = _agg()
    k1 = _cache_key(agg, "frame-wise", "zh")
    k2 = _cache_key(agg, "frame-wise", "zh", {})
    k3 = _cache_key(agg, "frame-wise", "zh", {"exclude_count": 1})
    assert k1.startswith("v3-")
    assert ":" not in k1                      # Windows-safe filename
    assert k1 == k2          # empty verdict dict == absent (back-compat shape)
    assert k1 != k3          # different verdicts -> different cache entries
    # REQ-3: chunk size participates in the key
    k4 = _cache_key(agg, "frame-wise", "zh", None, 16)
    k5 = _cache_key(agg, "frame-wise", "zh", None, 24)
    assert k4 != k1           # chunk size set vs absent
    assert k4 != k5           # different chunk sizes differ
