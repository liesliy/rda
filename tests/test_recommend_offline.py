"""Offline fallback tests for ``rda recommend``.

The offline fallback must be *deliberately boring*: when the server-side
rules engine is unreachable and no cache exists, the CLI previously
exited with an error even though all metrics were already computed
client-side. These tests pin the conservative grading behavior:

- temporal policies never get a trim suggestion offline
- frame-wise policies only get TRIM_INITIAL under the clearly-safe
  prefix condition (short, prefix-concentrated idle)
- everything else lands on DO_NOT_PRUNE
- every offline result is labeled rules_version="offline-fallback"

All aggregates below are hand-constructed DatasetTemporalSufficiency
objects — no dataset loading, no network, no statistical hand-waving.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.recommend.local_fallback import (  # noqa: E402
    OFFLINE_RULES_VERSION,
    build_offline_result,
)
from rda.recommend.temporal_metrics import DatasetTemporalSufficiency  # noqa: E402
from rda.recommend.types import (  # noqa: E402
    ConfidenceLevel,
    RecommendationAction,
    TargetPolicy,
)


def _agg(idle_total_median: float, idle_prefix_median: float,
         computed: int = 10) -> DatasetTemporalSufficiency:
    """Hand-build a dataset aggregate with known medians."""
    d = lambda v: {"p5": v, "median": v, "p95": v}
    return DatasetTemporalSufficiency(
        total_episodes=computed,
        total_frames=100_000,
        computed_episodes=computed,
        idle_total_ratio=d(idle_total_median),
        idle_prefix_ratio=d(idle_prefix_median),
        active_run_p50=d(10.0),
        active_run_p90=d(20.0),
        active_run_max=d(40.0),
        transition_count=d(5.0),
        valid_window_ratio_5=d(0.5),
        valid_window_ratio_10=d(0.3),
        valid_window_ratio_20=d(0.1),
    )


# --- labeling --------------------------------------------------------------


def test_offline_result_is_labeled():
    """Every offline result must carry the offline-fallback rules version."""
    for policy in (TargetPolicy.FRAME_WISE, TargetPolicy.TEMPORAL):
        r = build_offline_result(_agg(0.817, 0.015), policy)
        assert r.rules_version == OFFLINE_RULES_VERSION
        assert r.engine_version == "offline"


# --- temporal policy: never trim offline ------------------------------------


def test_temporal_policy_never_gets_trim():
    """Temporal policies depend on contiguous windows; offline says no pruning."""
    r = build_offline_result(_agg(0.02, 0.015), TargetPolicy.TEMPORAL)
    assert len(r.recommendations) == 1
    assert r.recommendations[0].action == RecommendationAction.DO_NOT_PRUNE


def test_temporal_policy_high_idle_also_no_prune():
    """Even with heavy idle, temporal policy stays DO_NOT_PRUNE offline."""
    r = build_offline_result(_agg(0.817, 0.015), TargetPolicy.TEMPORAL)
    assert r.recommendations[0].action == RecommendationAction.DO_NOT_PRUNE


# --- frame-wise: the clearly-safe TRIM_INITIAL branch ------------------------


def test_framewise_pure_prefix_gets_trim_initial():
    """Short prefix carrying most idle mass -> TRIM_INITIAL (EXPERIMENTAL)."""
    # prefix 5% of frames, total idle 6% -> prefix holds ~83% of idle mass
    r = build_offline_result(_agg(0.06, 0.05), TargetPolicy.FRAME_WISE)
    assert r.recommendations[0].action == RecommendationAction.TRIM_INITIAL
    assert r.recommendations[0].confidence == ConfidenceLevel.EXPERIMENTAL


def test_framewise_widespread_idle_stays_do_not_prune():
    """Mid-episode idle dominates -> conservative default, no trim."""
    # prefix 1.5%, total idle 81.7% -> prefix holds only ~2% of idle mass
    r = build_offline_result(_agg(0.817, 0.015), TargetPolicy.FRAME_WISE)
    assert r.recommendations[0].action == RecommendationAction.DO_NOT_PRUNE


def test_framewise_long_prefix_stays_do_not_prune():
    """Prefix too long (>10% of frames) -> outside the clearly-safe box."""
    # prefix 20%, total idle 22% -> prefix holds 91% of mass but is not short
    r = build_offline_result(_agg(0.22, 0.20), TargetPolicy.FRAME_WISE)
    assert r.recommendations[0].action == RecommendationAction.DO_NOT_PRUNE


def test_framewise_zero_idle_stays_do_not_prune():
    """Nothing to prune -> no trim suggestion."""
    r = build_offline_result(_agg(0.0, 0.0), TargetPolicy.FRAME_WISE)
    assert r.recommendations[0].action == RecommendationAction.DO_NOT_PRUNE


# --- degenerate inputs -------------------------------------------------------


def test_no_computed_episodes_defaults_to_no_prune():
    """computed_episodes == 0 (no action data) -> bare DO_NOT_PRUNE, no crash."""
    r = build_offline_result(_agg(0.0, 0.0, computed=0), TargetPolicy.FRAME_WISE)
    assert r.recommendations[0].action == RecommendationAction.DO_NOT_PRUNE
    assert r.rules_version == OFFLINE_RULES_VERSION


# --- copy labeling -----------------------------------------------------------


def test_offline_copy_discloses_mode_zh():
    """The zh copy must disclose that this was not graded by the server engine."""
    r = build_offline_result(_agg(0.817, 0.015), TargetPolicy.FRAME_WISE, lang="zh")
    joined = " ".join(rec.summary for rec in r.recommendations)
    assert "离线" in joined


def test_offline_copy_discloses_mode_en():
    """The en copy must disclose the offline grading."""
    r = build_offline_result(_agg(0.817, 0.015), TargetPolicy.FRAME_WISE, lang="en")
    joined = " ".join(rec.summary for rec in r.recommendations)
    assert "Offline" in joined or "offline" in joined
