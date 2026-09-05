"""REQ-3② (v0.7.3): fps-aware usable-run floor.

DROID's min_non_idle_len=16 encodes "about 1 second at 15-30 Hz". On
higher-fps data the bare frame floor under-counts usable retention:
at 50 Hz, 16 frames is 0.32s, letting sub-second twitches pass as
"usable". The floor is therefore max(16, 1s-in-frames) when the
episode's fps is known, and exactly 16 otherwise (byte-identical to
v0.6.0 behavior at <=16 Hz or when fps is missing/garbage).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.io.schema import EpisodeData  # noqa: E402
from rda.recommend.temporal_metrics import (  # noqa: E402
    USABLE_RUN_MIN_FRAMES,
    compute_temporal_sufficiency,
)


def _episode(fps=None, n: int = 100) -> EpisodeData:
    meta = {"fps": fps} if fps is not None else {}
    ep = EpisodeData(
        episode_index=0,
        timestamps=np.arange(n, dtype=np.float64) / 50.0,
        num_frames=n,
        meta=meta,
    )
    acts = np.random.RandomState(0).randn(n, 6).cumsum(axis=0) * 0.05
    acts[40:] = acts[39]  # idle tail: one long active run then static
    ep.action["joint_pos"] = acts
    return ep


def test_floor_is_16_without_fps():
    ts = compute_temporal_sufficiency(_episode(None))
    assert ts.usable_run_floor_frames == USABLE_RUN_MIN_FRAMES


def test_floor_is_fps_at_50hz():
    ts = compute_temporal_sufficiency(_episode(50))
    assert ts.usable_run_floor_frames == 50


def test_floor_stays_16_at_15hz():
    ts = compute_temporal_sufficiency(_episode(15))
    assert ts.usable_run_floor_frames == USABLE_RUN_MIN_FRAMES


def test_floor_stays_16_at_exact_16hz():
    ts = compute_temporal_sufficiency(_episode(16))
    assert ts.usable_run_floor_frames == USABLE_RUN_MIN_FRAMES


@pytest.mark.parametrize("bad", ["garbage", 0, -5, None, 2.5])
def test_floor_falls_back_on_bad_fps(bad):
    ts = compute_temporal_sufficiency(_episode(bad))
    assert ts.usable_run_floor_frames == USABLE_RUN_MIN_FRAMES


def test_50hz_data_usable_shrinks_vs_16_floor():
    """The aloha_insertion finding: at 50 Hz the stricter floor must
    not over-count sub-second runs as usable."""
    ep = _episode(50)
    ts50 = compute_temporal_sufficiency(ep)
    ep16 = _episode(None)
    ts16 = compute_temporal_sufficiency(ep16)
    assert ts50.usable_retention_ratio <= ts16.usable_retention_ratio


def test_dataset_aggregate_carries_floor():
    from rda.recommend.temporal_metrics import aggregate_temporal_sufficiency

    per = [compute_temporal_sufficiency(_episode(50)),
           compute_temporal_sufficiency(_episode(None))]
    agg = aggregate_temporal_sufficiency(per, total_episodes=2, total_frames=200)
    assert "usable_run_floor_frames" in agg.to_dict()
    median_floor = agg.usable_run_floor_frames.get("median")
    assert median_floor is not None and median_floor >= USABLE_RUN_MIN_FRAMES
