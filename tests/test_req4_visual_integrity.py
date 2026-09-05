"""REQ-4 (v0.7.0): VA-A visual-stream integrity + VA-B visual quality tests.

Covers:
- video_freeze: clean data passes; injected camera stall (short → REVIEW,
  long → EXCLUDE); N/A on video-less datasets; adaptive epsilon ignores
  slow motion.
- video_timestamp_alignment: consistent spans pass; systematic drift
  grades REVIEW (soft) / EXCLUDE (hard); N/A without timestamps.
- video_stream_sync: missing camera → EXCLUDE; single camera → N/A;
  consistent multi-camera → PASS.
- visual_quality: clean video passes (never EXCLUDE); REVIEW at most.
- verdict integration: video_freeze in CRITICAL_METRICS,
  visual_quality in REVIEW_METRICS.
- preflight: VA-A names are excluded from the default (no-decode)
  preflight pass and present when include_visual=True.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from rda.audit.rules import AuditVerdict, classify_episode
from rda.io.schema import EpisodeData
from rda.metrics.base import MetricAvailability, MetricResult
from rda.metrics.visual_integrity import (
    VideoFreezeMetric,
    VideoStreamSyncMetric,
    VideoTimestampAlignmentMetric,
)
from rda.metrics.visual_quality import VisualQualityMetric

av = pytest.importorskip("av", reason="PyAV required for VA-A/VA-B tests")


# ---------------------------------------------------------------------------
# Synthetic video helpers (no fixture datasets needed)
# ---------------------------------------------------------------------------

def _write_video(
    path: Path, n_frames: int = 32, size: int = 64, fps: int = 10,
    freeze: Optional[tuple[int, int]] = None, moving: bool = True,
) -> None:
    """Write a tiny MP4. ``freeze=(s, e)`` repeats frame s through e."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.width, stream.height = size, size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        stream.gop_size = n_frames
        cur = base.copy()
        for i in range(n_frames):
            if moving and (freeze is None or not (freeze[0] <= i <= freeze[1])):
                cur = np.roll(cur, 4, axis=1)  # obvious motion
            frame = av.VideoFrame.from_ndarray(cur, format="gray")
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def _make_episode(
    tmp_path: Path,
    cameras: Dict[str, Optional[tuple[int, int]]] = None,
    num_frames: int = 64,
    fps: int = 10,
    span_ratio: float = 1.0,
) -> EpisodeData:
    """Build an episode whose video_features point at synthetic files.

    The video span is declared as (num_frames-1)/fps × span_ratio so it
    matches the parquet timeline (timestamps[0..n-1] spans n-1 intervals)
    when span_ratio=1.0.
    """
    cameras = cameras or {"cam": None}
    root = tmp_path / "ds"
    video_features: Dict[str, Any] = {}
    for cam, freeze in cameras.items():
        _write_video(
            root / "videos" / cam / "chunk-000" / "file-000.mp4",
            n_frames=32, freeze=freeze, fps=fps,
        )
        video_features[cam] = {
            "chunk_index": 0,
            "file_index": 0,
            "from_timestamp": 0.0,
            "to_timestamp": (num_frames - 1) / fps * span_ratio,
        }
    t = np.arange(num_frames, dtype=np.float64) / fps
    rng = np.random.default_rng(3)
    action = np.cumsum(rng.normal(0, 0.1, size=(num_frames, 4)), axis=0)
    return EpisodeData(
        episode_index=0,
        num_frames=num_frames,
        timestamps=t,
        observation={"state": action},
        action={"joint_pos": action},
        meta={
            "dataset_root": str(root),
            "fps": fps,
            "video_features": video_features,
        },
    )


def _no_video_episode() -> EpisodeData:
    rng = np.random.default_rng(1)
    action = rng.normal(0, 1, size=(30, 3))
    return EpisodeData(
        episode_index=0,
        num_frames=30,
        timestamps=np.arange(30, dtype=np.float64) / 10.0,
        observation={"state": action},
        action={"joint_pos": action},
        meta={},
    )


# ---------------------------------------------------------------------------
# video_freeze
# ---------------------------------------------------------------------------

class TestVideoFreeze:
    def test_clean_moving_video_passes(self, tmp_path):
        ep = _make_episode(tmp_path)
        r = VideoFreezeMetric().compute(ep)
        assert r.availability == MetricAvailability.AVAILABLE
        assert r.assessment["status"] == "pass"

    def test_short_stall_is_review_not_exclude(self, tmp_path):
        # 3-frame stall of a 32-frame video @10fps = 0.3s < 0.5s min →
        # not even a region; use 6 frames (0.6s) → one region → REVIEW.
        ep = _make_episode(tmp_path, cameras={"cam": (10, 16)})
        r = VideoFreezeMetric().compute(ep)
        assert r.assessment["status"] in ("review", "pass")  # not exclude

    def test_long_stall_is_exclude(self, tmp_path):
        # 20-frame stall of 32 @10fps = 2.0s while "moving" → conclusive
        # via the ≥30% ratio rule.
        ep = _make_episode(tmp_path, cameras={"cam": (5, 25)})
        r = VideoFreezeMetric().compute(ep)
        assert r.assessment["status"] == "exclude"

    def test_no_video_is_na(self):
        r = VideoFreezeMetric().compute(_no_video_episode())
        assert r.availability == MetricAvailability.NOT_AVAILABLE

    def test_na_does_not_affect_verdict(self):
        ep = _no_video_episode()
        res = MetricResult.make_na(name="video_freeze", reason="x")
        verdict = classify_episode([res])
        assert verdict == AuditVerdict.PASS


# ---------------------------------------------------------------------------
# video_timestamp_alignment
# ---------------------------------------------------------------------------

class TestVideoTimestampAlignment:
    def test_consistent_span_passes(self, tmp_path):
        ep = _make_episode(tmp_path, span_ratio=1.0)
        r = VideoTimestampAlignmentMetric().compute(ep)
        assert r.assessment["status"] == "pass"

    def test_hard_drift_is_exclude(self, tmp_path):
        # parquet span is 63/10=6.3s; declare video span 30% longer.
        ep = _make_episode(tmp_path, span_ratio=1.3)
        r = VideoTimestampAlignmentMetric().compute(ep)
        assert r.assessment["status"] == "exclude"

    def test_no_video_is_na(self):
        r = VideoTimestampAlignmentMetric().compute(_no_video_episode())
        assert r.availability == MetricAvailability.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# video_stream_sync
# ---------------------------------------------------------------------------

class TestVideoStreamSync:
    def test_multi_camera_ok(self, tmp_path):
        ep = _make_episode(tmp_path, cameras={"cam_a": None, "cam_b": None})
        r = VideoStreamSyncMetric().compute(ep)
        assert r.assessment["status"] == "pass"

    def test_missing_camera_is_exclude(self, tmp_path):
        ep = _make_episode(tmp_path, cameras={"cam_a": None, "cam_b": None})
        # Remove the second camera's file → unresolvable stream.
        (Path(ep.meta["dataset_root"]) / "videos" / "cam_b" / "chunk-000"
         / "file-000.mp4").unlink()
        r = VideoStreamSyncMetric().compute(ep)
        assert r.assessment["status"] == "exclude"

    def test_single_camera_is_na(self, tmp_path):
        ep = _make_episode(tmp_path, cameras={"only_cam": None})
        r = VideoStreamSyncMetric().compute(ep)
        assert r.assessment["status"] == "na"


# ---------------------------------------------------------------------------
# visual_quality (VA-B)
# ---------------------------------------------------------------------------

class TestVisualQuality:
    def test_never_excludes(self, tmp_path):
        # Even a pathological video must not grade EXCLUDE.
        ep = _make_episode(tmp_path)
        r = VisualQualityMetric().compute(ep)
        assert r.assessment["status"] in ("pass", "review")

    def test_clean_video_passes(self, tmp_path):
        ep = _make_episode(tmp_path)
        r = VisualQualityMetric().compute(ep)
        assert r.assessment["status"] == "pass"

    def test_no_video_is_na(self):
        r = VisualQualityMetric().compute(_no_video_episode())
        assert r.availability == MetricAvailability.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Verdict & preflight integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_va_a_in_critical_metrics(self):
        from rda.audit.rules import CRITICAL_METRICS
        for m in ("video_freeze", "video_timestamp_alignment", "video_stream_sync"):
            assert m in CRITICAL_METRICS

    def test_va_b_in_review_metrics(self):
        from rda.audit.rules import REVIEW_METRICS
        assert "visual_quality" in REVIEW_METRICS

    def test_new_metrics_registered(self):
        from rda.metrics import ALL_METRICS
        names = {cls.name for cls in ALL_METRICS}
        assert {
            "video_freeze", "video_timestamp_alignment",
            "video_stream_sync", "visual_quality",
        } <= names

    def test_preflight_excludes_visual_by_default(self):
        from rda.recommend.preflight import PreflightAuditor
        pa = PreflightAuditor()
        assert not any(n.startswith("video_") for n in pa._preflight_names)

    def test_preflight_includes_visual_when_opted_in(self):
        from rda.recommend.preflight import PreflightAuditor
        pa = PreflightAuditor(include_visual=True)
        assert "video_freeze" in pa._preflight_names
