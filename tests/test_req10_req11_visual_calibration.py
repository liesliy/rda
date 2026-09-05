"""REQ-10/REQ-11 (v0.7.1): visual calibration closure + dep-missing UX.

REQ-10 regression anchors (from benchmarks/visual_inject.py calibration
on libero_10, 5 episodes × 11 scenarios, all acceptance checks passed):
- visual_quality detects injected blur / dark / blow-out (incl. the
  highlight-clipping blind spot: mean luminance stays < 230 while ≥20%
  of pixels clip at 255 — the clipped_frac signal catches it);
- video_freeze adaptive epsilon handles boundary cases: a majority-
  frozen episode (60%) and a fully static camera stream still grade
  EXCLUDE (epsilon does not desensitize).

REQ-11 contract: when PyAV is absent, the four visual metrics grade NA
with reason ``video_deps_missing`` (NOT skipped/pass), and the JSON
report carries a top-level ``skipped_by_missing_dep`` count.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricAvailability
from rda.metrics.visual_integrity import (
    VIDEO_DEPS_MISSING,
    VideoFreezeMetric,
)
from rda.metrics.visual_quality import VisualQualityMetric

av = pytest.importorskip("av", reason="PyAV required for VA-A/VA-B tests")


# ---------------------------------------------------------------------------
# helpers (same synthetic-video approach as test_req4_visual_integrity)
# ---------------------------------------------------------------------------

def _write_video(
    path: Path, n_frames: int = 32, size: int = 64, fps: int = 10,
    mode: str = "moving",
) -> None:
    """mode: moving | static | blurred | dark | blown"""
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
            if mode == "moving":
                cur = np.roll(cur, 4, axis=1)
            elif mode == "static":
                pass  # same frame every step
            elif mode == "blurred":
                cur = np.roll(cur, 4, axis=1)
            frame = av.VideoFrame.from_ndarray(cur, format="gray")
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def _postprocess(path: Path, mode: str) -> None:
    """In-place corruption applied AFTER encoding (decode→transform→re-encode)."""
    if mode not in ("blurred", "dark", "blown"):
        return
    import av

    with av.open(str(path)) as c:
        frames = [f.to_ndarray() for f in c.decode(c.streams.video[0])]
    arr = np.stack(frames)
    if mode == "blurred":
        from numpy.lib.stride_tricks import sliding_window_view

        # Strong blur: 7-tap binomial kernel applied twice per axis —
        # pushes the Laplacian variance well below the 80 floor
        # (calibrated: 5-tap×2 leaves var ≈ 44 → penalty 0.45 = pass).
        k = np.array([1, 6, 15, 20, 15, 6, 1], dtype=np.float32)
        k /= k.sum()
        r = 3
        f32 = arr.astype(np.float32)
        for _ in range(2):
            for axis in (1, 2):
                pad = [(0, 0)] * 3
                pad[axis] = (r, r)
                p = np.pad(f32, pad, mode="edge")
                win = sliding_window_view(p, len(k), axis=axis)
                f32 = (win * k.reshape((1,) * (win.ndim - 1) + (-1,))).sum(axis=-1)
        arr = np.clip(f32, 0, 255).astype(np.uint8)
    elif mode == "dark":
        arr = np.clip(arr.astype(np.float32) * 0.2, 0, 255).astype(np.uint8)
    elif mode == "blown":
        arr = np.clip(arr.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)

    tmp = path.with_suffix(".tmp.mp4")
    with av.open(str(tmp), mode="w") as c:
        stream = c.add_stream("libx264", rate=10)
        stream.width, stream.height = arr.shape[2], arr.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        for f in arr:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="gray")):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    tmp.replace(path)


def _make_episode(
    tmp_path: Path,
    mode: str = "moving",
    fps: int = 10,
) -> EpisodeData:
    root = tmp_path / "ds"
    vp = root / "videos" / "cam" / "chunk-000" / "file-000.mp4"
    _write_video(vp, mode=mode, fps=fps)
    _postprocess(vp, mode)
    t = np.arange(64, dtype=np.float64) / fps
    rng = np.random.default_rng(3)
    action = np.cumsum(rng.normal(0, 0.1, size=(64, 4)), axis=0)
    return EpisodeData(
        episode_index=0,
        num_frames=64,
        timestamps=t,
        observation={"state": action},
        action={"joint_pos": action},
        meta={
            "dataset_root": str(root),
            "fps": fps,
            "video_features": {
                "cam": {
                    "chunk_index": 0,
                    "file_index": 0,
                    "from_timestamp": 0.0,
                    "to_timestamp": 63 / fps,
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# REQ-10: injected corruption detection (calibration anchors)
# ---------------------------------------------------------------------------

class TestReq10Detection:
    def test_blur_is_review(self, tmp_path):
        r = VisualQualityMetric().compute(_make_episode(tmp_path, mode="blurred"))
        assert r.assessment["status"] == "review"
        assert r.measurement["penalty"] >= 0.5

    def test_dark_is_review(self, tmp_path):
        r = VisualQualityMetric().compute(_make_episode(tmp_path, mode="dark"))
        assert r.assessment["status"] == "review"

    def test_blown_is_review(self, tmp_path):
        # ×3 gain clips ≥20% of pixels to 255; mean luminance stays
        # below 230 — the clipped_frac signal must catch this (the
        # v0.7.0 threshold missed it, see REQ-10 calibration note).
        r = VisualQualityMetric().compute(_make_episode(tmp_path, mode="blown"))
        assert r.assessment["status"] == "review"

    def test_clean_passes(self, tmp_path):
        r = VisualQualityMetric().compute(_make_episode(tmp_path, mode="moving"))
        assert r.assessment["status"] == "pass"

    def test_static_camera_freeze_is_exclude(self, tmp_path):
        # Whole-stream stall: adaptive epsilon must not desensitize.
        r = VideoFreezeMetric().compute(_make_episode(tmp_path, mode="static"))
        assert r.assessment["status"] == "exclude"


# ---------------------------------------------------------------------------
# REQ-11: missing-dependency semantics
# ---------------------------------------------------------------------------

class TestReq11DepMissing:
    def test_dep_missing_reason_constant(self):
        assert VIDEO_DEPS_MISSING == "video_deps_missing"

    def test_metrics_grade_na_with_reason_when_no_av(self, tmp_path, monkeypatch):
        # Simulate absent PyAV for the metric-level guard.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "av":
                raise ImportError("No module named 'av'")
            return real_import(name, *args, **kwargs)

        ep = _make_episode(tmp_path)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        for metric in (VideoFreezeMetric(), VisualQualityMetric()):
            r = metric.compute(ep)
            assert r.availability == MetricAvailability.NOT_AVAILABLE
            assert r.assessment["reason"] == "video_deps_missing"
            assert "pip install av" in (r.message or "")

    def test_report_counts_skipped_by_missing_dep(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from rda.report.json_report import _skipped_by_missing_dep

        def _fake_ep(reason: Optional[str]):
            m = SimpleNamespace(assessment={"status": "na", "reason": reason})
            return SimpleNamespace(metric_results={"video_freeze": m})

        result = SimpleNamespace(episodes={
            "0": _fake_ep("video_deps_missing"),
            "1": _fake_ep("video_deps_missing"),
            "2": _fake_ep("no_video_features"),   # different NA → not counted
            "3": _fake_ep(None),                  # pass → not counted
        })
        assert _skipped_by_missing_dep(result) == {"av": 2}

    def test_clean_report_has_empty_skipped(self, tmp_path):
        from types import SimpleNamespace

        from rda.report.json_report import _skipped_by_missing_dep

        def _fake_ep(reason: Optional[str]):
            m = SimpleNamespace(assessment={"status": "pass", "reason": reason})
            return SimpleNamespace(metric_results={"x": m})

        result = SimpleNamespace(episodes={"0": _fake_ep(None)})
        assert _skipped_by_missing_dep(result) == {}
