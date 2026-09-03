"""Regression tests for the video frame-count fast paths.

``video_frame_integrity`` must NOT fall back to full-frame decoding for the
common MP4 case: LeRobot/ego4d videos usually lack the ``nb_frames`` metadata
key, but FFmpeg still exposes the exact count via ``stream.frames`` (read from
the MP4/MOV ``stts`` box). Before this was fixed, a video without the metadata
key caused a full decode of every frame, making the metric impractically slow
(>1h on ``lerobot/libero_10``). These tests pin the three code paths and assert
that the decode fallback is reached only when truly necessary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.metrics.video_integrity import (  # noqa: E402
    VideoFrameIntegrityMetric,
    _count_video_frames,
)
from rda.io.schema import EpisodeData  # noqa: E402


def _install_fake_av(monkeypatch, *, nb_frames, stream_frames, decode_count):
    """Install a fully-controlled fake ``av`` module.

    ``nb_frames`` is the value exposed via ``stream.metadata["nb_frames"]``
    (``None`` means the key is absent). ``stream_frames`` is
    ``stream.frames``. ``decode_count`` is how many frames the decode
    fallback would yield if reached.
    """
    stream = mock.MagicMock()
    stream.metadata = {"nb_frames": str(nb_frames)} if nb_frames else {}
    stream.frames = stream_frames

    container = mock.MagicMock()
    container.streams.video = [stream]
    container.decode.return_value = iter(range(decode_count))
    container.__enter__ = mock.MagicMock(return_value=container)
    container.__exit__ = mock.MagicMock(return_value=False)

    fake_av = mock.MagicMock()
    fake_av.open.return_value = container

    monkeypatch.setitem(sys.modules, "av", fake_av)
    return container


def test_metadata_nb_frames_path_skips_decode(tmp_path, monkeypatch):
    """When the ``nb_frames`` metadata key exists, use it, never decode."""
    container = _install_fake_av(
        monkeypatch, nb_frames=42, stream_frames=0, decode_count=0
    )
    assert _count_video_frames(tmp_path / "meta.mp4") == 42
    container.decode.assert_not_called()


def test_stream_frames_path_without_metadata_skips_decode(tmp_path, monkeypatch):
    """No metadata key but ``stream.frames`` present -> use it, never decode."""
    container = _install_fake_av(
        monkeypatch, nb_frames=None, stream_frames=101469, decode_count=0
    )
    assert _count_video_frames(tmp_path / "stts.mp4") == 101469
    container.decode.assert_not_called()


def test_decode_fallback_only_when_no_fast_source(tmp_path, monkeypatch):
    """Neither metadata nor ``stream.frames`` -> decode-count is the last resort."""
    container = _install_fake_av(
        monkeypatch, nb_frames=None, stream_frames=0, decode_count=777
    )
    assert _count_video_frames(tmp_path / "webm.mp4") == 777
    container.decode.assert_called_once()


# --- Chunked video frame-count cross-check (v3.0 shared MP4) ---
#
# In LeRobot v3.0 a *single* MP4 stores many episodes back-to-back (a
# "chunk"), so the file's total frame count is far larger than any one
# episode's frame count. Before the fix, ``video_frame_integrity`` compared
# the whole file's frame count against the episode's parquet count and
# flagged every episode as a hard mismatch (e.g. 101469 vs 214). The fix
# derives each episode's own span via ``(to_timestamp - from_timestamp) *
# fps``. These tests pin that behavior.


def _make_chunked_episode(tmp_path, *, num_frames, from_ts, to_ts, fps=10.0):
    """Build an :class:`EpisodeData` pointing at a shared chunked MP4."""
    import numpy as np

    feat = "observation.images.image"
    video_path = tmp_path / "videos" / feat / "chunk-000" / "file-000.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake-mp4")

    return EpisodeData(
        episode_index=0,
        num_frames=num_frames,
        timestamps=np.arange(num_frames, dtype=float) / fps,
        meta={
            "dataset_root": str(tmp_path),
            "fps": fps,
            "video_features": {
                feat: {
                    "chunk_index": 0,
                    "file_index": 0,
                    "from_timestamp": from_ts,
                    "to_timestamp": to_ts,
                }
            },
        },
    )


def test_chunked_video_uses_episode_span_not_file_frames(tmp_path, monkeypatch):
    """A shared MP4 with many frames must not false-fail an episode.

    The file holds 101469 frames total; the episode's own span is
    (21.4 - 0.0) * 10 = 214 frames, matching its parquet count.
    """
    ep = _make_chunked_episode(tmp_path, num_frames=214, from_ts=0.0, to_ts=21.4)

    monkeypatch.setattr(
        "rda.metrics.video_integrity._count_video_frames", lambda _p: 101469
    )

    result = VideoFrameIntegrityMetric().compute(ep)

    assert result.passed is True
    checked = result.details["checked"][0]
    assert checked["chunked"] is True
    assert checked["episode_video_frames"] == 214
    assert checked["delta"] == 0
    assert checked["file_truncated"] is False
    assert result.details["mismatched"] == []


def test_chunked_video_detects_truncated_file(tmp_path, monkeypatch):
    """If the episode's span runs past the end of the file, flag truncation."""
    ep = _make_chunked_episode(tmp_path, num_frames=214, from_ts=0.0, to_ts=21.4)

    # File only has 150 frames, but the episode span needs up to 214.
    monkeypatch.setattr(
        "rda.metrics.video_integrity._count_video_frames", lambda _p: 150
    )

    result = VideoFrameIntegrityMetric().compute(ep)

    assert result.passed is False
    checked = result.details["checked"][0]
    assert checked["file_truncated"] is True
    assert any(m["reason"] == "video_file_truncated" for m in result.details["mismatched"])
