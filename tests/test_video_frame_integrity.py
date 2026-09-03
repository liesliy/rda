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

from rda.metrics.video_integrity import _count_video_frames  # noqa: E402


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
