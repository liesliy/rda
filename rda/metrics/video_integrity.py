"""Video-frame integrity metric: MP4 frame count vs parquet frame count.

Cross-checks that the number of frames actually stored in each episode
video file matches the number of parquet rows (and the episode metadata
``length``). Mismatches here mean the visual modality is silently
misaligned with the state/action timeline — a failure mode that is
otherwise invisible until training, when frame indices go out of bounds.

This check was motivated by auditing EgoSuite/EgoDemo-Open100K
(video-first human datasets), but it applies to any LeRobot v3.0
dataset with ``dtype: video`` features.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult


# Tolerances (fraction of parquet frame count)
_SOFT_TOLERANCE = 0.01   # 1% — flag as review with explanation
_HARD_TOLERANCE = 0.05   # 5% — flag as strong mismatch


@lru_cache(maxsize=1024)
def _count_video_frames(path: Path) -> int:
    """Count frames in a video file using PyAV, falling back to OpenCV.

    Frame-count sources are tried in cheapest-to-most-expensive order:

    1. ``nb_frames`` metadata key — exact and zero-decode when the MP4
       writer populated it.
    2. ``stream.frames`` (PyAV's ``AVStream.nb_frames``) — FFmpeg fills
       this from the MP4/MOV ``stts`` box even when the metadata key is
       absent, which is the common case for LeRobot/ego4d videos. Without
       this path a video lacking the ``nb_frames`` key would fall through
       to a full decode of every frame, making ``video_frame_integrity``
       impractically slow on large datasets.
    3. Decode-count fallback — accurate for any container, but slow; only
       reached when FFmpeg cannot determine the frame count up front.

    Results are memoized because a single chunked MP4 is usually shared by
    many episodes, each of which would otherwise re-open and re-parse it.
    """
    try:
        import av  # type: ignore

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            nb = stream.metadata.get("nb_frames")
            if nb and int(nb) > 0:
                return int(nb)
            if stream.frames and int(stream.frames) > 0:
                return int(stream.frames)
            # Decode-count fallback (accurate for any container).
            return sum(1 for _ in container.decode(stream))
    except ImportError:
        pass

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
    except ImportError as exc:
        raise RuntimeError(
            "Cannot count video frames: neither 'av' (PyAV) nor 'cv2' "
            "(opencv-python) is installed. Install one of them to enable "
            "video_frame_integrity."
        ) from exc


class VideoFrameIntegrityMetric(MetricBase):
    """Verify that video files contain the same number of frames as parquet.

    Layer 1 (Data Integrity): deterministic hard check.

    Data sources, in priority order:
    1. ``episode.meta["dataset_root"]`` + ``meta["video_features"]``
       (populated by the v3.0 LeRobot loader) — resolves the actual MP4
       paths via the standard ``videos/<feature>/chunk-XXX/file-XXX.mp4``
       layout and counts real frames.
    2. If videos cannot be located or decoded, returns N/A (never a
       false fail).
    """

    name = "video_frame_integrity"
    description = (
        "Cross-check MP4 frame counts against parquet frame counts "
        "(Layer 1 integrity)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
        dataset_root = meta.get("dataset_root")

        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message=(
                    "No video features referenced by this episode; "
                    "video frame cross-check not applicable."
                ),
            )

        if not dataset_root:
            return MetricResult.make_na(
                name=self.name,
                reason="dataset_root_unknown",
                message=(
                    "Loader did not provide the dataset root; cannot "
                    "locate video files."
                ),
            )

        root = Path(dataset_root)
        fps = meta.get("fps")
        checked: List[Dict[str, Any]] = []
        mismatched: List[Dict[str, Any]] = []
        unreadable: List[str] = []

        for feature, info in sorted(video_features.items()):
            chunk = info.get("chunk_index")
            file_idx = info.get("file_index")
            if chunk is None or file_idx is None:
                unreadable.append(feature)
                continue

            video_path = (
                root
                / "videos"
                / feature
                / f"chunk-{int(chunk):03d}"
                / f"file-{int(file_idx):03d}.mp4"
            )
            if not video_path.exists():
                unreadable.append(feature)
                continue

            try:
                video_frames = _count_video_frames(video_path)
            except Exception:
                unreadable.append(feature)
                continue

            # Determine this episode's expected frame count *within* the
            # (possibly shared) chunked MP4.
            from_ts = info.get("from_timestamp")
            to_ts = info.get("to_timestamp")
            chunked = (
                from_ts is not None
                and to_ts is not None
                and fps is not None
                and math.isfinite(float(from_ts))
                and math.isfinite(float(to_ts))
            )

            if chunked:
                # Chunked video: multiple episodes share one MP4. The
                # episode's own frame span is [from_ts, to_ts) seconds, so
                # its expected frame count is (to - from) * fps — NOT the
                # whole file's frame count (which, when compared directly,
                # produced a false 100% "mismatch" on chunked datasets).
                episode_video_frames = int(round((float(to_ts) - float(from_ts)) * float(fps)))
                expected = episode_video_frames
                end_frame = int(round(float(to_ts) * float(fps)))
                file_truncated = video_frames < end_frame
            else:
                # Non-chunked (one episode = one video file): compare the
                # whole file's frame count directly.
                expected = video_frames
                file_truncated = False

            entry: Dict[str, Any] = {
                "feature": feature,
                "video_path": str(video_path.relative_to(root)),
                "video_frames": video_frames,
                "parquet_frames": episode.num_frames,
                "delta": expected - episode.num_frames,
            }
            if chunked:
                entry["chunked"] = True
                entry["from_timestamp"] = from_ts
                entry["to_timestamp"] = to_ts
                entry["episode_video_frames"] = expected
                entry["end_frame"] = end_frame
                entry["file_truncated"] = file_truncated
            meta_length = meta.get("episode_meta_length")
            if meta_length is not None:
                entry["episode_meta_length"] = meta_length
            checked.append(entry)

            # A truncated shared MP4 is a hard structural failure.
            if file_truncated:
                entry["level"] = "hard"
                entry["reason"] = "video_file_truncated"
                mismatched.append(entry)
                continue

            delta_ratio = abs(entry["delta"]) / max(episode.num_frames, 1)
            if delta_ratio > _HARD_TOLERANCE:
                entry["level"] = "hard"
                mismatched.append(entry)
            elif delta_ratio > _SOFT_TOLERANCE:
                entry["level"] = "soft"
                mismatched.append(entry)

        if not checked and not unreadable:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_files_resolved",
                message="No video files could be resolved for this episode.",
            )

        if unreadable and not checked:
            return MetricResult.make_na(
                name=self.name,
                reason="video_files_unreadable",
                message=(
                    f"Video files missing or unreadable for: "
                    f"{', '.join(unreadable)}."
                ),
                details={"unreadable_features": unreadable},
            )

        details: Dict[str, Any] = {
            "checked": checked,
            "mismatched": mismatched,
            "unreadable_features": unreadable,
            "tolerances": {"soft": _SOFT_TOLERANCE, "hard": _HARD_TOLERANCE},
        }

        if not mismatched:
            feat_summary = ", ".join(
                f"{c['feature']}={c['video_frames']}" for c in checked
            )
            return MetricResult.make_pass(
                name=self.name,
                measurement={
                    "score_compat": 1.0,
                    "checked_count": len(checked),
                    "mismatch_count": 0,
                },
                message=(
                    f"Video/parquet frame counts match across "
                    f"{len(checked)} video feature(s): {feat_summary}."
                ),
                details=details,
            )

        # Mismatch found — deterministic structural misalignment.
        worst = max(mismatched, key=lambda m: abs(m["delta"]))
        hard_count = sum(1 for m in mismatched if m["level"] == "hard")
        parts = []
        for m in mismatched:
            if m.get("chunked") and m.get("file_truncated"):
                parts.append(
                    f"{m['feature']}: file has {m['video_frames']} frames but "
                    f"episode needs up to {m['end_frame']} (truncated)"
                )
            elif m.get("chunked"):
                parts.append(
                    f"{m['feature']}: video span={m.get('episode_video_frames')} vs "
                    f"parquet={m['parquet_frames']} (delta {m['delta']:+d})"
                )
            else:
                parts.append(
                    f"{m['feature']}: video={m['video_frames']} vs "
                    f"parquet={m['parquet_frames']} (delta {m['delta']:+d})"
                )
        msg = (
            f"Video frame count mismatch in {len(mismatched)} feature(s): "
            + "; ".join(parts)
            + f"; worst delta ratio {abs(worst['delta']) / max(episode.num_frames, 1):.1%}."
        )
        return MetricResult.make_exclude(
            name=self.name,
            reason=(
                f"video_frame_mismatch ({hard_count} hard, "
                f"{len(mismatched) - hard_count} soft)"
            ),
            message=msg,
            details=details,
        )
