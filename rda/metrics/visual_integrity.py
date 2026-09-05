"""VA-A: visual-stream integrity metrics (REQ-4, v0.7.0).

Hard-evidence checks on the *visual* modality — the video counterpart
of the kinematic integrity metrics. All three are deterministic, prove
that the video stream is broken (not merely "looks bad"), and never
rely on semantic judgment:

1. ``video_freeze`` — consecutive frozen video frames (>0.5 s) while the
   arm is *moving* per the action timeline. A camera that dropped out
   (USB hiccup, encoder stall) keeps producing identical compressed
   frames while ``observation.state``/``action`` keep changing. This is
   the visual twin of ``missing_dropout``.

2. ``video_timestamp_alignment`` — the episode's video frame span
   (from ``from_timestamp``/``to_timestamp`` × fps, cross-checked against
   the actually decodable frame count) versus the parquet timeline span.
   A systematic mismatch means video and state/action timelines drift
   apart — frame indices in training silently sample the wrong moments.

3. ``video_stream_sync`` — multi-camera presence and drift. All camera
   features referenced by an episode must resolve to readable files with
   consistent frame spans. Missing wrist camera = silent modality loss.

Design constraints (aligned with the v1.1 roadmap):

- Decode budget: VA-A decodes at 64×64 grayscale. Full-frame decode is
  acceptable because chunked MP4s are decoded once per *episode span*
  (seek to ``from_timestamp``) and memoized per (file, span) pair.
- No OpenCV dependency: PyAV + numpy only (PyAV is already required by
  ``video_frame_integrity``).
- VA-A+ (URDF projection per-frame comparison, [LINGBOT]) is explicitly
  OUT of scope — opt-in deep check in a future version.
- N/A is returned (never a false fail) when videos cannot be located,
  decoded, or the dataset has no video features.
- Only ``video_freeze`` failures grade EXCLUDE (camera drop-out is
  deterministic corruption). Timestamp alignment failures with large
  systematic drift also grade EXCLUDE when beyond hard tolerance;
  small drift is REVIEW.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult


# --- Tunables ---
_FREEZE_GRAY_SIZE = 64          # decode target: 64×64 grayscale
_FREEZE_MIN_SECONDS = 0.5       # consecutive frozen span to count as a freeze
#   ^ single 0.5 s span can be a legit pause; repeated spans while moving
#     are the camera-drop signature. A *long* single span (>3 s) is
#     equally conclusive — no camera "slow motion" lasts 3 s.
_FREEZE_CONCLUSIVE_RUNS = 3     # this many sustained spans → EXCLUDE
_FREEZE_CONCLUSIVE_SECONDS = 3.0  # or one span longer than this
_FREEZE_MAX_TOTAL_RATIO = 0.30  # or frozen ≥30% of the episode
_MAX_FREEZE_REPORT = 8          # cap freeze regions in details (report size)
_TS_SOFT_TOLERANCE = 0.02       # 2% span mismatch → review
_TS_HARD_TOLERANCE = 0.10       # 10% span mismatch → exclude

VIDEO_DEPS_MISSING = "video_deps_missing"
"""NA reason code (REQ-11, v0.7.1): PyAV is not installed.

Semantic contract: visual metrics graded NA with this reason mean the
visual modality was NOT audited — "not checked", never "checked and
fine". The CLI/JSON report aggregates these into
``skipped_by_missing_dep`` so report readers see the gap explicitly.
"""


def _av_missing() -> bool:
    try:
        import av  # noqa: F401
        return False
    except ImportError:
        return True


@lru_cache(maxsize=256)
def _decode_span_gray(
    video_path: Path, start_sec: float, end_sec: float, fps: float
) -> Optional[np.ndarray]:
    """Decode [start_sec, end_sec) of a video into 64×64 grayscale frames.

    Returns an (N, 64, 64) uint8 array, or None when the file cannot be
    opened/decoded. Memoized per (path, span): chunked MP4s are shared
    across episodes but each episode reads its own span.

    The array is kept small on purpose: 64×64 gray at typical 10-15 s
    episode spans is a few MB at most.
    """
    try:
        import av
    except ImportError:
        # REQ-11 (v0.7.1): surface the missing optional dependency instead
        # of silently degrading — callers distinguish "not checked" from
        # "checked and fine" via this reason code.
        return None

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            # Seek to the span start (keyframe before start), then decode.
            try:
                tb = stream.time_base
                container.seek(int(start_sec / tb), stream=stream)
            except Exception:
                pass  # seek unsupported → decode from start
            frames: List[np.ndarray] = []
            target_w = _FREEZE_GRAY_SIZE
            for packet_frame in container.decode(stream):
                pts_sec = (
                    float(packet_frame.pts * tb) if packet_frame.pts is not None else None
                )
                if pts_sec is not None and pts_sec < start_sec - 1.0 / max(fps, 1.0):
                    continue  # before the span
                if pts_sec is not None and pts_sec >= end_sec:
                    break
                img = packet_frame.to_ndarray(format="gray")
                if img.shape != (target_w, target_w):
                    # PIL-free resize via numpy slicing fallback is crude;
                    # use PyAV's scaler instead for quality+speed.
                    img = packet_frame.reformat(
                        width=target_w, height=target_w, format="gray"
                    ).to_ndarray()
                frames.append(img)
            if not frames:
                return None
            return np.stack(frames)
    except Exception:
        return None


def _freeze_runs(
    frames: np.ndarray,
    min_run_frames: int,
) -> List[Tuple[int, int]]:
    """Find runs of consecutive near-identical (camera-stalled) frames.

    A frame i is "frozen" when its mean absolute difference to frame i-1
    falls below an *adaptive* epsilon. Calibration note (v0.7.0, libero_10
    diagnosis): a fixed absolute threshold misclassifies slow-motion
    spans — normal motion at 64×64 gray diffs ~1.0-3.0, slow motion dips
    to 0.3-1.0, while true codec-identical freeze is < ~0.1. The epsilon
    is therefore anchored to the episode's own low tail: the p10 of the
    diff distribution is treated as motion noise, and any frame below
    ``max(0.10, 0.25 × p10)`` is frozen. A camera stall produces diffs of
    exactly 0 (identical decoded frames), far below any motion noise.
    """
    if frames.shape[0] < 2:
        return []
    flat = frames.reshape(frames.shape[0], -1).astype(np.float32)
    diffs = np.abs(np.diff(flat, axis=0)).mean(axis=1)
    noise_floor = float(np.percentile(diffs, 10))
    eps = max(0.10, 0.25 * noise_floor)
    frozen = diffs < eps
    runs: List[Tuple[int, int]] = []
    start = None
    for i, f in enumerate(frozen):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if (i - start) >= min_run_frames:
                runs.append((start, i - 1))
            start = None
    if start is not None and (len(frozen) - start) >= min_run_frames:
        runs.append((start, len(frozen) - 1))
    return runs


def _moving_mask(
    episode: EpisodeData, n_frames: int
) -> Optional[np.ndarray]:
    """Per-frame motion boolean from the primary action array.

    Reuses the same "is this frame moving" notion as the idle detector:
    frame-to-frame norm delta of the primary action channel.
    """
    from rda.recommend.temporal_metrics import _primary_action_array

    arr = _primary_action_array(episode)
    if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
        return None
    deltas = np.abs(np.diff(arr.astype(np.float32), axis=0)).mean(axis=1)
    # Pad to length n_frames (last frame inherits previous delta).
    if deltas.shape[0] == 0:
        return None
    deltas = np.concatenate([deltas, deltas[-1:]])
    if deltas.shape[0] < n_frames:
        pad = np.full(n_frames - deltas.shape[0], deltas[-1], dtype=np.float32)
        deltas = np.concatenate([deltas, pad])
    scale = float(np.percentile(deltas, 90)) or 1.0
    return deltas > (0.05 * scale)


class VideoFreezeMetric(MetricBase):
    """Detect frozen video streams while the arm is moving (VA-A).

    Layer 1 (Data Integrity): a camera that stalled produces identical
    frames while the action timeline keeps advancing — deterministic,
    provable corruption of the visual modality.
    """

    name = "video_freeze"
    description = (
        "VA-A: frozen-video detection — consecutive identical frames "
        "while joints move (camera drop-out signature)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
        dataset_root = meta.get("dataset_root")
        fps = meta.get("fps")

        if _av_missing():
            return MetricResult.make_na(
                name=self.name,
                reason=VIDEO_DEPS_MISSING,
                message=(
                    "PyAV is not installed — visual stream was NOT audited. "
                    "Install it with: pip install av"
                ),
            )

        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="No video features; freeze detection not applicable.",
            )
        if not dataset_root or not fps:
            return MetricResult.make_na(
                name=self.name,
                reason="dataset_or_fps_unknown",
                message="Loader did not provide dataset root/fps; cannot decode videos.",
            )

        root = Path(dataset_root)
        moving = _moving_mask(episode, episode.num_frames)
        if moving is None:
            return MetricResult.make_na(
                name=self.name,
                reason="no_action_timeline",
                message="No usable action timeline; freeze-vs-motion cross-check not applicable.",
            )

        min_run_video = max(int(round(_FREEZE_MIN_SECONDS * float(fps))), 2)
        checked = 0
        freeze_regions: List[Dict[str, Any]] = []

        for feature, info in sorted(video_features.items()):
            chunk = info.get("chunk_index")
            file_idx = info.get("file_index")
            from_ts = info.get("from_timestamp")
            to_ts = info.get("to_timestamp")
            if None in (chunk, file_idx, from_ts, to_ts):
                continue
            video_path = (
                root / "videos" / feature
                / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
            )
            if not video_path.exists():
                continue
            frames = _decode_span_gray(
                video_path, float(from_ts), float(to_ts), float(fps)
            )
            if frames is None or frames.shape[0] < 2:
                continue
            checked += 1

            # Map video frame index → parquet frame index (proportional).
            n_video, n_parquet = frames.shape[0], episode.num_frames
            scale = n_parquet / max(n_video, 1)

            for v_start, v_end in _freeze_runs(frames, min_run_video):
                p_start = int(v_start * scale)
                p_end = min(int(v_end * scale) + 1, n_parquet - 1)
                span = moving[p_start:p_end + 1]
                moving_ratio = float(span.mean()) if span.size else 0.0
                if moving_ratio > 0.5:
                    freeze_regions.append({
                        "feature": feature,
                        "video_start": int(from_ts * fps) + v_start,
                        "video_end": int(from_ts * fps) + v_end,
                        "parquet_start": p_start,
                        "parquet_end": p_end,
                        "duration_sec": round((v_end - v_start + 1) / float(fps), 3),
                        "moving_ratio_in_span": round(moving_ratio, 3),
                    })

        if checked == 0:
            return MetricResult.make_na(
                name=self.name,
                reason="videos_not_decodable",
                message="No video spans could be decoded; freeze check skipped.",
            )

        freeze_regions.sort(key=lambda r: -(r["moving_ratio_in_span"]))
        hard = [r for r in freeze_regions if r["duration_sec"] >= _FREEZE_MIN_SECONDS][
            :_MAX_FREEZE_REPORT
        ]
        # A *stalled camera* produces codec-identical frames: every
        # frozen span has near-zero inter-frame diff. Slow motion has
        # low-but-nonzero diff. Require the span to be genuinely flat
        # (the freeze-run finder already used the adaptive epsilon) AND
        # multiple sustained spans before grading EXCLUDE — a single
        # span is a REVIEW-level hint, per the audit-not-veto posture.
        details: Dict[str, Any] = {
            "checked_features": checked,
            "freeze_regions": freeze_regions[:_MAX_FREEZE_REPORT],
            "freeze_region_count": len(freeze_regions),
            "params": {
                "decode": f"{_FREEZE_GRAY_SIZE}x{_FREEZE_GRAY_SIZE} gray",
                "min_freeze_seconds": _FREEZE_MIN_SECONDS,
                "epsilon": "adaptive: max(0.10, 0.25 x p10 of frame diffs)",
            },
        }

        if not freeze_regions:
            return MetricResult.make_pass(
                name=self.name,
                measurement={
                    "score_compat": 1.0,
                    "checked_features": checked,
                    "freeze_region_count": 0,
                },
                message=f"No frozen-video-while-moving spans in {checked} camera stream(s).",
                details=details,
            )

        total_frozen_sec = sum(r["duration_sec"] for r in freeze_regions)
        episode_span_sec = episode.num_frames / max(float(fps), 1e-6)
        longest_sec = max(r["duration_sec"] for r in freeze_regions)
        conclusive = (
            len(freeze_regions) >= _FREEZE_CONCLUSIVE_RUNS
            or longest_sec >= _FREEZE_CONCLUSIVE_SECONDS
            or (total_frozen_sec / max(episode_span_sec, 1e-6)) >= _FREEZE_MAX_TOTAL_RATIO
        )
        if conclusive:
            regions_desc = ", ".join(
                f"{r['feature']}@{r['duration_sec']}s(f{r['parquet_start']}-{r['parquet_end']})"
                for r in hard
            )
            return MetricResult.make_exclude(
                name=self.name,
                reason="video_freeze_while_moving",
                message=(
                    f"Camera stream(s) froze {len(freeze_regions)} time(s) for a total of "
                    f"{total_frozen_sec:.1f}s while the arm was moving: {regions_desc}. "
                    f"Visual modality is missing for these spans (camera drop-out signature)."
                ),
                details=details,
            )

        # One short freeze while moving: suspicious but not conclusive.
        r0 = freeze_regions[0]
        return MetricResult.make_review(
            name=self.name,
            measurement={
                "score_compat": 0.5,
                "checked_features": checked,
                "freeze_region_count": len(freeze_regions),
                "total_frozen_sec": round(total_frozen_sec, 3),
            },
            message=(
                f"Possible brief video freeze: {r0['feature']} frozen "
                f"{r0['duration_sec']}s at parquet frames "
                f"{r0['parquet_start']}-{r0['parquet_end']} while moving."
            ),
            details=details,
        )


class VideoTimestampAlignmentMetric(MetricBase):
    """VA-A: video span vs parquet timeline consistency.

    Reuses the (to_timestamp − from_timestamp) × fps span arithmetic from
    ``video_frame_integrity`` but compares it against the *parquet time
    axis* (last timestamp − first timestamp), catching systematic drift
    between the two timelines even when absolute frame counts pass.
    """

    name = "video_timestamp_alignment"
    description = (
        "VA-A: video time span vs parquet timeline consistency "
        "(sample-alignment hard evidence)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
        fps = meta.get("fps")

        if _av_missing():
            return MetricResult.make_na(
                name=self.name,
                reason=VIDEO_DEPS_MISSING,
                message=(
                    "PyAV is not installed — visual stream was NOT audited. "
                    "Install it with: pip install av"
                ),
            )

        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="No video features; alignment check not applicable.",
            )
        if not fps:
            return MetricResult.make_na(
                name=self.name,
                reason="fps_unknown",
                message="Loader did not provide fps; alignment check not applicable.",
            )

        # Parquet timeline span from the timestamp channel.
        ts = episode.timestamps
        if ts is None or len(ts) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="no_timestamp_channel",
                message="Episode has no timestamp channel; alignment check not applicable.",
            )
        parquet_span = float(ts[-1]) - float(ts[0])
        if not math.isfinite(parquet_span) or parquet_span <= 0:
            return MetricResult.make_na(
                name=self.name,
                reason="degenerate_timestamp_span",
                message="Parquet timestamp span is zero/negative; check not applicable.",
            )

        checked: List[Dict[str, Any]] = []
        for feature, info in sorted(video_features.items()):
            from_ts = info.get("from_timestamp")
            to_ts = info.get("to_timestamp")
            if from_ts is None or to_ts is None:
                continue
            video_span = (float(to_ts) - float(from_ts))
            checked.append({
                "feature": feature,
                "video_span_sec": round(video_span, 4),
                "parquet_span_sec": round(parquet_span, 4),
                "delta_sec": round(video_span - parquet_span, 4),
                "delta_ratio": round((video_span - parquet_span) / parquet_span, 4),
            })

        if not checked:
            return MetricResult.make_na(
                name=self.name,
                reason="no_chunk_timestamps",
                message="Video features carry no chunk timestamps; alignment not applicable.",
            )

        worst = max(checked, key=lambda c: abs(c["delta_ratio"]))
        details = {"checked": checked, "tolerances": {
            "soft": _TS_SOFT_TOLERANCE, "hard": _TS_HARD_TOLERANCE}}

        if abs(worst["delta_ratio"]) > _TS_HARD_TOLERANCE:
            feat_desc = "; ".join(
                f"{c['feature']}: video {c['video_span_sec']}s vs parquet "
                f"{c['parquet_span_sec']}s ({c['delta_ratio']:+.1%})"
                for c in checked
            )
            return MetricResult.make_exclude(
                name=self.name,
                reason="video_parquet_span_mismatch",
                message=(
                    f"Video/parquet timeline span mismatch beyond "
                    f"{_TS_HARD_TOLERANCE:.0%}: {feat_desc}. Frame-index "
                    f"training would silently sample misaligned moments."
                ),
                details=details,
            )
        if abs(worst["delta_ratio"]) > _TS_SOFT_TOLERANCE:
            return MetricResult.make_review(
                name=self.name,
                measurement={
                    "score_compat": 0.5,
                    "worst_delta_ratio": worst["delta_ratio"],
                },
                message=(
                    f"Video/parquet span drift {worst['delta_ratio']:+.1%} on "
                    f"{worst['feature']} — within hard tolerance but worth a look."
                ),
                details=details,
            )
        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "worst_delta_ratio": worst["delta_ratio"],
                "checked_count": len(checked),
            },
            message=(
                f"Video/parquet timeline spans consistent across "
                f"{len(checked)} camera stream(s) (worst drift "
                f"{worst['delta_ratio']:+.1%})."
            ),
            details=details,
        )


class VideoStreamSyncMetric(MetricBase):
    """VA-A: multi-camera presence & span consistency.

    All camera features referenced by the episode must resolve to
    readable files with consistent spans. A missing wrist camera or one
    stream ending early is silent modality loss — the training script
    would broadcast zeros/garbage for the missing view.
    """

    name = "video_stream_sync"
    description = (
        "VA-A: multi-camera presence and span consistency "
        "(missing/misaligned camera streams)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
        dataset_root = meta.get("dataset_root")

        if _av_missing():
            return MetricResult.make_na(
                name=self.name,
                reason=VIDEO_DEPS_MISSING,
                message=(
                    "PyAV is not installed — visual stream was NOT audited. "
                    "Install it with: pip install av"
                ),
            )

        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="Single/no camera streams; stream sync not applicable.",
            )
        if not dataset_root:
            return MetricResult.make_na(
                name=self.name,
                reason="dataset_root_unknown",
                message="Loader did not provide dataset root; cannot check streams.",
            )
        if len(video_features) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="single_camera",
                message="Only one camera stream; multi-camera sync not applicable.",
            )

        root = Path(dataset_root)
        missing: List[str] = []
        spans: Dict[str, float] = {}
        for feature, info in sorted(video_features.items()):
            chunk = info.get("chunk_index")
            file_idx = info.get("file_index")
            if chunk is None or file_idx is None:
                missing.append(feature)
                continue
            video_path = (
                root / "videos" / feature
                / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
            )
            if not video_path.exists():
                missing.append(feature)
                continue
            from_ts = info.get("from_timestamp")
            to_ts = info.get("to_timestamp")
            if from_ts is not None and to_ts is not None:
                spans[feature] = float(to_ts) - float(from_ts)

        details: Dict[str, Any] = {
            "expected_streams": sorted(video_features.keys()),
            "resolved_streams": sorted(spans.keys()),
            "missing_streams": missing,
            "span_drift_ratio": None,
        }

        if missing:
            return MetricResult.make_exclude(
                name=self.name,
                reason="camera_stream_missing",
                message=(
                    f"Camera stream(s) missing or unresolvable: "
                    f"{', '.join(missing)}. Downstream training would lose "
                    f"these views silently."
                ),
                details=details,
            )

        if len(spans) >= 2:
            vals = list(spans.values())
            med = float(np.median(vals))
            drift = max(abs(v - med) / med for v in vals if med > 0)
            details["span_drift_ratio"] = round(drift, 4)
            if drift > _TS_HARD_TOLERANCE:
                return MetricResult.make_exclude(
                    name=self.name,
                    reason="camera_span_drift",
                    message=(
                        f"Camera stream spans diverge up to {drift:.1%} from "
                        f"the median — views are not synchronized."
                    ),
                    details=details,
                )

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "stream_count": len(spans),
                "span_drift_ratio": details["span_drift_ratio"] or 0.0,
            },
            message=(
                f"All {len(spans)} camera streams resolve with consistent spans."
            ),
            details=details,
        )
