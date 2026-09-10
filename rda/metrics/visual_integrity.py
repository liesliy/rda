"""VA-A: visual-stream integrity metrics (v0.9).

Hard-evidence checks on the *visual* modality — the video counterpart
of the kinematic integrity metrics. All are deterministic, prove
that the video stream is broken (not merely "looks bad"), and never
rely on semantic judgment.

v0.9 changes:
  video_stream_sync (v0.8) has been split into four independent metrics:

  1. ``video_stream_presence`` (Layer 1 / CRITICAL) — all required camera
     streams exist and are readable. Missing stream = silent modality loss.
  2. ``video_stream_span_consistency`` (Layer 2 / Diagnostic) — per-camera
     video spans are consistent with each other.
  3. ``video_stream_temporal_offset`` (Layer 2 / Diagnostic) — frame-level
     pairwise temporal offset between camera streams.
  4. ``video_stream_temporal_drift`` (Layer 2 / Diagnostic) — clock drift
     rate between camera streams over the episode.

  Existing metrics (unchanged):
  5. ``video_freeze`` — consecutive frozen video frames while the arm moves.
  6. ``video_timestamp_alignment`` — video span vs parquet timeline span.

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
"""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult


# --- Tunables ---
_FREEZE_GRAY_SIZE = 64          # decode target: 64×64 grayscale
_FREEZE_MIN_SECONDS = 0.5       # consecutive frozen span to count as a freeze
_FREEZE_CONCLUSIVE_RUNS = 3     # this many sustained spans → EXCLUDE
_FREEZE_CONCLUSIVE_SECONDS = 3.0  # or one span longer than this
_FREEZE_MAX_TOTAL_RATIO = 0.30  # or frozen ≥30% of the episode
_MAX_FREEZE_REPORT = 8          # cap freeze regions in details (report size)
_TS_SOFT_TOLERANCE = 0.02       # 2% span mismatch → review
_TS_HARD_TOLERANCE = 0.10       # 10% span mismatch → exclude (legacy; span_consistency is diagnostic by default)
_DRIFT_WINDOW_SEC = 5.0         # window size for temporal drift computation

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
    """
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            try:
                tb = stream.time_base
                container.seek(int(start_sec / tb), stream=stream)
            except Exception:
                pass
            frames: List[np.ndarray] = []
            target_w = _FREEZE_GRAY_SIZE
            for packet_frame in container.decode(stream):
                pts_sec = (
                    float(packet_frame.pts * tb) if packet_frame.pts is not None else None
                )
                if pts_sec is not None and pts_sec < start_sec - 1.0 / max(fps, 1.0):
                    continue
                if pts_sec is not None and pts_sec >= end_sec:
                    break
                img = packet_frame.to_ndarray(format="gray")
                if img.shape != (target_w, target_w):
                    img = packet_frame.reformat(
                        width=target_w, height=target_w, format="gray"
                    ).to_ndarray()
                frames.append(img)
            if not frames:
                return None
            return np.stack(frames)
    except Exception:
        return None


@lru_cache(maxsize=256)
def _get_frame_timestamps(
    video_path: Path, start_sec: float, end_sec: float
) -> Optional[np.ndarray]:
    """Extract per-frame PTS timestamps (in seconds) from a video span.

    Returns a 1-D float64 array of timestamps, or None when the file
    cannot be opened/decoded or frames lack PTS values.
    Memoized per (path, start, end).
    """
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base)
            try:
                container.seek(int(start_sec / tb), stream=stream)
            except Exception:
                pass
            timestamps: List[float] = []
            for packet_frame in container.decode(stream):
                if packet_frame.pts is None:
                    continue
                pts_sec = float(packet_frame.pts) * tb
                if pts_sec < start_sec - 0.01:
                    continue
                if pts_sec >= end_sec:
                    break
                timestamps.append(pts_sec)
            if not timestamps:
                return None
            return np.array(timestamps, dtype=np.float64)
    except Exception:
        return None


def _freeze_runs(
    frames: np.ndarray,
    min_run_frames: int,
) -> List[Tuple[int, int]]:
    """Find runs of consecutive near-identical (camera-stalled) frames."""
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
    """Per-frame motion boolean from the primary action array."""
    from rda.recommend.temporal_metrics import _primary_action_array

    arr = _primary_action_array(episode)
    if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
        return None
    deltas = np.abs(np.diff(arr.astype(np.float32), axis=0)).mean(axis=1)
    if deltas.shape[0] == 0:
        return None
    deltas = np.concatenate([deltas, deltas[-1:]])
    if deltas.shape[0] < n_frames:
        pad = np.full(n_frames - deltas.shape[0], deltas[-1], dtype=np.float32)
        deltas = np.concatenate([deltas, pad])
    scale = float(np.percentile(deltas, 90)) or 1.0
    return deltas > (0.05 * scale)


def _resolve_video_path(
    root: Path, feature: str, info: Dict[str, Any]
) -> Optional[Path]:
    """Resolve the video file path for a feature. Returns None if unresolvable."""
    chunk = info.get("chunk_index")
    file_idx = info.get("file_index")
    if chunk is None or file_idx is None:
        return None
    video_path = (
        root / "videos" / feature
        / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
    )
    return video_path if video_path.exists() else None


def _pairwise_nearest_offsets(
    ts_a: np.ndarray, ts_b: np.ndarray
) -> np.ndarray:
    """For each timestamp in ts_a, find the nearest timestamp in ts_b.

    Returns array of offsets (ts_a_i - nearest_ts_b_i) in seconds.
    """
    if len(ts_a) == 0 or len(ts_b) == 0:
        return np.array([])
    indices = np.searchsorted(ts_b, ts_a, side="left")
    offsets = np.empty(len(ts_a), dtype=np.float64)
    for i, idx in enumerate(indices):
        candidates = []
        if idx < len(ts_b):
            candidates.append(ts_b[idx])
        if idx > 0:
            candidates.append(ts_b[idx - 1])
        if candidates:
            nearest = min(candidates, key=lambda t: abs(t - ts_a[i]))
            offsets[i] = ts_a[i] - nearest
        else:
            offsets[i] = np.nan
    return offsets


# =========================================================================
# 1. VideoStreamPresenceMetric (Layer 1 — CRITICAL)
# =========================================================================

class VideoStreamPresenceMetric(MetricBase):
    """Check that all required camera streams exist and are readable.

    Layer 1 (Data Integrity): a missing or unreadable camera stream means
    silent modality loss — the training script would broadcast zeros/garbage
    for the missing view.
    """

    name = "video_stream_presence"
    description = (
        "Multi-camera presence check — all required camera streams must "
        "exist and be readable (silent modality loss detection)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
        dataset_root = meta.get("dataset_root")

        if _av_missing():
            return MetricResult.make_na(
                name=self.name,
                reason=VIDEO_DEPS_MISSING,
                message="PyAV is not installed — visual stream was NOT audited.",
            )
        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="No video features; stream presence check not applicable.",
            )
        if not dataset_root:
            return MetricResult.make_na(
                name=self.name,
                reason="dataset_root_unknown",
                message="Loader did not provide dataset root; cannot check streams.",
            )

        root = Path(dataset_root)
        missing: List[str] = []
        verified: List[str] = []

        for feature, info in sorted(video_features.items()):
            video_path = _resolve_video_path(root, feature, info)
            if video_path is None:
                missing.append(feature)
                continue
            # Try to verify the file is readable (at least openable)
            try:
                import av
                with av.open(str(video_path)) as container:
                    if len(container.streams.video) == 0:
                        missing.append(feature)
                        continue
                verified.append(feature)
            except Exception:
                missing.append(feature)

        details: Dict[str, Any] = {
            "expected_streams": sorted(video_features.keys()),
            "verified": verified,
            "missing": missing,
        }

        if missing:
            return MetricResult.make_exclude(
                name=self.name,
                reason="camera_stream_missing",
                message=(
                    f"Camera stream(s) missing or unreadable: "
                    f"{', '.join(missing)}. Downstream training would lose "
                    f"these views silently."
                ),
                details=details,
            )

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "stream_count": len(verified),
            },
            message=f"All {len(verified)} camera stream(s) present and readable.",
            details=details,
        )


# =========================================================================
# 2. VideoStreamSpanConsistencyMetric (Layer 2 — Diagnostic)
# =========================================================================

class VideoStreamSpanConsistencyMetric(MetricBase):
    """Check multi-camera span consistency (diagnostic).

    Compares per-camera video durations (to_timestamp - from_timestamp)
    to detect if streams have significantly different spans. Default:
    measurement only, does not affect verdict.
    """

    name = "video_stream_span_consistency"
    description = (
        "Multi-camera span consistency — checks whether video spans "
        "across cameras are consistent (diagnostic, does not affect verdict by default)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}

        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="No video features; span consistency check not applicable.",
            )
        if len(video_features) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="single_camera",
                message="Only one camera stream; span consistency not applicable.",
            )

        spans: Dict[str, float] = {}
        for feature, info in sorted(video_features.items()):
            from_ts = info.get("from_timestamp")
            to_ts = info.get("to_timestamp")
            if from_ts is not None and to_ts is not None:
                spans[feature] = float(to_ts) - float(from_ts)

        if len(spans) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="insufficient_span_data",
                message="Fewer than 2 streams have timestamp data; cannot compare spans.",
            )

        vals = list(spans.values())
        median_span = float(np.median(vals))
        drifts: Dict[str, float] = {}
        for feat, span in spans.items():
            if median_span > 0:
                drifts[feat] = round(abs(span - median_span) / median_span, 4)
            else:
                drifts[feat] = 0.0
        max_drift = max(drifts.values()) if drifts else 0.0

        measurement = {
            "spans_sec": {k: round(v, 4) for k, v in spans.items()},
            "median_span_sec": round(median_span, 4),
            "drifts": drifts,
            "max_drift": round(max_drift, 4),
        }

        return MetricResult.make_pass(
            name=self.name,
            measurement=measurement,
            message=(
                f"Span consistency: median={median_span:.2f}s, "
                f"max drift={max_drift:.2%} across {len(spans)} streams."
            ),
            details={"drift_threshold_note": "diagnostic only — does not affect verdict by default"},
        )


# =========================================================================
# 3. VideoStreamTemporalOffsetMetric (Layer 2 — Diagnostic)
# =========================================================================

class VideoStreamTemporalOffsetMetric(MetricBase):
    """Frame-level pairwise temporal offset between camera streams.

    For each pair of cameras, extracts per-frame PTS timestamps and
    computes nearest-neighbor offsets. Reports median, p95, p99, max,
    std, and signed median per pair.
    """

    name = "video_stream_temporal_offset"
    description = (
        "Frame-level pairwise temporal offset between camera streams "
        "(diagnostic measurement)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
        dataset_root = meta.get("dataset_root")

        if _av_missing():
            return MetricResult.make_na(
                name=self.name,
                reason=VIDEO_DEPS_MISSING,
                message="PyAV is not installed — temporal offset not audited.",
            )
        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="No video features; temporal offset not applicable.",
            )
        if not dataset_root:
            return MetricResult.make_na(
                name=self.name,
                reason="dataset_root_unknown",
                message="Loader did not provide dataset root.",
            )
        if len(video_features) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="single_camera",
                message="Only one camera stream; pairwise offset not applicable.",
            )

        root = Path(dataset_root)

        # Extract per-frame timestamps for each stream
        stream_timestamps: Dict[str, np.ndarray] = {}
        for feature, info in sorted(video_features.items()):
            video_path = _resolve_video_path(root, feature, info)
            if video_path is None:
                continue
            from_ts = info.get("from_timestamp", 0.0)
            to_ts = info.get("to_timestamp", 0.0)
            if from_ts is None or to_ts is None:
                continue
            ts = _get_frame_timestamps(video_path, float(from_ts), float(to_ts))
            if ts is not None and len(ts) > 0:
                stream_timestamps[feature] = ts

        if len(stream_timestamps) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="insufficient_frame_timestamps",
                message=(
                    "Fewer than 2 streams have frame-level timestamps; "
                    "cannot compute pairwise offset. "
                    "Suggestion: re-record with hardware-triggered cameras "
                    "or enable per-frame PTS in video encoder."
                ),
            )

        # Compute pairwise offsets
        pairwise: Dict[str, Dict[str, float]] = {}
        all_p95: List[float] = []
        all_max: List[float] = []

        features = sorted(stream_timestamps.keys())
        for feat_a, feat_b in combinations(features, 2):
            ts_a = stream_timestamps[feat_a]
            ts_b = stream_timestamps[feat_b]
            offsets = _pairwise_nearest_offsets(ts_a, ts_b)
            if len(offsets) == 0:
                continue

            offsets_ms = offsets * 1000.0  # convert to ms
            abs_offsets_ms = np.abs(offsets_ms)
            pair_name = f"{feat_a}_vs_{feat_b}"
            pair_stats = {
                "offset_median_ms": round(float(np.median(abs_offsets_ms)), 2),
                "offset_p95_ms": round(float(np.percentile(abs_offsets_ms, 95)), 2),
                "offset_p99_ms": round(float(np.percentile(abs_offsets_ms, 99)), 2),
                "offset_max_ms": round(float(np.max(abs_offsets_ms)), 2),
                "offset_std_ms": round(float(np.std(offsets_ms)), 2),
                "signed_median_ms": round(float(np.median(offsets_ms)), 2),
            }
            pairwise[pair_name] = pair_stats
            all_p95.append(pair_stats["offset_p95_ms"])
            all_max.append(pair_stats["offset_max_ms"])

        if not pairwise:
            return MetricResult.make_na(
                name=self.name,
                reason="no_pairwise_data",
                message="Could not compute pairwise offsets for any stream pair.",
            )

        worst_p95 = max(all_p95)
        worst_max = max(all_max)

        measurement = {
            "measured": True,
            "pairwise": pairwise,
            "worst_p95_offset_ms": round(worst_p95, 2),
            "worst_max_offset_ms": round(worst_max, 2),
        }

        return MetricResult.make_pass(
            name=self.name,
            measurement=measurement,
            message=(
                f"Temporal offset: worst p95={worst_p95:.1f}ms, "
                f"worst max={worst_max:.1f}ms across {len(pairwise)} pair(s)."
            ),
        )


# =========================================================================
# 4. VideoStreamTemporalDriftMetric (Layer 2 — Diagnostic)
# =========================================================================

class VideoStreamTemporalDriftMetric(MetricBase):
    """Clock drift rate between camera streams over the episode.

    Splits the episode into time windows, computes pairwise offset per
    window, fits a linear trend to detect systematic clock drift.
    Reports drift rate in ms/min per stream pair.
    """

    name = "video_stream_temporal_drift"
    description = (
        "Multi-camera clock drift rate over the episode "
        "(diagnostic measurement)."
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
                message="PyAV is not installed — temporal drift not audited.",
            )
        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="No video features; temporal drift not applicable.",
            )
        if not dataset_root:
            return MetricResult.make_na(
                name=self.name,
                reason="dataset_root_unknown",
                message="Loader did not provide dataset root.",
            )
        if len(video_features) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="single_camera",
                message="Only one camera stream; drift not applicable.",
            )

        root = Path(dataset_root)

        # Extract per-frame timestamps for each stream
        stream_timestamps: Dict[str, np.ndarray] = {}
        for feature, info in sorted(video_features.items()):
            video_path = _resolve_video_path(root, feature, info)
            if video_path is None:
                continue
            from_ts = info.get("from_timestamp", 0.0)
            to_ts = info.get("to_timestamp", 0.0)
            if from_ts is None or to_ts is None:
                continue
            ts = _get_frame_timestamps(video_path, float(from_ts), float(to_ts))
            if ts is not None and len(ts) > 0:
                stream_timestamps[feature] = ts

        if len(stream_timestamps) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="insufficient_frame_timestamps",
                message="Fewer than 2 streams with frame timestamps; cannot compute drift.",
            )

        # Determine episode time range
        all_starts = [ts[0] for ts in stream_timestamps.values()]
        all_ends = [ts[-1] for ts in stream_timestamps.values()]
        episode_start = float(min(all_starts))
        episode_end = float(max(all_ends))
        episode_duration = episode_end - episode_start

        if episode_duration < _DRIFT_WINDOW_SEC * 2:
            return MetricResult.make_na(
                name=self.name,
                reason="episode_too_short",
                message=(
                    f"Episode duration ({episode_duration:.1f}s) is too short "
                    f"for drift analysis (need ≥ {_DRIFT_WINDOW_SEC * 2:.0f}s)."
                ),
            )

        # Compute per-window offsets and fit linear trend
        features = sorted(stream_timestamps.keys())
        pairwise: Dict[str, Dict[str, float]] = {}

        for feat_a, feat_b in combinations(features, 2):
            ts_a = stream_timestamps[feat_a]
            ts_b = stream_timestamps[feat_b]

            # Slide windows
            window_centers: List[float] = []
            window_offsets: List[float] = []

            t = episode_start
            while t + _DRIFT_WINDOW_SEC <= episode_end:
                w_start = t
                w_end = t + _DRIFT_WINDOW_SEC
                mask_a = (ts_a >= w_start) & (ts_a < w_end)
                mask_b = (ts_b >= w_start) & (ts_b < w_end)
                if mask_a.sum() >= 3 and mask_b.sum() >= 3:
                    offsets = _pairwise_nearest_offsets(ts_a[mask_a], ts_b[mask_b])
                    valid = offsets[~np.isnan(offsets)]
                    if len(valid) > 0:
                        window_centers.append(w_start + _DRIFT_WINDOW_SEC / 2)
                        window_offsets.append(float(np.median(np.abs(valid))) * 1000.0)  # ms
                t += _DRIFT_WINDOW_SEC

            if len(window_centers) < 3:
                pairwise[f"{feat_a}_vs_{feat_b}"] = {
                    "drift_rate_ms_per_min": None,
                    "r_squared": None,
                    "note": "insufficient windows for trend fitting",
                }
                continue

            # Linear fit: offset(t) = a + b * t
            centers = np.array(window_centers)
            offsets_arr = np.array(window_offsets)
            # Normalize time to minutes from episode start
            t_min = (centers - episode_start) / 60.0
            coeffs = np.polyfit(t_min, offsets_arr, 1)
            slope = coeffs[0]  # ms per minute
            predicted = np.polyval(coeffs, t_min)
            ss_res = np.sum((offsets_arr - predicted) ** 2)
            ss_tot = np.sum((offsets_arr - np.mean(offsets_arr)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            pairwise[f"{feat_a}_vs_{feat_b}"] = {
                "drift_rate_ms_per_min": round(abs(slope), 3),
                "r_squared": round(r_squared, 3),
            }

        measurement = {
            "measured": True,
            "pairwise": pairwise,
            "window_size_sec": _DRIFT_WINDOW_SEC,
            "num_windows": len(window_centers) if 'window_centers' in dir() else 0,
        }

        return MetricResult.make_pass(
            name=self.name,
            measurement=measurement,
            message=(
                f"Temporal drift computed across {len(pairwise)} pair(s) "
                f"with {_DRIFT_WINDOW_SEC}s windows."
            ),
        )


# =========================================================================
# 5. VideoFreezeMetric (Layer 1 — unchanged)
# =========================================================================

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


# =========================================================================
# 6. VideoTimestampAlignmentMetric (Layer 1 — unchanged)
# =========================================================================

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


# =========================================================================
# Backward compatibility alias
# =========================================================================

# Deprecated: use VideoStreamPresenceMetric instead.
# Kept so that existing imports don't break during migration.
VideoStreamSyncMetric = VideoStreamPresenceMetric
