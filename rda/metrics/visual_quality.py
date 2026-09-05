"""VA-B: visual quality measurements (REQ-4, v0.7.0).

Measurement-level vision checks — REVIEW grade, never a verdict veto.
Per the roadmap's GIGO position ([GIGO原文], HF blog): RDA measures and
presents; it does not judge semantic "goodness". Everything here is a
physical measurement of the captured image:

- blur:      Laplacian variance on the center crop of sampled frames
             (parameter reference: [SLE] score_lerobot_episodes, whose
             normalized ``max_var=80`` floor we treat as a starting
             prior — thresholds MUST be calibrated on real data before
             grading, see REQ-4 risk note).
- exposure:  mean luminance vs dark (<50) and blown-out (>230) bands,
             plus contrast (P95 − P5 of luminance) for low-contrast
             scenes.
- sampling:  up to 10 uniformly spaced frames per episode (matches
             [SLE]'s 10-frame sampling).
- aggregation: per-episode quality penalty ``max(blur_penalty,
             exposure_penalty)`` — [SLE]'s aggregation rule — plus the
             worst-frame timestamp so a human can jump straight to the
             offending moment.

Output is observational: the metric reports measurements and a
REVIEW-level finding only when values cross the calibrated thresholds.
It never grades EXCLUDE.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult


# --- Sampling & decode budget ---
_SAMPLE_FRAMES = 10             # uniform samples per episode [SLE]
_DECODE_SIZE = 128              # decode at 128×128 gray (VA-B needs real texture)

# --- Starting thresholds (SLE priors; calibrate before trusting grades) ---
_BLUR_VAR_FLOOR = 80.0          # Laplacian-variance normalization floor [SLE max_var]
_DARK_MEAN = 50.0               # mean luminance below = dark [SLE]
_BRIGHT_MEAN = 230.0            # mean luminance above = blown out
_CONTRAST_FLOOR = 20.0          # P95-P5 luminance below = low contrast
_PENALTY_REVIEW = 0.5           # aggregate penalty >= this → REVIEW finding


@lru_cache(maxsize=256)
def _laplacian_kernel() -> np.ndarray:
    """3×3 Laplacian kernel (variance-of-Laplacian blur measure)."""
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian response (classic blur proxy)."""
    k = _laplacian_kernel()
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(gray.astype(np.float32), (3, 3))
    resp = (win * k).sum(axis=(-2, -1))
    return float(resp.var())


def _sample_timestamps(
    episode: EpisodeData, n: int
) -> List[Optional[float]]:
    """Uniform timestamps for n samples (parquet timeline when present)."""
    ts = episode.timestamps
    total = episode.num_frames
    out: List[Optional[float]] = []
    for i in range(n):
        f = int(round(i * (total - 1) / max(n - 1, 1)))
        if ts is not None and len(ts) > f:
            out.append(float(ts[f]))
        else:
            out.append(None)
    return out


class VisualQualityMetric(MetricBase):
    """VA-B: blur / exposure / contrast measurements on sampled frames.

    Layer 2-style observational metric: REVIEW at most, never EXCLUDE.
    Thresholds are SLE-derived starting priors — see the REQ-4 risk note
    about calibrating per camera resolution/focus before trusting the
    grades.
    """

    name = "visual_quality"
    description = (
        "VA-B: blur (Laplacian variance), exposure and contrast on 10 "
        "uniformly sampled frames (measurement only, never a veto)."
    )

    def compute(self, episode: EpisodeData) -> MetricResult:
        meta = episode.meta or {}
        video_features: Dict[str, Dict[str, Any]] = meta.get("video_features") or {}
        dataset_root = meta.get("dataset_root")
        fps = meta.get("fps")

        if not video_features:
            return MetricResult.make_na(
                name=self.name,
                reason="no_video_features",
                message="No video features; visual quality not applicable.",
            )
        if not dataset_root or not fps:
            return MetricResult.make_na(
                name=self.name,
                reason="dataset_or_fps_unknown",
                message="Loader did not provide dataset root/fps; cannot sample videos.",
            )

        root = Path(dataset_root)
        n_samples = min(_SAMPLE_FRAMES, max(episode.num_frames, 1))
        timestamps = _sample_timestamps(episode, n_samples)

        per_feature: Dict[str, Dict[str, Any]] = {}
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

            span = float(to_ts) - float(from_ts)
            samples: List[Dict[str, Any]] = []
            for k in range(n_samples):
                # Uniform sample position within the episode's video span.
                frac = k / max(n_samples - 1, 1)
                at = float(from_ts) + frac * max(span - 1.0 / float(fps), 0.0)
                frame = _decode_one_gray(video_path, at, float(fps))
                if frame is None:
                    continue
                # Blur on center crop (edges are usually irrelevant).
                h, w = frame.shape
                cy, cx = h // 2, w // 2
                r = min(h, w) // 4
                crop = frame[cy - r:cy + r, cx - r:cx + r]
                blur_var = _laplacian_variance(crop)
                mean_l = float(frame.mean())
                p5, p95 = (float(np.percentile(frame, q)) for q in (5, 95))
                samples.append({
                    "t": round(frac * span, 3),
                    "parquet_t": timestamps[k] if timestamps else None,
                    "blur_var": round(blur_var, 2),
                    "mean_lum": round(mean_l, 1),
                    "contrast_p5_p95": round(p95 - p5, 1),
                })

            if samples:
                per_feature[feature] = _aggregate_samples(samples)

        if not per_feature:
            return MetricResult.make_na(
                name=self.name,
                reason="videos_not_decodable",
                message="No video frames could be sampled; visual quality skipped.",
            )

        # Aggregate across features: worst feature drives the penalty.
        worst = max(
            per_feature.values(), key=lambda s: s["penalty"]
        )
        details: Dict[str, Any] = {
            "per_feature": per_feature,
            "sample_count": n_samples,
            "thresholds": {
                "blur_var_floor": _BLUR_VAR_FLOOR,
                "dark_mean": _DARK_MEAN,
                "bright_mean": _BRIGHT_MEAN,
                "contrast_floor": _CONTRAST_FLOOR,
                "note": "SLE-derived priors; calibrate per camera before trusting grades",
            },
        }

        penalty = worst["penalty"]
        measurement = {
            "score_compat": round(1.0 - penalty, 3),
            "penalty": round(penalty, 3),
            "worst_feature": max(
                per_feature.items(), key=lambda kv: kv[1]["penalty"]
            )[0],
            "worst_frame_t": worst.get("worst_frame_t"),
        }

        if penalty >= _PENALTY_REVIEW:
            return MetricResult.make_review(
                name=self.name,
                measurement=measurement,
                message=(
                    f"Visual quality penalty {penalty:.2f} "
                    f"({worst['dominant_issue']}); worst sample at t="
                    f"{worst.get('worst_frame_t')}s — human review suggested."
                ),
                details=details,
            )
        return MetricResult.make_pass(
            name=self.name,
            measurement=measurement,
            message=(
                f"Visual quality OK (penalty {penalty:.2f}, "
                f"{len(per_feature)} camera stream(s) sampled)."
            ),
            details=details,
        )


def _aggregate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-frame samples into penalties per [SLE]: max(blur, exposure).

    blur_penalty     := 1 − clamp(blur_var / floor)  (median over frames)
    exposure_penalty := max(dark_frac, bright_frac, low_contrast_frac)
    """
    blur_vars = np.array([s["blur_var"] for s in samples], dtype=np.float32)
    med_blur = float(np.median(blur_vars))
    blur_penalty = 1.0 - min(med_blur / _BLUR_VAR_FLOOR, 1.0)

    dark_frac = float(np.mean([s["mean_lum"] < _DARK_MEAN for s in samples]))
    bright_frac = float(np.mean([s["mean_lum"] > _BRIGHT_MEAN for s in samples]))
    low_contrast_frac = float(
        np.mean([s["contrast_p5_p95"] < _CONTRAST_FLOOR for s in samples])
    )
    exposure_penalty = max(dark_frac, bright_frac, low_contrast_frac)

    penalty = max(blur_penalty, exposure_penalty)
    if penalty == blur_penalty and blur_penalty >= exposure_penalty:
        dominant = "blur"
    elif dark_frac == exposure_penalty and dark_frac > 0:
        dominant = "dark"
    elif bright_frac == exposure_penalty and bright_frac > 0:
        dominant = "blown_out"
    elif low_contrast_frac > 0:
        dominant = "low_contrast"
    else:
        dominant = "blur"

    worst_idx = int(np.argmin(blur_vars)) if dominant == "blur" else int(
        np.argmax([
            (s["mean_lum"] < _DARK_MEAN) or (s["mean_lum"] > _BRIGHT_MEAN)
            or (s["contrast_p5_p95"] < _CONTRAST_FLOOR)
            for s in samples
        ])
    )
    return {
        "median_blur_var": round(med_blur, 2),
        "blur_penalty": round(blur_penalty, 3),
        "dark_frac": round(dark_frac, 2),
        "bright_frac": round(bright_frac, 2),
        "low_contrast_frac": round(low_contrast_frac, 2),
        "exposure_penalty": round(exposure_penalty, 3),
        "penalty": round(penalty, 3),
        "dominant_issue": dominant,
        "worst_frame_t": samples[worst_idx]["t"],
        "samples": samples,
    }


@lru_cache(maxsize=1024)
def _decode_one_gray(video_path: Path, at_sec: float, fps: float) -> Optional[np.ndarray]:
    """Decode a single frame near ``at_sec`` as 128×128 grayscale."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            tb = stream.time_base
            try:
                container.seek(int(at_sec / tb), stream=stream)
            except Exception:
                pass
            for frame in container.decode(stream):
                pts = float(frame.pts * tb) if frame.pts is not None else None
                if pts is not None and pts < at_sec - 1.0 / max(fps, 1.0):
                    continue
                return (
                    frame.reformat(width=_DECODE_SIZE, height=_DECODE_SIZE,
                                   format="gray").to_ndarray()
                )
    except Exception:
        return None
    return None
