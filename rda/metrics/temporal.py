"""Temporal metrics: TimestampValidityMetric, SensorSyncMetric, JitterMetric.

These metrics answer the temporal half of Q2: "Are the robot data's
time and motion sane?" — focusing on timestamp correctness and
cross-modal synchronization.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult, MetricAvailability


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _percentiles(
    values: np.ndarray,
    ps: Tuple[float, ...] = (50.0, 95.0, 99.0),
) -> Dict[str, float]:
    """Compute named percentile statistics from a 1-D array.

    Percentile 50 is always named ``"median"``; others are named
    ``"p{int(p)}"`` (e.g. ``"p95"``).

    Args:
        values: 1-D array of numeric values.
        ps: Tuple of percentile levels (0–100).

    Returns:
        Dict mapping percentile name to float value. If the input array
        is empty, all values default to 0.0.
    """
    if values.size == 0:
        out: Dict[str, float] = {}
        for p in ps:
            key = "median" if p == 50.0 else f"p{int(p) if p == int(p) else p}"
            out[key] = 0.0
        return out
    result: Dict[str, float] = {}
    for p in ps:
        key = "median" if p == 50.0 else f"p{int(p) if p == int(p) else p}"
        result[key] = float(np.percentile(values, p))
    return result


def _dt_stats(timestamps: np.ndarray) -> Dict[str, Any]:
    """Compute first-difference statistics for a timestamp array.

    Args:
        timestamps: 1-D array of timestamps in seconds.

    Returns:
        Dict with keys ``median_ms``, ``p95_ms``, ``p99_ms``, ``max_ms``,
        ``mean_ms``, ``std_ms``, and ``count`` — all derived from the
        absolute value of ``diff(timestamps)`` in milliseconds.
    """
    if timestamps.size < 2:
        return {
            "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
            "max_ms": 0.0, "mean_ms": 0.0, "std_ms": 0.0, "count": 0,
        }
    dt = np.diff(timestamps.astype(np.float64))
    dt_ms = dt * 1000.0
    pcts = _percentiles(np.abs(dt_ms), (50.0, 95.0, 99.0))
    return {
        "median_ms": pcts["median"],
        "p95_ms": pcts["p95"],
        "p99_ms": pcts["p99"],
        "max_ms": float(np.max(np.abs(dt_ms))),
        "mean_ms": float(np.mean(dt_ms)),
        "std_ms": float(np.std(dt_ms)),
        "count": int(dt.size),
    }


def _collect_stream_timestamps(episode: EpisodeData) -> Dict[str, np.ndarray]:
    """Extract per-stream timestamps from episode metadata.

    Looks for ``episode.meta["stream_timestamps"]``, which should be a
    dict mapping stream name to an array / list of timestamps in seconds.

    Args:
        episode: The episode to read stream timestamps from.

    Returns:
        Dict mapping stream name to float64 numpy array of timestamps.
        Empty dict if no stream timestamps are available.
    """
    streams = episode.meta.get("stream_timestamps")
    if isinstance(streams, dict):
        return {
            name: np.asarray(ts, dtype=np.float64)
            for name, ts in streams.items()
            if isinstance(ts, np.ndarray) or isinstance(ts, list)
        }
    return {}


# ---------------------------------------------------------------------------
# Metric 04 — Timestamp Validity
# ---------------------------------------------------------------------------

class TimestampValidityMetric(MetricBase):
    name = "timestamp_validity"
    description = "Check timestamp monotonicity, duplicates, negative deltas, and extreme inter-frame gaps."

    def compute(self, episode: EpisodeData) -> MetricResult:
        ts = episode.timestamps
        details: Dict[str, Any] = {
            "num_frames": episode.num_frames,
            "monotonic": True,
            "duplicate_count": 0,
            "negative_delta_count": 0,
            "dt_ms": {},
        }

        if ts is None or ts.size < 2:
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "num_timestamps": ts.size if ts is not None else 0},
                message="Fewer than 2 timestamps — nothing to validate.",
                details=details,
            )

        ts_arr = np.asarray(ts, dtype=np.float64)
        dt = np.diff(ts_arr)

        monotonic = bool(np.all(dt >= 0))
        details["monotonic"] = monotonic

        duplicates = int(np.sum(dt == 0))
        details["duplicate_count"] = duplicates

        negative = int(np.sum(dt < 0))
        details["negative_delta_count"] = negative

        stats = _dt_stats(ts_arr)
        details["dt_ms"] = {
            "median": stats["median_ms"], "p95": stats["p95_ms"],
            "p99": stats["p99_ms"], "max": stats["max_ms"],
            "mean": stats["mean_ms"], "std": stats["std_ms"],
        }

        # Fail hard on non-monotonic or negative deltas
        if not monotonic or negative > 0:
            reasons = []
            if not monotonic:
                reasons.append("non-monotonic")
            if negative > 0:
                reasons.append(f"{negative} negative delta(s)")
            msg = "Timestamp validity FAILED: " + "; ".join(reasons) + "."
            return MetricResult.make_exclude(
                name=self.name,
                reason="; ".join(reasons),
                message=msg,
                details=details,
            )
        else:
            # Score decreases with duplicate ratio (backward compat)
            n_intervals = max(dt.size, 1)
            dup_ratio = duplicates / n_intervals
            score_compat = float(np.clip(1.0 - dup_ratio, 0.0, 1.0))
            msg = (
                f"Timestamps are monotonically increasing; "
                f"median dt = {stats['median_ms']:.2f} ms, "
                f"max dt = {stats['max_ms']:.2f} ms."
            )
            if duplicates > 0:
                msg = f"Timestamps monotonic but {duplicates} duplicate(s) ({dup_ratio:.1%} of intervals)."
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": score_compat, "duplicates": duplicates, "median_dt_ms": stats["median_ms"]},
                message=msg,
                details=details,
            )


# ---------------------------------------------------------------------------
# Metric 05 — Sensor Synchronization (Hero Metric)
# ---------------------------------------------------------------------------

class SensorSyncMetric(MetricBase):
    """Measure temporal offset and its stability across sensor streams.
    
    CRITICAL: If no per-stream timestamps are available, returns N/A
    (NOT pass). This is the #1 bug fix from the review.
    """

    name = "sensor_synchronization"
    description = "Measure temporal offset stability between every pair of sensor/modality streams."

    def _nearest_neighbour_offsets(self, ts_a: np.ndarray, ts_b: np.ndarray) -> np.ndarray:
        if ts_a.size == 0 or ts_b.size == 0:
            return np.array([], dtype=np.float64)
        idx = np.searchsorted(ts_b, ts_a)
        idx = np.clip(idx, 1, ts_b.size - 1)
        left = ts_b[idx - 1]
        right = ts_b[idx]
        closer = np.where(
            np.abs(ts_a - left) <= np.abs(ts_a - right),
            left, right,
        )
        return ts_a - closer

    def compute(self, episode: EpisodeData) -> MetricResult:
        streams = _collect_stream_timestamps(episode)
        details: Dict[str, Any] = {
            "streams": list(streams.keys()),
            "pairs": [],
        }

        # --- N/A cases: data not available ---
        if not streams:
            return MetricResult.make_na(
                name=self.name,
                reason="stream_timestamps_not_provided",
                message="Sensor sync: N/A (no per-stream timestamps provided)",
                details=details,
            )

        if len(streams) < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="fewer_than_two_streams",
                message=f"Sensor sync: N/A (only {len(streams)} stream available, need ≥2)",
                details=details,
            )

        pair_results: List[Dict[str, Any]] = []
        worst_p95_ms = 0.0
        worst_max_ms = 0.0

        stream_names = sorted(streams.keys())
        for a_name, b_name in combinations(stream_names, 2):
            ts_a = streams[a_name]
            ts_b = streams[b_name]

            if ts_a.size > ts_b.size:
                ts_a, ts_b = ts_b, ts_a
                a_name, b_name = b_name, a_name

            offsets_s = self._nearest_neighbour_offsets(ts_a, ts_b)
            offsets_ms = offsets_s * 1000.0
            abs_offsets_ms = np.abs(offsets_ms)

            if offsets_ms.size == 0:
                continue

            pcts = _percentiles(abs_offsets_ms, (50.0, 95.0, 99.0))
            pair_info: Dict[str, Any] = {
                "stream_a": a_name,
                "stream_b": b_name,
                "sample_count": int(offsets_ms.size),
                "offset_ms": {
                    "mean": float(np.mean(offsets_ms)),
                    "median": pcts["median"],
                    "p95": pcts["p95"],
                    "p99": pcts["p99"],
                    "max": float(np.max(abs_offsets_ms)),
                    "std": float(np.std(offsets_ms)),
                    "signed_median": float(np.median(offsets_ms)),
                },
            }
            pair_results.append(pair_info)

            if pcts["p95"] > worst_p95_ms:
                worst_p95_ms = pcts["p95"]
            if float(np.max(abs_offsets_ms)) > worst_max_ms:
                worst_max_ms = float(np.max(abs_offsets_ms))

        details["pairs"] = pair_results

        if not pair_results:
            return MetricResult.make_na(
                name=self.name,
                reason="no_valid_pairs",
                message="Sensor sync: N/A (no valid stream pairs to compare)",
                details=details,
            )

        # Data is available — always pass (observational, no hard threshold)
        msg = (
            f"Analyzed {len(pair_results)} stream pair(s); "
            f"worst p95 offset = {worst_p95_ms:.2f} ms, "
            f"worst max offset = {worst_max_ms:.2f} ms."
        )

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "worst_p95_offset_ms": worst_p95_ms,
                "worst_max_offset_ms": worst_max_ms,
                "num_pairs": len(pair_results),
            },
            message=msg,
            details=details,
        )


# ---------------------------------------------------------------------------
# Metric 06 — Sampling Jitter
# ---------------------------------------------------------------------------

class JitterMetric(MetricBase):
    name = "sampling_jitter"
    description = "Measure inter-frame timestamp jitter (CV = std/mean)."

    def compute(self, episode: EpisodeData) -> MetricResult:
        ts = episode.timestamps
        if ts is None or ts.size < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="insufficient_timestamps",
                message="Fewer than 2 timestamps — no jitter to measure.",
                details={"mean_interval_ms": 0.0, "jitter_ms": 0.0, "jitter_ratio": 0.0},
            )

        dt = np.diff(np.asarray(ts, dtype=np.float64))
        dt_ms = dt * 1000.0
        mean_dt = float(np.mean(dt_ms))
        std_dt = float(np.std(dt_ms))
        cv = std_dt / mean_dt if mean_dt > 0 else 0.0

        stats = _dt_stats(ts)
        details = {
            "mean_interval_ms": mean_dt,
            "jitter_ms": std_dt,
            "jitter_ratio": cv,
            "dt_ms": {
                "median": stats["median_ms"], "p95": stats["p95_ms"],
                "p99": stats["p99_ms"], "max": stats["max_ms"],
            },
        }

        msg = f"Sampling jitter CV = {cv:.4f} (mean dt = {mean_dt:.2f} ms, std = {std_dt:.2f} ms)."

        # Pure observational
        return MetricResult.make_pass(
            name=self.name,
            measurement={"score_compat": 1.0, "cv": cv, "jitter_ms": std_dt},
            message=msg,
            details=details,
            baseline={
                "method": "coefficient_of_variation",
                "scope": "episode",
                "reference_population": int(dt.size),
            },
        )


# ---------------------------------------------------------------------------
# Metric — Temporal Sufficiency
# ---------------------------------------------------------------------------

class TemporalSufficiencyMetric(MetricBase):
    """Measure temporal sufficiency of action data for idle-frame pruning.

    Computes per-episode idle structure metrics (idle prefix, active run
    distribution, valid window ratios) that indicate whether and how much
    idle-frame pruning is safe.
    """

    name = "temporal_sufficiency"
    description = "Measure temporal sufficiency: idle structure, active runs, and valid window ratios."

    def compute(self, episode: EpisodeData) -> MetricResult:
        from rda.recommend.temporal_metrics import compute_temporal_sufficiency

        ts = compute_temporal_sufficiency(episode)

        if ts is None:
            # No usable action data — return NOT_AVAILABLE
            return MetricResult.make_na(
                name=self.name,
                reason="no_usable_action_data",
                message="Temporal sufficiency: N/A (no usable action data).",
            )

        ts_dict = ts.to_dict()

        measurement = {
            "idle_total_ratio": ts.idle_total_ratio,
            "idle_prefix_ratio": ts.idle_prefix_ratio,
            "active_run_p50": ts.active_run_p50,
            "active_run_p90": ts.active_run_p90,
            "active_run_max": ts.active_run_max,
            "transition_count": ts.transition_count,
            "idle_to_active_ratio": ts.idle_to_active_ratio,
            "valid_window_ratio_5": ts.valid_window_ratio_5,
            "valid_window_ratio_10": ts.valid_window_ratio_10,
            "valid_window_ratio_20": ts.valid_window_ratio_20,
            "total_frames": ts.total_frames,
        }

        msg = (
            f"Temporal sufficiency: idle_total={ts.idle_total_ratio:.1%}, "
            f"idle_prefix={ts.idle_prefix_ratio:.1%}, "
            f"active_run_p50={ts.active_run_p50:.0f}f, "
            f"valid_window(seq=10)={ts.valid_window_ratio_10:.1%}."
        )

        return MetricResult.make_pass(
            name=self.name,
            measurement=measurement,
            message=msg,
            details=ts_dict,
            baseline={
                "method": "action_delta_idle_detection",
                "scope": "episode",
                "threshold": ts.threshold,
                "threshold_method": ts.threshold_method,
                "reference_population": ts.total_frames,
            },
        )
