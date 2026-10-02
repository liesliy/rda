"""Integrity metrics: MissingFramesMetric, NaNInfMetric, SchemaShapeMetric.

These three metrics answer Q1: "Is the data broken?" at the most basic
level — frame completeness, value validity, and structural consistency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult, MetricAvailability


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _collect_numeric_features(episode: EpisodeData) -> Dict[str, np.ndarray]:
    """Collect all numeric observation and action features from an episode.

    Iterates over ``episode.observation`` and ``episode.action``, and
    returns a flat dict of ``{full_key: ndarray}`` for every feature
    whose dtype is a numeric kind (float / int / uint, excluding uint8
    which is typically image data).

    Keys are prefixed with their source: ``"observation.state"``,
    ``"action.joint_pos"``, etc.

    Args:
        episode: The episode to extract numeric features from.

    Returns:
        Mapping from dotted feature path to numpy array.
    """
    features: Dict[str, np.ndarray] = {}
    for source_name, source_dict in (
        ("observation", episode.observation),
        ("action", episode.action),
    ):
        for key, arr in source_dict.items():
            full_key = f"{source_name}.{key}"
            if not isinstance(arr, np.ndarray):
                continue
            if arr.dtype.kind in ("f", "i", "u") and arr.dtype != np.uint8:
                features[full_key] = arr
    return features


def _feature_shape_signature(arr: np.ndarray) -> Tuple[int, ...]:
    """Return the per-frame shape signature of a feature array.

    For a 1-D array (scalar per frame), returns an empty tuple ``()``.
    For an N-D array, returns ``arr.shape[1:]`` — the shape of each
    individual frame, excluding the time dimension.

    Args:
        arr: Feature array with shape ``(T, ...)``.

    Returns:
        A tuple describing the per-frame dimensionality.
    """
    if arr.ndim <= 1:
        return ()
    return tuple(arr.shape[1:])


# ---------------------------------------------------------------------------
# Metric 01 — Missing / Dropout
# ---------------------------------------------------------------------------

class MissingFramesMetric(MetricBase):
    name = "missing_dropout"
    description = "Detect missing frames and per-feature sensor dropout."

    def compute(self, episode: EpisodeData) -> MetricResult:
        n_frames = episode.num_frames
        details: Dict[str, Any] = {
            "expected_frames": n_frames,
            "missing_frames": 0,
            "dropout_features": [],
            "by_feature_frame_count": {},
        }

        frame_index = episode.meta.get("frame_index")
        if frame_index is not None and isinstance(frame_index, np.ndarray):
            expected = np.arange(frame_index.size)
            missing_mask = ~np.isin(expected, frame_index)
            details["missing_frames"] = int(missing_mask.sum())
            details["missing_detection_method"] = "frame_index"
        elif n_frames > 0:
            # Path B: infer missing frames from timestamp gaps
            timestamps = getattr(episode, "timestamps", None)
            _ts_arr: np.ndarray | None = None
            if timestamps is not None:
                try:
                    _ts_arr = np.asarray(timestamps, dtype=np.float64)
                except (ValueError, TypeError):
                    _ts_arr = None

            if _ts_arr is not None and _ts_arr.size >= 2:
                dt = np.diff(_ts_arr)
                median_dt = float(np.median(dt))
                if median_dt > 0:
                    # Gap threshold: 2.5× median dt.
                    # Derived from T-02 decision in governance. Not re-calibrated
                    # in v0.9.17 — ArmnetBench has 0 gap events across 2499 episodes
                    # (all pass Layer 1 integrity), so no discriminative signal
                    # for threshold optimization on this metric.
                    gap_threshold = median_dt * 2.5
                    gap_mask = dt > gap_threshold
                    gap_indices = np.where(gap_mask)[0]
                    gap_count = int(gap_mask.sum())
                    inferred_missing = 0
                    gap_positions: List[int] = []
                    gap_total_duration = 0.0
                    for idx in gap_indices:
                        missing_here = int(round(float(dt[idx]) / median_dt)) - 1
                        if missing_here > 0:
                            inferred_missing += missing_here
                            gap_positions.append(int(idx))
                            gap_total_duration += float(dt[idx])
                    details["missing_frames"] = inferred_missing
                    details["missing_detection_method"] = "timestamp_gap_inference"
                    details["gap_count"] = gap_count
                    details["gap_positions"] = gap_positions
                    details["gap_total_duration"] = gap_total_duration
                else:
                    details["missing_frames"] = 0
                    details["missing_detection_method"] = "timestamp_gap_inference"
            else:
                details["missing_frames"] = 0
                details["missing_detection_method"] = "timestamp_gap_inference"

        features = _collect_numeric_features(episode)

        ref_len: int | None = None
        if "observation.state" in features:
            ref_len = features["observation.state"].shape[0]
        elif features:
            ref_len = max(arr.shape[0] for arr in features.values())

        by_feature: Dict[str, int] = {}
        dropout_features: List[str] = []
        for key, arr in features.items():
            count = arr.shape[0]
            by_feature[key] = count
            if ref_len is not None and count != ref_len:
                dropout_features.append(key)

        details["by_feature_frame_count"] = by_feature
        details["dropout_features"] = dropout_features

        missing_count = details["missing_frames"]
        dropout_count = len(dropout_features)

        if n_frames <= 0:
            return MetricResult.make_na(
                name=self.name,
                reason="empty_episode",
                message="Empty episode, no frames to check.",
                details=details,
            )

        passed = missing_count == 0 and dropout_count == 0

        if passed:
            msg = f"No missing frames or dropout detected across {len(features)} features."
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "missing_frames": missing_count, "dropout_count": dropout_count},
                message=msg,
                details=details,
            )
        else:
            parts = []
            if missing_count > 0:
                parts.append(f"{missing_count} missing frame(s)")
            if dropout_count > 0:
                parts.append(f"{dropout_count} feature(s) with dropout")
            msg = "Detected: " + "; ".join(parts) + "."
            return MetricResult.make_exclude(
                name=self.name,
                reason="; ".join(parts),
                message=msg,
                details=details,
            )


# ---------------------------------------------------------------------------
# Metric 04 — Physical Plausibility (L1 WARNING-level)
# ---------------------------------------------------------------------------

# Default thresholds (configurable via constructor kwargs).
# These are intentionally宽松 (lenient) — they catch obviously wrong data
# (e.g., joint angles in degrees mislabelled as radians, force spikes of
# 1e6 N) without false-flagging legitimate platform-specific ranges.
_DEFAULT_ANGLE_LIMIT: float = 3.2  # slightly > π, catches degree-vs-radian errors
_DEFAULT_FORCE_LIMIT: float = 1e4  # 10 kN — most lab robots are < 500 N
_DEFAULT_TORQUE_LIMIT: float = 1e3  # 1000 Nm — most lab robots are < 50 Nm
_DEFAULT_VELOCITY_LIMIT: float = 100.0  # 100 rad/s or m/s — generous ceiling

# Feature name patterns for auto-detection (case-insensitive substring match)
_ANGLE_PATTERNS = ("joint_pos", "joint_angle", "angle", "position")
_FORCE_PATTERNS = ("force", "wrench_force")
_TORQUE_PATTERNS = ("torque", "wrench_torque")
_VELOCITY_PATTERNS = ("velocity", "vel", "joint_vel")


class NumericSanityMetric(MetricBase):
    """Basic numeric sanity check for action/state fields.

    This is a Layer 1 safeguard that detects obviously wrong numeric values
    (e.g., angles in degrees when radians are expected, forces in orders of
    magnitude beyond typical lab robots). It does NOT affect the episode
    verdict — warnings are recorded in ``details["warnings"]`` for
    diagnostic reference only.

    Checks performed:
      - Joint angles: |value| > angle_limit (default ±π + margin)
      - Force/torque: |value| > force/torque_limit
      - Velocity: |value| > velocity_limit

    Thresholds are configurable via constructor kwargs:
      angle_limit, force_limit, torque_limit, velocity_limit

    The metric always returns PASS with warnings attached
    in the ``details["warnings"]`` list. Each warning is a dict with:
      - feature: the feature key
      - check: the type of check (e.g., "angle_range")
      - max_abs: the maximum absolute value found
      - threshold: the threshold that was exceeded
      - count: number of frames exceeding the threshold
    """

    name = "numeric_sanity"
    description = "Basic numeric range check for extreme/outlier values (WARNING-level, does not affect verdict)."

    def __init__(
        self,
        angle_limit: float = _DEFAULT_ANGLE_LIMIT,
        force_limit: float = _DEFAULT_FORCE_LIMIT,
        torque_limit: float = _DEFAULT_TORQUE_LIMIT,
        velocity_limit: float = _DEFAULT_VELOCITY_LIMIT,
    ) -> None:
        self.angle_limit = angle_limit
        self.force_limit = force_limit
        self.torque_limit = torque_limit
        self.velocity_limit = velocity_limit

    def compute(self, episode: EpisodeData) -> MetricResult:
        features = _collect_numeric_features(episode)
        details: Dict[str, Any] = {
            "checked_features": list(features.keys()),
            "warnings": [],
            "thresholds": {
                "angle_limit": self.angle_limit,
                "force_limit": self.force_limit,
                "torque_limit": self.torque_limit,
                "velocity_limit": self.velocity_limit,
            },
        }

        if not features:
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "warning_count": 0},
                message="No numeric features to check.",
                details=details,
            )

        warnings: List[Dict[str, Any]] = []

        for key, arr in features.items():
            if arr.size == 0:
                continue

            key_lower = key.lower()

            # Check angle-like features
            if any(p in key_lower for p in _ANGLE_PATTERNS):
                max_abs = float(np.nanmax(np.abs(arr)))
                if max_abs > self.angle_limit:
                    count = int(np.sum(np.abs(arr) > self.angle_limit))
                    warnings.append({
                        "feature": key,
                        "check": "angle_range",
                        "max_abs": max_abs,
                        "threshold": self.angle_limit,
                        "count": count,
                    })

            # Check force-like features
            if any(p in key_lower for p in _FORCE_PATTERNS):
                max_abs = float(np.nanmax(np.abs(arr)))
                if max_abs > self.force_limit:
                    count = int(np.sum(np.abs(arr) > self.force_limit))
                    warnings.append({
                        "feature": key,
                        "check": "force_range",
                        "max_abs": max_abs,
                        "threshold": self.force_limit,
                        "count": count,
                    })

            # Check torque-like features
            if any(p in key_lower for p in _TORQUE_PATTERNS):
                max_abs = float(np.nanmax(np.abs(arr)))
                if max_abs > self.torque_limit:
                    count = int(np.sum(np.abs(arr) > self.torque_limit))
                    warnings.append({
                        "feature": key,
                        "check": "torque_range",
                        "max_abs": max_abs,
                        "threshold": self.torque_limit,
                        "count": count,
                    })

            # Check velocity-like features
            if any(p in key_lower for p in _VELOCITY_PATTERNS):
                max_abs = float(np.nanmax(np.abs(arr)))
                if max_abs > self.velocity_limit:
                    count = int(np.sum(np.abs(arr) > self.velocity_limit))
                    warnings.append({
                        "feature": key,
                        "check": "velocity_range",
                        "max_abs": max_abs,
                        "threshold": self.velocity_limit,
                        "count": count,
                    })

        details["warnings"] = warnings

        warning_count = len(warnings)
        if warning_count == 0:
            msg = f"All {len(features)} numeric features are within expected ranges."
        else:
            features_with_warnings = {w["feature"] for w in warnings}
            msg = (
                f"Numeric sanity warnings in {len(features_with_warnings)} "
                f"feature(s): {', '.join(sorted(features_with_warnings))}."
            )

        # Always PASS — warnings are informational only (L1 WARNING-level)
        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "warning_count": warning_count,
                "features_checked": len(features),
            },
            message=msg,
            details=details,
        )


# ---------------------------------------------------------------------------
# Metric 02 — NaN / Inf / Invalid
# ---------------------------------------------------------------------------

class NaNInfMetric(MetricBase):
    name = "invalid_values"
    description = "Check for NaN, Inf, and -Inf values in numeric features."

    def compute(self, episode: EpisodeData) -> MetricResult:
        features = _collect_numeric_features(episode)
        details: Dict[str, Any] = {
            "checked_features": list(features.keys()),
            "nan_count": 0,
            "inf_count": 0,
            "total_cells": 0,
            "by_feature": {},
        }

        total_nan = 0
        total_inf = 0
        total_cells = 0
        by_feature: Dict[str, Dict[str, int]] = {}

        for key, arr in features.items():
            if np.issubdtype(arr.dtype, np.floating):
                nan_count = int(np.isnan(arr).sum())
                inf_count = int(np.isinf(arr).sum())
            else:
                nan_count = 0
                inf_count = 0

            cells = int(arr.size)
            total_nan += nan_count
            total_inf += inf_count
            total_cells += cells
            by_feature[key] = {"nan_count": nan_count, "inf_count": inf_count, "total_cells": cells}

        details["nan_count"] = total_nan
        details["inf_count"] = total_inf
        details["total_cells"] = total_cells
        details["by_feature"] = by_feature

        invalid_total = total_nan + total_inf

        if total_cells == 0:
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "invalid_count": 0, "total_cells": 0},
                message="No numeric features to check.",
                details=details,
            )

        passed = invalid_total == 0
        invalid_ratio = invalid_total / total_cells

        if passed:
            msg = f"No NaN or Inf values found across {len(features)} features."
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "invalid_count": 0, "total_cells": total_cells},
                message=msg,
                details=details,
            )
        else:
            msg = f"Found {total_nan} NaN and {total_inf} Inf values ({invalid_ratio:.4%} of cells)."
            return MetricResult.make_exclude(
                name=self.name,
                reason=f"{total_nan} NaN + {total_inf} Inf values",
                message=msg,
                details=details,
            )


# ---------------------------------------------------------------------------
# Metric 03 — Schema / Shape Consistency
# ---------------------------------------------------------------------------

class SchemaShapeMetric(MetricBase):
    name = "schema_consistency"
    description = "Validate per-episode schema and record structural signatures."

    def compute(self, episode: EpisodeData) -> MetricResult:
        details: Dict[str, Any] = {
            "num_frames": episode.num_frames,
            "features": {},
            "length_mismatches": [],
        }

        length_mismatches: List[str] = []
        features_info: Dict[str, Dict[str, Any]] = {}

        all_features: Dict[str, np.ndarray] = {}
        for source_name, source_dict in (
            ("observation", episode.observation),
            ("action", episode.action),
        ):
            for key, arr in source_dict.items():
                if isinstance(arr, np.ndarray):
                    all_features[f"{source_name}.{key}"] = arr

        for key, arr in all_features.items():
            shape_sig = list(_feature_shape_signature(arr))
            features_info[key] = {
                "dtype": str(arr.dtype),
                "shape": shape_sig,
                "first_dim": int(arr.shape[0]),
            }
            if arr.shape[0] != episode.num_frames:
                length_mismatches.append(key)

        details["features"] = features_info
        details["length_mismatches"] = length_mismatches

        # Cross-validate actual data dimensions with declared shapes from info.json
        dimension_mismatches: List[str] = []
        declared_features = episode.meta.get("declared_features")
        if declared_features:
            all_obs: Dict[str, Any] = {}
            all_obs.update(episode.observation)
            all_obs.update(episode.action)

            for key, spec in declared_features.items():
                declared_shape = spec.get("shape", [])
                if not declared_shape:
                    continue

                # Map declared feature keys to actual episode data keys
                # info.json uses "observation.state" but episode.observation uses "state"
                lookup_key = key
                if key.startswith("observation."):
                    lookup_key = key[len("observation."):]
                elif key == "action":
                    lookup_key = "action"  # action is stored as episode.action["action"]

                arr = all_obs.get(lookup_key)
                if arr is None:
                    continue

                if hasattr(arr, 'shape'):
                    actual_shape = list(arr.shape)
                else:
                    actual_shape = [len(arr)]

                # Compare dimensions: declared_shape from info.json does NOT include frame dim,
                # actual_shape does. Strip frame dim from actual only.
                declared_dims = list(declared_shape)
                actual_dims = list(actual_shape[1:]) if len(actual_shape) > 1 else []

                if declared_dims and actual_dims and declared_dims != actual_dims:
                    dimension_mismatches.append(
                        f"{key}_declared_{declared_dims}_actual_{actual_dims}"
                    )

        details["dimension_mismatches"] = dimension_mismatches

        all_mismatches = length_mismatches + dimension_mismatches
        passed = len(all_mismatches) == 0

        if passed:
            msg = f"All {len(all_features)} features have consistent shape within the episode."
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "mismatch_count": 0},
                message=msg,
                details=details,
            )
        else:
            parts: List[str] = []
            if length_mismatches:
                parts.append(f"{len(length_mismatches)} first-dim length mismatch(es): {', '.join(length_mismatches)}")
            if dimension_mismatches:
                parts.append(f"{len(dimension_mismatches)} dimension mismatch(es): {', '.join(dimension_mismatches)}")
            msg = "Schema check failed: " + "; ".join(parts) + "."
            return MetricResult.make_exclude(
                name=self.name,
                reason="; ".join(parts),
                message=msg,
                details=details,
            )
