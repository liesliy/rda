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
        elif n_frames > 0:
            details["missing_frames"] = 0

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
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "missing_frames": 0, "dropout_count": 0},
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
            return MetricResult.make_review(
                name=self.name,
                measurement={"score_compat": 0.0, "missing_frames": missing_count, "dropout_count": dropout_count},
                reason="; ".join(parts),
                message=msg,
                details=details,
                severity="high",
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

        passed = len(length_mismatches) == 0

        if passed:
            msg = f"All {len(all_features)} features have consistent shape within the episode."
            return MetricResult.make_pass(
                name=self.name,
                measurement={"score_compat": 1.0, "mismatch_count": 0},
                message=msg,
                details=details,
            )
        else:
            msg = f"{len(length_mismatches)} feature(s) have first-dimension length mismatch: {', '.join(length_mismatches)}."
            return MetricResult.make_exclude(
                name=self.name,
                reason=f"{len(length_mismatches)} schema mismatches",
                message=msg,
                details=details,
            )
