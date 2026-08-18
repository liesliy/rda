"""Distribution metrics: DistributionMetric (11) and CoverageMetric (12).

These metrics answer the "what does the data look like?" half of the
RDA audit — describing the statistical shape of action and state
trajectories, and measuring how much of the state space an episode
covers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult, MetricAvailability


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _feature_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute column-wise statistical summary of a 2-D feature array.

    Args:
        values: Array of shape ``(T, D)`` or ``(T,)`` — T timesteps,
            D feature dimensions.

    Returns:
        Dict with keys ``mean``, ``std``, ``percentiles`` (p5/p25/p50/p75/p95),
        ``min``, ``max`` — each is a list of length D (per-column value).
    """
    if values.size == 0 or values.shape[0] == 0:
        return {
            "mean": [], "std": [],
            "percentiles": {"p5": [], "p25": [], "p50": [], "p75": [], "p95": []},
            "min": [], "max": [],
        }
    pcts = np.percentile(values, [5, 25, 50, 75, 95], axis=0)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=0).tolist(),
        "percentiles": {
            "p5": pcts[0].tolist(), "p25": pcts[1].tolist(),
            "p50": pcts[2].tolist(), "p75": pcts[3].tolist(),
            "p95": pcts[4].tolist(),
        },
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure an array is at least 2-D (time × features).

    Args:
        arr: 1-D or 2-D numpy array.

    Returns:
        2-D view / copy with shape ``(T, 1)`` if input was 1-D,
        otherwise the original array.
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


# ---------------------------------------------------------------------------
# Metric 11 — Action / Trajectory Distribution
# ---------------------------------------------------------------------------

class DistributionMetric(MetricBase):
    name = "distribution"
    description = "Compute statistical distribution summary of episode action features and trajectory properties."

    default_hz: float = 30.0

    def __init__(self, default_hz: float = 30.0) -> None:
        self.default_hz = float(default_hz)

    def _action_stats(self, episode: EpisodeData) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, arr in episode.action.items():
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                continue
            arr_2d = _ensure_2d(arr.astype(np.float64))
            out[key] = _feature_stats(arr_2d)
        return out

    def _trajectory_stats(self, episode: EpisodeData) -> Dict[str, Any]:
        state = episode.observation.get("state")
        result: Dict[str, Any] = {
            "duration_sec": 0.0, "path_length": 0.0,
            "velocity": {"mean": 0.0, "std": 0.0, "p95": 0.0},
            "endpoint": [], "timestamps_available": False,
        }
        T = episode.num_frames
        ts = episode.timestamps
        if ts is not None and isinstance(ts, np.ndarray) and ts.size == T and T > 0:
            ts_arr = ts.astype(np.float64)
            result["duration_sec"] = float(ts_arr[-1] - ts_arr[0])
            result["timestamps_available"] = True
        else:
            result["duration_sec"] = max(T - 1, 0) / self.default_hz if T > 0 else 0.0

        if state is None or not isinstance(state, np.ndarray) or state.ndim < 2 or state.shape[0] == 0:
            return result

        state_arr = state.astype(np.float64)
        T_state = state_arr.shape[0]
        result["endpoint"] = state_arr[-1].tolist()

        if T_state < 2:
            return result

        diffs = np.diff(state_arr, axis=0)
        step_dists = np.linalg.norm(diffs, axis=1)
        result["path_length"] = float(np.sum(step_dists))

        if result["timestamps_available"] and T_state <= ts.size:
            dt = np.diff(ts.astype(np.float64)[:T_state])
            dt = np.where(dt <= 0, 1e-9, dt)
            velocity = step_dists / dt
        else:
            velocity = step_dists * self.default_hz

        if velocity.size > 0:
            result["velocity"] = {
                "mean": float(np.mean(velocity)),
                "std": float(np.std(velocity, ddof=0)),
                "p95": float(np.percentile(velocity, 95)),
            }
        return result

    def compute(self, episode: EpisodeData) -> MetricResult:
        action_stats = self._action_stats(episode)
        trajectory = self._trajectory_stats(episode)

        details: Dict[str, Any] = {
            "action_stats": action_stats,
            "trajectory": trajectory,
        }

        n_actions = len(action_stats)
        has_state = episode.observation.get("state") is not None
        dur = trajectory.get("duration_sec", 0.0)
        plen = trajectory.get("path_length", 0.0)

        if n_actions == 0 and not has_state:
            return MetricResult.make_na(
                name=self.name,
                reason="no_action_or_state_data",
                message="No action or state data available for distribution analysis.",
                details=details,
            )

        msg = f"Distribution summary: {n_actions} action feature(s); duration = {dur:.2f}s, path_length = {plen:.4f}."

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "num_action_features": n_actions,
                "duration_sec": dur,
                "path_length": plen,
            },
            message=msg,
            details=details,
        )


# ---------------------------------------------------------------------------
# Metric 12 — Workspace / State Coverage  ★ Hero Metric #3
# ---------------------------------------------------------------------------

class CoverageMetric(MetricBase):
    """Evaluate state-space coverage using an occupancy grid.
    
    P1 CHANGE: Pure observational metric — always passes.
    Renamed semantic: state_space_occupancy at episode level.
    """

    name = "coverage"
    description = "Assess state-space coverage using an occupancy grid over observation.state (Hero Metric #3)."

    default_resolution: Tuple[int, ...] = (10, 10, 10)
    underrepresented_threshold: float = 0.1

    def __init__(self, resolution: Optional[Tuple[int, ...]] = None,
                 underrepresented_threshold: float = 0.1) -> None:
        if resolution is not None:
            self.default_resolution = tuple(int(r) for r in resolution)
        self.underrepresented_threshold = float(underrepresented_threshold)

    def _effective_resolution(self, ndims: int) -> List[int]:
        base = list(self.default_resolution)
        if ndims <= len(base):
            return base[:ndims]
        return base + [base[-1]] * (ndims - len(base))

    def compute(self, episode: EpisodeData) -> MetricResult:
        details: Dict[str, Any] = {
            "dimensions": [],
            "grid": {"resolution": [], "occupied": 0, "total": 0, "occupancy_rate": 0.0},
            "underrepresented_regions": [],
        }

        state = episode.observation.get("state")
        if state is None or not isinstance(state, np.ndarray) or state.ndim < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="observation_state_missing",
                message="observation.state not available; skipping coverage analysis.",
                details=details,
            )

        state_arr = state.astype(np.float64)
        T, D = state_arr.shape

        if T == 0:
            return MetricResult.make_na(
                name=self.name,
                reason="no_frames",
                message="Zero frames; cannot compute coverage.",
                details=details,
            )

        use_dims = min(D, 3)
        points = state_arr[:, :use_dims]
        dim_names = [f"dim_{i}" for i in range(use_dims)]
        details["dimensions"] = dim_names

        resolution = self._effective_resolution(use_dims)
        total_cells = int(np.prod(resolution))

        try:
            counts, edges = np.histogramdd(points, bins=resolution)
        except (ValueError, RuntimeError) as exc:
            return MetricResult.make_na(
                name=self.name,
                reason=f"histogram_failed: {exc}",
                message=f"Failed to build occupancy grid: {exc}",
                details=details,
            )

        occupied = int(np.sum(counts > 0))
        coverage = occupied / total_cells if total_cells > 0 else 0.0

        details["grid"] = {
            "resolution": resolution,
            "occupied": occupied,
            "total": total_cells,
            "coverage": float(coverage),
        }

        # Underrepresented regions
        mean_count = T / total_cells if total_cells > 0 else 0.0
        threshold = self.underrepresented_threshold * mean_count
        under_idx = np.argwhere(counts < threshold)

        under_regions: List[Dict[str, Any]] = []
        for idx in under_idx:
            cell_idx = tuple(int(i) for i in idx)
            cell_count = int(counts[cell_idx])
            center = []
            for d_i in range(use_dims):
                bin_i = cell_idx[d_i]
                edge = edges[d_i]
                center.append(float((edge[bin_i] + edge[bin_i + 1]) / 2))
            under_regions.append({
                "cell_index": list(cell_idx),
                "count": cell_count,
                "center": center,
            })
        details["underrepresented_regions"] = under_regions

        # --- OBSERVATIONAL: always pass ---
        msg = (
            f"State-space occupancy: {coverage:.1%} "
            f"({occupied}/{total_cells} cells, "
            f"resolution {'×'.join(str(r) for r in resolution)}); "
            f"{len(under_regions)} underrepresented region(s)."
        )

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "occupancy_rate": float(coverage),
                "occupied_cells": occupied,
                "total_cells": total_cells,
                "grid_resolution": resolution,
            },
            message=msg,
            details=details,
        )
