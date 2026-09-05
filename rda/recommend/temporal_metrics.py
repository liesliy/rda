"""Temporal sufficiency metrics for the recommendation engine.

Computes idle/active segmentation and various temporal-structure metrics
that help determine whether idle-frame pruning is safe for a given dataset.

Key metrics:
  - ``idle_prefix_ratio``  : fraction of frames that are initial idle (leading pause)
  - ``idle_total_ratio``   : total idle ratio (initial + mid-episode pauses)
  - ``active_run_p50/p90/max`` : distribution of consecutive active run lengths (in frames)
  - ``valid_window_ratio(seq=N)`` : fraction of all-N sliding windows that are fully active
  - ``transition_count``   : number of idle → active transitions
  - ``idle_to_active_ratio``: transitions per total frame (normalized transition density)

All metrics are pure observational — no pass/fail judgment. They feed
into the recommendation engine which decides what level of pruning,
if any, is appropriate.

Note on idle detection strategy
-------------------------------
Idle detection uses action-frame deltas (matching the existing
``IdleRatioMetric`` in rda.metrics.motion). A frame is considered
"active" if the action-norm delta exceeds a data-driven threshold
(bimodal-gap or MAD-based). The initial idle prefix is the leading
consecutive-idle segment from frame 0.

This is consistent with the experimental report (Exp 5b) which
used action displacement as the idle signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rda.io.schema import EpisodeData

# REQ-3 (v0.6.0): minimum length of a consecutive non-idle segment for it
# to count as "usable" in usable_retention_ratio. Aligned with openpi's
# DROID chunked filter (min_non_idle_len=16 ≈ 1s at typical 15-30Hz).
USABLE_RUN_MIN_FRAMES = 16


# ---------------------------------------------------------------------------
# Idle detection (reuses the same approach as IdleRatioMetric)
# ---------------------------------------------------------------------------

def _primary_action_array(episode: EpisodeData) -> Optional[np.ndarray]:
    """Extract the primary action array from an episode.

    Mirrors ``rda.metrics.motion._primary_action_array`` to stay
    consistent with the existing idle-detection pipeline.
    """
    if not episode.action:
        return None
    for preferred in ("joint_pos", "position", "action"):
        if preferred in episode.action:
            arr = episode.action[preferred]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                return arr
    first_arr: Optional[np.ndarray] = None
    for arr in episode.action.values():
        if not isinstance(arr, np.ndarray):
            continue
        if first_arr is None:
            first_arr = arr
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            return arr
    return first_arr


def _mad(values: np.ndarray) -> float:
    """Median Absolute Deviation."""
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _find_bimodal_threshold(motion: np.ndarray) -> Optional[float]:
    """Find a bimodal-gap threshold in the motion distribution.

    Mirrors ``IdleRatioMetric._find_bimodal_threshold``.
    """
    if motion.size < 10:
        return None
    finite_mask = np.isfinite(motion)
    finite_motion = motion[finite_mask]
    if finite_motion.size < 10:
        return None
    max_motion = float(np.max(finite_motion))
    abs_floor = 1e-6
    num_bins = 30
    min_low_cluster_fraction = 0.1
    if max_motion <= abs_floor:
        return None
    hist, bin_edges = np.histogram(finite_motion, bins=num_bins, range=(0.0, max_motion))
    total = motion.size
    low_cluster_count = 0
    threshold_candidate: Optional[float] = None
    in_valley = False
    valley_min = float("inf")
    for i in range(len(hist)):
        frac = hist[i] / total
        low_cluster_count += hist[i]
        low_frac = low_cluster_count / total
        if low_frac >= min_low_cluster_fraction and frac < 0.01:
            in_valley = True
            if hist[i] < valley_min:
                valley_min = hist[i]
                threshold_candidate = float(bin_edges[i + 1])
        elif in_valley and frac > 0.02:
            if threshold_candidate is not None and threshold_candidate > 0:
                return threshold_candidate
            in_valley = False
            valley_min = float("inf")
    # Valley extends to end without confirmed second peak → fall back to MAD
    return None


def compute_idle_mask(
    episode: EpisodeData,
    mad_multiplier: float = 3.0,
    abs_threshold_floor: float = 1e-6,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Compute the per-frame idle mask for an episode.

    The idle mask has length ``num_frames - 1`` (one entry per
    inter-frame step). A step is "idle" if the action-norm delta
    is below the data-driven threshold.

    Args:
        episode: The episode to analyze.
        mad_multiplier: Multiplier for MAD-based threshold fallback.
        abs_threshold_floor: Absolute minimum threshold.

    Returns:
        A tuple ``(idle_mask, info)`` where ``idle_mask`` is a 1-D
        boolean array (True = idle step) or ``None`` if the episode
        has no action data / too few frames, and ``info`` contains
        metadata about the threshold selection.
    """
    action = _primary_action_array(episode)
    info: Dict[str, Any] = {}

    if action is None or action.size == 0:
        return None, {"reason": "no_action_data"}

    if action.ndim == 1:
        action = action.reshape(-1, 1)
    action = action.astype(np.float64)

    T = action.shape[0]
    if T < 2:
        return None, {"reason": "too_few_frames"}

    delta = np.diff(action, axis=0)
    if not np.all(np.isfinite(delta)):
        return None, {"reason": "non_finite_values"}

    motion = np.linalg.norm(delta, axis=1)

    # Threshold selection — same logic as IdleRatioMetric
    threshold = _find_bimodal_threshold(motion)
    method = "bimodal_gap"
    mad = _mad(motion)

    if threshold is None:
        threshold = mad_multiplier * mad
        method = "mad"
    if threshold < abs_threshold_floor:
        threshold = abs_threshold_floor
        method = "floor" if method == "mad" else method

    idle_mask = motion < threshold

    info.update({
        "threshold": float(threshold),
        "threshold_method": method,
        "motion_mad": float(mad),
        "total_steps": int(motion.size),
    })

    return idle_mask, info


# ---------------------------------------------------------------------------
# Temporal sufficiency data class & computation
# ---------------------------------------------------------------------------

@dataclass
class TemporalSufficiency:
    """Temporal sufficiency metrics for a single episode.

    Attributes:
        total_frames: Total number of frames in the episode.
        idle_total_ratio: Fraction of all steps that are idle.
        idle_prefix_ratio: Fraction of frames that are initial idle prefix.
            The prefix length is the number of consecutive idle steps
            starting from step 0, plus one (frames = steps + 1 for the
            first frame which has no predecessor delta).
        active_run_p50: Median length of consecutive active runs (frames).
        active_run_p90: P90 length of consecutive active runs (frames).
        active_run_max: Maximum length of a single active run (frames).
        transition_count: Number of idle → active transitions.
        idle_to_active_ratio: transition_count / total_steps.
        valid_window_ratio_5: Fraction of 5-frame windows that are fully active.
        valid_window_ratio_10: Fraction of 10-frame windows that are fully active.
        valid_window_ratio_20: Fraction of 20-frame windows that are fully active.
        active_runs: List of active run lengths (for downstream analysis).
        threshold_method: How the idle threshold was determined.
        threshold: Numeric idle threshold value.
    """
    total_frames: int = 0
    idle_total_ratio: float = 0.0
    idle_prefix_ratio: float = 0.0
    idle_prefix_frames: int = 0
    active_run_p50: float = 0.0
    active_run_p90: float = 0.0
    active_run_max: int = 0
    transition_count: int = 0
    idle_to_active_ratio: float = 0.0
    valid_window_ratio_5: float = 0.0
    valid_window_ratio_10: float = 0.0
    valid_window_ratio_20: float = 0.0
    active_runs: List[int] = field(default_factory=list)
    threshold_method: str = ""
    threshold: float = 0.0
    # REQ-3 (v0.6.0): DROID-aligned retention metrics.
    # usable_retention_ratio: fraction of frames inside "usable" runs —
    #   consecutive non-idle segments of at least USABLE_RUN_MIN_FRAMES
    #   frames (DROID min_non_idle_len=16 ≈ 1s at 15-30Hz). Directly
    #   answers "how much data survives a chunk-aligned prune".
    usable_retention_ratio: float = 0.0
    # Longest run of consecutive all-idle chunks (in chunks). A static
    # stretch longer than chunk_size-1 chunks cannot fit any valid
    # chunk window (DROID min_idle_len semantics).
    max_idle_run_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (JSON-friendly)."""
        return {
            "total_frames": self.total_frames,
            "idle_total_ratio": self.idle_total_ratio,
            "idle_prefix_ratio": self.idle_prefix_ratio,
            "idle_prefix_frames": self.idle_prefix_frames,
            "active_run_p50": self.active_run_p50,
            "active_run_p90": self.active_run_p90,
            "active_run_max": self.active_run_max,
            "transition_count": self.transition_count,
            "idle_to_active_ratio": self.idle_to_active_ratio,
            "valid_window_ratio_5": self.valid_window_ratio_5,
            "valid_window_ratio_10": self.valid_window_ratio_10,
            "valid_window_ratio_20": self.valid_window_ratio_20,
            "usable_retention_ratio": self.usable_retention_ratio,
            "max_idle_run_frames": self.max_idle_run_frames,
            "threshold_method": self.threshold_method,
            "threshold": self.threshold,
        }


def compute_temporal_sufficiency(
    episode: EpisodeData,
    mad_multiplier: float = 3.0,
    abs_threshold_floor: float = 1e-6,
    window_sizes: Tuple[int, ...] = (5, 10, 20),
) -> Optional[TemporalSufficiency]:
    """Compute temporal sufficiency metrics for a single episode.

    Args:
        episode: The episode to analyze.
        mad_multiplier: MAD multiplier for idle threshold fallback.
        abs_threshold_floor: Minimum absolute threshold.
        window_sizes: Window sizes for valid-window-ratio computation.

    Returns:
        A :class:`TemporalSufficiency` object, or ``None`` if the
        episode has no usable action data.
    """
    idle_mask, info = compute_idle_mask(
        episode,
        mad_multiplier=mad_multiplier,
        abs_threshold_floor=abs_threshold_floor,
    )

    if idle_mask is None:
        return None

    T_frames = episode.num_frames
    T_steps = idle_mask.size  # T_frames - 1

    result = TemporalSufficiency(total_frames=T_frames)
    result.threshold = info.get("threshold", 0.0)
    result.threshold_method = info.get("threshold_method", "")

    # --- Total idle ratio ---
    idle_steps = int(np.sum(idle_mask))
    result.idle_total_ratio = idle_steps / T_steps if T_steps > 0 else 0.0

    # --- Idle prefix ---
    # Find the first non-idle step; everything before is the initial idle prefix.
    first_active = np.argmax(~idle_mask) if not np.all(idle_mask) else T_steps
    # If all steps are idle, argmax returns 0, but all is idle so prefix = all
    if np.all(idle_mask):
        first_active = T_steps
    # Prefix frames = prefix steps + 1 (the first frame counts too)
    idle_prefix_frames = first_active + 1 if first_active > 0 else 0
    result.idle_prefix_frames = idle_prefix_frames
    result.idle_prefix_ratio = idle_prefix_frames / T_frames if T_frames > 0 else 0.0

    # --- Active run lengths ---
    # Find runs of consecutive active (non-idle) steps.
    # An active "run" of k steps spans k+1 frames.
    runs: List[int] = []
    current_run = 0
    for is_idle in idle_mask:
        if not is_idle:
            current_run += 1
        else:
            if current_run > 0:
                # Convert steps→frames: k steps → k+1 frames
                runs.append(current_run + 1)
                current_run = 0
    if current_run > 0:
        runs.append(current_run + 1)

    result.active_runs = runs
    if runs:
        runs_arr = np.array(runs, dtype=np.float64)
        result.active_run_p50 = float(np.percentile(runs_arr, 50))
        result.active_run_p90 = float(np.percentile(runs_arr, 90))
        result.active_run_max = int(np.max(runs_arr))
    else:
        result.active_run_p50 = 0.0
        result.active_run_p90 = 0.0
        result.active_run_max = 0

    # --- Transitions (idle → active) ---
    # A transition happens when step i-1 is idle and step i is active.
    if T_steps >= 2:
        # idle_mask[i-1] and not idle_mask[i]
        transitions = np.sum(idle_mask[:-1] & ~idle_mask[1:])
        # Also check if the very first step is active (count as "start active")
        if not idle_mask[0]:
            transitions += 1
        result.transition_count = int(transitions)
    else:
        result.transition_count = 0 if T_steps == 0 else (0 if idle_mask[0] else 1)

    result.idle_to_active_ratio = result.transition_count / T_steps if T_steps > 0 else 0.0

    # --- REQ-3 (v0.6.0): DROID-aligned retention metrics ---
    # usable_retention_ratio: frames inside active runs of at least
    # USABLE_RUN_MIN_FRAMES frames (DROID min_non_idle_len=16 ≈ 1s).
    # An active run of k steps spans k+1 frames (see runs above).
    usable_frames = sum(r for r in runs if r >= USABLE_RUN_MIN_FRAMES)
    result.usable_retention_ratio = (
        usable_frames / T_frames if T_frames > 0 else 0.0
    )
    # max_idle_run_frames: longest run of consecutive idle steps, in
    # frames (k idle steps ≈ k+1 consecutive static frames).
    max_idle_steps = 0
    current_idle = 0
    for is_idle in idle_mask:
        if is_idle:
            current_idle += 1
            max_idle_steps = max(max_idle_steps, current_idle)
        else:
            current_idle = 0
    result.max_idle_run_frames = max_idle_steps + 1 if max_idle_steps > 0 else 0

    # --- Valid window ratios ---
    # A window of size N (frames) is "fully active" if none of its
    # N-1 internal steps are idle.
    active_mask_steps = ~idle_mask  # True = active step
    for ws in window_sizes:
        if T_frames < ws:
            # Not enough frames for any window of this size
            ratio = 0.0
        else:
            n_windows = T_frames - ws + 1
            # For window starting at frame i to be fully active,
            # steps i through i+ws-2 must all be active.
            # Using 1-D convolution / cumsum trick.
            # Number of active steps in each window of size ws-1:
            active_steps_int = active_mask_steps.astype(np.int32)
            if ws - 1 == 0:
                # Trivial — window of 1 frame is always "all active"
                valid_count = n_windows
            else:
                # cumsum trick for sliding window sum
                cumsum = np.concatenate([[0], np.cumsum(active_steps_int)])
                window_sums = cumsum[ws - 1:] - cumsum[:-(ws - 1)] if ws - 1 <= len(active_steps_int) else np.array([])
                valid_count = int(np.sum(window_sums == ws - 1))
            ratio = valid_count / n_windows if n_windows > 0 else 0.0

        if ws == 5:
            result.valid_window_ratio_5 = ratio
        elif ws == 10:
            result.valid_window_ratio_10 = ratio
        elif ws == 20:
            result.valid_window_ratio_20 = ratio

    return result


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class DatasetTemporalSufficiency:
    """Aggregated temporal sufficiency across a full dataset.

    All distribution fields use median (p50) as the central measure,
    with p5 and p95 giving the spread.
    """
    total_episodes: int = 0
    total_frames: int = 0
    computed_episodes: int = 0

    idle_total_ratio: Dict[str, float] = field(default_factory=dict)
    idle_prefix_ratio: Dict[str, float] = field(default_factory=dict)
    active_run_p50: Dict[str, float] = field(default_factory=dict)
    active_run_p90: Dict[str, float] = field(default_factory=dict)
    active_run_max: Dict[str, float] = field(default_factory=dict)
    transition_count: Dict[str, float] = field(default_factory=dict)
    valid_window_ratio_5: Dict[str, float] = field(default_factory=dict)
    valid_window_ratio_10: Dict[str, float] = field(default_factory=dict)
    valid_window_ratio_20: Dict[str, float] = field(default_factory=dict)
    # REQ-3 (v0.6.0): DROID-aligned retention metrics (dataset level).
    usable_retention_ratio: Dict[str, float] = field(default_factory=dict)
    max_idle_run_frames: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "computed_episodes": self.computed_episodes,
            "idle_total_ratio": self.idle_total_ratio,
            "idle_prefix_ratio": self.idle_prefix_ratio,
            "active_run_p50": self.active_run_p50,
            "active_run_p90": self.active_run_p90,
            "active_run_max": self.active_run_max,
            "transition_count": self.transition_count,
            "valid_window_ratio_5": self.valid_window_ratio_5,
            "valid_window_ratio_10": self.valid_window_ratio_10,
            "valid_window_ratio_20": self.valid_window_ratio_20,
            "usable_retention_ratio": self.usable_retention_ratio,
            "max_idle_run_frames": self.max_idle_run_frames,
        }


def aggregate_temporal_sufficiency(
    per_episode: List[TemporalSufficiency],
    total_episodes: int,
    total_frames: int,
) -> DatasetTemporalSufficiency:
    """Aggregate per-episode temporal sufficiency to dataset level.

    Args:
        per_episode: List of per-episode results (may contain None entries
            for episodes where the metric couldn't be computed).
        total_episodes: Total number of episodes in the dataset.
        total_frames: Total number of frames in the dataset.

    Returns:
        :class:`DatasetTemporalSufficiency` with distribution stats.
    """
    valid = [ts for ts in per_episode if ts is not None]

    result = DatasetTemporalSufficiency(
        total_episodes=total_episodes,
        total_frames=total_frames,
        computed_episodes=len(valid),
    )

    if not valid:
        return result

    def _dist(values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {"p5": 0.0, "median": 0.0, "p95": 0.0}
        return {
            "p5": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }

    result.idle_total_ratio = _dist(
        np.array([ts.idle_total_ratio for ts in valid], dtype=np.float64)
    )
    result.idle_prefix_ratio = _dist(
        np.array([ts.idle_prefix_ratio for ts in valid], dtype=np.float64)
    )
    result.active_run_p50 = _dist(
        np.array([ts.active_run_p50 for ts in valid], dtype=np.float64)
    )
    result.active_run_p90 = _dist(
        np.array([ts.active_run_p90 for ts in valid], dtype=np.float64)
    )
    result.active_run_max = _dist(
        np.array([float(ts.active_run_max) for ts in valid], dtype=np.float64)
    )
    result.transition_count = _dist(
        np.array([float(ts.transition_count) for ts in valid], dtype=np.float64)
    )
    result.valid_window_ratio_5 = _dist(
        np.array([ts.valid_window_ratio_5 for ts in valid], dtype=np.float64)
    )
    result.valid_window_ratio_10 = _dist(
        np.array([ts.valid_window_ratio_10 for ts in valid], dtype=np.float64)
    )
    result.valid_window_ratio_20 = _dist(
        np.array([ts.valid_window_ratio_20 for ts in valid], dtype=np.float64)
    )
    # REQ-3 (v0.6.0): DROID-aligned retention metrics at dataset level.
    result.usable_retention_ratio = _dist(
        np.array([ts.usable_retention_ratio for ts in valid], dtype=np.float64)
    )
    result.max_idle_run_frames = _dist(
        np.array([float(ts.max_idle_run_frames) for ts in valid], dtype=np.float64)
    )

    return result
