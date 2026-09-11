"""Motion metrics: JointLimitMetric, VelocityMetric, ActionDiscontinuityMetric, IdleRatioMetric.

These metrics answer the motion half of Q2: "Are the robot data's
time and motion sane?" — focusing on joint limits, action
smoothness, and idle/effective-motion ratio.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rda.io.schema import EpisodeData
from rda.metrics.base import MetricBase, MetricResult, MetricAvailability


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _primary_action_array(episode: EpisodeData) -> Optional[np.ndarray]:
    """Find the primary action array from an episode.

    Looks for well-known action keys (``joint_pos``, ``position``,
    ``action``) first, then falls back to the first 2-D floating-point
    action array, then to any action array at all.

    Args:
        episode: The episode to search for an action array in.

    Returns:
        The primary action ndarray, or ``None`` if no action data exists.
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


def _percentiles(values: np.ndarray, ps: Tuple[float, ...] = (50.0, 95.0, 99.0)) -> Dict[str, float]:
    """Compute named percentile statistics from a 1-D array.

    Percentile 50 is named ``"median"``; others are named ``"p{int(p)}"``.

    Args:
        values: 1-D array of numeric values.
        ps: Tuple of percentile levels (0–100).

    Returns:
        Dict mapping percentile name to float value.
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


def _mad(values: np.ndarray) -> float:
    """Compute the Median Absolute Deviation (MAD) of a 1-D array.

    MAD is a robust measure of statistical dispersion, less sensitive
    to outliers than standard deviation.

    Args:
        values: 1-D array of numeric values.

    Returns:
        The MAD value as a float. Returns 0.0 for empty arrays.
    """
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


# ---------------------------------------------------------------------------
# Metric 07 — Joint Limit Violation
# ---------------------------------------------------------------------------

class JointLimitMetric(MetricBase):
    """Three-level joint-limit check (D-14).

    Classification
    --------------
    PASS    – all frames inside limits; inter-frame jumps normal.
    REVIEW  – any frame within ``approach_threshold`` (±2 %) of a limit
              boundary, OR >= ``consecutive_frames`` consecutive frames
              glued to a boundary.
    EXCLUDE – any frame actually exceeds a limit, OR inter-frame jump
              exceeds ``jump_multiplier`` x theoretical max increment.

    Configurable parameters
    -----------------------
    approach_threshold : float = 0.02
        Fraction of full range considered "approaching".
    consecutive_frames : int = 3
        Minimum consecutive frames at boundary for REVIEW.
    jump_multiplier : float = 2.0
        Jump / theoretical-max-increment ratio above which -> EXCLUDE.
    """

    name = "joint_limit"
    description = "Three-level joint-limit check: PASS / REVIEW / EXCLUDE."

    def __init__(
        self,
        approach_threshold: float = 0.02,
        consecutive_frames: int = 3,
        jump_multiplier: float = 2.0,
    ) -> None:
        self.approach_threshold = approach_threshold
        self.consecutive_frames = consecutive_frames
        self.jump_multiplier = jump_multiplier

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _max_consecutive_run(mask: np.ndarray) -> int:
        """Return length of the longest True-run in a boolean array."""
        if mask.size == 0:
            return 0
        changes = np.diff(mask.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        # Handle edge: starts with True
        if mask[0]:
            starts = np.concatenate([[0], starts])
        # Handle edge: ends with True
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        if len(starts) == 0 or len(ends) == 0:
            return 0
        n = min(len(starts), len(ends))
        if n == 0:
            return 0
        runs = ends[:n] - starts[:n]
        return int(runs.max()) if runs.size else 0

    # ----- main compute ---------------------------------------------------

    def compute(self, episode: EpisodeData) -> MetricResult:
        details: Dict[str, Any] = {
            "violations": 0,
            "approaching": 0,
            "by_joint": {},
            "joints_checked": 0,
            "approach_threshold_used": self.approach_threshold,
            "consecutive_frames_used": self.consecutive_frames,
            "jump_multiplier_used": self.jump_multiplier,
            "min_margin_ratio": 1.0,
            "max_jump_ratio": 0.0,
        }

        limits = episode.meta.get("joint_limits")
        if limits is None:
            return MetricResult.make_na(
                name=self.name,
                reason="joint_limits_not_provided",
                message="Joint limits not provided; skipping check.",
                details=details,
            )

        state_arr = episode.observation.get("state")
        if state_arr is None:
            return MetricResult.make_na(
                name=self.name,
                reason="observation_state_missing",
                message="observation.state not available; skipping check.",
                details=details,
            )

        if not isinstance(state_arr, np.ndarray) or state_arr.ndim < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="state_array_invalid_shape",
                message="observation.state has invalid shape; skipping check.",
                details=details,
            )

        n_frames, n_joints_state = state_arr.shape
        n_joints_limits = len(limits)
        n_checked = min(n_joints_state, n_joints_limits)
        details["joints_checked"] = n_checked

        if n_checked == 0:
            return MetricResult.make_na(
                name=self.name,
                reason="no_matching_joint_dimensions",
                message="No matching joint dimensions; skipping check.",
                details=details,
            )

        # Velocity limits for jump detection (optional metadata)
        velocity_limits = episode.meta.get("velocity_limits")

        total_violations = 0
        total_approaching = 0
        by_joint: Dict[str, Any] = {}
        global_min_margin = 1.0
        global_max_jump_ratio = 0.0

        for j in range(n_checked):
            low, high = limits[j]
            joint_range = high - low
            if joint_range <= 0:
                continue

            col = state_arr[:, j].astype(np.float64)

            # margin_ratio: distance to nearest boundary as fraction of range
            #   > 0 means inside, < 0 means outside, 0 means exactly at boundary
            margin_low = (col - low) / joint_range
            margin_high = (high - col) / joint_range
            margin_ratio = np.minimum(margin_low, margin_high)

            frame_min_margin = float(np.min(margin_ratio))
            if frame_min_margin < global_min_margin:
                global_min_margin = frame_min_margin

            # --- violation: any frame beyond limit ---
            violation_mask = margin_ratio < 0.0
            violation_count = int(violation_mask.sum())

            # --- approaching: within threshold but not yet violated ---
            approaching_mask = (margin_ratio >= 0.0) & (margin_ratio < self.approach_threshold)
            approaching_count = int(approaching_mask.sum())

            # --- consecutive at boundary (margin < 0.001 = 0.1% of range) ---
            at_boundary = margin_ratio < 0.001
            max_consec = self._max_consecutive_run(at_boundary)

            # --- inter-frame jump detection ---
            max_jump_ratio_j = 0.0
            if n_frames > 1:
                diffs = np.abs(np.diff(col))
                max_jump = float(np.max(diffs))
                # Theoretical max increment from velocity_limits or fallback
                if velocity_limits is not None and j < len(velocity_limits):
                    v_limit = float(velocity_limits[j])
                    fps = float(episode.meta.get("fps", 10.0))
                    dt = 1.0 / fps if fps > 0 else 0.1
                    max_theoretical_jump = v_limit * dt
                else:
                    # Fallback: assume max velocity = joint_range per second
                    fps = float(episode.meta.get("fps", 10.0))
                    dt = 1.0 / fps if fps > 0 else 0.1
                    max_theoretical_jump = joint_range * dt
                if max_theoretical_jump > 0:
                    max_jump_ratio_j = max_jump / max_theoretical_jump
                if max_jump_ratio_j > global_max_jump_ratio:
                    global_max_jump_ratio = max_jump_ratio_j

            # Per-joint record (only if noteworthy)
            if violation_count > 0 or approaching_count > 0 or max_consec >= self.consecutive_frames:
                by_joint[f"joint_{j}"] = {
                    "violations": violation_count,
                    "approaching": approaching_count,
                    "max_consecutive_at_boundary": max_consec,
                    "max_jump_ratio": round(max_jump_ratio_j, 4),
                    "min_margin": round(frame_min_margin, 6),
                }

            total_violations += violation_count
            total_approaching += approaching_count

        details["violations"] = total_violations
        details["approaching"] = total_approaching
        details["by_joint"] = by_joint
        details["min_margin_ratio"] = round(global_min_margin, 6)
        details["max_jump_ratio"] = round(global_max_jump_ratio, 4)

        measurement = {
            "score_compat": 1.0 if (total_violations == 0 and total_approaching == 0) else 0.0,
            "violations": total_violations,
            "approaching": total_approaching,
            "joints_checked": n_checked,
            "min_margin_ratio": round(global_min_margin, 6),
            "max_jump_ratio": round(global_max_jump_ratio, 4),
        }

        # === EXCLUDE: actual violations or excessive jumps ===
        if total_violations > 0 or global_max_jump_ratio > self.jump_multiplier:
            parts: List[str] = []
            if total_violations > 0:
                parts.append(f"{total_violations} frame(s) exceeding joint limits")
            if global_max_jump_ratio > self.jump_multiplier:
                parts.append(
                    f"max inter-frame jump {global_max_jump_ratio:.1f}x "
                    f"theoretical max"
                )
            msg = "Joint limit check failed: " + "; ".join(parts) + "."
            measurement["score_compat"] = 0.0
            return MetricResult.make_exclude(
                name=self.name,
                reason="; ".join(parts),
                message=msg,
                details=details,
            )

        # === REVIEW: approaching boundary or consecutive at boundary ===
        has_approaching = total_approaching > 0
        has_consecutive = any(
            v.get("max_consecutive_at_boundary", 0) >= self.consecutive_frames
            for v in by_joint.values()
        )
        if has_approaching or has_consecutive:
            parts = []
            if has_approaching:
                parts.append(
                    f"{total_approaching} frame(s) within "
                    f"{self.approach_threshold * 100:.0f}% of limit boundary"
                )
            if has_consecutive:
                parts.append(
                    f"{self.consecutive_frames}+ consecutive frames "
                    f"at boundary"
                )
            msg = "Joint limit check: " + "; ".join(parts) + "."
            measurement["score_compat"] = 0.5
            return MetricResult.make_review(
                name=self.name,
                measurement=measurement,
                reason="; ".join(parts),
                message=msg,
                details=details,
                severity="medium",
            )

        # === PASS: all clear ===
        msg = (
            f"No joint-limit violations or boundary approaches "
            f"across {n_checked} joint(s)."
        )
        measurement["score_compat"] = 1.0
        return MetricResult.make_pass(
            name=self.name,
            measurement=measurement,
            message=msg,
            details=details,
        )


# ---------------------------------------------------------------------------
# Metric 09 — Action Discontinuity / Jerk (Hero Metric #2)
# ---------------------------------------------------------------------------

class ActionDiscontinuityMetric(MetricBase):
    """Detect abrupt jumps in action trajectories.
    
    P1 CHANGE: Pure observational metric — always passes.
    Reports spike_count, rates, per-joint breakdown as measurement.
    """

    name = "action_discontinuity"
    description = "Detect abrupt jumps / discontinuities in action trajectories using MAD-based outlier detection."

    spike_threshold: float = 5.0

    def __init__(self, spike_threshold: float = 5.0) -> None:
        self.spike_threshold = float(spike_threshold)

    def _mad_zscore(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.array([], dtype=np.float64)
        med = float(np.median(values))
        mad = _mad(values)
        if mad == 0.0:
            return np.zeros_like(values, dtype=np.float64)
        return 0.6745 * (values - med) / mad

    def _detect_spikes(self, series: np.ndarray) -> Tuple[int, float, np.ndarray]:
        if series.size == 0:
            return 0, 0.0, np.array([], dtype=np.int64)
        z = np.abs(self._mad_zscore(series))
        spike_mask = z > self.spike_threshold
        indices = np.where(spike_mask)[0]
        return int(indices.size), float(np.max(series)), indices

    def compute(self, episode: EpisodeData) -> MetricResult:
        action = _primary_action_array(episode)
        details: Dict[str, Any] = {
            "action_shape": None,
            "spike_count": 0,
            "max_delta": 0.0,
            "max_second_delta": 0.0,
            "spike_indices": [],
            "delta_stats": {},
            "delta2_stats": {},
            "by_joint": {},
        }

        if action is None or action.size == 0:
            return MetricResult.make_na(
                name=self.name,
                reason="action_array_missing",
                message="No action array available; skipping.",
                details=details,
            )

        if action.ndim == 1:
            action = action.reshape(-1, 1)
        action = action.astype(np.float64)

        # Guard against NaN/Inf
        if not np.all(np.isfinite(action)):
            nan_count = int(~np.isfinite(action).any(axis=1).sum())
            return MetricResult.make_exclude(
                name=self.name,
                reason=f"non_finite_values_in_action ({nan_count} frames contain NaN/Inf)",
                message=f"Action data contains {nan_count} frames with NaN/Inf values; data is corrupted.",
                details={"nan_frames": nan_count, "total_frames": action.shape[0]},
            )

        T, D = action.shape
        details["action_shape"] = [T, D]

        if T < 3:
            return MetricResult.make_na(
                name=self.name,
                reason="too_few_frames",
                message="Fewer than 3 frames — cannot detect discontinuities.",
                details=details,
            )

        # First difference
        delta = np.diff(action, axis=0)
        delta_norm = np.linalg.norm(delta, axis=1)

        # Second difference
        delta2 = np.diff(delta, axis=0)
        delta2_norm = np.linalg.norm(delta2, axis=1)

        # Spike detection on second diff (jerk proxy)
        spike_count, max_delta2, spike_idx = self._detect_spikes(delta2_norm)
        details["spike_count"] = spike_count
        details["max_delta"] = float(np.max(delta_norm)) if delta_norm.size else 0.0
        details["max_second_delta"] = float(max_delta2)
        details["spike_indices"] = spike_idx.tolist()

        # Statistics
        delta_stats = _percentiles(delta_norm, (50.0, 95.0, 99.0))
        delta_stats["max"] = float(np.max(delta_norm)) if delta_norm.size else 0.0
        delta_stats["mean"] = float(np.mean(delta_norm)) if delta_norm.size else 0.0
        details["delta_stats"] = delta_stats

        delta2_stats = _percentiles(delta2_norm, (50.0, 95.0, 99.0))
        delta2_stats["max"] = float(max_delta2)
        delta2_stats["mean"] = float(np.mean(delta2_norm)) if delta2_norm.size else 0.0
        details["delta2_stats"] = delta2_stats

        # Per-joint breakdown
        by_joint: Dict[str, Dict[str, Any]] = {}
        for j in range(D):
            j_delta2 = delta2[:, j] if D > 1 else delta2.ravel()
            j_spikes, j_max, _ = self._detect_spikes(np.abs(j_delta2))
            if j_spikes > 0:
                by_joint[f"joint_{j}"] = {"spike_count": j_spikes, "max_second_delta": j_max}
        details["by_joint"] = by_joint

        # --- OBSERVATIONAL: always pass, report as measurement ---
        n_steps = max(delta2_norm.size, 1)
        spike_ratio = spike_count / n_steps
        affected_joints = len(by_joint)

        msg = (
            f"Action discontinuity: {spike_count} spike(s) "
            f"({spike_ratio:.2%} of steps, {affected_joints} joint(s) affected); "
            f"median Δa = {delta_stats['median']:.4f}, "
            f"median Δ²a = {delta2_stats['median']:.4f}."
        )

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "spike_count": spike_count,
                "spike_rate": spike_ratio,
                "p95_delta": delta_stats.get("p95", 0.0),
                "p99_delta": delta_stats.get("p99", 0.0),
                "max_delta": details["max_delta"],
                "max_second_delta": float(max_delta2),
                "affected_joints": affected_joints,
                "total_joints": D,
            },
            message=msg,
            details=details,
            baseline={
                "method": "MAD_zscore",
                "scope": "episode",
                "threshold": self.spike_threshold,
                "reference_population": int(delta2_norm.size),
            },
        )


# ---------------------------------------------------------------------------
# Metric 10 — Idle / Effective-motion Ratio
# ---------------------------------------------------------------------------

class IdleRatioMetric(MetricBase):
    """Measure the proportion of idle / near-static frames.
    
    P1 CHANGE: Pure observational metric.
    Reports effective_motion_ratio = 1 - idle_ratio prominently.
    """

    name = "idle_ratio"
    description = "Measure the ratio of idle / near-static frames with robust data-driven threshold."

    mad_multiplier: float = 3.0
    abs_threshold_floor: float = 1e-6
    num_bins: int = 30
    min_low_cluster_fraction: float = 0.1

    def __init__(self, mad_multiplier: float = 3.0, abs_threshold_floor: float = 1e-6,
                 num_bins: int = 30, min_low_cluster_fraction: float = 0.1) -> None:
        self.mad_multiplier = float(mad_multiplier)
        self.abs_threshold_floor = float(abs_threshold_floor)
        self.num_bins = int(num_bins)
        self.min_low_cluster_fraction = float(min_low_cluster_fraction)

    def _find_bimodal_threshold(self, motion: np.ndarray) -> Optional[float]:
        if motion.size < 10:
            return None
        # Guard against NaN/Inf — filter to finite values only
        finite_mask = np.isfinite(motion)
        finite_motion = motion[finite_mask]
        if finite_motion.size < 10:
            return None
        max_motion = float(np.max(finite_motion))
        if max_motion <= self.abs_threshold_floor:
            return None
        hist, bin_edges = np.histogram(finite_motion, bins=self.num_bins, range=(0.0, max_motion))
        total = motion.size
        low_cluster_count = 0
        threshold_candidate: Optional[float] = None
        in_valley = False
        valley_min = float("inf")

        for i in range(len(hist)):
            frac = hist[i] / total
            low_cluster_count += hist[i]
            low_frac = low_cluster_count / total
            if low_frac >= self.min_low_cluster_fraction and frac < 0.01:
                in_valley = True
                if hist[i] < valley_min:
                    valley_min = hist[i]
                    threshold_candidate = float(bin_edges[i + 1])
            elif in_valley and frac > 0.02:
                if threshold_candidate is not None and threshold_candidate > 0:
                    return threshold_candidate
                in_valley = False
                valley_min = float("inf")
        return None

    def compute(self, episode: EpisodeData) -> MetricResult:
        action = _primary_action_array(episode)
        details: Dict[str, Any] = {
            "idle_ratio": 0.0,
            "idle_steps": 0,
            "total_steps": 0,
            "threshold": 0.0,
            "motion_stats": {},
        }

        if action is None or action.size == 0:
            return MetricResult.make_na(
                name=self.name,
                reason="action_array_missing",
                message="No action array available; skipping.",
                details=details,
            )

        if action.ndim == 1:
            action = action.reshape(-1, 1)
        action = action.astype(np.float64)

        T = action.shape[0]
        details["total_steps"] = max(T - 1, 0)

        if T < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="too_few_frames",
                message="Fewer than 2 frames — cannot compute idle ratio.",
                details=details,
            )

        delta = np.diff(action, axis=0)

        # Guard against NaN/Inf in action data
        if not np.all(np.isfinite(delta)):
            nan_count = int(~np.isfinite(delta).any(axis=1).sum()) if delta.ndim == 2 else int(~np.isfinite(delta).sum())
            return MetricResult.make_exclude(
                name=self.name,
                reason=f"non_finite_values_in_action ({nan_count} frames contain NaN/Inf)",
                message=f"Action data contains {nan_count} frames with NaN/Inf values; data is corrupted.",
                details={"nan_frames": nan_count, "total_frames": T},
            )

        motion = np.linalg.norm(delta, axis=1)

        # Threshold selection
        threshold = self._find_bimodal_threshold(motion)
        method = "bimodal_gap"
        mad = _mad(motion)

        if threshold is None:
            threshold = self.mad_multiplier * mad
            method = "mad"
        if threshold < self.abs_threshold_floor:
            threshold = self.abs_threshold_floor
            method = "floor" if method == "mad" else method

        details["threshold"] = float(threshold)
        details["threshold_method"] = method

        idle_mask = motion < threshold
        idle_steps = int(idle_mask.sum())
        idle_ratio = idle_steps / motion.size
        effective_motion_ratio = 1.0 - idle_ratio
        details["idle_steps"] = idle_steps
        details["idle_ratio"] = float(idle_ratio)
        details["effective_motion_ratio"] = float(effective_motion_ratio)
        details["active_frames"] = int(motion.size - idle_steps)

        stats = _percentiles(motion, (50.0, 95.0, 99.0))
        stats["max"] = float(np.max(motion)) if motion.size else 0.0
        stats["mean"] = float(np.mean(motion)) if motion.size else 0.0
        stats["mad"] = float(mad)
        details["motion_stats"] = stats

        # --- OBSERVATIONAL: always pass ---
        total_steps = motion.size
        active_frames = total_steps - idle_steps
        msg = (
            f"Effective motion: {effective_motion_ratio:.1%} "
            f"(active {active_frames} frames / total {total_steps} frames); "
            f"idle ratio = {idle_ratio:.2%}, threshold = {threshold:.6f}."
        )

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "idle_ratio": float(idle_ratio),
                "effective_motion_ratio": float(effective_motion_ratio),
                "idle_steps": idle_steps,
                "total_steps": total_steps,
                "active_frames": active_frames,
            },
            message=msg,
            details=details,
            baseline={
                "method": method,
                "scope": "episode",
                "threshold": float(threshold),
                "reference_population": int(motion.size),
            },
        )


# ---------------------------------------------------------------------------
# Metric 08 — Velocity / Acceleration Anomaly
# ---------------------------------------------------------------------------

class VelocityMetric(MetricBase):
    """Analyze velocity profiles and detect extreme spikes.
    
    P1 CHANGE: Pure observational metric — always passes.
    Reports all stats as measurement, no pass/fail judgment.
    """

    name = "velocity_acceleration"
    description = "Compute velocity / acceleration statistics and detect extreme spikes."

    def compute(self, episode: EpisodeData) -> MetricResult:
        state = episode.observation.get("state")
        details: Dict[str, Any] = {
            "velocity_stats": {},
            "acceleration_stats": {},
            "extreme_spike_count": 0,
        }

        if state is None or not isinstance(state, np.ndarray) or state.ndim < 2:
            return MetricResult.make_na(
                name=self.name,
                reason="observation_state_missing",
                message="observation.state not available; skipping.",
                details=details,
            )

        state = state.astype(np.float64)

        # Guard against NaN/Inf
        if not np.all(np.isfinite(state)):
            nan_count = int(~np.isfinite(state).any(axis=1).sum())
            return MetricResult.make_exclude(
                name=self.name,
                reason=f"non_finite_values_in_state ({nan_count} frames contain NaN/Inf)",
                message=f"observation.state contains {nan_count} frames with NaN/Inf values; data is corrupted.",
                details={"nan_frames": nan_count, "total_frames": state.shape[0]},
            )

        T = state.shape[0]

        if T < 3:
            return MetricResult.make_na(
                name=self.name,
                reason="too_few_frames",
                message="Fewer than 3 frames; skipping.",
                details=details,
            )

        ts = episode.timestamps
        if ts is not None and ts.size == T:
            dt = np.diff(ts.astype(np.float64))
            dt = np.where(dt == 0, 1e-9, dt)
            vel = np.diff(state, axis=0) / dt[:, None]
        else:
            vel = np.diff(state, axis=0)

        vel_norm = np.linalg.norm(vel, axis=1)
        vel_stats = _percentiles(vel_norm, (50.0, 95.0, 99.0))
        vel_stats["max"] = float(np.max(vel_norm)) if vel_norm.size else 0.0
        vel_stats["mean"] = float(np.mean(vel_norm)) if vel_norm.size else 0.0
        details["velocity_stats"] = vel_stats

        extreme_spike_count = 0
        if vel_norm.size >= 2:
            acc = np.diff(vel, axis=0)
            acc_norm = np.linalg.norm(acc, axis=1)
            acc_stats = _percentiles(acc_norm, (50.0, 95.0, 99.0))
            acc_stats["max"] = float(np.max(acc_norm)) if acc_norm.size else 0.0
            details["acceleration_stats"] = acc_stats

            mad = _mad(acc_norm)
            if mad > 0:
                med = float(np.median(acc_norm))
                z = 0.6745 * np.abs(acc_norm - med) / mad
                extreme_spike_count = int(np.sum(z > 5.0))
        details["extreme_spike_count"] = extreme_spike_count

        # --- OBSERVATIONAL: always pass ---
        msg = (
            f"Velocity: p50={vel_stats.get('median', 0):.4f}, "
            f"p95={vel_stats.get('p95', 0):.4f}, "
            f"p99={vel_stats.get('p99', 0):.4f}, "
            f"extreme accel spikes = {extreme_spike_count}."
        )

        return MetricResult.make_pass(
            name=self.name,
            measurement={
                "score_compat": 1.0,
                "velocity_p50": vel_stats.get("median", 0.0),
                "velocity_p95": vel_stats.get("p95", 0.0),
                "velocity_p99": vel_stats.get("p99", 0.0),
                "velocity_max": vel_stats.get("max", 0.0),
                "acceleration_spikes": extreme_spike_count,
                "acceleration_spike_rate": extreme_spike_count / max(T, 1),
            },
            message=msg,
            details=details,
        )
