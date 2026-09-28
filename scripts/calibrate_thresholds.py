#!/usr/bin/env python3
"""ArmnetBench threshold calibration script.

Loads the ArmnetBench dataset (SO-101, 2499 episodes), computes RDA metrics
for each episode, and uses the ground-truth reward labels to find optimal
thresholds via Youden's J statistic.

Usage:
    python scripts/calibrate_thresholds.py <dataset_path> [--output <json_path>]

Example:
    python scripts/calibrate_thresholds.py /path/to/armnetbench_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure RDA is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from rda.io.schema import EpisodeData
from rda.metrics.motion import (
    ActionDiscontinuityMetric,
    IdleRatioMetric,
    VelocityMetric,
)


def load_episodes_from_parquet(dataset_path: str) -> Tuple[List[EpisodeData], List[float]]:
    """Load all episodes from a LeRobot-format dataset directory.

    Returns:
        Tuple of (episodes, labels) where labels are the last-frame rewards.
    """
    data_dir = Path(dataset_path) / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Read all parquet files and group by episode
    all_dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(str(pf))
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: failed to read {pf}: {e}", file=sys.stderr)

    if not all_dfs:
        raise RuntimeError("No parquet files could be read")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Loaded {len(combined)} frames from {len(parquet_files)} parquet files")

    episodes = []
    labels = []

    for ep_idx, group in combined.groupby("episode_index"):
        group = group.sort_values("frame_index")
        ep_idx = int(ep_idx)

        # Build EpisodeData
        timestamps = group["timestamp"].values.astype(np.float64) if "timestamp" in group.columns else np.arange(len(group), dtype=np.float64) / 20.0

        # Action
        action_data = np.stack(group["action"].values)

        # Observation state
        obs = {}
        if "observation.state" in group.columns:
            obs["state"] = np.stack(group["observation.state"].values)

        # Reward
        reward = group["next.reward"].values.astype(np.float64) if "next.reward" in group.columns else None

        episode = EpisodeData(
            episode_index=ep_idx,
            num_frames=len(group),
            timestamps=timestamps,
            observation=obs,
            action={"action": action_data.astype(np.float64)},
            reward=reward,
        )
        episodes.append(episode)

        # Label: last frame reward (1.0 = success, 0.0 = failure)
        last_reward = float(group.iloc[-1]["next.reward"]) if "next.reward" in group.columns else 0.0
        labels.append(last_reward)

    print(f"  Parsed {len(episodes)} episodes")
    return episodes, labels


def compute_metrics_for_episodes(
    episodes: List[EpisodeData],
) -> Dict[str, List[float]]:
    """Compute key metrics for all episodes.

    Returns dict of metric_name -> list of values (one per episode).
    """
    spike_metric = ActionDiscontinuityMetric()
    idle_metric = IdleRatioMetric()
    velocity_metric = VelocityMetric()

    results: Dict[str, List[float]] = {
        "spike_count": [],
        "effective_motion_ratio": [],
        "idle_ratio": [],
        "velocity_p95": [],
        "duration_sec": [],
    }

    for i, ep in enumerate(episodes):
        # Action discontinuity
        ad_result = spike_metric.compute(ep)
        results["spike_count"].append(float(ad_result.measurement.get("spike_count", 0)))

        # Idle ratio
        ir_result = idle_metric.compute(ep)
        results["effective_motion_ratio"].append(float(ir_result.measurement.get("effective_motion_ratio", 1.0)))
        results["idle_ratio"].append(float(ir_result.measurement.get("idle_ratio", 0.0)))

        # Velocity
        vel_result = velocity_metric.compute(ep)
        results["velocity_p95"].append(float(vel_result.measurement.get("velocity_p95", 0.0)))

        # Duration
        if ep.timestamps is not None and len(ep.timestamps) >= 2:
            dur = float(ep.timestamps[-1] - ep.timestamps[0])
        else:
            dur = ep.num_frames / 20.0  # Assume 20 FPS
        results["duration_sec"].append(dur)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(episodes)} episodes")

    return results


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC using Mann-Whitney U statistic.

    labels: 1 = positive, 0 = negative
    scores: higher = more likely positive
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Count concordant pairs
    n_concordant = 0
    n_tied = 0
    for p in pos:
        n_concordant += int((p > neg).sum())
        n_tied += int((p == neg).sum())
    return float((n_concordant + 0.5 * n_tied) / (n_pos * n_neg))


def youden_j_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    direction: str = "higher_is_positive",
) -> Tuple[float, float, float, float]:
    """Find optimal threshold using Youden's J statistic.

    J = sensitivity + specificity - 1 = TPR + TNR - 1

    Args:
        scores: metric values
        labels: binary labels (1 = positive, 0 = negative)
        direction: "higher_is_positive" means higher scores → more likely label=1

    Returns:
        (threshold, J, sensitivity, specificity) at optimal point
    """
    if direction == "higher_is_negative":
        scores = -scores

    unique_scores = np.unique(scores)
    # Add midpoints between unique values for better resolution
    thresholds = []
    for i in range(len(unique_scores) - 1):
        thresholds.append((unique_scores[i] + unique_scores[i + 1]) / 2)
    thresholds = np.array(thresholds)

    if len(thresholds) == 0:
        return float(np.median(scores)), 0.0, 0.5, 0.5

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float(np.median(scores)), 0.0, 0.5, 0.5

    best_j = -1
    best_thresh = float(np.median(scores))
    best_sens = 0.5
    best_spec = 0.5

    for thresh in thresholds:
        predicted_positive = scores >= thresh
        tp = (predicted_positive & (labels == 1)).sum()
        tn = (~predicted_positive & (labels == 0)).sum()
        sens = tp / n_pos
        spec = tn / n_neg
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = float(thresh)
            best_sens = float(sens)
            best_spec = float(spec)

    return best_thresh, float(best_j), best_sens, best_spec


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size.

    delta = P(X > Y) - P(X < Y)
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    # Vectorized computation
    gt = 0
    lt = 0
    for xi in x:
        gt += (xi > y).sum()
        lt += (xi < y).sum()
    return float((gt - lt) / (n_x * n_y))


def percentile_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute distribution statistics."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="ArmnetBench threshold calibration")
    parser.add_argument("dataset_path", help="Path to ArmnetBench dataset directory")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from: {dataset_path}")

    # Load episodes and labels
    episodes, labels = load_episodes_from_parquet(str(dataset_path))
    labels_arr = np.array(labels)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics_for_episodes(episodes)

    # Separate by label
    success_mask = labels_arr == 1.0
    failure_mask = labels_arr == 0.0

    n_success = success_mask.sum()
    n_failure = failure_mask.sum()
    print(f"\nDistribution: {n_success} successful, {n_failure} failure")

    # For calibration, we treat success=0, failure=1 (failure is the "positive" class)
    binary_labels = failure_mask.astype(int)

    report: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "n_episodes": len(episodes),
        "n_success": int(n_success),
        "n_failure": int(n_failure),
        "metrics": {},
        "optimal_thresholds": {},
    }

    # Analyze each metric
    for metric_name, values in metrics.items():
        values_arr = np.array(values)
        success_vals = values_arr[success_mask]
        failure_vals = values_arr[failure_mask]

        print(f"\n{'='*60}")
        print(f"Metric: {metric_name}")
        print(f"  Success: mean={success_vals.mean():.4f}, median={np.median(success_vals):.4f}")
        print(f"  Failure: mean={failure_vals.mean():.4f}, median={np.median(failure_vals):.4f}")

        # Distribution stats
        report["metrics"][metric_name] = {
            "success": percentile_stats(success_vals),
            "failure": percentile_stats(failure_vals),
        }

        # ROC-AUC: P(metric_value_f > metric_value_s) for concordant pairs
        # If failure has higher values: AUC > 0.5 when scoring directly
        # If failure has lower values: AUC < 0.5 when scoring directly, use 1-AUC
        auc_raw = compute_roc_auc(values_arr, binary_labels)
        auc = auc_raw if auc_raw >= 0.5 else 1.0 - auc_raw
        report["metrics"][metric_name]["roc_auc"] = auc
        print(f"  ROC-AUC: {auc:.4f}")

        # Cliff's delta: P(failure > success) - P(failure < success)
        # Positive = failure has higher values
        cd = cliffs_delta(failure_vals, success_vals)
        report["metrics"][metric_name]["cliffs_delta"] = cd
        abs_cd = abs(cd)
        interpretation = "negligible" if abs_cd < 0.147 else "small" if abs_cd < 0.33 else "medium" if abs_cd < 0.474 else "large"
        print(f"  Cliff's δ: {cd:+.4f} (|δ|={abs_cd:.4f}, {interpretation})")

        # Optimal threshold (Youden's J)
        # Determine direction: does higher or lower value predict failure?
        if failure_vals.mean() > success_vals.mean():
            direction_label = "higher_is_worse"
            thresh, j, sens, spec = youden_j_threshold(
                values_arr, binary_labels, direction="higher_is_positive",
            )
        else:
            direction_label = "lower_is_worse"
            thresh, j, sens, spec = youden_j_threshold(
                -values_arr, binary_labels, direction="higher_is_positive",
            )
            thresh = -thresh  # Convert back to original scale

        report["optimal_thresholds"][metric_name] = {
            "threshold": thresh,
            "youden_j": j,
            "sensitivity": sens,
            "specificity": spec,
            "direction": direction_label,
        }
        print(f"  Optimal threshold: {thresh:.4f} (J={j:.4f}, sens={sens:.4f}, spec={spec:.4f})")

    # Summary recommendations
    print(f"\n{'='*60}")
    print("THRESHOLD RECOMMENDATIONS")
    print(f"{'='*60}")

    # spike_count: optimal threshold for failure detection
    sc_thresh = report["optimal_thresholds"]["spike_count"]
    print(f"\n1. spike_count → failure detection:")
    print(f"   Optimal threshold: {sc_thresh['threshold']:.1f}")
    print(f"   Youden's J: {sc_thresh['youden_j']:.4f}")
    print(f"   Sensitivity: {sc_thresh['sensitivity']:.4f}, Specificity: {sc_thresh['specificity']:.4f}")
    print(f"   ROC-AUC: {report['metrics']['spike_count']['roc_auc']:.4f}")

    # effective_motion_ratio: frozen episode detection
    emr_thresh = report["optimal_thresholds"]["effective_motion_ratio"]
    print(f"\n2. effective_motion_ratio → frozen episode detection:")
    print(f"   Optimal threshold: {emr_thresh['threshold']:.4f}")
    print(f"   Youden's J: {emr_thresh['youden_j']:.4f}")
    print(f"   Note: For FROZEN detection, we use a more conservative threshold")
    print(f"   (episodes with ~0% motion, not just below the Youden optimum)")

    # For frozen episodes, look at the extreme low end
    failure_emr = np.array(metrics["effective_motion_ratio"])[failure_mask]
    success_emr = np.array(metrics["effective_motion_ratio"])[success_mask]
    # Find a threshold that catches 95% of frozen episodes while minimizing false positives
    # Frozen episodes should have effective_motion_ratio very close to 0
    frozen_candidates = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    print(f"\n   Frozen episode threshold analysis:")
    for t in frozen_candidates:
        # How many failure episodes have EMR < t?
        caught_failures = (failure_emr < t).sum()
        caught_successes = (success_emr < t).sum()
        print(f"   EMR < {t:.2f}: catches {caught_failures}/{n_failure} failures, {caught_successes}/{n_success} false positives")

    # Output
    output_json = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(output_json)
        print(f"\nReport saved to: {args.output}")
    else:
        print(f"\n{output_json}")


if __name__ == "__main__":
    main()
