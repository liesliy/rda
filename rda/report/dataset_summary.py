"""Dataset-level summary aggregation for RDA v0.9.

Aggregates L2 diagnostic metrics across all episodes to produce
dataset-level distribution statistics for three key indicators:
- idle_ratio: median, P10, P90
- sensor_sync: worst_p95_offset_ms median, P90, max
- visual_quality: median_blur_var median + exposure anomaly ratio

This module is independent from per-episode reporting and runs
after all episodes have been audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSummaryResult:
    """Container for dataset-level summary statistics."""

    total_episodes: int = 0
    episodes_with_data: int = 0

    # idle_ratio distribution
    idle_ratio_median: Optional[float] = None
    idle_ratio_p10: Optional[float] = None
    idle_ratio_p90: Optional[float] = None

    # sensor_sync distribution
    sensor_sync_median_p95_offset_ms: Optional[float] = None
    sensor_sync_p90_p95_offset_ms: Optional[float] = None
    sensor_sync_max_p95_offset_ms: Optional[float] = None

    # visual_quality distribution
    visual_quality_median_blur_var: Optional[float] = None
    visual_quality_exposure_anomaly_ratio: Optional[float] = None
    visual_quality_exposure_anomaly_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary matching v0.9 spec format."""
        return {
            "dataset_summary": {
                "total_episodes": self.total_episodes,
                "episodes_with_data": self.episodes_with_data,
                "idle_ratio": _optional_dict(
                    self.idle_ratio_median,
                    median=self.idle_ratio_median,
                    p10=self.idle_ratio_p10,
                    p90=self.idle_ratio_p90,
                ),
                "sensor_sync": _optional_dict(
                    self.sensor_sync_median_p95_offset_ms,
                    median_p95_offset_ms=self.sensor_sync_median_p95_offset_ms,
                    p90_p95_offset_ms=self.sensor_sync_p90_p95_offset_ms,
                    max_p95_offset_ms=self.sensor_sync_max_p95_offset_ms,
                ),
                "visual_quality": _optional_dict(
                    self.visual_quality_median_blur_var,
                    median_blur_var=self.visual_quality_median_blur_var,
                    exposure_anomaly_ratio=self.visual_quality_exposure_anomaly_ratio,
                    exposure_anomaly_count=self.visual_quality_exposure_anomaly_count,
                ),
            }
        }


def _optional_dict(primary_value: Optional[float], **kwargs) -> Optional[Dict[str, Any]]:
    """Return None if primary value is None, otherwise return the dict."""
    if primary_value is None:
        return None
    return {k: v for k, v in kwargs.items() if v is not None}


def _percentile(values: List[float], p: float) -> Optional[float]:
    """Compute percentile using linear interpolation.

    Args:
        values: Sorted list of float values (must be non-empty).
        p: Percentile in [0, 100].

    Returns:
        Interpolated percentile value, or None if values is empty.
    """
    if not values:
        return None
    n = len(values)
    if n == 1:
        return values[0]
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return values[-1]
    d = k - f
    return values[f] + d * (values[c] - values[f])


def _median(values: List[float]) -> Optional[float]:
    """Compute median of a sorted list."""
    return _percentile(values, 50)


def compute_dataset_summary(
    episode_results: List[Dict[str, Any]],
) -> DatasetSummaryResult:
    """Compute dataset-level summary from all episode audit results.

    Args:
        episode_results: List of dicts, each containing:
            - "episode_id": str
            - "verdict": str (PASS/EXCLUDE)
            - "measurements": dict mapping metric_name -> measurement dict
            - "findings": list of findings (optional)

    Returns:
        DatasetSummaryResult with aggregated statistics.
    """
    result = DatasetSummaryResult()
    result.total_episodes = len(episode_results)

    if not episode_results:
        return result

    # Collect per-episode values for each summary metric
    idle_ratios: List[float] = []
    sensor_sync_p95_offsets: List[float] = []
    blur_vars: List[float] = []
    exposure_anomaly_count = 0
    visual_quality_count = 0

    for ep in episode_results:
        measurements = ep.get("measurements", {})
        has_any_data = False

        # --- idle_ratio ---
        idle_data = measurements.get("idle_ratio", {})
        if idle_data:
            ir = idle_data.get("idle_ratio")
            if ir is not None:
                idle_ratios.append(float(ir))
                has_any_data = True

        # --- sensor_synchronization ---
        sync_data = measurements.get("sensor_synchronization", {})
        if sync_data:
            worst_p95 = sync_data.get("worst_p95_offset_ms")
            if worst_p95 is not None:
                sensor_sync_p95_offsets.append(float(worst_p95))
                has_any_data = True

        # --- visual_quality ---
        vq_data = measurements.get("visual_quality", {})
        if vq_data:
            blur_var = vq_data.get("median_blur_var")
            if blur_var is not None:
                blur_vars.append(float(blur_var))
                visual_quality_count += 1
                has_any_data = True

            # Check exposure anomaly
            exposure_issues = vq_data.get("exposure_issues", [])
            if exposure_issues:
                exposure_anomaly_count += 1

        if has_any_data:
            result.episodes_with_data += 1

    # Compute idle_ratio distribution
    if idle_ratios:
        idle_ratios.sort()
        result.idle_ratio_median = round(_median(idle_ratios) or 0, 4)
        result.idle_ratio_p10 = round(_percentile(idle_ratios, 10) or 0, 4)
        result.idle_ratio_p90 = round(_percentile(idle_ratios, 90) or 0, 4)

    # Compute sensor_sync distribution
    if sensor_sync_p95_offsets:
        sensor_sync_p95_offsets.sort()
        result.sensor_sync_median_p95_offset_ms = round(
            _median(sensor_sync_p95_offsets) or 0, 2
        )
        result.sensor_sync_p90_p95_offset_ms = round(
            _percentile(sensor_sync_p95_offsets, 90) or 0, 2
        )
        result.sensor_sync_max_p95_offset_ms = round(
            sensor_sync_p95_offsets[-1], 2
        )

    # Compute visual_quality distribution
    if blur_vars:
        blur_vars.sort()
        result.visual_quality_median_blur_var = round(_median(blur_vars) or 0, 2)

    if visual_quality_count > 0:
        result.visual_quality_exposure_anomaly_ratio = round(
            exposure_anomaly_count / result.total_episodes, 4
        )
        result.visual_quality_exposure_anomaly_count = exposure_anomaly_count

    return result


def format_dataset_summary_text(summary: DatasetSummaryResult) -> str:
    """Format dataset summary as human-readable text for report appendix."""
    lines = ["=" * 60, "Dataset Summary (v0.9)", "=" * 60]
    lines.append(f"Total episodes: {summary.total_episodes}")
    lines.append(f"Episodes with usable data: {summary.episodes_with_data}")
    lines.append("")

    # idle_ratio
    lines.append("--- idle_ratio distribution ---")
    if summary.idle_ratio_median is not None:
        lines.append(f"  median = {summary.idle_ratio_median:.2%}")
        lines.append(f"  P10    = {summary.idle_ratio_p10:.2%}")
        lines.append(f"  P90    = {summary.idle_ratio_p90:.2%}")
    else:
        lines.append("  No data available")
    lines.append("")

    # sensor_sync
    lines.append("--- sensor_sync distribution ---")
    if summary.sensor_sync_median_p95_offset_ms is not None:
        lines.append(
            f"  median worst_p95_offset = {summary.sensor_sync_median_p95_offset_ms:.2f} ms"
        )
        lines.append(
            f"  P90 worst_p95_offset    = {summary.sensor_sync_p90_p95_offset_ms:.2f} ms"
        )
        lines.append(
            f"  max worst_p95_offset    = {summary.sensor_sync_max_p95_offset_ms:.2f} ms"
        )
    else:
        lines.append("  No data available")
    lines.append("")

    # visual_quality
    lines.append("--- visual_quality distribution ---")
    if summary.visual_quality_median_blur_var is not None:
        lines.append(
            f"  median blur variance = {summary.visual_quality_median_blur_var:.2f}"
        )
    else:
        lines.append("  No blur data available")

    if summary.visual_quality_exposure_anomaly_ratio is not None:
        lines.append(
            f"  exposure anomaly ratio = {summary.visual_quality_exposure_anomaly_ratio:.2%} "
            f"({summary.visual_quality_exposure_anomaly_count}/{summary.total_episodes} episodes)"
        )
    else:
        lines.append("  No visual quality data available")

    lines.append("")
    return "\n".join(lines)
