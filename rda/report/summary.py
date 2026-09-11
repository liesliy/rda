"""Summary generation with three-layer output and N/A handling.

Provides:
1. AuditSummary + build_summary — compact structured summary
2. format_enhanced_summary_text — full three-layer text report
3. Chinese-language narrator for hero metrics (handles N/A)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from rda.audit.dataset_audit import DatasetAuditResult
from rda.audit.rules import CRITICAL_METRICS, REVIEW_METRICS, AuditVerdict
from rda.metrics.base import MetricAvailability
from rda.report.aggregation import aggregate_dataset_metrics
from rda.report.top_issues import compute_hero_metrics, compute_top_observations


# ---------------------------------------------------------------------------
# Portable vs Platform scoring helpers
# ---------------------------------------------------------------------------


def compute_behavioral_score_summary(
    result: DatasetAuditResult,
) -> Dict[str, Any]:
    """Aggregate per-episode behavioral scores into dataset-level stats.

    Returns a dict with portable_score, platform_score, and combined_score
    statistics (median, p95, mean), plus a ``has_platform_metrics`` flag
    indicating whether platform-specific metrics were calibrated.
    """
    portable_scores: List[float] = []
    platform_scores: List[float] = []
    combined_scores: List[float] = []
    has_platform = False

    for ep in result.episodes.values():
        if ep.portable_score is not None:
            portable_scores.append(ep.portable_score)
        if ep.platform_score is not None:
            platform_scores.append(ep.platform_score)
        if ep.combined_score is not None:
            combined_scores.append(ep.combined_score)
        if ep.has_platform_metrics:
            has_platform = True

    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"median": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}
        a = np.array(arr, dtype=np.float64)
        return {
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(np.mean(a)),
            "max": float(np.max(a)),
        }

    return {
        "portable_score": _stats(portable_scores),
        "platform_score": _stats(platform_scores) if has_platform else None,
        "combined_score": _stats(combined_scores),
        "has_platform_metrics": has_platform,
        "n_scored": len(portable_scores),
    }


# ---------------------------------------------------------------------------
# Legacy compact summary
# ---------------------------------------------------------------------------

@dataclass
class AuditSummary:
    total_episodes: int = 0
    verdict_counts: Dict[str, int] = field(default_factory=dict)
    pass_rate: float = 0.0
    failing_metrics: Dict[str, int] = field(default_factory=dict)
    exclude_episodes: List[int] = field(default_factory=list)
    review_episodes: List[int] = field(default_factory=list)


def build_summary(result: DatasetAuditResult) -> AuditSummary:
    summary = AuditSummary(total_episodes=result.num_episodes)
    summary.verdict_counts = {v.value: c for v, c in result.verdict_counts.items()}

    pass_count = result.verdict_counts.get(AuditVerdict.PASS, 0)
    summary.pass_rate = pass_count / result.num_episodes if result.num_episodes > 0 else 0.0

    failing_metrics: Dict[str, int] = {}
    for ep_result in result.episodes.values():
        if ep_result.verdict == AuditVerdict.EXCLUDE:
            summary.exclude_episodes.append(ep_result.episode_index)
        elif ep_result.verdict == AuditVerdict.REVIEW:
            summary.review_episodes.append(ep_result.episode_index)

        for m_name, m_result in ep_result.metrics.items():
            # Only count failures for AVAILABLE metrics
            if m_result.availability != MetricAvailability.AVAILABLE:
                continue
            if not m_result.passed:
                failing_metrics[m_name] = failing_metrics.get(m_name, 0) + 1

    summary.failing_metrics = failing_metrics
    summary.exclude_episodes.sort()
    summary.review_episodes.sort()

    return summary


# ---------------------------------------------------------------------------
# Three-layer text report
# ---------------------------------------------------------------------------

def format_enhanced_summary_text(result: DatasetAuditResult) -> str:
    """Format the full v0.9 four-section audit report as text.

    Sections:
        1. Header with dataset path and stats
        2. Verdict breakdown
        3. Integrity Gate (L1 — hard checks)
        4. Trajectory Diagnostics (L2 — observational measurements)
        5. Dataset Profile (L3 — dataset-level per-episode metrics)
        6. Top Observations
        7. Hero Metrics (with N/A handling)
        8. Video Temporal Verification (verifiability levels)
        9. Dataset Summary (cross-episode aggregation)
    """
    dataset_metrics = aggregate_dataset_metrics(result)
    top_obs = compute_top_observations(result, dataset_metrics=dataset_metrics)
    hero_metrics = compute_hero_metrics(dataset_metrics)
    compact = build_summary(result)

    total = result.num_episodes
    total_frames = result.dataset_info.total_frames
    dataset_path = result.dataset_info.path

    lines: List[str] = []

    # --- Header ---
    lines.append("=" * 60)
    from rda import __version__ as _rda_version
    lines.append(f"  RDA — Robot Data Audit Report (v{_rda_version})")
    lines.append("=" * 60)
    lines.append(f"  Dataset: {dataset_path}")
    lines.append(f"  Episodes: {total} | Frames: {total_frames:,}")
    lines.append("")

    # --- Verdict ---
    lines.append("  ── Verdict ──")
    for verdict in ["PASS", "REVIEW", "EXCLUDE"]:
        count = compact.verdict_counts.get(verdict, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        lines.append(f"  {verdict + ':':8s} {count:>3d} ({pct:>4.1f}%)")
    lines.append("")

    # ── Integrity Gate (L1) ──
    lines.append("  ── Integrity Gate ──")
    integrity = dataset_metrics.get("integrity", {})
    for metric_name, stats in integrity.items():
        avail = stats.get("available", 0)
        passed = stats.get("passed", 0)
        failed = stats.get("failed", 0)
        na = stats.get("na", 0)
        pass_rate = stats.get("pass_rate")
        verif = _verifiability_text_for(metric_name, stats)
        if pass_rate is not None:
            lines.append(f"  {metric_name:28s} {verif}  {passed}/{avail} pass ({pass_rate:.0%})")
        else:
            lines.append(f"  {metric_name:28s} {verif}  N/A ({na} episodes)")
    lines.append("")

    # ── Trajectory Diagnostics (L2) ──
    lines.append("  ── Trajectory Diagnostics ──")
    temporal = dataset_metrics.get("temporal_motion", {})

    # Sensor sync with N/A handling
    sync = temporal.get("sensor_synchronization", {})
    sync_avail = sync.get("available_episodes", 0)
    sync_na = sync.get("na_episodes", 0)
    if sync_avail == 0:
        lines.append(f"  {'sensor_synchronization':28s} ⚠ Not verifiable  ({sync_na} episodes, no stream timestamps)")
    else:
        p95 = sync.get("worst_p95_offset_ms", {})
        median_p95 = p95.get("median", 0.0)
        lines.append(f"  {'sensor_synchronization':28s} ✓ Measured  median p95 offset = {median_p95:.1f}ms ({sync_avail}/{total} episodes)")

    # Video stream diagnostics (v0.9 split)
    for vname in ("video_stream_span_consistency", "video_stream_temporal_offset", "video_stream_temporal_drift"):
        vdata = temporal.get(vname, {})
        if vdata:
            v_avail = vdata.get("available_episodes", 0)
            v_na = vdata.get("na_episodes", 0)
            if v_avail == 0:
                lines.append(f"  {vname:28s} — N/A ({v_na} episodes)")
            else:
                lines.append(f"  {vname:28s} ✓ Measured  ({v_avail}/{total} episodes)")
        # If no data at all, skip (not yet computed)

    # Action discontinuity
    disc = temporal.get("action_discontinuity", {})
    if disc:
        if disc.get("available_episodes", 0) == 0:
            lines.append(
                f"  {'action_discontinuity':28s} — N/A (no action arrays — video-only dataset)"
            )
        else:
            total_spikes = disc.get("total_spikes", 0)
            affected = disc.get("episodes_with_spikes", 0)
            lines.append(
                f"  {'action_discontinuity':28s} ✓ Measured  {total_spikes} spikes in {affected} episodes"
            )

    # Velocity
    vel = temporal.get("velocity_acceleration", {})
    if vel:
        vel_p95 = vel.get("velocity_p95", {})
        median_v = vel_p95.get("median", 0.0)
        lines.append(
            f"  {'velocity_acceleration':28s} ✓ Measured  median velocity p95 = {median_v:.4f}"
        )

    # Idle ratio (L2 diagnostic in v0.9)
    idle = temporal.get("idle_ratio", {})
    if idle:
        idle_med = idle.get("idle_ratio", {}).get("median", 0.0)
        lines.append(
            f"  {'idle_ratio':28s} ✓ Measured  median = {idle_med:.1%}"
        )

    # Visual quality
    vq = temporal.get("visual_quality", {})
    if vq:
        blur_med = vq.get("median_blur_var", {}).get("median", 0.0)
        lines.append(
            f"  {'visual_quality':28s} ✓ Measured  median blur var = {blur_med:.1f}"
        )
    lines.append("")

    # ── Dataset Profile (L3) ──
    lines.append("  ── Dataset Profile ──")
    utility = dataset_metrics.get("dataset_utility", {})

    # State-space occupancy (aggregation key is "coverage")
    sso = utility.get("coverage", {})
    if sso:
        if sso.get("available_episodes", 0) == 0:
            lines.append(
                f"  {'state_space_occupancy':28s} — N/A (no observation.state — video-only dataset)"
            )
        else:
            occ = sso.get("state_space_occupancy", {})
            median_occ = occ.get("median", 0.0)
            lines.append(f"  {'state_space_occupancy':28s} median = {median_occ:.1%}")

    # Distribution
    dist = utility.get("distribution", {})
    if dist:
        dur = dist.get("duration_sec", {})
        median_dur = dur.get("median", 0.0)
        lines.append(f"  {'distribution':28s} median duration = {median_dur:.2f}s")

    # Temporal structure (v0.9: renamed from temporal_sufficiency, moved to L3)
    ts = utility.get("temporal_structure", utility.get("temporal_sufficiency", {}))
    if ts:
        ts_avail = ts.get("available_episodes", 0)
        if ts_avail > 0:
            idle_med = ts.get("idle_total_ratio", {}).get("median", 0.0)
            prefix_med = ts.get("idle_prefix_ratio", {}).get("median", 0.0)
            active_p50_med = ts.get("active_run_p50", {}).get("median", 0.0)
            vw10_med = ts.get("valid_window_ratio_10", {}).get("median", 0.0)
            lines.append(
                f"  {'temporal_structure':28s} idle_total={idle_med:.1%}, "
                f"idle_prefix={prefix_med:.1%}, "
                f"active_run_p50={active_p50_med:.0f}f, "
                f"valid_window(seq=10)={vw10_med:.1%} "
                f"({ts_avail}/{total} episodes)"
            )
    lines.append("")

    # --- Top Observations ---
    lines.append("  ── Top Observations ──")
    if not top_obs:
        lines.append("  No notable observations — all metrics nominal.")
    else:
        for obs in top_obs:
            sig = obs["significance"].upper()
            hero_mark = " ★" if obs.get("hero") else ""
            lines.append(f"  {obs['rank']}. [{sig}{hero_mark}] {obs['description']}")
    lines.append("")

    # --- Hero Metrics ---
    lines.append("  ── Hero Metrics ──")

    # Sensor sync hero
    sync_hero = hero_metrics.get("sensor_synchronization", {})
    sync_interp = sync_hero.get("interpretation", "")
    if sync_interp == "na":
        lines.append("  ★ Sensor Sync:      N/A (no per-stream timestamps provided)")
    else:
        sync_interp_cn = {
            "excellent": "优秀", "acceptable": "可接受",
            "needs_check": "需检查", "severe": "严重",
        }.get(sync_interp, sync_interp)
        lines.append(
            f"  ★ Sensor Sync:      median p95 offset = "
            f"{sync_hero.get('median_p95_offset_ms', 0):.1f}ms ({sync_interp_cn})"
        )

    # Action disc hero
    disc_hero = hero_metrics.get("action_discontinuity", {})
    if disc_hero.get("interpretation") == "na":
        lines.append(
            "  ★ Action Discontinuity: N/A (no action arrays — video-only dataset)"
        )
    else:
        lines.append(
            f"  ★ Action Discontinuity: "
            f"{disc_hero.get('total_spikes', 0)} spikes total, "
            f"{disc_hero.get('affected_episodes', 0)} episodes affected"
        )

    # State-space occupancy hero
    sso_hero = hero_metrics.get("state_space_occupancy", {})
    if sso_hero.get("interpretation") == "na":
        lines.append(
            "  ★ State Space Occupancy: N/A (no observation.state — video-only dataset)"
        )
    else:
        sso_range = sso_hero.get("range", [0, 0])
        lines.append(
            f"  ★ State Space Occupancy: "
            f"median {sso_hero.get('median_occupancy', 0):.1%}, "
            f"range [{sso_range[0]:.1%}, {sso_range[1]:.1%}]"
        )
    lines.append("")

    # ── Video Temporal Verification ──
    lines.append("  ── Video Temporal Verification ──")
    _append_verifiability_section(lines, result, dataset_metrics)
    lines.append("")

    # ── Dataset Summary ──
    if result.dataset_summary is not None:
        from rda.report.dataset_summary import format_dataset_summary_text
        lines.append(format_dataset_summary_text(result.dataset_summary))

    # --- EXCLUDE Episodes ---
    if compact.exclude_episodes:
        lines.append("  ── EXCLUDE Episodes ──")
        ep_list = ", ".join(f"#{e}" for e in compact.exclude_episodes[:20])
        extra = ""
        if len(compact.exclude_episodes) > 20:
            extra = f" ... (+{len(compact.exclude_episodes) - 20} more)"
        lines.append(f"  {ep_list}{extra}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def _verifiability_text_for(metric_name: str, stats: dict) -> str:
    """Return a short verifiability marker for text reports."""
    avail = stats.get("available", 0)
    na = stats.get("na", 0)
    if avail == 0 and na > 0:
        return "— N/A"
    if avail == 0:
        return "⚠ Not verifiable"
    # Video-related L2 diagnostics use "Measured"
    video_measured = {
        "sensor_synchronization",
        "video_stream_span_consistency",
        "video_stream_temporal_offset",
        "video_stream_temporal_drift",
        "visual_quality",
    }
    if metric_name in video_measured:
        return "✓ Measured"
    return "✓ Verified"


def _append_verifiability_section(
    lines: List[str],
    result: DatasetAuditResult,
    dataset_metrics: Dict[str, Any],
) -> None:
    """Append the Video Temporal Verification block."""
    # Collect verifiability from per-episode results
    video_metrics = [
        "video_stream_presence",
        "video_stream_span_consistency",
        "video_stream_temporal_offset",
        "video_stream_temporal_drift",
        "video_frame_integrity",
    ]

    # Determine status for each metric across the dataset
    status_map: Dict[str, str] = {}
    for m_name in video_metrics:
        found_verified = False
        found_measured = False
        found_na = False
        found_not_verifiable = False

        for ep in result.episodes.values():
            m = ep.metrics.get(m_name)
            if m is None:
                continue
            from rda.metrics.base import MetricAvailability
            if m.availability == MetricAvailability.AVAILABLE:
                if m_name in ("video_stream_span_consistency", "video_stream_temporal_offset", "video_stream_temporal_drift"):
                    found_measured = True
                else:
                    found_verified = True
            elif m.availability == MetricAvailability.NOT_AVAILABLE:
                # N/A vs "Not verifiable" depends on WHY the metric is
                # unavailable: metric not applicable to this episode
                # (single camera, no video) => N/A; metric should apply but
                # required inputs are missing/errored => Not verifiable.
                # MetricAvailability has no NA member (only AVAILABLE /
                # NOT_AVAILABLE / ERROR).
                reason = ""
                try:
                    reason = (m.assessment or {}).get("reason") or ""
                except Exception:
                    reason = ""
                if reason in ("single_camera", "no_video_features"):
                    found_na = True
                else:
                    found_not_verifiable = True

        if found_verified:
            status_map[m_name] = "✓ Verified"
        elif found_measured:
            status_map[m_name] = "✓ Measured"
        elif found_not_verifiable:
            status_map[m_name] = "⚠ Not verifiable"
        elif found_na:
            status_map[m_name] = "— N/A"
        else:
            status_map[m_name] = "— N/A"

    display_names = {
        "video_stream_presence": "Camera presence",
        "video_stream_span_consistency": "Stream coverage",
        "video_stream_temporal_offset": "Frame-level synchronization",
        "video_stream_temporal_drift": "Temporal drift",
        "video_frame_integrity": "Frame integrity",
    }

    for m_name in video_metrics:
        label = display_names.get(m_name, m_name)
        status = status_map.get(m_name, "— N/A")
        lines.append(f"  {label:30s} {status}")


# Keep old name for backward compat
def format_summary_text(summary: AuditSummary) -> str:
    lines = [
        "=" * 60,
        "  RDA — Robot Data Audit Summary",
        "=" * 60,
        f"  Total episodes audited : {summary.total_episodes}",
        f"  PASS rate              : {summary.pass_rate:.1%}",
        "",
        "  Verdict breakdown:",
    ]
    for verdict in ["PASS", "REVIEW", "EXCLUDE"]:
        count = summary.verdict_counts.get(verdict, 0)
        lines.append(f"    {verdict:8s} : {count}")
    if summary.failing_metrics:
        lines.append("")
        lines.append("  Top failing metrics:")
        sorted_metrics = sorted(summary.failing_metrics.items(), key=lambda x: x[1], reverse=True)
        for name, count in sorted_metrics[:10]:
            lines.append(f"    {name:20s} : {count} episodes")
    if summary.exclude_episodes:
        ep_list = ", ".join(str(e) for e in summary.exclude_episodes[:20])
        lines.append("")
        lines.append(f"  EXCLUDE episodes: {ep_list}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
