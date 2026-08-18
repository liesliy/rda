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
    """Format the full three-layer enhanced audit report as text.

    Sections:
        1. Header with dataset path and stats
        2. Verdict breakdown
        3. Layer 1 — Data Integrity
        4. Layer 2 — Temporal & Motion Anomaly
        5. Layer 3 — Dataset Utility
        6. Top Observations
        7. Hero Metrics (with N/A handling)
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

    # --- Layer 1: Data Integrity ---
    lines.append("  ── Layer 1: Data Integrity ──")
    integrity = dataset_metrics.get("integrity", {})
    for metric_name, stats in integrity.items():
        avail = stats.get("available", 0)
        passed = stats.get("passed", 0)
        failed = stats.get("failed", 0)
        na = stats.get("na", 0)
        pass_rate = stats.get("pass_rate")
        if pass_rate is not None:
            lines.append(f"  {metric_name:24s} {passed}/{avail} pass ({pass_rate:.0%})")
        else:
            lines.append(f"  {metric_name:24s} N/A ({na} episodes)")
    lines.append("")

    # --- Layer 2: Temporal & Motion ---
    lines.append("  ── Layer 2: Temporal & Motion Anomaly ──")
    temporal = dataset_metrics.get("temporal_motion", {})

    # Sensor sync with N/A handling
    sync = temporal.get("sensor_synchronization", {})
    sync_avail = sync.get("available_episodes", 0)
    sync_na = sync.get("na_episodes", 0)
    if sync_avail == 0:
        lines.append(f"  sensor_synchronization    N/A ({sync_na} episodes, no stream timestamps)")
    else:
        p95 = sync.get("worst_p95_offset_ms", {})
        median_p95 = p95.get("median", 0.0)
        lines.append(f"  sensor_synchronization    median p95 offset = {median_p95:.1f}ms ({sync_avail}/{total} episodes)")

    # Action discontinuity
    disc = temporal.get("action_discontinuity", {})
    if disc:
        total_spikes = disc.get("total_spikes", 0)
        affected = disc.get("episodes_with_spikes", 0)
        lines.append(f"  action_discontinuity      {total_spikes} spikes in {affected} episodes (observational)")

    # Velocity
    vel = temporal.get("velocity_acceleration", {})
    if vel:
        vel_p95 = vel.get("velocity_p95", {})
        median_v = vel_p95.get("median", 0.0)
        lines.append(f"  velocity_acceleration     median velocity p95 = {median_v:.4f} (observational)")

    # Temporal sufficiency
    ts = temporal.get("temporal_sufficiency", {})
    if ts:
        ts_avail = ts.get("available_episodes", 0)
        if ts_avail > 0:
            idle_med = ts.get("idle_total_ratio", {}).get("median", 0.0)
            prefix_med = ts.get("idle_prefix_ratio", {}).get("median", 0.0)
            active_p50_med = ts.get("active_run_p50", {}).get("median", 0.0)
            vw10_med = ts.get("valid_window_ratio_10", {}).get("median", 0.0)
            lines.append(
                f"  temporal_sufficiency     idle_total={idle_med:.1%}, "
                f"idle_prefix={prefix_med:.1%}, "
                f"active_run_p50={active_p50_med:.0f}f, "
                f"valid_window(seq=10)={vw10_med:.1%} "
                f"({ts_avail}/{total} episodes)"
            )
    lines.append("")

    # --- Layer 3: Dataset Utility ---
    lines.append("  ── Layer 3: Dataset Utility ──")
    utility = dataset_metrics.get("dataset_utility", {})

    # State-space occupancy
    sso = utility.get("state_space_occupancy", {})
    if sso:
        occ = sso.get("state_space_occupancy", {})
        median_occ = occ.get("median", 0.0)
        lines.append(f"  state_space_occupancy     median = {median_occ:.1%}")

    # Idle ratio / effective motion
    idle = utility.get("idle_ratio", {})
    if idle:
        idle_stats = idle.get("idle_ratio", {})
        effective_stats = idle.get("effective_motion_ratio", {})
        median_idle = idle_stats.get("median", 0.0)
        median_eff = effective_stats.get("median", 0.0)
        lines.append(f"  idle_ratio                median idle = {median_idle:.1%}, effective motion = {median_eff:.1%}")

    # Distribution
    dist = utility.get("distribution", {})
    if dist:
        dur = dist.get("duration_sec", {})
        median_dur = dur.get("median", 0.0)
        lines.append(f"  distribution              median duration = {median_dur:.2f}s")
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
    lines.append(
        f"  ★ Action Discontinuity: "
        f"{disc_hero.get('total_spikes', 0)} spikes total, "
        f"{disc_hero.get('affected_episodes', 0)} episodes affected"
    )

    # State-space occupancy hero
    sso_hero = hero_metrics.get("state_space_occupancy", {})
    sso_range = sso_hero.get("range", [0, 0])
    lines.append(
        f"  ★ State Space Occupancy: "
        f"median {sso_hero.get('median_occupancy', 0):.1%}, "
        f"range [{sso_range[0]:.1%}, {sso_range[1]:.1%}]"
    )
    lines.append("")

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
