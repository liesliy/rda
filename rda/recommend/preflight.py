"""Recommendation preflight verdicts (OPEN SOURCE).

REQ-1 (v0.5.9): wire the audit verdict into the recommendation pipeline.

Problem being fixed
-------------------
``classify_episode`` results never entered the recommendation chain — a
dataset whose episodes were EXCLUDEd for NaN/Inf or non-monotonic
timestamps could still receive ``TRIM_INITIAL (HIGH)`` pruning advice.
Pruning presumes the data is healthy; when the premise is unverified the
recommendation chain is logically unsound. Repair must come before
optimization.

Design
------
Rather than forcing users to run ``rda audit`` first, the recommend pass
recomputes the deterministic CRITICAL metrics inline. The recommend
pipeline already streams every episode for temporal sufficiency; running
five more O(T) array checks in the same pass has near-zero marginal
cost (no second dataset traversal, no forced two-step workflow).

For every episode that fails a CRITICAL check we also emit a coarse
``reason_code`` distinguishing:

- ``INVALID``    — irrecoverable at audit time (NaN/Inf, broken time
                   axis, schema mismatch). The truth value of affected
                   frames cannot be reconstructed without domain
                   knowledge RDA does not have.
- ``REPAIRABLE`` — locally damaged but recoverable by trimming /
                   re-export (frame loss, joint-limit excursion).

The server grades the overall verdict (REPAIR_FIRST vs proceed); this
module only supplies evidence. No rule thresholds live here.

Copyright (c) 2026 Niu Su Tech. MIT License.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rda.audit.rules import CRITICAL_METRICS, AuditVerdict
from rda.metrics.base import MetricAvailability
from rda.metrics.integrity import (
    MissingFramesMetric,
    NaNInfMetric,
    SchemaShapeMetric,
)
from rda.metrics.motion import JointLimitMetric
from rda.metrics.temporal import TimestampValidityMetric

# Note: rules.py CRITICAL_METRICS lists five metrics. v0.5.6 added
# VideoFrameIntegrityMetric to the audit Layer-1 list, but its name is
# not yet in rules.CRITICAL_METRICS and it needs video decoding, so
# preflight deliberately keeps the five classical deterministic checks.
PREFLIGHT_METRIC_NAMES: List[str] = list(CRITICAL_METRICS)

# reason_code classification per metric (aligned with the v1.1 design
# doc): only damage whose truth values are unrecoverable at audit time
# is INVALID. Schema mismatch is REPAIRABLE — it may be a trimmable
# tail, a re-export artifact, or a feature-mapping error, all fixable
# without re-acquisition.
REASON_CODE_BY_METRIC: Dict[str, str] = {
    # Irrecoverable at audit time: the true values cannot be restored
    # without domain knowledge (invalid_values), or the time axis itself
    # is broken so every derived ordering is unreliable (timestamp).
    "invalid_values": "INVALID",
    "timestamp_validity": "INVALID",
    # Locally damaged, recoverable via trimming or re-export:
    "missing_dropout": "REPAIRABLE",
    "schema_consistency": "REPAIRABLE",
    "joint_limit": "REPAIRABLE",
    # REQ-4 (v0.7.0) VA-A: camera drop-out and missing camera streams are
    # re-acquisition problems for the affected spans — treat as REPAIRABLE
    # (re-record) rather than INVALID; timeline misalignment likewise
    # (re-export from source).
    "video_freeze": "REPAIRABLE",
    "video_stream_sync": "REPAIRABLE",
    "video_timestamp_alignment": "REPAIRABLE",
}


@dataclass
class EpisodeVerdictSummary:
    """Per-episode preflight verdict evidence for one episode."""

    episode_index: int
    verdict: str                       # AuditVerdict value ("PASS"/"REVIEW"/"EXCLUDE")
    reason_code: Optional[str] = None  # "INVALID"/"REPAIRABLE" when EXCLUDE
    failed_metrics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "episode_index": self.episode_index,
            "verdict": self.verdict,
        }
        if self.reason_code is not None:
            d["reason_code"] = self.reason_code
        if self.failed_metrics:
            d["failed_metrics"] = list(self.failed_metrics)
        return d


@dataclass
class PreflightSummary:
    """Aggregated preflight verdict evidence for the whole dataset.

    This is what gets attached to the recommendation API payload. Size
    stays well under 1KB even for large datasets: excluded episodes are
    reported in full, but PASS/REVIEW episodes are only counted.
    """

    episodes_total: int = 0
    exclude_count: int = 0
    review_count: int = 0
    pass_count: int = 0
    excluded_episodes: List[EpisodeVerdictSummary] = field(default_factory=list)
    review_episode_indices: List[int] = field(default_factory=list)
    # Worst reason code across excluded episodes: INVALID dominates
    # REPAIRABLE (unrecoverable damage outranks recoverable damage).
    dominant_reason_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the API payload (and for cache keys)."""
        d: Dict[str, Any] = {
            "episodes_total": self.episodes_total,
            "exclude_count": self.exclude_count,
            "review_count": self.review_count,
            "pass_count": self.pass_count,
        }
        if self.excluded_episodes:
            d["excluded_episodes"] = [e.to_dict() for e in self.excluded_episodes]
        if self.review_episode_indices:
            d["review_episode_indices"] = list(self.review_episode_indices)
        if self.dominant_reason_code is not None:
            d["dominant_reason_code"] = self.dominant_reason_code
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreflightSummary":
        """Reconstruct from a dict (tests, cached payloads)."""
        excluded = [
            EpisodeVerdictSummary(
                episode_index=e.get("episode_index", -1),
                verdict=e.get("verdict", "EXCLUDE"),
                reason_code=e.get("reason_code"),
                failed_metrics=list(e.get("failed_metrics", [])),
            )
            for e in data.get("excluded_episodes", [])
        ]
        return cls(
            episodes_total=data.get("episodes_total", 0),
            exclude_count=data.get("exclude_count", 0),
            review_count=data.get("review_count", 0),
            pass_count=data.get("pass_count", 0),
            excluded_episodes=excluded,
            review_episode_indices=list(data.get("review_episode_indices", [])),
            dominant_reason_code=data.get("dominant_reason_code"),
        )


class PreflightAuditor:
    """Runs the deterministic CRITICAL metrics over streamed episodes.

    Single-pass: the recommend pipeline already iterates episodes for
    temporal sufficiency; this auditor consumes the same stream.
    Metric failures are swallowed per-metric (same policy as the audit
    pipeline) — a broken metric must never abort a recommendation run.
    """

    def __init__(self, include_visual: bool = False) -> None:
        self._metric_instances = {
            "missing_dropout": MissingFramesMetric(),
            "invalid_values": NaNInfMetric(),
            "schema_consistency": SchemaShapeMetric(),
            "timestamp_validity": TimestampValidityMetric(),
            "joint_limit": JointLimitMetric(),
        }
        # REQ-4 (v0.7.0): VA-A visual-stream integrity checks decode
        # video (64×64 gray spans). Off by default in preflight — the
        # recommend pass must stay cheap on video-heavy datasets; the
        # audit pass always runs them. ``include_visual=True`` opts in
        # (e.g. explicit visual-audit recommendation runs).
        self._preflight_names = list(PREFLIGHT_METRIC_NAMES)
        if include_visual:
            from rda.metrics.visual_integrity import (
                VideoFreezeMetric,
                VideoStreamSyncMetric,
                VideoTimestampAlignmentMetric,
            )
            self._metric_instances.update({
                "video_freeze": VideoFreezeMetric(),
                "video_timestamp_alignment": VideoTimestampAlignmentMetric(),
                "video_stream_sync": VideoStreamSyncMetric(),
            })
        else:
            self._preflight_names = [
                n for n in self._preflight_names
                if not n.startswith("video_") or n == "video_frame_integrity"
            ]

    def evaluate(self, episode) -> EpisodeVerdictSummary:
        """Evaluate one episode against the CRITICAL checks."""
        failed: List[str] = []
        zero_frames = episode.num_frames == 0

        for name in self._preflight_names:
            metric = self._metric_instances.get(name)
            if metric is None:
                continue
            try:
                result = metric.compute(episode)
            except Exception:
                # Mirror EpisodeAuditor: metric errors don't affect verdict.
                continue
            if result.availability != MetricAvailability.AVAILABLE:
                continue
            if not result.has_finding:
                continue
            failed.append(name)

        if zero_frames:
            # Same guard as audit (v0.4.12): 0-frame episodes must not
            # silently PASS. Classified REPAIRABLE (re-export fixes it).
            verdict = AuditVerdict.EXCLUDE.value
            return EpisodeVerdictSummary(
                episode_index=episode.episode_index,
                verdict=verdict,
                reason_code="REPAIRABLE",
                failed_metrics=list(failed) or ["_zero_frame_guard"],
            )

        if failed:
            reason_code = next(
                (REASON_CODE_BY_METRIC[m] for m in failed
                 if REASON_CODE_BY_METRIC.get(m) == "INVALID"),
                "REPAIRABLE",
            )
            return EpisodeVerdictSummary(
                episode_index=episode.episode_index,
                verdict=AuditVerdict.EXCLUDE.value,
                reason_code=reason_code,
                failed_metrics=failed,
            )

        return EpisodeVerdictSummary(
            episode_index=episode.episode_index,
            verdict=AuditVerdict.PASS.value,
        )


def aggregate_preflight(verdicts: List[EpisodeVerdictSummary]) -> PreflightSummary:
    """Fold per-episode verdicts into a dataset-level PreflightSummary."""
    summary = PreflightSummary(episodes_total=len(verdicts))
    for v in verdicts:
        if v.verdict == AuditVerdict.EXCLUDE.value:
            summary.exclude_count += 1
            summary.excluded_episodes.append(v)
        elif v.verdict == AuditVerdict.REVIEW.value:
            summary.review_count += 1
            summary.review_episode_indices.append(v.episode_index)
        else:
            summary.pass_count += 1

    if summary.excluded_episodes:
        codes = {e.reason_code for e in summary.excluded_episodes}
        summary.dominant_reason_code = (
            "INVALID" if "INVALID" in codes else "REPAIRABLE"
        )
    return summary


# ---------------------------------------------------------------------------
# Client-side verdict gate (REQ-1)
# ---------------------------------------------------------------------------

def gate_result_by_verdict(
    result,
    summary: Optional[PreflightSummary],
    lang: str = "zh",
):
    """Final client-side gate: no TRIM may survive broken-episode evidence.

    Applied to EVERY result path (server response, cache, offline
    fallback). This is what makes the REQ-1 guarantee independent of
    the server upgrade order: even a v1 server (or a stale cache, or
    the offline fallback) cannot hand a TRIM_* suggestion to a dataset
    whose episodes failed the deterministic CRITICAL checks.

    This gate consumes only client-computed evidence — no rule
    thresholds live here, so it stays in the open-source package.

    Args:
        result: A RecommendationResult (mutated recommendations in place).
        summary: PreflightSummary from the same run, or None.
        lang: Copy language ("zh" or "en").

    Returns:
        The (possibly gated) result. Also annotated with
        ``result.verdict_summary`` (dict) for JSON reports.
    """
    if summary is None or summary.exclude_count == 0:
        return result

    from rda.recommend.types import (
        ConfidenceLevel,
        GENERAL_CAVEATS,
        HELD_OUT_VALIDATION,
        Recommendation,
        RecommendationAction,
    )

    lang = lang if lang in ("zh", "en") else "zh"
    zh = lang == "zh"

    # Evidence text: worst reason code, affected episode count, and up
    # to three example episodes with their failed metrics.
    n = summary.exclude_count
    dominant = summary.dominant_reason_code or "REPAIRABLE"

    if dominant == "INVALID":
        damage_zh = "不可恢复损坏（NaN/Inf、时间轴断裂或结构不一致）"
        damage_en = ("irrecoverable damage (NaN/Inf, broken time axis, "
                     "or schema inconsistency)")
    else:
        damage_zh = "可修复损坏（缺帧、传感器掉线或关节限位越界）"
        damage_en = ("repairable damage (frame loss, sensor dropout, or "
                     "joint-limit excursion)")

    examples = []
    for ev in summary.excluded_episodes[:3]:
        failed = ",".join(ev.failed_metrics[:2]) if ev.failed_metrics else "critical check"
        examples.append(
            f"episode {ev.episode_index} ({failed})" if not zh
            else f"episode {ev.episode_index}（{failed}）"
        )
    more = n - len(examples)
    example_line = (
        "; ".join(examples) + (f" 等 {n} 条" if more > 0 else "")
        if zh
        else "; ".join(examples) + (f" (+{more} more)" if more > 0 else "")
    )

    if zh:
        title = "先修复数据，再考虑剪枝 (REPAIR_FIRST)"
        summary_text = (
            f"预检发现 {n} 条 episode 存在{damage_zh}：{example_line}。"
            "在修复这些问题之前，所有剪枝类建议不予出具——"
            "修剪建立在数据健康的前提上，当前前提不成立。"
            "建议先运行 rda audit 查看逐条证据，修复或剔除问题 episode 后再获取优化建议。"
        )
        details = [
            f"EXCLUDE episode 数：{n} / {summary.episodes_total}",
            f"主要损坏类型：{dominant}",
            "REPAIRABLE 类（缺帧/限位越界）可尝试裁剪损坏段或重新导出；"
            "INVALID 类（NaN/时间轴/结构）需要回到采集或导出环节排查。",
            f"REVIEW episode 数：{summary.review_count}（不阻断建议，但建议人工过目）。",
        ]
    else:
        title = "REPAIR_FIRST"
        summary_text = (
            f"Preflight found {n} episode(s) with {damage_en}: {example_line}. "
            "No pruning advice is issued until these are repaired — pruning "
            "presumes healthy data, and that premise does not hold here. "
            "Run rda audit for per-episode evidence, repair or drop the "
            "affected episodes, then re-run recommend."
        )
        details = [
            f"EXCLUDE episodes: {n} / {summary.episodes_total}",
            f"Dominant damage type: {dominant}",
            "REPAIRABLE damage (frame loss / joint excursion) may be fixed "
            "by trimming or re-export; INVALID damage (NaN / time axis / "
            "schema) requires tracing back to acquisition or export.",
            f"REVIEW episodes: {summary.review_count} (advice not blocked, "
            "but human review is advised).",
        ]

    gate = Recommendation(
        action=RecommendationAction.REPAIR_FIRST,
        confidence=ConfidenceLevel.HIGH,
        title=title,
        summary=summary_text,
        details=details,
        caveats=[HELD_OUT_VALIDATION] + GENERAL_CAVEATS,
    )

    result.recommendations = [gate]
    # Attach the evidence for JSON consumers (report/pipeline use);
    # verdict_summary is a real field on RecommendationResult (REQ-1).
    result.verdict_summary = summary.to_dict()
    return result
