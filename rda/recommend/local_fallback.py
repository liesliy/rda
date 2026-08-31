"""Offline fallback for ``rda recommend`` (OPEN SOURCE).

When the remote rules API is unreachable AND no cached result exists,
this module provides a deliberately conservative local evaluation so the
CLI degrades gracefully instead of exiting with an error.

Design boundary
---------------
This module intentionally does NOT re-implement the closed-source rules
engine. It applies one obviously-safe branch plus a conservative
default, using only dataset-level aggregates that are already computed
client-side:

- TRIM_INITIAL  : only when the initial-idle prefix is essentially pure
  waiting (short prefix carrying most of the idle mass) and the target
  model is frame-wise. Never for temporal policies.
- DO_NOT_PRUNE  : the default. When the offline path cannot establish a
  clearly safe condition, it says "don't prune".

Every offline result is labeled with rules_version "offline-fallback"
so downstream consumers never mistake it for a server-graded verdict.

Copyright (c) 2026 Niu Su Tech. MIT License.
"""
from __future__ import annotations

from typing import List

from rda.recommend.types import (
    ConfidenceLevel,
    GENERAL_CAVEATS,
    HELD_OUT_VALIDATION,
    Recommendation,
    RecommendationAction,
    RecommendationResult,
    TargetPolicy,
)

OFFLINE_RULES_VERSION = "offline-fallback"


def _dist_median(d, default: float = 0.0) -> float:
    return float(d.get("median", default)) if d else default


def build_offline_result(
    agg,
    target_policy: TargetPolicy,
    lang: str = "zh",
) -> RecommendationResult:
    """Build a conservative RecommendationResult from local aggregates only.

    Args:
        agg: DatasetTemporalSufficiency computed client-side.
        target_policy: The user's intended model architecture.
        lang: Copy language ("zh" or "en").

    Returns:
        RecommendationResult with rules_version = "offline-fallback".
    """
    lang = lang if lang in ("zh", "en") else "zh"
    zh = lang == "zh"

    idle_total = _dist_median(agg.idle_total_ratio)
    idle_prefix = _dist_median(agg.idle_prefix_ratio)
    computed = agg.computed_episodes

    caveats = [HELD_OUT_VALIDATION] + GENERAL_CAVEATS
    recommendations: List[Recommendation] = []

    # ---- Guard: no computable metrics -> bare conservative default ----
    if computed == 0:
        summary = (
            "缺少 action 数据，无法计算时序指标。离线模式下无法评估，"
            "默认建议不要剪枝。" if zh else
            "No action data available, temporal metrics could not be computed. "
            "Cannot evaluate offline — defaulting to no pruning."
        )
        recommendations.append(Recommendation(
            action=RecommendationAction.DO_NOT_PRUNE,
            confidence=ConfidenceLevel.HIGH,
            title="保守起见，不剪枝 (DO_NOT_PRUNE)" if zh else "DO_NOT_PRUNE",
            summary=summary,
            details=[],
            caveats=caveats,
        ))
        return _wrap(agg, target_policy, recommendations)

    # ---- Branch 1 (the only offline "trim" suggestion) ----
    # Pure leading idle prefix, frame-wise policy only, conservative bounds:
    # the prefix is short, carries >=80% of all idle mass, and the
    # remaining mid-episode idle share is small.
    prefix_is_pure_waiting = (
        idle_prefix > 0.0
        and idle_total > 0.0
        and idle_prefix / idle_total >= 0.8   # most idle mass is in the prefix
        and idle_prefix <= 0.10               # prefix is <=10% of frames
    )
    rest_idle_small = (idle_total - idle_prefix) <= 0.10

    if target_policy == TargetPolicy.FRAME_WISE and prefix_is_pure_waiting and rest_idle_small:
        if zh:
            title = "可考虑仅去除初始等待帧 (TRIM_INITIAL, 离线保守判定)"
            summary = (
                f"离线模式下仅执行明显安全的判定：初始等待帧约占 {idle_prefix:.1%}，"
                f"且占全部 idle 帧的大头（{idle_prefix / idle_total:.0%}），"
                "去除每个 episode 开头的静止等待帧风险极低。"
                "其余帧间 idle 未做评估，不做任何剪枝建议。"
            )
            details = [
                f"初始等待帧占比（中位数）：{idle_prefix:.1%}",
                f"总 idle 占比（中位数）：{idle_total:.1%}",
                "仅删除每个 episode 的前导 idle 帧，保留所有主动作序列。",
                "完整规则引擎（含 TRIM_IDLE_MILD 等分级）仅在服务端可用，"
                "恢复联网后重新运行可获得完整评估。",
            ]
        else:
            title = "TRIM_INITIAL (offline conservative check)"
            summary = (
                f"Offline mode only asserts clearly-safe conditions: the initial "
                f"idle prefix is ~{idle_prefix:.1%} of frames and carries "
                f"{idle_prefix / idle_total:.0%} of all idle mass. Removing the "
                "leading static frames is very low risk. No other pruning is "
                "suggested offline."
            )
            details = [
                f"Initial idle prefix ratio (median): {idle_prefix:.1%}",
                f"Total idle ratio (median): {idle_total:.1%}",
                "Only leading idle frames are removed; active sequences stay intact.",
                "The full rules engine (incl. TRIM_IDLE_MILD grading) is "
                "server-side only; re-run online for the complete evaluation.",
            ]
        recommendations.append(Recommendation(
            action=RecommendationAction.TRIM_INITIAL,
            confidence=ConfidenceLevel.EXPERIMENTAL,
            title=title,
            summary=summary,
            details=details,
            caveats=caveats,
        ))
        return _wrap(agg, target_policy, recommendations)

    # ---- Conservative default (DO_NOT_PRUNE) with reason-specific copy ----
    if target_policy != TargetPolicy.FRAME_WISE:
        reason = (
            "时序模型（ACT / Diffusion Policy 等）依赖连续帧窗口，"
            "剪枝会破坏时序结构，离线模式一律不建议剪枝。" if zh else
            "Temporal policies (ACT / Diffusion Policy etc.) depend on contiguous "
            "frame windows; pruning breaks temporal structure. Offline mode "
            "never suggests pruning for temporal policies."
        )
        details = (
            [
                "时序策略使用连续帧序列作为输入，任何剪枝都改变帧间结构。",
                "离线模式无法评估窗口完整性（valid_window_ratio 的分级判定在服务端）。",
            ] if zh else [
                "Temporal policies consume contiguous frame sequences.",
                "Window-integrity grading (valid_window_ratio) is server-side only.",
            ]
        )
    elif idle_total > 0.5:
        reason = (
            f"数据集 idle 占比中位数约 {idle_total:.0%}，idle 分布复杂"
            "（大量 idle 在 episode 中段而非仅开头），离线模式不做评估。" if zh else
            f"Median idle ratio is ~{idle_total:.0%} with a complex idle layout "
            "(much of it mid-episode, not just the prefix). Offline mode does "
            "not evaluate this."
        )
        details = (
            [
                f"总 idle 占比（中位数）：{idle_total:.1%}",
                f"初始等待帧占比（中位数）：{idle_prefix:.1%}",
                "中段 idle 的处理属于完整规则引擎的判定范围，离线不可用。",
            ] if zh else [
                f"Total idle ratio (median): {idle_total:.1%}",
                f"Initial idle prefix ratio (median): {idle_prefix:.1%}",
                "Mid-episode idle handling belongs to the full server-side engine.",
            ]
        )
    else:
        reason = (
            f"数据集 idle 占比中位数约 {idle_total:.1%}，未满足离线模式下的"
            "明显安全条件（初始 idle 占比低且集中），保守起见不建议剪枝。" if zh else
            f"Median idle ratio is ~{idle_total:.1%} — the clearly-safe offline "
            "condition (low, prefix-concentrated idle) does not hold. "
            "Defaulting to no pruning."
        )
        details = (
            [
                f"总 idle 占比（中位数）：{idle_total:.1%}",
                f"初始等待帧占比（中位数）：{idle_prefix:.1%}",
            ] if zh else [
                f"Total idle ratio (median): {idle_total:.1%}",
                f"Initial idle prefix ratio (median): {idle_prefix:.1%}",
            ]
        )

    extra_note = (
        "恢复联网后重新运行 recommend 可获得服务端规则引擎的完整评估。" if zh else
        "Re-run recommend when online to get the full server-side evaluation."
    )

    recommendations.append(Recommendation(
        action=RecommendationAction.DO_NOT_PRUNE,
        confidence=ConfidenceLevel.HIGH,
        title="保守起见，不剪枝 (DO_NOT_PRUNE)" if zh else "DO_NOT_PRUNE",
        summary=reason,
        details=details + [extra_note],
        caveats=caveats,
    ))
    return _wrap(agg, target_policy, recommendations)


def _wrap(agg, target_policy: TargetPolicy,
          recommendations: List[Recommendation]) -> RecommendationResult:
    return RecommendationResult(
        target_policy=target_policy,
        temporal_sufficiency=agg,
        recommendations=recommendations,
        rules_version=OFFLINE_RULES_VERSION,
        engine_version="offline",
    )
