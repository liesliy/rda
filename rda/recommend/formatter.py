"""Text formatting for recommendation output (OPEN SOURCE).

Formats RecommendationResult as human-readable text for CLI output.
This module is intentionally separate from the engine so that it
can be distributed freely without exposing rule logic.

Bilingual: ``lang="zh"`` (default) or ``lang="en"`` selects the
template language. The recommendation copy itself comes from the
rules engine, which is also language-aware.

Copyright (c) 2026 Niu Su Tech. MIT License.
"""
from __future__ import annotations

from typing import List

from rda.recommend.types import (
    ConfidenceLevel,
    RecommendationResult,
    TargetPolicy,
)


def format_recommendation_text(result: RecommendationResult, lang: str = "zh") -> str:
    """Format recommendation results as human-readable text.

    Uses restrained, conservative language per product positioning.

    Args:
        result: The recommendation result to format.
        lang: Output language ("zh" or "en"). Defaults to "zh".

    Returns:
        A multi-line string suitable for CLI output.
    """
    lang = lang if lang in ("zh", "en") else "zh"
    ts = result.temporal_sufficiency
    lines: List[str] = []

    # Header with rules version
    is_offline = result.rules_version == "offline-fallback"
    rules_tag = f"  (rules v{result.rules_version})" if result.rules_version else ""
    lines.append("=" * 60)
    if is_offline:
        if lang == "zh":
            lines.append("  RDA Recommend  [离线保守模式]")
        else:
            lines.append("  RDA Recommend  [offline conservative mode]")
        if lang == "zh":
            lines.append("  注意：本次结果由本地保守规则生成，未经服务端规则引擎评估。")
        else:
            lines.append("  Note: graded by built-in local rules, not the server-side engine.")
    else:
        lines.append(f"  RDA Recommend{rules_tag}")
    lines.append("=" * 60)
    lines.append("")

    # Policy context
    policy_label = {
        TargetPolicy.FRAME_WISE: {
            "zh": "逐帧模型 (MLP / BC)",
            "en": "Frame-wise model (MLP / BC)",
        },
        TargetPolicy.TEMPORAL: {
            "zh": "时序模型 (ACT / Diffusion Policy / Transformer)",
            "en": "Temporal model (ACT / Diffusion Policy / Transformer)",
        },
    }.get(result.target_policy, {lang: result.target_policy.value})[lang]
    if lang == "zh":
        lines.append(f"  目标模型类型：{policy_label}")
    else:
        lines.append(f"  Target model type: {policy_label}")
    lines.append("")

    # Temporal sufficiency snapshot
    if lang == "zh":
        lines.append("  -- Temporal Sufficiency 概况 --")
    else:
        lines.append("  -- Temporal Sufficiency Overview --")
    if ts.computed_episodes > 0:
        idle_total = ts.idle_total_ratio.get("median", 0.0)
        idle_prefix = ts.idle_prefix_ratio.get("median", 0.0)
        active_p50 = ts.active_run_p50.get("median", 0.0)
        active_p90 = ts.active_run_p90.get("median", 0.0)
        vw5 = ts.valid_window_ratio_5.get("median", 0.0)
        vw10 = ts.valid_window_ratio_10.get("median", 0.0)
        vw20 = ts.valid_window_ratio_20.get("median", 0.0)
        transitions = ts.transition_count.get("median", 0.0)

        if lang == "zh":
            lines.append(f"  idle_total_ratio:      {idle_total:.1%}  (总 idle 帧占比)")
            lines.append(f"  idle_prefix_ratio:     {idle_prefix:.1%}  (初始等待帧占比)")
            lines.append(f"  active_run_p50/p90:    {active_p50:.0f} / {active_p90:.0f} 帧  (连续 active 段长度)")
            lines.append(f"  transition_count:      {transitions:.0f}  (idle->active 转换次数/episode)")
        else:
            lines.append(f"  idle_total_ratio:      {idle_total:.1%}  (overall idle frame ratio)")
            lines.append(f"  idle_prefix_ratio:     {idle_prefix:.1%}  (initial waiting prefix ratio)")
            lines.append(f"  active_run_p50/p90:    {active_p50:.0f} / {active_p90:.0f} frames  (active-run length)")
            lines.append(f"  transition_count:      {transitions:.0f}  (idle->active transitions per episode)")
        lines.append(f"  valid_window_ratio:    seq=5: {vw5:.1%} | seq=10: {vw10:.1%} | seq=20: {vw20:.1%}")
        # REQ-3 (v0.6.0): DROID-aligned retention metrics
        usable = getattr(ts, "usable_retention_ratio", {}).get("median", 0.0)
        max_idle = getattr(ts, "max_idle_run_frames", {}).get("median", 0.0)
        if lang == "zh":
            lines.append(f"  usable_retention:      {usable:.1%}  (连续 ≥16 帧非静止段帧占比, 对齐 DROID)")
            lines.append(f"  max_idle_run:          {max_idle:.0f} 帧  (最长连续静止段)")
        else:
            lines.append(f"  usable_retention:      {usable:.1%}  (frames in runs >= 16 non-idle, DROID-aligned)")
            lines.append(f"  max_idle_run:          {max_idle:.0f} frames  (longest static stretch)")
    else:
        if lang == "zh":
            lines.append("  无法计算 temporal sufficiency 指标（缺少 action 数据）。")
        else:
            lines.append("  Temporal sufficiency metrics could not be computed (missing action data).")
    lines.append("")

    # Recommendations
    lines.append(f"  -- {'建议' if lang == 'zh' else 'Recommendations'} --")
    lines.append("")

    rec_word = "建议" if lang == "zh" else "Recommendation"
    for i, rec in enumerate(result.recommendations, 1):
        conf_mark = {
            ConfidenceLevel.HIGH: "[HIGH]",
            ConfidenceLevel.EXPERIMENTAL: "[EXPERIMENTAL]",
            ConfidenceLevel.NOT_RECOMMENDED: "[NOT_RECOMMENDED]",
        }.get(rec.confidence, rec.confidence.value)

        # REQ-1: REPAIR_FIRST is a verdict gate, not a normal
        # recommendation — render it as a warning block.
        if rec.action.value == "REPAIR_FIRST":
            lines.append("  " + "!" * 56)
            if lang == "zh":
                lines.append("  ⚠ 数据预检未通过 — 已抑制全部剪枝建议")
            else:
                lines.append("  ⚠ DATA PREFLIGHT FAILED — all pruning advice suppressed")
            lines.append("  " + "!" * 56)
            lines.append(f"  {conf_mark} {rec.title}")
            lines.append(f"         {rec.summary}")
            lines.append("")
            if rec.details:
                for detail in rec.details:
                    lines.append(f"       - {detail}")
            lines.append("")
            continue

        # REQ-3/REQ-2 (v0.6.0): review-type actions render with a
        # "human review" marker instead of the generic layout.
        if rec.action.value in (
            "DISCARD_STATIC", "SMOOTHING_REVIEW", "CALIBRATION_CHECK",
            "COVERAGE_SUGGESTION",
        ):
            lines.append(f"  {rec_word} {i}: {conf_mark} {rec.title}")
            lines.append(f"         {rec.summary}")
            lines.append("")
            if rec.details:
                for detail in rec.details:
                    lines.append(f"       - {detail}")
                lines.append("")
            if lang == "zh":
                lines.append("       ⚑ 此建议为人工复核信号，不构成自动处理指令。")
            else:
                lines.append("       ⚑ Human-review signal — not an automated processing instruction.")
            lines.append("")
            continue

        if lang == "zh":
            lines.append(f"  {rec_word} {i}: {conf_mark} {rec.title}")
        else:
            lines.append(f"  {rec_word} {i}: {conf_mark} {rec.title}")
        lines.append(f"         {rec.summary}")
        lines.append("")

        if rec.details:
            lines.append(f"     {'细节' if lang == 'zh' else 'Details'}：")
            for detail in rec.details:
                lines.append(f"       - {detail}")
            lines.append("")

        if rec.expected_impact:
            label = "预期影响" if lang == "zh" else "Expected impact"
            sep = "：" if lang == "zh" else ": "
            lines.append(f"     {label}{sep}{rec.expected_impact}")
            lines.append("")

        if rec.caveats:
            lines.append(f"     {'注意事项' if lang == 'zh' else 'Caveats'}：")
            for caveat in rec.caveats:
                lines.append(f"       ! {caveat}")
            lines.append("")

    # Disclaimer
    if lang == "zh":
        lines.append("  -- 免责声明 --")
        lines.append("  RDA 提供数据质量诊断和低风险优化建议，")
        lines.append("  不是保证成功率提升的自动优化器。")
        lines.append("  所有建议均基于特定实验条件下的观察，")
        lines.append("  实际效果随任务域、数据分布和模型架构变化。")
        lines.append("  应用任何优化策略前，请务必在 held-out 集上验证。")
    else:
        lines.append("  -- Disclaimer --")
        lines.append("  RDA provides data quality diagnostics and low-risk")
        lines.append("  optimization advice — it is not an optimizer that")
        lines.append("  guarantees success-rate improvement.")
        lines.append("  All recommendations are observations under specific")
        lines.append("  experimental conditions; actual effects vary with task")
        lines.append("  domain, data distribution, and model architecture.")
        lines.append("  Validate on a held-out set before applying anything.")
    lines.append("")

    # Cache indicator
    if result.rules_version:
        lines.append(f"  Rules version: {result.rules_version}")
    if is_offline:
        if lang == "zh":
            lines.append("  恢复联网后重新运行 recommend 可获得完整服务端评估。")
        else:
            lines.append("  Re-run recommend when online for the full server-side evaluation.")
    lines.append("=" * 60)

    return "\n".join(lines)
