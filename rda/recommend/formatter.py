"""Text formatting for recommendation output (OPEN SOURCE).

Formats RecommendationResult as human-readable text for CLI output.
This module is intentionally separate from the engine so that it
can be distributed freely without exposing rule logic.

Copyright (c) 2026 Niu Su Tech. MIT License.
"""
from __future__ import annotations

from typing import List

from rda.recommend.types import (
    ConfidenceLevel,
    RecommendationResult,
    TargetPolicy,
)


def format_recommendation_text(result: RecommendationResult) -> str:
    """Format recommendation results as human-readable text.

    Uses restrained, conservative language per product positioning.

    Args:
        result: The recommendation result to format.

    Returns:
        A multi-line string suitable for CLI output.
    """
    ts = result.temporal_sufficiency
    lines: List[str] = []

    # Header with rules version
    rules_tag = f"  (rules v{result.rules_version})" if result.rules_version else ""
    lines.append("=" * 60)
    lines.append(f"  RDA Recommend{rules_tag}")
    lines.append("=" * 60)
    lines.append("")

    # Policy context
    policy_label = {
        TargetPolicy.FRAME_WISE: "逐帧模型 (MLP / BC)",
        TargetPolicy.TEMPORAL: "时序模型 (ACT / Diffusion Policy / Transformer)",
    }.get(result.target_policy, result.target_policy.value)
    lines.append(f"  目标模型类型：{policy_label}")
    lines.append("")

    # Temporal sufficiency snapshot
    lines.append("  -- Temporal Sufficiency 概况 --")
    if ts.computed_episodes > 0:
        idle_total = ts.idle_total_ratio.get("median", 0.0)
        idle_prefix = ts.idle_prefix_ratio.get("median", 0.0)
        active_p50 = ts.active_run_p50.get("median", 0.0)
        active_p90 = ts.active_run_p90.get("median", 0.0)
        vw5 = ts.valid_window_ratio_5.get("median", 0.0)
        vw10 = ts.valid_window_ratio_10.get("median", 0.0)
        vw20 = ts.valid_window_ratio_20.get("median", 0.0)
        transitions = ts.transition_count.get("median", 0.0)

        lines.append(f"  idle_total_ratio:      {idle_total:.1%}  (总 idle 帧占比)")
        lines.append(f"  idle_prefix_ratio:     {idle_prefix:.1%}  (初始等待帧占比)")
        lines.append(f"  active_run_p50/p90:    {active_p50:.0f} / {active_p90:.0f} 帧  (连续 active 段长度)")
        lines.append(f"  transition_count:      {transitions:.0f}  (idle->active 转换次数/episode)")
        lines.append(f"  valid_window_ratio:    seq=5: {vw5:.1%} | seq=10: {vw10:.1%} | seq=20: {vw20:.1%}")
    else:
        lines.append("  无法计算 temporal sufficiency 指标（缺少 action 数据）。")
    lines.append("")

    # Recommendations
    lines.append("  -- 建议 --")
    lines.append("")

    for i, rec in enumerate(result.recommendations, 1):
        conf_mark = {
            ConfidenceLevel.HIGH: "[HIGH]",
            ConfidenceLevel.EXPERIMENTAL: "[EXPERIMENTAL]",
            ConfidenceLevel.NOT_RECOMMENDED: "[NOT_RECOMMENDED]",
        }.get(rec.confidence, rec.confidence.value)

        lines.append(f"  建议 {i}: {conf_mark} {rec.title}")
        lines.append(f"         {rec.summary}")
        lines.append("")

        if rec.details:
            lines.append(f"     细节：")
            for detail in rec.details:
                lines.append(f"       - {detail}")
            lines.append("")

        if rec.expected_impact:
            lines.append(f"     预期影响：{rec.expected_impact}")
            lines.append("")

        if rec.caveats:
            lines.append(f"     注意事项：")
            for caveat in rec.caveats:
                lines.append(f"       ! {caveat}")
            lines.append("")

    # Disclaimer
    lines.append("  -- 免责声明 --")
    lines.append("  RDA 提供数据质量诊断和低风险优化建议，")
    lines.append("  不是保证成功率提升的自动优化器。")
    lines.append("  所有建议均基于特定实验条件下的观察，")
    lines.append("  实际效果随任务域、数据分布和模型架构变化。")
    lines.append("  应用任何优化策略前，请务必在 held-out 集上验证。")
    lines.append("")

    # Cache indicator
    if result.rules_version:
        lines.append(f"  Rules version: {result.rules_version}")
    lines.append("=" * 60)

    return "\n".join(lines)
