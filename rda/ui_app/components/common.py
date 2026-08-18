"""可复用 UI 组件。

提供雷达图、健康指数计算、问题统计等通用组件，供各页面调用。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rda.audit.dataset_audit import DatasetAuditResult
from rda.audit.rules import AuditVerdict, CRITICAL_METRICS, REVIEW_METRICS
from rda.metrics.base import MetricAvailability
from rda.report.aggregation import aggregate_dataset_metrics

# ---------------------------------------------------------------------------
# 四维健康指数计算
# ---------------------------------------------------------------------------

def _dhi_detail(key: str, lang: str, **kwargs) -> str:
    """Localized readiness-detail strings for compute_dhi."""
    table = {
        "zh": {
            "empty": "数据集中无 episode",
            "ready": f"数据可直接用于训练，共 {kwargs.get('total', 0)} 个 episode",
            "cond": f"审核 {kwargs.get('review', 0)} 个 episode 后可用于训练",
            "not_ready": "存在显著质量问题，建议先处理数据",
        },
        "en": {
            "empty": "No episodes in dataset",
            "ready": f"Data is ready for training — {kwargs.get('total', 0)} episodes",
            "cond": f"Usable for training after reviewing {kwargs.get('review', 0)} episodes",
            "not_ready": "Significant quality issues found — clean the data first",
        },
    }
    return table.get(lang, table["zh"]).get(key, key)


def compute_dhi(result: DatasetAuditResult, lang: str = "zh") -> Dict[str, Any]:
    """计算 Dataset Health Index 与四维维度得分。

    权重：Integrity 0.4 / Temporal 0.2 / Motion 0.2 / Consistency 0.2
    每个维度归一化到 0-100。

    Returns:
        {
            "dhi": float,
            "grade": str,
            "dimensions": {"integrity": float, "temporal": float,
                           "motion": float, "consistency": float},
            "training_readiness": str,
            "training_readiness_detail": str,
            "estimated_post_cleanup_quality": float,
        }
    """
    total = result.num_episodes
    if total == 0:
        return {
            "dhi": 0.0,
            "grade": "Poor",
            "dimensions": {"integrity": 0.0, "temporal": 0.0,
                           "motion": 0.0, "consistency": 0.0},
            "training_readiness": "Not Ready",
            "training_readiness_detail": _dhi_detail("empty", lang),
            "estimated_post_cleanup_quality": 0.0,
        }

    counts = {v: result.verdict_counts.get(v, 0) for v in AuditVerdict}

    # ---------- Integrity 维度 ----------
    # Layer 1A: 完整性检查通过率（critical metric 全通过的 episode 比例）
    integrity_pass = 0
    for ep in result.episodes.values():
        has_critical_fail = False
        for m_name in CRITICAL_METRICS:
            m = ep.metrics.get(m_name)
            if m is None:
                continue
            if m.availability != MetricAvailability.AVAILABLE:
                continue
            if not m.passed:
                has_critical_fail = True
                break
        if not has_critical_fail:
            integrity_pass += 1
    integrity_score = (integrity_pass / total) * 100.0

    # ---------- Temporal 维度 ----------
    # 时间戳有效性 + FPS 稳定性
    ts_pass = 0
    ts_available = 0
    for ep in result.episodes.values():
        m = ep.metrics.get("timestamp_validity")
        if m is None:
            continue
        if m.availability != MetricAvailability.AVAILABLE:
            continue
        ts_available += 1
        if m.passed:
            ts_pass += 1

    # FPS 稳定性（通过 dt 标准差推断）
    fps_stability_scores: List[float] = []
    for ep in result.episodes.values():
        m = ep.metrics.get("timestamp_validity")
        if m is None or m.availability != MetricAvailability.AVAILABLE:
            continue
        # dt 的变异系数越小越稳定
        dt_stats = m.measurement
        if "std_dt_ms" in dt_stats and "median_dt_ms" in dt_stats:
            median_dt = dt_stats["median_dt_ms"]
            std_dt = dt_stats["std_dt_ms"]
            if median_dt > 0:
                cv = std_dt / median_dt
                # cv=0 → 100, cv=0.5 → 0
                stability = max(0.0, min(100.0, 100.0 * (1 - cv / 0.5)))
                fps_stability_scores.append(stability)

    ts_pass_rate = (ts_pass / ts_available * 100.0) if ts_available > 0 else 100.0
    fps_stability = float(np.mean(fps_stability_scores)) if fps_stability_scores else 85.0
    temporal_score = 0.7 * ts_pass_rate + 0.3 * fps_stability

    # ---------- Motion 维度 ----------
    # 运动质量：有效运动比例 + 动作不连续性
    idle_ratio_scores: List[float] = []
    disc_scores: List[float] = []

    for ep in result.episodes.values():
        m = ep.metrics.get("idle_ratio")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            eff = m.measurement.get("effective_motion_ratio", 0.0)
            # 有效运动比例直接映射
            idle_ratio_scores.append(min(100.0, float(eff) * 100.0))

        m = ep.metrics.get("action_discontinuity")
        if m is not None and m.availability == MetricAvailability.AVAILABLE:
            spikes = m.measurement.get("spike_count", 0)
            # 0 spikes → 100, >50 → 0
            score = max(0.0, min(100.0, 100.0 - (spikes / 50.0) * 100.0))
            disc_scores.append(score)

    motion_eff = float(np.mean(idle_ratio_scores)) if idle_ratio_scores else 90.0
    motion_smooth = float(np.mean(disc_scores)) if disc_scores else 95.0
    motion_score = 0.5 * motion_eff + 0.5 * motion_smooth

    # ---------- Consistency 维度 ----------
    # 行为一致性：基于 verdict 分布，PASS 比例越高越一致
    pass_count = counts.get(AuditVerdict.PASS, 0)
    review_count = counts.get(AuditVerdict.REVIEW, 0)
    exclude_count = counts.get(AuditVerdict.EXCLUDE, 0)

    # PASS 全分，REVIEW 扣一半，EXCLUDE 全扣
    consistency_score = (
        (pass_count + review_count * 0.5) / total * 100.0
    ) if total > 0 else 0.0

    # ---------- 加权 DHI ----------
    dhi = (
        0.4 * integrity_score
        + 0.2 * temporal_score
        + 0.2 * motion_score
        + 0.2 * consistency_score
    )

    # ---------- 等级 ----------
    if dhi >= 90:
        grade = "Excellent"
    elif dhi >= 75:
        grade = "Good"
    elif dhi >= 60:
        grade = "Fair"
    else:
        grade = "Poor"

    # ---------- Training Readiness ----------
    exclude_pct = exclude_count / total if total > 0 else 0.0
    review_pct = review_count / total if total > 0 else 0.0

    if exclude_pct < 0.05 and review_pct < 0.15:
        readiness = "Ready"
        readiness_detail = _dhi_detail("ready", lang, total=total)
    elif exclude_pct < 0.10 and review_pct < 0.30:
        readiness = "Conditionally Ready"
        readiness_detail = _dhi_detail("cond", lang, review=review_count)
    else:
        readiness = "Not Ready"
        readiness_detail = _dhi_detail("not_ready", lang)

    # ---------- 预估清洗后质量 ----------
    if total > 0 and (total - exclude_count) > 0:
        post_cleanup = (
            (pass_count + review_count * 0.7) / (total - exclude_count) * 100.0
        )
        # 完整性也会提升（排除 integrity fail 的）
        post_cleanup = min(100.0, post_cleanup * 0.6 + integrity_score * 0.4)
    else:
        post_cleanup = dhi

    return {
        "dhi": round(dhi, 1),
        "grade": grade,
        "dimensions": {
            "integrity": round(integrity_score, 1),
            "temporal": round(temporal_score, 1),
            "motion": round(motion_score, 1),
            "consistency": round(consistency_score, 1),
        },
        "training_readiness": readiness,
        "training_readiness_detail": readiness_detail,
        "estimated_post_cleanup_quality": round(post_cleanup, 1),
    }


# ---------------------------------------------------------------------------
# 雷达图
# ---------------------------------------------------------------------------

def make_radar_chart(dimensions: Dict[str, float], lang: str = "zh") -> go.Figure:
    """生成四维雷达图。"""
    labels_local = {
        "zh": ["数据完整性", "时间质量", "运动质量", "行为一致性"],
        "en": ["Integrity", "Temporal", "Motion", "Consistency"],
    }
    score_label = "得分" if lang == "zh" else "Score"
    labels = labels_local.get(lang, labels_local["zh"])
    values = [
        dimensions.get("integrity", 0),
        dimensions.get("temporal", 0),
        dimensions.get("motion", 0),
        dimensions.get("consistency", 0),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.3)",
        line=dict(color="rgb(99, 102, 241)", width=2),
        name=score_label,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[0, 25, 50, 75, 100]),
            angularaxis=dict(tickfont=dict(size=13)),
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# 问题统计（Critical / Warning / Info）
# ---------------------------------------------------------------------------

def compute_issue_stats(result: DatasetAuditResult, lang: str = "zh") -> Dict[str, Any]:
    """统计问题分布。

    Returns:
        {
            "critical": int,   # 有 critical 问题的 episode 数
            "warning": int,    # 有 review 级别问题的 episode 数
            "info": int,       # 提示性观察
            "top_issues": [{"code": str, "count": int, "severity": str,
                            "description": str, "pattern_type": str|None}, ...]
        }
    """
    total = result.num_episodes
    if total == 0:
        return {"critical": 0, "warning": 0, "info": 0, "top_issues": []}

    # 按 metric 统计失败数
    fail_counts: Dict[str, int] = {}
    critical_eps = 0
    warning_eps = 0

    for ep in result.episodes.values():
        ep_critical = False
        ep_warning = False
        for m_name, m in ep.metrics.items():
            if m.availability != MetricAvailability.AVAILABLE:
                continue
            if not m.passed:
                fail_counts[m_name] = fail_counts.get(m_name, 0) + 1
                if m_name in CRITICAL_METRICS:
                    ep_critical = True
                elif m_name in REVIEW_METRICS:
                    ep_warning = True
        if ep_critical:
            critical_eps += 1
        elif ep_warning:
            warning_eps += 1

    # 问题描述映射（双语）
    issue_desc = {
        "zh": {
            "missing_dropout": "缺失帧 / 传感器断连",
            "invalid_values": "NaN/Inf 异常值",
            "schema_consistency": "数据格式不匹配",
            "timestamp_validity": "时间戳异常（非单调/间隔不均）",
            "joint_limit": "关节限位超限",
            "sensor_synchronization": "多传感器同步偏差",
            "sampling_jitter": "采样抖动",
            "velocity_acceleration": "速度/加速度异常",
            "action_discontinuity": "动作不连续（抖动）",
            "idle_ratio": "有效运动比例过低",
            "distribution": "轨迹分布离群",
            "coverage": "状态空间覆盖不足",
        },
        "en": {
            "missing_dropout": "Missing frames / sensor dropout",
            "invalid_values": "NaN/Inf invalid values",
            "schema_consistency": "Schema mismatch",
            "timestamp_validity": "Timestamp anomalies (non-monotonic / uneven)",
            "joint_limit": "Joint limit violation",
            "sensor_synchronization": "Multi-sensor sync deviation",
            "sampling_jitter": "Sampling jitter",
            "velocity_acceleration": "Velocity/acceleration anomalies",
            "action_discontinuity": "Action discontinuity (jitter)",
            "idle_ratio": "Low effective-motion ratio",
            "distribution": "Trajectory distribution outlier",
            "coverage": "Insufficient state-space coverage",
        },
    }
    desc_table = issue_desc.get(lang, issue_desc["zh"])

    # 映射到 Pattern Type
    metric_to_pattern = {
        "missing_dropout": None,
        "invalid_values": None,
        "schema_consistency": None,
        "timestamp_validity": None,
        "joint_limit": None,
        "action_discontinuity": "Jittery",
        "idle_ratio": "Stuck/Frozen",
        "distribution": "Inefficient",
        "velocity_acceleration": "Jittery",
        "coverage": "Unusual",
    }

    top_issues = []
    sorted_fail = sorted(fail_counts.items(), key=lambda x: x[1], reverse=True)
    for code, count in sorted_fail[:10]:
        severity = "critical" if code in CRITICAL_METRICS else (
            "warning" if code in REVIEW_METRICS else "info"
        )
        top_issues.append({
            "code": code.upper().replace("_", "-"),
            "metric_name": code,
            "count": count,
            "severity": severity,
            "description": desc_table.get(code, code),
            "pattern_type": metric_to_pattern.get(code),
        })

    return {
        "critical": critical_eps,
        "warning": warning_eps,
        "info": 0,  # info 级别的作为观察而非问题计数
        "top_issues": top_issues,
    }


# ---------------------------------------------------------------------------
# Episode 列表 DataFrame
# ---------------------------------------------------------------------------

def build_episodes_dataframe(result: DatasetAuditResult, lang: str = "zh") -> pd.DataFrame:
    """构建 Episode 列表 DataFrame，用于 Episode Explorer。"""
    rows: List[Dict[str, Any]] = []

    for ep_idx, ep in result.episodes.items():
        # Integrity check
        integrity_pass = True
        integrity_issues: List[str] = []
        for m_name in CRITICAL_METRICS:
            m = ep.metrics.get(m_name)
            if m is None:
                continue
            if m.availability != MetricAvailability.AVAILABLE:
                continue
            if not m.passed:
                integrity_pass = False
                integrity_issues.append(m_name.upper().replace("_", "-"))

        # Deviation score (基于 score 推导的综合异常度)
        scores = []
        for m_name, m in ep.metrics.items():
            if m.availability == MetricAvailability.AVAILABLE:
                scores.append(m.score)
        avg_score = float(np.mean(scores)) if scores else 1.0
        deviation_score = round((1.0 - avg_score) * 100, 2)

        # Reference-calibrated behavioral scores (portable / platform / combined)
        portable_score = ep.portable_score
        platform_score = ep.platform_score
        combined_score = ep.combined_score
        has_platform_metrics = ep.has_platform_metrics

        # Pattern type
        pattern_type = _detect_pattern_type(ep)

        # Top issues
        top_issues = []
        for m_name, m in ep.metrics.items():
            if m.availability != MetricAvailability.AVAILABLE:
                continue
            if not m.passed:
                top_issues.append(m_name.upper().replace("_", "-"))
        top_issues_str = ", ".join(top_issues[:3]) if top_issues else (
            "无" if lang != "en" else "None"
        )

        rows.append({
            "episode_id": ep_idx,
            "num_frames": ep.num_frames,
            "integrity_check": "PASS" if integrity_pass else "FAIL",
            "integrity_issues": ", ".join(integrity_issues) if integrity_issues else "—",
            "behavior_verdict": ep.verdict.value,
            "deviation_score": deviation_score,
            "portable_score": round(portable_score, 2) if portable_score is not None else None,
            "platform_score": round(platform_score, 2) if platform_score is not None else None,
            "combined_score": round(combined_score, 2) if combined_score is not None else None,
            "has_platform_metrics": has_platform_metrics,
            "pattern_type": pattern_type or "—",
            "top_issues": top_issues_str,
            "issue_count": len(top_issues),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # 按 verdict 优先级排序（EXCLUDE 优先），同 verdict 按偏离度降序
        verdict_order = {"EXCLUDE": 0, "REVIEW": 1, "PASS": 2}
        df["_verdict_order"] = df["behavior_verdict"].map(verdict_order).fillna(3)
        df = df.sort_values(
            by=["_verdict_order", "deviation_score"],
            ascending=[True, False],
        ).drop(columns=["_verdict_order"]).reset_index(drop=True)
    return df


def _detect_pattern_type(ep_result) -> Optional[str]:
    """根据 metric 结果推断 Pattern Type（简化版）。"""
    m_idle = ep_result.metrics.get("idle_ratio")
    m_disc = ep_result.metrics.get("action_discontinuity")
    m_dist = ep_result.metrics.get("distribution")

    idle_low = False
    spikes_high = False
    duration_long = False

    if m_idle and m_idle.availability == MetricAvailability.AVAILABLE:
        eff = m_idle.measurement.get("effective_motion_ratio", 1.0)
        idle_low = eff < 0.3

    if m_disc and m_disc.availability == MetricAvailability.AVAILABLE:
        spikes = m_disc.measurement.get("spike_count", 0)
        spikes_high = spikes > 20

    if m_dist and m_dist.availability == MetricAvailability.AVAILABLE:
        # 简单用 duration 是否异常判断
        if not m_dist.passed:
            duration_long = True

    # 判定
    if idle_low and spikes_high:
        return "Stuck"
    if idle_low and not spikes_high:
        return "Frozen"
    if spikes_high and not idle_low:
        return "Jittery"
    if duration_long and not idle_low:
        return "Inefficient"

    # 检查是否有 review 级别的行为异常
    has_behavior_issue = False
    for m_name, m in ep_result.metrics.items():
        if m_name in REVIEW_METRICS and m.availability == MetricAvailability.AVAILABLE and not m.passed:
            has_behavior_issue = True
            break
    if has_behavior_issue:
        return "Unusual"

    return None


# ---------------------------------------------------------------------------
# 训练就绪度徽章样式
# ---------------------------------------------------------------------------

def get_readiness_badge(readiness: str) -> Tuple[str, str]:
    """返回 (emoji, color) 用于展示 Training Readiness。"""
    mapping = {
        "Ready": ("🟢", "#10b981"),
        "Conditionally Ready": ("🟡", "#f59e0b"),
        "Not Ready": ("🔴", "#ef4444"),
    }
    return mapping.get(readiness, ("⚪", "#9ca3af"))


def get_grade_badge(grade: str) -> Tuple[str, str]:
    """返回 (emoji, color) 用于展示 DHI 等级。"""
    mapping = {
        "Excellent": ("🟢", "#10b981"),
        "Good": ("🟡", "#f59e0b"),
        "Fair": ("🟠", "#f97316"),
        "Poor": ("🔴", "#ef4444"),
    }
    return mapping.get(grade, ("⚪", "#9ca3af"))


def get_verdict_badge(verdict: str) -> Tuple[str, str]:
    """返回 (emoji, color) 用于展示 verdict。"""
    mapping = {
        "PASS": ("🟢", "#10b981"),
        "REVIEW": ("🟡", "#f59e0b"),
        "EXCLUDE": ("🔴", "#ef4444"),
    }
    return mapping.get(verdict, ("⚪", "#9ca3af"))
