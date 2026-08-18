"""Page 3: Health Overview · 数据集健康概览（首页）。

展示：
- Dataset Health Score 四维雷达图
- DHI + 等级
- Training Readiness
- 问题统计（Critical / Warning / Info）
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from components.common import (  # noqa: E402
    compute_dhi,
    compute_issue_stats,
    get_grade_badge,
    get_readiness_badge,
    make_radar_chart,
)

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _get_recommendation(metric_name: str) -> str:
    """根据 metric 返回中文建议。"""
    recs = {
        "missing_dropout": "检查采集系统的存储和传感器连接，确保录制过程中无丢包。",
        "invalid_values": "检查数据采集管线的数值处理逻辑，排查 NaN/Inf 的产生源头。",
        "schema_consistency": "检查数据格式版本一致性，确认所有 episode 的张量维度和 dtype 统一。",
        "timestamp_validity": "检查采集系统的时间戳源，确保时钟同步和单调递增。",
        "joint_limit": "检查机器人控制策略，避免关节打到机械限位。",
        "sensor_synchronization": "校准多传感器的时间同步，或使用硬件触发同步。",
        "sampling_jitter": "检查采集系统的采样时钟稳定性，可能需要优化调度优先级。",
        "velocity_acceleration": "检查控制器参数，动作可能过于激进；或检查传感器噪声。",
        "action_discontinuity": "检查动作平滑度，可能需要调整控制器或加入动作滤波。",
        "idle_ratio": "检查 episode 中是否存在长时间停顿，可能是任务失败或等待状态。",
        "distribution": "检查轨迹分布，离群 episode 可能对应失败或异常的任务执行。",
        "coverage": "数据集状态空间覆盖度较低，建议增加多样化的采集场景。",
    }
    return recs.get(metric_name, "建议人工审核对应 episodes，确认具体原因。")



st.title("💚 数据集健康概览")
st.caption("数据集整体质量评估 · 不展示算法细节，只展示结论")

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if st.session_state.audit_result is None:
    st.warning("⚠️ 请先运行审计", icon="🔍")
    st.page_link("pages/2_Audit.py", label="前往 Audit 页面 →", icon="🔍")
    st.stop()

result = st.session_state.audit_result
info = result.dataset_info

# 确保已计算 DHI
if st.session_state.dataset_report is None:
    st.session_state.dataset_report = compute_dhi(result)

report = st.session_state.dataset_report
dhi = report["dhi"]
grade = report["grade"]
dims = report["dimensions"]
readiness = report["training_readiness"]
readiness_detail = report["training_readiness_detail"]

# ---------------------------------------------------------------------------
# Training Readiness 大横幅
# ---------------------------------------------------------------------------
emoji_r, color_r = get_readiness_badge(readiness)
st.markdown(f"""
<div style="padding: 24px; border-radius: 12px; background: linear-gradient(135deg, {color_r}15 0%, {color_r}05 100%); border: 1px solid {color_r}30;">
    <div style="font-size: 14px; color: #6b7280; margin-bottom: 8px;">训练就绪度 · Training Readiness</div>
    <div style="font-size: 28px; font-weight: 700; color: {color_r};">{emoji_r} {readiness}</div>
    <div style="font-size: 15px; color: #4b5563; margin-top: 4px;">{readiness_detail}</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# Verdict 分布
counts = {v.value: c for v, c in result.verdict_counts.items()}
total = result.num_episodes
pass_c = counts.get("PASS", 0)
review_c = counts.get("REVIEW", 0)
exclude_c = counts.get("EXCLUDE", 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🟢 Keep", f"{pass_c}",
              f"{pass_c / max(total, 1) * 100:.1f}% of episodes")
with col2:
    st.metric("🟡 Review", f"{review_c}",
              f"{review_c / max(total, 1) * 100:.1f}% of episodes")
with col3:
    st.metric("🔴 Remove", f"{exclude_c}",
              f"{exclude_c / max(total, 1) * 100:.1f}% of episodes")

st.divider()

# ---------------------------------------------------------------------------
# DHI + 雷达图
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("数据健康指数 (DHI)")
    emoji_g, color_g = get_grade_badge(grade)

    st.markdown(f"""
    <div style="text-align: center; padding: 30px 20px; border-radius: 16px; background: #f8fafc; border: 1px solid #e2e8f0;">
        <div style="font-size: 64px; font-weight: 800; color: {color_g}; line-height: 1;">{dhi}<span style="font-size: 24px; color: #94a3b8;"> / 100</span></div>
        <div style="font-size: 20px; margin-top: 8px; color: #1e293b;">{emoji_g} {grade} 级</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 12px; font-style: italic;">
            基于数据完整性与行为一致性的相对评估<br/>
            Relative assessment, not absolute score
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # 维度进度条
    st.markdown("**四维质量得分**")
    dim_labels = [
        ("数据完整性 (Integrity)", dims["integrity"], 0.4),
        ("时间质量 (Temporal)", dims["temporal"], 0.2),
        ("运动质量 (Motion)", dims["motion"], 0.2),
        ("行为一致性 (Consistency)", dims["consistency"], 0.2),
    ]
    for name, score, weight in dim_labels:
        cols = st.columns([2, 3, 1])
        with cols[0]:
            st.caption(name)
        with cols[1]:
            st.progress(score / 100.0)
        with cols[2]:
            st.caption(f"**{score:.0f}** / 100")

with col_right:
    st.subheader("雷达图")
    fig = make_radar_chart(dims)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 可复用性统计
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("当前 DHI", f"{dhi:.0f} / 100")
with col2:
    usable_pct = (pass_c + review_c) / max(total, 1) * 100
    st.metric("Potentially usable after review", f"{usable_pct:.1f}%",
              f"{pass_c + review_c} / {total} episodes",
              delta_color="off")
with col3:
    keep_pct = pass_c / max(total, 1) * 100
    st.metric("Clear Keep (无疑问)", f"{keep_pct:.1f}%",
              f"{pass_c} episodes",
              delta_color="off")

st.caption(f"Potentially usable = KEEP + REVIEW（经人工审核后可保留的部分）；不包含明确 REMOVE 的 episodes")

st.divider()

# ---------------------------------------------------------------------------
# Top Issues
# ---------------------------------------------------------------------------
st.subheader("🔴 主要数据质量问题")

issue_stats = compute_issue_stats(result)
top_issues = issue_stats["top_issues"]

if not top_issues:
    st.success("🎉 未发现明显的数据质量问题", icon="✅")
else:
    # 统计 Critical / Warning / Info
    critical_count = sum(1 for i in top_issues if i["severity"] == "critical")
    warning_count = sum(1 for i in top_issues if i["severity"] == "warning")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 严重 (Critical)", issue_stats["critical"])
    with col2:
        st.metric("🟡 警告 (Warning)", issue_stats["warning"])
    with col3:
        st.metric("📋 问题类型数", len(top_issues))

    st.write("")

    # 问题列表
    for idx, issue in enumerate(top_issues[:8], 1):
        sev_color = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🔵",
        }.get(issue["severity"], "⚪")

        sev_label = {
            "critical": "严重",
            "warning": "警告",
            "info": "提示",
        }.get(issue["severity"], "未知")

        with st.expander(
            f"{sev_color} **{idx}. {issue['description']}**  "
            f"`{issue['code']}`  "
            f"· 影响 {issue['count']} 个 episodes "
            f"({issue['count'] / max(total, 1) * 100:.1f}%)",
            expanded=idx <= 3,
        ):
            cols = st.columns(4)
            with cols[0]:
                st.caption("问题等级")
                st.write(f"{sev_color} {sev_label}")
            with cols[1]:
                st.caption("影响范围")
                st.write(f"{issue['count']} / {total} episodes")
            with cols[2]:
                st.caption("问题编码")
                st.code(issue["code"])
            with cols[3]:
                if issue["pattern_type"]:
                    st.caption("Pattern Type")
                    st.write(f"`{issue['pattern_type']}`")

            st.caption("**建议**")
            recommendation = _get_recommendation(issue["metric_name"])
            st.info(recommendation)

    st.page_link(
        "pages/4_Episode_Explorer.py",
        label="逐 episode 查看详情 →",
        icon="📋",
    )


# ---------------------------------------------------------------------------
# 建议模板
# ---------------------------------------------------------------------------

