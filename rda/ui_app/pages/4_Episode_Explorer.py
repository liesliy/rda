"""Page 4: Episode Explorer · 逐 episode 查看。

功能：
- Episode 列表（可排序、可筛选）
- Review Queue 优先展示
- 点击查看详情（metrics、issues、diagnosis）
- Your Decision 列（Keep / Remove / Uncertain）
- Notes 输入框，决策持久化到 session_state
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from components.common import (  # noqa: E402
    build_episodes_dataframe,
    get_verdict_badge,
)

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _render_review_queue(table_df: pd.DataFrame) -> None:
    """渲染 Review Queue — 带有 Your Decision 列和快速决策按钮。"""
    if table_df.empty:
        st.info("没有需要审核的 episode")
        return

    for _, row in table_df.iterrows():
        ep_id = row["episode_id"]
        verdict = row["behavior_verdict"]
        sev_score = row["_view_score"]
        pattern = row["pattern_type"]
        top_issues = row["top_issues"]
        num_frames = row["num_frames"]

        # 取用户决策
        user_v = st.session_state.user_verdicts.get(ep_id, {})
        user_decision = user_v.get("decision")
        user_notes = user_v.get("notes", "")

        # 决策徽章
        if user_decision == "KEEP":
            decision_badge = "🟢 **KEEP**"
        elif user_decision == "REMOVE":
            decision_badge = "🔴 **REMOVE**"
        elif user_decision == "UNCERTAIN":
            decision_badge = "🟡 **UNCERTAIN**"
        else:
            decision_badge = "⚪ 未决策"

        # AI verdict 展示名称
        ai_verdict_display = {"PASS": "KEEP", "REVIEW": "REVIEW", "EXCLUDE": "REMOVE"}.get(verdict, verdict)

        with st.expander(
            f"Episode #{ep_id}  ·  AI: {ai_verdict_display}  ·  "
            f"Issue Severity: {sev_score:.1f}  ·  "
            f"Your Decision: {decision_badge}",
            expanded=False,
        ):
            col_info, col_actions = st.columns([2, 1])

            with col_info:
                st.markdown(f"**帧数**: {num_frames}  ·  **Pattern Type**: `{pattern}`")
                st.markdown(f"**主要问题**: {top_issues}")

                # Notes 输入
                notes_key = f"queue_notes_{ep_id}"
                new_notes = st.text_area(
                    "审核备注",
                    value=user_notes,
                    height=60,
                    placeholder="记录审核原因...",
                    key=notes_key,
                    label_visibility="collapsed",
                )
                if new_notes != user_notes:
                    existing = st.session_state.user_verdicts.get(ep_id, {"decision": None, "notes": ""})
                    st.session_state.user_verdicts[ep_id] = {
                        "decision": existing.get("decision"),
                        "notes": new_notes,
                    }

            with col_actions:
                st.markdown("**Your Decision**")
                if st.button("✅ KEEP", key=f"q_keep_{ep_id}", use_container_width=True,
                             type="primary" if user_decision == "KEEP" else "secondary"):
                    st.session_state.user_verdicts[ep_id] = {
                        "decision": "KEEP",
                        "notes": new_notes,
                    }
                    st.rerun()
                if st.button("🗑️ REMOVE", key=f"q_remove_{ep_id}", use_container_width=True,
                             type="primary" if user_decision == "REMOVE" else "secondary"):
                    st.session_state.user_verdicts[ep_id] = {
                        "decision": "REMOVE",
                        "notes": new_notes,
                    }
                    st.rerun()
                if st.button("❓ UNCERTAIN", key=f"q_uncertain_{ep_id}", use_container_width=True,
                             type="primary" if user_decision == "UNCERTAIN" else "secondary"):
                    st.session_state.user_verdicts[ep_id] = {
                        "decision": "UNCERTAIN",
                        "notes": new_notes,
                    }
                    st.rerun()
                if st.button("📖 查看详情", key=f"q_detail_{ep_id}", use_container_width=True):
                    st.session_state.selected_episode = ep_id
                    st.rerun()

def _render_episode_table(table_df: pd.DataFrame, prefix: str) -> None:
    """渲染 episode 列表表格（全部 Episodes 标签页使用）。"""
    if table_df.empty:
        st.info("没有符合条件的 episode")
        return

    display_df = table_df[[
        "episode_id", "behavior_verdict", "integrity_check",
        "_view_score", "pattern_type", "num_frames", "top_issues",
    ]].copy()

    # 美化列名
    display_df.columns = [
        "Episode #", "Verdict", "完整性", "Behavior Score", "Pattern Type",
        "帧数", "主要问题",
    ]

    # 配置数据表格
    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        column_config={
            "Episode #": st.column_config.NumberColumn(format="%d"),
            "Verdict": st.column_config.TextColumn(),
            "完整性": st.column_config.TextColumn(),
            "Behavior Score": st.column_config.NumberColumn(format="%.2f"),
            "帧数": st.column_config.NumberColumn(format="%d"),
        },
        hide_index=True,
        key=f"episode_table_{prefix}",
    )

    # Episode 选择器
    col1, col2 = st.columns([2, 1])
    with col1:
        ep_options = [f"Episode #{row['episode_id']} · {row['behavior_verdict']}"
                      for _, row in table_df.iterrows()]
        ep_indices = table_df["episode_id"].tolist()
        if ep_indices:
            selected = st.selectbox(
                "选择 episode 查看详情",
                options=range(len(ep_indices)),
                format_func=lambda i: ep_options[i],
                key=f"detail_select_{prefix}",
                index=0,
            )
            st.session_state.selected_episode = ep_indices[selected]
    with col2:
        st.write("")
        st.write("")
        if st.button("📖 查看详情", type="primary", key=f"view_detail_{prefix}"):
            st.session_state.selected_episode = ep_indices[selected]
            st.rerun()

def _render_episode_detail(ep_id: int) -> None:
    """渲染单个 episode 的详情面板。"""
    st.divider()
    st.subheader(f"🔍 Episode #{ep_id} 详情")

    ep_result = result.episodes.get(ep_id)
    if ep_result is None:
        st.error(f"未找到 Episode #{ep_id}")
        return

    # 找到对应的行
    ep_row = df[df["episode_id"] == ep_id].iloc[0] if not df[df["episode_id"] == ep_id].empty else None

    # 双组件：Integrity + Behavior
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 数据完整性 (Layer 1A)")
        integrity_pass = ep_row["integrity_check"] == "PASS" if ep_row is not None else True
        emoji = "🟢" if integrity_pass else "🔴"
        status_text = "PASS · 无硬检查问题" if integrity_pass else "FAIL · 存在完整性问题"
        st.markdown(f"""
        <div style="padding: 16px; border-radius: 10px; background: {'#dcfce7' if integrity_pass else '#fee2e2'}; color: {'#166534' if integrity_pass else '#991b1b'};">
            <div style="font-size: 24px; font-weight: 700;">{emoji} {status_text}</div>
        </div>
        """, unsafe_allow_html=True)

        if ep_row is not None and ep_row["integrity_issues"] != "—":
            st.caption("存在的问题：")
            for issue in str(ep_row["integrity_issues"]).split(", "):
                st.warning(f"🔴 {issue}")

    with col2:
        st.markdown("### 行为质量 (Layer 1B)")
        verdict = ep_result.verdict.value
        emoji_v, color_v = get_verdict_badge(verdict)
        verdict_label = {"PASS": "KEEP · 建议保留",
                         "REVIEW": "REVIEW · 建议人工审核",
                         "EXCLUDE": "REMOVE · 建议排除"}.get(verdict, verdict)
        sev_score = ep_row['_view_score'] if ep_row is not None else '—'

        # 多维度评分展示
        portable_s = ep_row.get('portable_score') if ep_row is not None else None
        platform_s = ep_row.get('platform_score') if ep_row is not None else None
        combined_s = ep_row.get('combined_score') if ep_row is not None else None
        has_plat = bool(ep_row.get('has_platform_metrics')) if ep_row is not None else False

        score_lines = [f"Issue Severity Score: {sev_score}"]
        if portable_s is not None:
            score_lines.append(f"Portable Score: {portable_s:.2f}")
        if has_plat and platform_s is not None:
            score_lines.append(f"Platform Score: {platform_s:.2f}")
            score_lines.append(f"Combined Score: {combined_s:.2f}")

        st.markdown(f"""
        <div style="padding: 16px; border-radius: 10px; background: {color_v}20; color: {color_v};">
            <div style="font-size: 24px; font-weight: 700;">{emoji_v} {verdict_label.split(' · ')[0]}</div>
            <div style="font-size: 14px; margin-top: 4px;">{verdict_label.split(' · ')[1] if ' · ' in verdict_label else ''}</div>
            <div style="font-size: 12px; margin-top: 6px; opacity: 0.85; line-height: 1.6;">
                {'<br/>'.join(score_lines)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Pattern Type
        if ep_row is not None and ep_row["pattern_type"] != "—":
            st.caption("Pattern Type")
            st.info(f"`{ep_row['pattern_type']}`")

    st.divider()

    # Metrics 详情
    st.markdown("### 📊 各项指标详情")

    metric_rows = []
    for m_name, m in sorted(ep_result.metrics.items()):
        if m.availability.value == "available":
            status = "✅ PASS" if not m.has_finding else "⚠️ FLAGGED"
            measurement_str = _format_measurement(m.measurement)
            metric_rows.append({
                "指标": m_name,
                "状态": status,
                "测量值": measurement_str,
                "说明": m.message or m.assessment.get("reason", ""),
            })
        elif m.availability.value == "not_available":
            metric_rows.append({
                "指标": m_name,
                "状态": "N/A",
                "测量值": "—",
                "说明": m.assessment.get("reason", "数据不可用"),
            })
        else:
            metric_rows.append({
                "指标": m_name,
                "状态": "ERROR",
                "测量值": "—",
                "说明": m.assessment.get("reason", "计算出错"),
            })

    metric_df = pd.DataFrame(metric_rows)
    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "指标": st.column_config.TextColumn(width="small"),
            "状态": st.column_config.TextColumn(width="small"),
            "测量值": st.column_config.TextColumn(width="medium"),
            "说明": st.column_config.TextColumn(width="large"),
        },
    )

    # 诊断
    st.markdown("### 💡 诊断与建议")

    if ep_result.verdict.value == "PASS":
        st.success("该 episode 各项指标正常，无明显质量问题。")
    else:
        diagnosis = _generate_diagnosis(ep_result, ep_row)
        with st.container():
            st.markdown(f"**WHAT · 问题描述**")
            st.info(diagnosis["what"])
            st.markdown(f"**WHY · 原因分析**")
            st.warning(diagnosis["why"])
            st.markdown(f"**NEXT · 处理建议**")
            st.success(diagnosis["next"])

    st.divider()

    # ──────────────────────────────────────────────────────────────────────
    # 轨迹可视化 (Trajectory Visualization)
    # ──────────────────────────────────────────────────────────────────────
    _render_trajectory_visualization(ep_id, ep_result, ep_row)

    st.divider()

    # 用户决策（Review Queue 核心功能）
    st.markdown("### ✍️ 人工审核决策 · Your Decision")

    current = st.session_state.user_verdicts.get(ep_id, {"decision": None, "notes": ""})
    current_decision = current.get("decision")
    current_notes = current.get("notes", "")

    # 决策按钮
    col_k, col_r, col_u = st.columns(3)
    with col_k:
        keep_pressed = st.button(
            "✅ KEEP · 保留",
            type="primary" if current_decision == "KEEP" else "secondary",
            use_container_width=True,
            key=f"btn_keep_{ep_id}",
            help="确认该 episode 可用于训练",
        )
    with col_r:
        remove_pressed = st.button(
            "🗑️ REMOVE · 排除",
            type="primary" if current_decision == "REMOVE" else "secondary",
            use_container_width=True,
            key=f"btn_remove_{ep_id}",
            help="确认该 episode 应从训练集中排除",
        )
    with col_u:
        uncertain_pressed = st.button(
            "❓ UNCERTAIN · 存疑",
            type="primary" if current_decision == "UNCERTAIN" else "secondary",
            use_container_width=True,
            key=f"btn_uncertain_{ep_id}",
            help="暂时标记为存疑，待进一步确认",
        )

    if keep_pressed:
        st.session_state.user_verdicts[ep_id] = {
            "decision": "KEEP",
            "notes": current_notes,
        }
        st.success(f"Episode #{ep_id} 已标记为 KEEP（保留）")
        st.rerun()
    if remove_pressed:
        st.session_state.user_verdicts[ep_id] = {
            "decision": "REMOVE",
            "notes": current_notes,
        }
        st.error(f"Episode #{ep_id} 已标记为 REMOVE（排除）")
        st.rerun()
    if uncertain_pressed:
        st.session_state.user_verdicts[ep_id] = {
            "decision": "UNCERTAIN",
            "notes": current_notes,
        }
        st.warning(f"Episode #{ep_id} 已标记为 UNCERTAIN（存疑）")
        st.rerun()

    # 当前决策状态展示
    if current_decision:
        badge = {"KEEP": ("🟢", "#10b981", "保留"),
                 "REMOVE": ("🔴", "#ef4444", "排除"),
                 "UNCERTAIN": ("🟡", "#f59e0b", "存疑")}.get(current_decision, ("⚪", "#9ca3af", ""))
        st.caption(f"当前决策：<span style='color:{badge[1]}; font-weight:600;'>{badge[0]} {current_decision} · {badge[2]}</span>",
                   unsafe_allow_html=True)
    else:
        st.caption("当前状态：未决策")

    # Notes 输入框
    notes_val = st.text_area(
        "审核备注（可选）",
        value=current_notes,
        height=80,
        placeholder="记录审核原因、上下文或后续动作...",
        key=f"notes_{ep_id}",
    )

    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if st.button("💾 保存备注", key=f"save_notes_{ep_id}", use_container_width=True):
            existing = st.session_state.user_verdicts.get(ep_id, {"decision": None, "notes": ""})
            st.session_state.user_verdicts[ep_id] = {
                "decision": existing.get("decision"),
                "notes": notes_val,
            }
            st.success("备注已保存")
            st.rerun()
    with col_clear:
        if st.button("🗑️ 清除决策", key=f"clear_verdict_{ep_id}", use_container_width=True):
            if ep_id in st.session_state.user_verdicts:
                del st.session_state.user_verdicts[ep_id]
            st.info("决策已清除")
            st.rerun()

    # 关闭按钮
    if st.button("关闭详情", key="close_detail"):
        st.session_state.selected_episode = None
        st.rerun()

@st.cache_data(show_spinner="正在加载 episode 原始数据...")
def _load_episode_data(dataset_path: str, episode_index: int):
    """加载单个 episode 的原始数据（带缓存）。"""
    from rda.io.lerobot_loader import iter_episodes

    for ep in iter_episodes(dataset_path):
        if ep.episode_index == episode_index:
            return ep
    return None

def _render_trajectory_visualization(ep_id, ep_result, ep_row):
    """渲染轨迹可视化区域（State / Action 时序图 + 异常高亮）。"""
    st.markdown("### 📈 轨迹可视化")

    dataset_path = st.session_state.get("dataset_path")
    if not dataset_path:
        st.info("💡 数据集路径未设置，无法加载原始轨迹数据")
        return

    try:
        episode_data = _load_episode_data(str(dataset_path), int(ep_id))
    except Exception as e:
        st.warning(f"⚠️ 加载轨迹数据失败：{e}")
        return

    if episode_data is None:
        st.warning(f"⚠️ 未找到 Episode #{ep_id} 的原始数据")
        return

    has_state = bool(episode_data.observation)
    has_action = bool(episode_data.action)

    if not has_state and not has_action:
        st.info("该 episode 没有 state 或 action 数据，跳过可视化")
        return

    # 选择显示模式
    view_options = []
    if has_state:
        view_options.append("State")
    if has_action:
        view_options.append("Action")
    if has_state and has_action:
        view_options.append("Both")

    selected_view = st.selectbox(
        "显示模式",
        options=view_options,
        index=len(view_options) - 1,  # 默认选 Both（如果有）
        key=f"traj_view_{ep_id}",
    )

    # 时间轴
    timestamps = episode_data.timestamps
    joint_limits = episode_data.meta.get("joint_limits") if episode_data.meta else None

    # 收集异常区间
    anomaly_regions = _collect_anomaly_regions(ep_result, episode_data.num_frames)

    show_state = selected_view in ("State", "Both") and has_state
    show_action = selected_view in ("Action", "Both") and has_action

    if show_state:
        _plot_timeseries(
            data_dict=episode_data.observation,
            timestamps=timestamps,
            title="Joint States 时序图",
            ep_id=ep_id,
            suffix="state",
            joint_limits=joint_limits,
            anomaly_regions=anomaly_regions,
        )

    if show_action:
        _plot_timeseries(
            data_dict=episode_data.action,
            timestamps=timestamps,
            title="Actions 时序图",
            ep_id=ep_id,
            suffix="action",
            joint_limits=joint_limits,
            anomaly_regions=anomaly_regions,
        )

def _collect_anomaly_regions(ep_result, num_frames: int):
    """从审计结果中提取异常区间，用于在图上高亮。

    Returns:
        List of (start_frame, end_frame, label) tuples.
    """
    regions = []

    for m_name, m in ep_result.metrics.items():
        if m.passed:
            continue

        # action_discontinuity — 标记动作跳变较大的帧区间
        if "discontinuity" in m_name and m.measurement:
            spike_frames = m.measurement.get("spike_frames", [])
            window = max(1, num_frames // 100)  # ±1% 窗口
            for frame_idx in spike_frames:
                start = max(0, int(frame_idx) - window)
                end = min(num_frames - 1, int(frame_idx) + window)
                regions.append((start, end, m_name))

        # spike 相关指标
        if "spike" in m_name and m.measurement:
            spike_indices = m.measurement.get("spike_indices", [])
            window = max(1, num_frames // 100)
            for frame_idx in spike_indices:
                start = max(0, int(frame_idx) - window)
                end = min(num_frames - 1, int(frame_idx) + window)
                regions.append((start, end, m_name))

    return regions

def _plot_timeseries(
    data_dict,
    timestamps,
    title,
    ep_id,
    suffix,
    joint_limits=None,
    anomaly_regions=None,
):
    """绘制时序图（State 或 Action），支持关节限制线和异常高亮。"""
    fig = go.Figure()

    # 绘制每个维度的曲线
    for key, arr in data_dict.items():
        if arr.ndim == 1:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=arr,
                mode="lines",
                name=key,
                line=dict(width=1.5),
            ))
        elif arr.ndim == 2:
            for dim_idx in range(arr.shape[1]):
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=arr[:, dim_idx],
                    mode="lines",
                    name=f"{key}[{dim_idx}]",
                    line=dict(width=1.5),
                ))

    # 关节限制虚线
    if joint_limits is not None:
        if isinstance(joint_limits, (list, np.ndarray)) and len(joint_limits) > 0:
            jl = np.array(joint_limits)
            if jl.ndim == 2 and jl.shape[1] >= 2:
                # joint_limits: shape (n_joints, 2) — [low, high] per joint
                for j_idx in range(jl.shape[0]):
                    for bound_idx, bound_val in enumerate(jl[j_idx]):
                        fig.add_hline(
                            y=float(bound_val),
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.4,
                        )

    # 异常区域红色背景高亮
    if anomaly_regions:
        for start, end, label in anomaly_regions:
            fig.add_vrect(
                x0=timestamps[start] if start < len(timestamps) else start,
                x1=timestamps[end] if end < len(timestamps) else end,
                fillcolor="red",
                opacity=0.12,
                line_width=0,
                annotation_text=label,
                annotation_position="top",
            )

    fig.update_layout(
        title=f"{title}  ·  Episode #{ep_id}",
        xaxis_title="时间 (s)",
        yaxis_title="值",
        legend_title="维度",
        hovermode="x unified",
        height=350,
        margin=dict(l=40, r=20, t=40, b=30),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"traj_{suffix}_{ep_id}")

def _format_measurement(measurement: dict) -> str:
    """将 measurement dict 格式化为可读字符串。"""
    if not measurement:
        return "—"
    parts = []
    for k, v in measurement.items():
        if isinstance(v, float):
            parts.append(f"{k}: {v:.4f}")
        elif isinstance(v, (int, str, bool)):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: {type(v).__name__}")
    return " · ".join(parts[:5]) if parts else "—"

def _generate_diagnosis(ep_result, ep_row) -> dict:
    """生成三层诊断（WHAT / WHY / NEXT）。"""
    pattern = ep_row["pattern_type"] if ep_row is not None else None
    sev_score = ep_row["_view_score"] if ep_row is not None else 0

    # 收集失败的 metrics
    failed_metrics = []
    for m_name, m in ep_result.metrics.items():
        if m.availability.value == "available" and not m.passed:
            failed_metrics.append(m_name)

    what_parts = []
    why_parts = []
    next_parts = []

    # Integrity 问题
    critical_failed = [m for m in failed_metrics if m in {
        "missing_dropout", "invalid_values", "schema_consistency",
        "timestamp_validity", "joint_limit",
    }]
    if critical_failed:
        what_parts.append(f"完整性检查未通过（{', '.join(critical_failed)}）")
        why_parts.append("数据采集或存储过程中出现确定性错误")
        next_parts.append("检查数据采集管线，修复后重新采集")

    # Behavior 问题
    if "action_discontinuity" in failed_metrics:
        what_parts.append("动作不连续性异常偏高")
        why_parts.append("可能是控制器抖动或传感器噪声")
        next_parts.append("检查控制器参数或动作平滑处理")

    if "idle_ratio" in failed_metrics:
        what_parts.append("有效运动比例偏低")
        why_parts.append("机器人可能长时间处于停顿或等待状态")
        next_parts.append("检查是否为任务失败，考虑重采")

    if "velocity_acceleration" in failed_metrics:
        what_parts.append("速度/加速度超出正常范围")
        why_parts.append("动作过于激进或轨迹异常")
        next_parts.append("检查示教数据或控制器增益")

    if "distribution" in failed_metrics:
        what_parts.append("轨迹分布偏离参考分布")
        why_parts.append("可能是任务执行方式不同或异常行为")
        next_parts.append("人工确认是否为有效但罕见的行为模式")

    if not what_parts:
        what_parts.append(f"Issue Severity Score {sev_score:.2f}，略高于参考分布")
        why_parts.append("多项指标轻微异常的累积效应")
        next_parts.append("建议人工确认是否影响训练质量")

    return {
        "what": "；".join(what_parts),
        "why": "；".join(why_parts) if why_parts else "具体原因需结合数据上下文分析",
        "next": "；".join(next_parts) if next_parts else "人工审核后决定保留或排除",
    }



st.title("📋 Episode Explorer · 逐集查看")
st.caption("浏览每个 episode 的审计结果、问题类型和诊断详情，并记录人工审核决策")

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if st.session_state.audit_result is None:
    st.warning("⚠️ 请先运行审计", icon="🔍")
    st.page_link("pages/2_Audit.py", label="前往 Audit 页面 →", icon="🔍")
    st.stop()

result = st.session_state.audit_result

# 确保 episodes_df 存在
if st.session_state.episodes_df is None:
    st.session_state.episodes_df = build_episodes_dataframe(result)

df = st.session_state.episodes_df

# 确保 user_verdicts 存在（防御性，app.py 中已初始化）
if "user_verdicts" not in st.session_state:
    st.session_state.user_verdicts = {}

# ---------------------------------------------------------------------------
# 评分视图切换（Portable / Platform-specific / Combined）
# ---------------------------------------------------------------------------
_has_platform = bool(df["has_platform_metrics"].any()) if not df.empty else False

_score_view_options = ["Portable Core (跨平台)", "Combined (深度分析)"]
_score_view_keys = ["portable_score", "combined_score"]
if _has_platform:
    _score_view_options.insert(1, "Platform-specific (同平台)")
    _score_view_keys.insert(1, "platform_score")

score_view_label = st.radio(
    "📊 行为评分视图",
    options=_score_view_options,
    index=0,  # 默认 Portable Core
    horizontal=True,
    help=(
        "Portable Core：仅使用跨平台通用指标（duration / spike / effective_motion），"
        "适用于跨平台比较。\n"
        "Platform-specific：仅使用平台特有指标（velocity / path_length 等），"
        "仅在同一平台下有意义。\n"
        "Combined：所有指标综合，用于同平台深度分析。"
    ),
)
_score_view_key = _score_view_keys[_score_view_options.index(score_view_label)]

# 动态计算当前视图下的 deviation_score（用于排序和展示）
if not df.empty and df[_score_view_key].notna().any():
    df = df.copy()
    df["_view_score"] = df[_score_view_key].fillna(df["deviation_score"])
else:
    df = df.copy()
    df["_view_score"] = df["deviation_score"]

# ---------------------------------------------------------------------------
# 统计概览
# ---------------------------------------------------------------------------
total = len(df)
keep_count = len(df[df["behavior_verdict"] == "PASS"])
review_count = len(df[df["behavior_verdict"] == "REVIEW"])
remove_count = len(df[df["behavior_verdict"] == "EXCLUDE"])

# 已决策统计
verdicts = st.session_state.user_verdicts
decided_count = sum(1 for v in verdicts.values() if v.get("decision"))
user_keep = sum(1 for v in verdicts.values() if v.get("decision") == "KEEP")
user_remove = sum(1 for v in verdicts.values() if v.get("decision") == "REMOVE")
user_uncertain = sum(1 for v in verdicts.values() if v.get("decision") == "UNCERTAIN")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总 Episodes", total)
with col2:
    st.metric("🟢 KEEP (AI)", keep_count)
with col3:
    st.metric("🟡 REVIEW (AI)", review_count)
with col4:
    st.metric("🔴 REMOVE (AI)", remove_count)

# 人工审核进度
st.caption(
    f"人工审核进度：**{decided_count} / {review_count + remove_count}** "
    f"（KEEP: {user_keep} · REMOVE: {user_remove} · UNCERTAIN: {user_uncertain}）"
)

# ---------------------------------------------------------------------------
# Review Queue 标签
# ---------------------------------------------------------------------------
tab_queue, tab_all = st.tabs(["🚨 Review Queue（优先处理）", "📋 全部 Episodes"])

with tab_queue:
    queue_df = df[df["behavior_verdict"].isin(["EXCLUDE", "REVIEW"])].copy()
    if queue_df.empty:
        st.success("🎉 没有需要审核的 episode，数据集质量良好")
    else:
        st.info(
            f"共 **{len(queue_df)}** 个 episode 需要关注 "
            f"（{len(queue_df[queue_df['behavior_verdict'] == 'EXCLUDE'])} 个 REMOVE（AI建议），"
            f"{len(queue_df[queue_df['behavior_verdict'] == 'REVIEW'])} 个 REVIEW）",
            icon="⚠️",
        )
        _render_review_queue(queue_df)

with tab_all:
    # 筛选器
    col1, col2, col3 = st.columns(3)
    with col1:
        verdict_filter = st.multiselect(
            "Verdict 筛选",
            options=["PASS", "REVIEW", "EXCLUDE"],
            default=["PASS", "REVIEW", "EXCLUDE"],
            format_func=lambda x: {"PASS": "KEEP", "REVIEW": "REVIEW", "EXCLUDE": "REMOVE"}.get(x, x),
        )
    with col2:
        integrity_filter = st.multiselect(
            "完整性筛选",
            options=["PASS", "FAIL"],
            default=["PASS", "FAIL"],
        )
    with col3:
        pattern_filter = st.multiselect(
            "Pattern Type 筛选",
            options=[p for p in df["pattern_type"].unique() if p != "—"],
            default=[],
        )

    filtered = df[df["behavior_verdict"].isin(verdict_filter)]
    filtered = filtered[filtered["integrity_check"].isin(integrity_filter)]
    if pattern_filter:
        filtered = filtered[filtered["pattern_type"].isin(pattern_filter)]

    st.caption(f"显示 {len(filtered)} / {total} 个 episodes")
    _render_episode_table(filtered, prefix="all")


# ---------------------------------------------------------------------------
# Episode 详情模态（使用展开式）
# ---------------------------------------------------------------------------
if "selected_episode" in st.session_state and st.session_state.selected_episode is not None:
    ep_id = st.session_state.selected_episode
    _render_episode_detail(ep_id)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# 轨迹可视化 — 数据加载
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 轨迹可视化 — 渲染
# ---------------------------------------------------------------------------










