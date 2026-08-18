"""Page 6: Audit History · 审计历史趋势。

展示多次审计的结果变化趋势：
- 审计历史列表（时间倒序）
- Verdict 分布变化趋势图
- 平均 deviation score 趋势线
- EXCLUDE 数量趋势线
- 支持按数据集过滤
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

st.title("📈 审计历史")
st.caption("追踪数据质量变化趋势 · 基于多次审计结果对比分析")

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
from rda.report.audit_history import (  # noqa: E402
    load_audit_history,
    compute_trend,
)


def _load_history_for_current() -> list:
    """加载当前数据集的审计历史，或尝试获取所有已知历史。"""
    dataset_path = st.session_state.get("dataset_path")
    if dataset_path:
        return load_audit_history(dataset_path)
    return []


history = _load_history_for_current()

# ---------------------------------------------------------------------------
# 无数据提示
# ---------------------------------------------------------------------------
if not history:
    st.info(
        "暂无审计历史记录。\n\n"
        "每次在 Audit 页面完成审计后，系统会自动保存一份快照。\n"
        "完成至少一次审计后即可在此查看历史趋势。",
        icon="📭",
    )
    st.stop()

# ---------------------------------------------------------------------------
# 侧边栏：数据集过滤
# ---------------------------------------------------------------------------
# 收集所有出现过的数据集名称
dataset_names = sorted(set(s.get("dataset_name", "unknown") for s in history))

if len(dataset_names) > 1:
    with st.sidebar:
        st.divider()
        selected_ds = st.selectbox(
            "📂 按数据集过滤",
            options=["全部"] + dataset_names,
            index=0,
        )
    if selected_ds != "全部":
        history = [s for s in history if s.get("dataset_name") == selected_ds]

# 按时间倒序排列
history_sorted = sorted(history, key=lambda s: s.get("timestamp", ""), reverse=True)

# ---------------------------------------------------------------------------
# 概览指标
# ---------------------------------------------------------------------------
st.subheader("📊 历史概览")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("审计次数", len(history))
with col2:
    latest = history_sorted[0]
    st.metric("最近审计", latest.get("timestamp", "N/A")[:10])
with col3:
    st.metric("Episodes", latest.get("total_episodes", 0))
with col4:
    vc = latest.get("verdict_counts", {})
    total = latest.get("total_episodes", 1)
    pass_rate = vc.get("PASS", 0) / total * 100 if total > 0 else 0
    st.metric("最新 PASS 率", f"{pass_rate:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# 趋势图
# ---------------------------------------------------------------------------
if len(history) < 2:
    st.info("⚠️ 至少需要 2 次审计才能查看趋势变化", icon="📊")
else:
    st.subheader("📈 趋势分析")

    # --- Verdict 分布变化（堆叠面积图）---
    st.markdown("**Verdict 分布变化**")
    history_asc = sorted(history, key=lambda s: s.get("timestamp", ""))
    timestamps = [s.get("timestamp", "")[:16] for s in history_asc]
    labels = [f"#{i+1}" for i in range(len(timestamps))]

    import plotly.graph_objects as go

    pass_counts = [s.get("verdict_counts", {}).get("PASS", 0) for s in history_asc]
    review_counts = [s.get("verdict_counts", {}).get("REVIEW", 0) for s in history_asc]
    exclude_counts = [s.get("verdict_counts", {}).get("EXCLUDE", 0) for s in history_asc]

    fig_verdict = go.Figure()
    fig_verdict.add_trace(go.Scatter(
        x=labels, y=pass_counts,
        mode="lines+markers",
        name="PASS · 保留",
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.2)",
        line=dict(color="#10b981", width=2),
        marker=dict(size=6),
    ))
    fig_verdict.add_trace(go.Scatter(
        x=labels, y=review_counts,
        mode="lines+markers",
        name="REVIEW · 审核",
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.2)",
        line=dict(color="#f59e0b", width=2),
        marker=dict(size=6),
    ))
    fig_verdict.add_trace(go.Scatter(
        x=labels, y=exclude_counts,
        mode="lines+markers",
        name="EXCLUDE · 排除",
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.2)",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=6),
    ))
    fig_verdict.update_layout(
        xaxis_title="审计次序",
        yaxis_title="Episode 数量",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(fig_verdict, use_container_width=True)

    # --- Deviation Score 趋势 ---
    st.markdown("**平均 Deviation Score 趋势**")
    mean_devs = [s.get("mean_deviation_score", 0) for s in history_asc]
    median_devs = [s.get("median_deviation_score", 0) for s in history_asc]

    fig_dev = go.Figure()
    fig_dev.add_trace(go.Scatter(
        x=labels, y=mean_devs,
        mode="lines+markers",
        name="平均值",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=7),
    ))
    fig_dev.add_trace(go.Scatter(
        x=labels, y=median_devs,
        mode="lines+markers",
        name="中位数",
        line=dict(color="#a78bfa", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    fig_dev.update_layout(
        xaxis_title="审计次序",
        yaxis_title="Deviation Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        hovermode="x unified",
    )
    st.plotly_chart(fig_dev, use_container_width=True)

    # --- EXCLUDE 数量趋势 ---
    st.markdown("**EXCLUDE 数量趋势**")
    fig_exclude = go.Figure()
    fig_exclude.add_trace(go.Scatter(
        x=labels, y=exclude_counts,
        mode="lines+markers",
        name="EXCLUDE 数量",
        line=dict(color="#ef4444", width=2.5),
        marker=dict(size=7, color="#ef4444"),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.1)",
    ))
    fig_exclude.update_layout(
        xaxis_title="审计次序",
        yaxis_title="EXCLUDE Episode 数量",
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig_exclude, use_container_width=True)

    # --- Integrity Pass Rate 趋势 ---
    st.markdown("**完整性通过率趋势**")
    integrity_rates = [s.get("integrity_pass_rate", 0) * 100 for s in history_asc]
    fig_integrity = go.Figure()
    fig_integrity.add_trace(go.Scatter(
        x=labels, y=integrity_rates,
        mode="lines+markers",
        name="完整性通过率",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=7),
    ))
    fig_integrity.update_layout(
        xaxis_title="审计次序",
        yaxis_title="通过率 (%)",
        yaxis_range=[0, 105],
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig_integrity, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 审计历史列表
# ---------------------------------------------------------------------------
st.subheader("📋 审计记录")

for i, snapshot in enumerate(history_sorted):
    ts = snapshot.get("timestamp", "N/A")
    ds_name = snapshot.get("dataset_name", "unknown")
    total_eps = snapshot.get("total_episodes", 0)
    vc = snapshot.get("verdict_counts", {})
    pass_count = vc.get("PASS", 0)
    review_count = vc.get("REVIEW", 0)
    exclude_count = vc.get("EXCLUDE", 0)
    pass_pct = pass_count / total_eps * 100 if total_eps > 0 else 0
    mean_dev = snapshot.get("mean_deviation_score", 0)
    median_dev = snapshot.get("median_deviation_score", 0)
    integrity_rate = snapshot.get("integrity_pass_rate", 0) * 100
    rda_ver = snapshot.get("rda_version", "?")
    top_patterns = snapshot.get("top_patterns", [])

    with st.expander(f"#{len(history_sorted) - i}  {ts[:19]}  ·  {ds_name}  ·  {total_eps} episodes", expanded=(i == 0)):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 PASS", f"{pass_count} ({pass_pct:.1f}%)")
        with col2:
            review_pct = review_count / total_eps * 100 if total_eps > 0 else 0
            st.metric("🟡 REVIEW", f"{review_count} ({review_pct:.1f}%)")
        with col3:
            exclude_pct = exclude_count / total_eps * 100 if total_eps > 0 else 0
            st.metric("🔴 EXCLUDE", f"{exclude_count} ({exclude_pct:.1f}%)")
        with col4:
            st.metric("完整性通过率", f"{integrity_rate:.1f}%")

        st.caption(f"平均 Deviation: {mean_dev:.2f} · 中位数: {median_dev:.2f} · RDA v{rda_ver}")

        if top_patterns:
            pattern_str = " · ".join(f"{p['pattern']}({p['count']})" for p in top_patterns)
            st.caption(f"🔍 主要 Pattern: {pattern_str}")

        # Hero metrics summary
        hero = snapshot.get("hero_metrics_summary", {})
        if hero:
            hero_parts = []
            sync_interp = hero.get("sensor_sync_interpretation", "na")
            if sync_interp != "na":
                hero_parts.append(f"传感器同步: {hero.get('sensor_sync_median_p95_ms', 0):.1f}ms ({sync_interp})")
            else:
                hero_parts.append("传感器同步: N/A")
            hero_parts.append(f"动作抖动: {hero.get('action_disc_total_spikes', 0)} spikes")
            hero_parts.append(f"空间覆盖: {hero.get('state_space_median_occupancy', 0):.1%} ({hero.get('state_space_interpretation', 'low')})")
            st.caption("  |  ".join(hero_parts))
