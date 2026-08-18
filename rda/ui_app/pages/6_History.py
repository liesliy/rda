"""Page 6: Audit History · audit trend tracking (bilingual zh/en).

Shows trends across multiple audits:
- Audit history list (newest first)
- Verdict distribution trend chart
- Mean deviation score trend line
- EXCLUDE count trend line
- Filter by dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rda.ui_app.i18n import t  # noqa: E402

st.title(t("history_title"))
st.caption(t("history_caption"))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
from rda.report.audit_history import (  # noqa: E402
    load_audit_history,
    compute_trend,
)


def _load_history_for_current() -> list:
    """Load audit history for the current dataset, or all known history."""
    dataset_path = st.session_state.get("dataset_path")
    if dataset_path:
        return load_audit_history(dataset_path)
    return []


history = _load_history_for_current()

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not history:
    st.info(t("history_empty"), icon="📭")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: dataset filter
# ---------------------------------------------------------------------------
dataset_names = sorted(set(s.get("dataset_name", "unknown") for s in history))

if len(dataset_names) > 1:
    with st.sidebar:
        st.divider()
        _all_opt = t("history_filter_all")
        selected_ds = st.selectbox(
            t("history_filter_label"),
            options=[_all_opt] + dataset_names,
            index=0,
        )
    if selected_ds != _all_opt:
        history = [s for s in history if s.get("dataset_name") == selected_ds]

# Newest first
history_sorted = sorted(history, key=lambda s: s.get("timestamp", ""), reverse=True)

# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------
st.subheader(t("history_overview"))
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(t("history_metric_runs"), len(history))
with col2:
    latest = history_sorted[0]
    st.metric(t("history_metric_latest"), latest.get("timestamp", "N/A")[:10])
with col3:
    st.metric("Episodes", latest.get("total_episodes", 0))
with col4:
    vc = latest.get("verdict_counts", {})
    total = latest.get("total_episodes", 1)
    pass_rate = vc.get("PASS", 0) / total * 100 if total > 0 else 0
    st.metric(t("history_metric_pass"), f"{pass_rate:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Trend charts
# ---------------------------------------------------------------------------
if len(history) < 2:
    st.info(t("history_need_two"), icon="📊")
else:
    st.subheader(t("history_trend_header"))

    # --- Verdict distribution (stacked area chart) ---
    st.markdown(t("history_verdict_chart"))
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
        name=t("history_verdict_keep"),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.2)",
        line=dict(color="#10b981", width=2),
        marker=dict(size=6),
    ))
    fig_verdict.add_trace(go.Scatter(
        x=labels, y=review_counts,
        mode="lines+markers",
        name=t("history_verdict_review"),
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.2)",
        line=dict(color="#f59e0b", width=2),
        marker=dict(size=6),
    ))
    fig_verdict.add_trace(go.Scatter(
        x=labels, y=exclude_counts,
        mode="lines+markers",
        name=t("history_verdict_exclude"),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.2)",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=6),
    ))
    fig_verdict.update_layout(
        xaxis_title=t("history_axis_run"),
        yaxis_title=t("history_axis_eps"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(fig_verdict, use_container_width=True)

    # --- Deviation Score trend ---
    st.markdown(t("history_dev_chart"))
    mean_devs = [s.get("mean_deviation_score", 0) for s in history_asc]
    median_devs = [s.get("median_deviation_score", 0) for s in history_asc]

    fig_dev = go.Figure()
    fig_dev.add_trace(go.Scatter(
        x=labels, y=mean_devs,
        mode="lines+markers",
        name=t("history_dev_mean"),
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=7),
    ))
    fig_dev.add_trace(go.Scatter(
        x=labels, y=median_devs,
        mode="lines+markers",
        name=t("history_dev_median"),
        line=dict(color="#a78bfa", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    fig_dev.update_layout(
        xaxis_title=t("history_axis_run"),
        yaxis_title="Deviation Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        hovermode="x unified",
    )
    st.plotly_chart(fig_dev, use_container_width=True)

    # --- EXCLUDE count trend ---
    st.markdown(t("history_excl_chart"))
    fig_exclude = go.Figure()
    fig_exclude.add_trace(go.Scatter(
        x=labels, y=exclude_counts,
        mode="lines+markers",
        name=t("history_excl_series"),
        line=dict(color="#ef4444", width=2.5),
        marker=dict(size=7, color="#ef4444"),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.1)",
    ))
    fig_exclude.update_layout(
        xaxis_title=t("history_axis_run"),
        yaxis_title=t("history_excl_axis"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig_exclude, use_container_width=True)

    # --- Integrity pass rate trend ---
    st.markdown(t("history_integrity_chart"))
    integrity_rates = [s.get("integrity_pass_rate", 0) * 100 for s in history_asc]
    fig_integrity = go.Figure()
    fig_integrity.add_trace(go.Scatter(
        x=labels, y=integrity_rates,
        mode="lines+markers",
        name=t("history_integrity_series"),
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=7),
    ))
    fig_integrity.update_layout(
        xaxis_title=t("history_axis_run"),
        yaxis_title=t("history_rate_axis"),
        yaxis_range=[0, 105],
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig_integrity, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Audit history list
# ---------------------------------------------------------------------------
st.subheader(t("history_list_header"))

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
            st.metric(t("history_integrity_pass_metric"), f"{integrity_rate:.1f}%")

        st.caption(t("history_mean_dev", mean=mean_dev, median=median_dev, ver=rda_ver))

        if top_patterns:
            pattern_str = " · ".join(f"{p['pattern']}({p['count']})" for p in top_patterns)
            st.caption(t("history_top_patterns", patterns=pattern_str))

        # Hero metrics summary
        hero = snapshot.get("hero_metrics_summary", {})
        if hero:
            hero_parts = []
            sync_interp = hero.get("sensor_sync_interpretation", "na")
            if sync_interp != "na":
                hero_parts.append(t("history_sensor_sync", v=hero.get('sensor_sync_median_p95_ms', 0), interp=sync_interp))
            else:
                hero_parts.append(t("history_sensor_sync_na"))
            hero_parts.append(t("history_action_jitter", n=hero.get('action_disc_total_spikes', 0)))
            hero_parts.append(t("history_space_coverage", v=hero.get('state_space_median_occupancy', 0), interp=hero.get('state_space_interpretation', 'low')))
            st.caption("  |  ".join(hero_parts))
