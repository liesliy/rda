"""Page 3: Health Overview · dataset health summary (bilingual zh/en).

Shows:
- Dataset Health Score 4-dimension radar chart
- DHI + grade
- Training Readiness
- Issue stats (Critical / Warning / Info)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rda.ui_app.i18n import get_lang, t  # noqa: E402

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
    """Return a localized recommendation for a metric."""
    key = f"rec_{metric_name}"
    if metric_name in (
        "missing_dropout", "invalid_values", "schema_consistency",
        "timestamp_validity", "joint_limit", "sensor_synchronization",
        "sampling_jitter", "velocity_acceleration", "action_discontinuity",
        "idle_ratio", "distribution", "coverage",
    ):
        return t(key)
    return t("rec_generic")



st.title(t("health_title"))
st.caption(t("health_caption"))

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
if st.session_state.audit_result is None:
    st.warning(t("not_audited"), icon="🔍")
    st.page_link("pages/2_Audit.py", label=t("go_audit"), icon="🔍")
    st.stop()

result = st.session_state.audit_result
info = result.dataset_info

# Make sure the DHI report exists in the active language
if st.session_state.dataset_report is None:
    st.session_state.dataset_report = compute_dhi(result, lang=get_lang())

report = st.session_state.dataset_report
dhi = report["dhi"]
grade = report["grade"]
dims = report["dimensions"]
readiness = report["training_readiness"]
readiness_detail = report["training_readiness_detail"]

# ---------------------------------------------------------------------------
# Training Readiness banner
# ---------------------------------------------------------------------------
emoji_r, color_r = get_readiness_badge(readiness)
st.markdown(f"""
<div style="padding: 24px; border-radius: 12px; background: linear-gradient(135deg, {color_r}15 0%, {color_r}05 100%); border: 1px solid {color_r}30;">
    <div style="font-size: 14px; color: #6b7280; margin-bottom: 8px;">{t("health_readiness_label")}</div>
    <div style="font-size: 28px; font-weight: 700; color: {color_r};">{emoji_r} {readiness}</div>
    <div style="font-size: 15px; color: #4b5563; margin-top: 4px;">{readiness_detail}</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# Verdict distribution
counts = {v.value: c for v, c in result.verdict_counts.items()}
total = result.num_episodes
pass_c = counts.get("PASS", 0)
review_c = counts.get("REVIEW", 0)
exclude_c = counts.get("EXCLUDE", 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🟢 Keep", f"{pass_c}",
              t("health_verdict_keep_pct", pct=pass_c / max(total, 1) * 100))
with col2:
    st.metric("🟡 Review", f"{review_c}",
              t("health_verdict_keep_pct", pct=review_c / max(total, 1) * 100))
with col3:
    st.metric("🔴 Remove", f"{exclude_c}",
              t("health_verdict_keep_pct", pct=exclude_c / max(total, 1) * 100))

st.divider()

# ---------------------------------------------------------------------------
# DHI + radar chart
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(t("health_dhi_header"))
    emoji_g, color_g = get_grade_badge(grade)

    st.markdown(f"""
    <div style="text-align: center; padding: 30px 20px; border-radius: 16px; background: #f8fafc; border: 1px solid #e2e8f0;">
        <div style="font-size: 64px; font-weight: 800; color: {color_g}; line-height: 1;">{dhi}<span style="font-size: 24px; color: #94a3b8;"> / 100</span></div>
        <div style="font-size: 20px; margin-top: 8px; color: #1e293b;">{emoji_g} {grade}{t("health_grade_suffix")}</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 12px; font-style: italic;">
            {t("health_dhi_note")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Dimension progress bars
    st.markdown(f"**{t('health_dims_header')}**")
    dim_labels = [
        (t("health_dim_integrity"), dims["integrity"], 0.4),
        (t("health_dim_temporal"), dims["temporal"], 0.2),
        (t("health_dim_motion"), dims["motion"], 0.2),
        (t("health_dim_consistency"), dims["consistency"], 0.2),
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
    st.subheader(t("health_radar_header"))
    fig = make_radar_chart(dims, lang=get_lang())
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Reusability stats
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(t("health_usable_metric"), f"{dhi:.0f} / 100")
with col2:
    usable_pct = (pass_c + review_c) / max(total, 1) * 100
    st.metric(t("health_usable_after_review"), f"{usable_pct:.1f}%",
              t("health_usable_after_review_sub", n=pass_c + review_c, total=total),
              delta_color="off")
with col3:
    keep_pct = pass_c / max(total, 1) * 100
    st.metric(t("health_clear_keep"), f"{keep_pct:.1f}%",
              t("health_clear_keep_sub", n=pass_c),
              delta_color="off")

st.caption(t("health_usable_caption"))

st.divider()

# ---------------------------------------------------------------------------
# Top Audit Observations
# ---------------------------------------------------------------------------
st.subheader(t("health_top_issues_header"))
st.caption(
    "确定性结构问题可支持规则排除；Risk Signal 只表示统计异常，"
    "Unverifiable 表示当前输入不足以判断，三者都不替代人工确认。"
    if get_lang() == "zh" else
    "Deterministic failures can support a rule-based exclusion; Risk Signals are statistical observations, "
    "and Unverifiable means the current input is insufficient. None replaces manual confirmation."
)

issue_stats = compute_issue_stats(result, lang=get_lang())
top_issues = issue_stats["top_issues"]

if not top_issues:
    st.success(t("health_no_issues"), icon="✅")
else:
    # Count Critical / Warning / Info
    critical_count = sum(1 for i in top_issues if i["severity"] == "critical")
    warning_count = sum(1 for i in top_issues if i["severity"] == "warning")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t("health_metric_critical"), issue_stats["critical"])
    with col2:
        st.metric(t("health_metric_warning"), issue_stats["risk_signal"])
    with col3:
        st.metric(t("health_metric_types"), len(top_issues))

    if issue_stats.get("unverifiable"):
        st.info(
            (
                f"当前有 {issue_stats['unverifiable']} 个指标无法验证；这不等于通过。"
                if get_lang() == "zh" else
                f"{issue_stats['unverifiable']} metric evaluations were unavailable; this does not mean PASS."
            ),
            icon="ℹ️",
        )

    st.write("")

    # Issue list
    for idx, issue in enumerate(top_issues[:8], 1):
        sev_color = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🔵",
        }.get(issue["severity"], "⚪")

        sev_label = {
            "critical": t("health_sev_critical"),
            "warning": t("health_sev_warning"),
            "info": t("health_sev_info"),
        }.get(issue["severity"], t("health_sev_unknown"))

        with st.expander(
            t("health_issue_expander",
              sev=sev_color, idx=idx, desc=issue['description'],
              code=issue['code'], count=issue['count'],
              pct=issue['count'] / max(total, 1) * 100),
            expanded=idx <= 3,
        ):
            cols = st.columns(4)
            with cols[0]:
                st.caption(t("health_lbl_severity"))
                st.write(f"{sev_color} {sev_label} · {issue.get('evidence_level', 'RISK_SIGNAL')}")
            with cols[1]:
                st.caption(t("health_lbl_scope"))
                st.write(f"{issue['count']} / {total} episodes")
            with cols[2]:
                st.caption(t("health_lbl_code"))
                st.code(issue["code"])
            with cols[3]:
                if issue["pattern_type"]:
                    st.caption("Pattern Type")
                    st.write(f"`{issue['pattern_type']}`")

            st.caption(t("health_recommendation"))
            recommendation = _get_recommendation(issue["metric_name"])
            st.info(recommendation)

    st.page_link(
        "pages/4_Episode_Explorer.py",
        label=t("health_view_episodes"),
        icon="📋",
    )


# ---------------------------------------------------------------------------
# Recommendation templates
# ---------------------------------------------------------------------------
