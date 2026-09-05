"""Page 4: Episode Explorer · per-episode browsing (bilingual zh/en).

Features:
- Episode table (sortable, filterable)
- Review Queue shown first
- Click for details (metrics, issues, diagnosis)
- Your Decision column (Keep / Remove / Uncertain)
- Notes input, decisions persisted to session_state
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

from rda.ui_app.i18n import get_lang, t  # noqa: E402

from components.common import (  # noqa: E402
    build_episodes_dataframe,
    get_verdict_badge,
)

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _render_review_queue(table_df: pd.DataFrame) -> None:
    """Render the Review Queue with Your Decision column and quick-action buttons."""
    if table_df.empty:
        st.info(t("ep_queue_no_items"))
        return

    for _, row in table_df.iterrows():
        ep_id = row["episode_id"]
        verdict = row["behavior_verdict"]
        sev_score = row["_view_score"]
        pattern = row["pattern_type"]
        top_issues = row["top_issues"]
        num_frames = row["num_frames"]

        # User decision
        user_v = st.session_state.user_verdicts.get(ep_id, {})
        user_decision = user_v.get("decision")
        user_notes = user_v.get("notes", "")

        # Decision badge
        if user_decision == "KEEP":
            decision_badge = "🟢 **KEEP**"
        elif user_decision == "REMOVE":
            decision_badge = "🔴 **REMOVE**"
        elif user_decision == "UNCERTAIN":
            decision_badge = "🟡 **UNCERTAIN**"
        else:
            decision_badge = t("ep_queue_undecided")

        # AI verdict display name
        ai_verdict_display = {"PASS": "KEEP", "REVIEW": "REVIEW", "EXCLUDE": "REMOVE"}.get(verdict, verdict)

        with st.expander(
            f"Episode #{ep_id}  ·  AI: {ai_verdict_display}  ·  "
            f"Issue Severity: {sev_score:.1f}  ·  "
            f"Your Decision: {decision_badge}",
            expanded=False,
        ):
            col_info, col_actions = st.columns([2, 1])

            with col_info:
                st.markdown(f"{t('ep_frames')}: {num_frames}  ·  **Pattern Type**: `{pattern}`")
                st.caption(
                    "Pattern Type 是统计风险模式，不是已确认的故障；请结合轨迹和任务语义人工确认。"
                    if get_lang() == "zh" else
                    "Pattern Type is a statistical risk pattern, not a confirmed failure; confirm it with trajectory and task context."
                )
                st.markdown(f"{t('ep_main_issues')}: {top_issues}")

                # Notes input
                notes_key = f"queue_notes_{ep_id}"
                new_notes = st.text_area(
                    t("ep_notes_label"),
                    value=user_notes,
                    height=60,
                    placeholder=t("ep_notes_placeholder"),
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
                if st.button(t("ep_view_detail"), key=f"q_detail_{ep_id}", use_container_width=True):
                    st.session_state.selected_episode = ep_id
                    st.rerun()

def _render_episode_table(table_df: pd.DataFrame, prefix: str) -> None:
    """Render the episode table (used by the All Episodes tab)."""
    if table_df.empty:
        st.info(t("ep_table_empty"))
        return

    col_ep = t("ep_col_episode")
    col_verdict = t("ep_col_verdict")
    col_integrity = t("ep_col_integrity")
    col_behavior = t("ep_col_behavior")
    col_pattern = t("ep_col_pattern")
    col_frames = t("ep_col_frames")
    col_issues = t("ep_col_issues")

    display_df = table_df[[
        "episode_id", "behavior_verdict", "integrity_check",
        "_view_score", "pattern_type", "num_frames", "top_issues",
    ]].copy()

    # Localized column names
    display_df.columns = [
        col_ep, col_verdict, col_integrity, col_behavior,
        col_pattern, col_frames, col_issues,
    ]

    # Data table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        column_config={
            col_ep: st.column_config.NumberColumn(format="%d"),
            col_verdict: st.column_config.TextColumn(),
            col_integrity: st.column_config.TextColumn(),
            col_behavior: st.column_config.NumberColumn(format="%.2f"),
            col_frames: st.column_config.NumberColumn(format="%d"),
        },
        hide_index=True,
        key=f"episode_table_{prefix}",
    )

    # Episode selector
    col1, col2 = st.columns([2, 1])
    with col1:
        ep_options = [f"Episode #{row['episode_id']} · {row['behavior_verdict']}"
                      for _, row in table_df.iterrows()]
        ep_indices = table_df["episode_id"].tolist()
        if ep_indices:
            selected = st.selectbox(
                t("ep_select_detail"),
                options=range(len(ep_indices)),
                format_func=lambda i: ep_options[i],
                key=f"detail_select_{prefix}",
                index=0,
            )
            st.session_state.selected_episode = ep_indices[selected]
    with col2:
        st.write("")
        st.write("")
        if st.button(t("ep_view_detail"), type="primary", key=f"view_detail_{prefix}"):
            st.session_state.selected_episode = ep_indices[selected]
            st.rerun()

def _status_chip(m) -> str:
    """One-line status chip for a metric result."""
    if m is None:
        return "—"
    status = (m.assessment or {}).get("status", "?")
    if m.availability.value == "not_available":
        reason = (m.assessment or {}).get("reason", "")
        return f"⚪ N/A · {reason}" if reason else "⚪ N/A"
    icon = {"pass": "✅", "review": "🟡", "exclude": "🔴"}.get(status, "⚪")
    return f"{icon} {status.upper()}"


def _render_visual_audit(ep_id: int, ep_result) -> None:
    """Render the visual audit panel (VA-A integrity + VA-B quality)."""
    st.markdown(f"### {t('vis_header')}")

    vis_metrics = {
        "video_freeze": ep_result.metrics.get("video_freeze"),
        "video_timestamp_alignment": ep_result.metrics.get("video_timestamp_alignment"),
        "video_stream_sync": ep_result.metrics.get("video_stream_sync"),
        "visual_quality": ep_result.metrics.get("visual_quality"),
    }
    present = {k: v for k, v in vis_metrics.items() if v is not None}
    if not present:
        st.info(t("vis_no_visual"))
        return

    # Dependency-missing gate: visual metrics are "not audited", never "pass".
    dep_missing = [
        m for m in present.values()
        if m.availability.value == "not_available"
        and (m.assessment or {}).get("reason") == "video_deps_missing"
    ]
    if len(dep_missing) == len(present):
        st.warning(t("vis_dep_missing"), icon="⚠️")
        return
    if dep_missing:
        st.warning(t("vis_dep_missing"), icon="⚠️")

    # ---- VA-A: hard-evidence integrity trio ----
    st.markdown(f"#### {t('vis_va_a_header')}")
    name_map = {
        "video_freeze": t("vis_vf_name"),
        "video_timestamp_alignment": t("vis_vta_name"),
        "video_stream_sync": t("vis_vsync_name"),
        "visual_quality": t("vis_vq_name"),
    }
    for key in ("video_freeze", "video_timestamp_alignment", "video_stream_sync"):
        m = vis_metrics.get(key)
        if m is None:
            continue
        msg = (m.message or "").strip()
        chip = _status_chip(m)
        if msg:
            st.markdown(f"- **{name_map[key]}** — {chip}  \n  {msg}")
        else:
            st.markdown(f"- **{name_map[key]}** — {chip}")

    # Freeze regions table (the actionable evidence)
    m_vf = vis_metrics.get("video_freeze")
    if m_vf is not None and m_vf.availability.value == "available":
        details = m_vf.details or {}
        regions = details.get("freeze_regions") or []
        if regions:
            st.caption(t("vis_freeze_regions", n=len(regions)))
            rows = []
            for r in regions[:8]:
                rows.append({
                    t("vis_col_feature"): r.get("feature", "—"),
                    t("vis_col_span"): f"{r.get('parquet_start', '—')}–{r.get('parquet_end', '—')}",
                    t("vis_col_duration"): r.get("duration_sec", "—"),
                    t("vis_col_moving"): r.get("moving_ratio_in_span", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption(t("vis_freeze_none"))

    # ---- VA-B: observational quality measurement ----
    m_vq = vis_metrics.get("visual_quality")
    if m_vq is not None and m_vq.availability.value == "available":
        st.markdown(f"#### {t('vis_va_b_header')}")
        details = m_vq.details or {}
        per_feature = details.get("per_feature") or {}
        worst_feature = (m_vq.measurement or {}).get("worst_feature")
        worst = per_feature.get(worst_feature) or {}

        pen = (m_vq.measurement or {}).get("penalty", "—")
        dominant = worst.get("dominant_issue", "—")
        n_samples = details.get("sample_count", "—")
        st.markdown(t("vis_penalty_line", pen=pen, issue=dominant, n=n_samples))
        if worst.get("worst_frame_t") is not None:
            st.caption(t("vis_worst_frame", t=worst["worst_frame_t"]))

        if per_feature:
            st.caption(t("vis_vq_cameras"))
            rows = []
            for feat, s in sorted(
                per_feature.items(),
                key=lambda kv: -kv[1].get("penalty", 0.0),
            ):
                rows.append({
                    t("vis_col_feature"): feat + (" ★" if feat == worst_feature else ""),
                    t("vis_col_blur"): s.get("median_blur_var", "—"),
                    t("vis_col_blur_pen"): s.get("blur_penalty", "—"),
                    t("vis_col_lum"): s.get("dark_frac", "—"),
                    t("vis_col_bright"): s.get("bright_frac", "—"),
                    t("vis_col_contrast"): s.get("low_contrast_frac", "—"),
                    t("vis_col_exposure_pen"): s.get("exposure_penalty", "—"),
                    t("vis_col_penalty"): s.get("penalty", "—"),
                    t("vis_col_dominant"): s.get("dominant_issue", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            th = details.get("thresholds") or {}
            st.caption(t(
                "vis_thresholds",
                blur=th.get("blur_var_floor", "—"),
                dark=th.get("dark_mean", "—"),
                bright=th.get("bright_mean", "—"),
                contrast=th.get("contrast_floor", "—"),
            ))

            # Per-sample curves of the worst camera stream
            samples = worst.get("samples") or []
            if len(samples) >= 2:
                st.caption(t("vis_samples_chart"))
                df_s = pd.DataFrame(samples)
                if "t" in df_s.columns:
                    fig_vq = go.Figure()
                    if "blur_var" in df_s.columns:
                        bv = df_s["blur_var"].astype(float)
                        bv_max = bv.max() if float(bv.max()) > 0 else 1.0
                        fig_vq.add_trace(go.Scatter(
                            x=df_s["t"], y=(bv / bv_max * 100.0).round(1),
                            mode="lines+markers", name=t("vis_blur_var_norm"),
                            line=dict(width=1.8),
                        ))
                    if "mean_lum" in df_s.columns:
                        fig_vq.add_trace(go.Scatter(
                            x=df_s["t"], y=df_s["mean_lum"],
                            mode="lines+markers", name=t("vis_mean_lum"),
                            line=dict(width=1.5, dash="dot"),
                        ))
                    if "contrast_p5_p95" in df_s.columns:
                        fig_vq.add_trace(go.Scatter(
                            x=df_s["t"], y=df_s["contrast_p5_p95"],
                            mode="lines+markers", name=t("vis_contrast"),
                            line=dict(width=1.5, dash="dash"),
                        ))
                    if "clipped_frac" in df_s.columns:
                        fig_vq.add_trace(go.Scatter(
                            x=df_s["t"], y=df_s["clipped_frac"],
                            mode="lines+markers", name=t("vis_clipped_frac"),
                            line=dict(width=1.5, dash="dot"),
                            yaxis="y2",
                        ))
                    fig_vq.update_layout(
                        xaxis_title=t("ep_traj_time_axis"),
                        yaxis=dict(title=t("ep_traj_value_axis")),
                        yaxis2=dict(
                            title=t("vis_clipped_frac"), overlaying="y",
                            side="right", range=[0, 1], showgrid=False,
                        ),
                        hovermode="x unified",
                        height=300,
                        margin=dict(l=40, r=50, t=20, b=30),
                        legend=dict(orientation="h", y=-0.25),
                    )
                    st.plotly_chart(
                        fig_vq, use_container_width=True,
                        key=f"visq_{ep_id}",
                    )


def _render_episode_detail(ep_id: int) -> None:
    """Render the detail panel of a single episode."""
    st.divider()
    st.subheader(t("ep_detail_title", ep=ep_id))

    ep_result = result.episodes.get(ep_id)
    if ep_result is None:
        st.error(t("ep_not_found", ep=ep_id))
        return

    # Find the matching row
    ep_row = df[df["episode_id"] == ep_id].iloc[0] if not df[df["episode_id"] == ep_id].empty else None

    # Two panels: Integrity + Behavior
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {t('ep_integrity_header')}")
        integrity_pass = ep_row["integrity_check"] == "PASS" if ep_row is not None else True
        emoji = "🟢" if integrity_pass else "🔴"
        status_text = t("ep_integrity_pass") if integrity_pass else t("ep_integrity_fail")
        st.markdown(f"""
        <div style="padding: 16px; border-radius: 10px; background: {'#dcfce7' if integrity_pass else '#fee2e2'}; color: {'#166534' if integrity_pass else '#991b1b'};">
            <div style="font-size: 24px; font-weight: 700;">{emoji} {status_text}</div>
        </div>
        """, unsafe_allow_html=True)

        if ep_row is not None and ep_row["integrity_issues"] != "—":
            st.caption(t("ep_integrity_issues"))
            for issue in str(ep_row["integrity_issues"]).split(", "):
                st.warning(f"🔴 {issue}")

    with col2:
        st.markdown(f"### {t('ep_behavior_header')}")
        verdict = ep_result.verdict.value
        emoji_v, color_v = get_verdict_badge(verdict)
        verdict_label = {"PASS": t("ep_verdict_keep"),
                         "REVIEW": t("ep_verdict_review"),
                         "EXCLUDE": t("ep_verdict_exclude")}.get(verdict, verdict)
        sev_score = ep_row['_view_score'] if ep_row is not None else '—'

        # Multi-dimension scores
        portable_s = ep_row.get('portable_score') if ep_row is not None else None
        platform_s = ep_row.get('platform_score') if ep_row is not None else None
        combined_s = ep_row.get('combined_score') if ep_row is not None else None
        has_plat = bool(ep_row.get('has_platform_metrics')) if ep_row is not None else False

        score_lines = [t("ep_score_severity", v=sev_score)]
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
            st.caption("Pattern Type · Risk Signal")
            st.info(
                f"`{ep_row['pattern_type']}`\n\n"
                + (
                    "这是统计模式，不等于已确认的数据错误。"
                    if get_lang() == "zh" else
                    "This is a statistical pattern, not a confirmed data error."
                )
            )

    st.divider()

    # Metrics detail
    st.markdown(f"### {t('ep_metrics_header')}")

    col_metric = t("ep_col_metric")
    col_status = t("ep_col_status")
    col_value = t("ep_col_value")
    col_note = t("ep_col_note")

    metric_rows = []
    for m_name, m in sorted(ep_result.metrics.items()):
        if m.availability.value == "available":
            status = "✅ PASS" if not m.has_finding else "⚠️ FLAGGED"
            measurement_str = _format_measurement(m.measurement)
            metric_rows.append({
                col_metric: m_name,
                col_status: status,
                col_value: measurement_str,
                col_note: m.message or m.assessment.get("reason", ""),
            })
        elif m.availability.value == "not_available":
            metric_rows.append({
                col_metric: m_name,
                col_status: "N/A",
                col_value: "—",
                col_note: m.assessment.get("reason", t("ep_na_reason")),
            })
        else:
            metric_rows.append({
                col_metric: m_name,
                col_status: "ERROR",
                col_value: "—",
                col_note: m.assessment.get("reason", t("ep_error_reason")),
            })

    metric_df = pd.DataFrame(metric_rows)
    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            col_metric: st.column_config.TextColumn(width="small"),
            col_status: st.column_config.TextColumn(width="small"),
            col_value: st.column_config.TextColumn(width="medium"),
            col_note: st.column_config.TextColumn(width="large"),
        },
    )

    st.divider()

    # Visual audit panel (v0.7.x UI catch-up, REQ-4)
    _render_visual_audit(ep_id, ep_result)

    st.divider()

    # Diagnosis
    st.markdown(f"### {t('ep_diag_header')}")

    if ep_result.verdict.value == "PASS":
        st.success(t("ep_diag_normal"))
    else:
        diagnosis = _generate_diagnosis(ep_result, ep_row)
        with st.container():
            st.markdown(t("ep_diag_what"))
            st.info(diagnosis["what"])
            st.markdown(t("ep_diag_why"))
            st.warning(diagnosis["why"])
            st.markdown(t("ep_diag_next"))
            st.success(diagnosis["next"])

    st.divider()

    # ──────────────────────────────────────────────────────────────────────
    # Trajectory Visualization
    # ──────────────────────────────────────────────────────────────────────
    _render_trajectory_visualization(ep_id, ep_result, ep_row)

    st.divider()

    # User decision (core Review Queue feature)
    st.markdown(f"### {t('ep_decision_header')}")

    current = st.session_state.user_verdicts.get(ep_id, {"decision": None, "notes": ""})
    current_decision = current.get("decision")
    current_notes = current.get("notes", "")

    # Decision buttons
    col_k, col_r, col_u = st.columns(3)
    with col_k:
        keep_pressed = st.button(
            t("ep_btn_keep"),
            type="primary" if current_decision == "KEEP" else "secondary",
            use_container_width=True,
            key=f"btn_keep_{ep_id}",
            help=t("ep_btn_keep_help"),
        )
    with col_r:
        remove_pressed = st.button(
            t("ep_btn_remove"),
            type="primary" if current_decision == "REMOVE" else "secondary",
            use_container_width=True,
            key=f"btn_remove_{ep_id}",
            help=t("ep_btn_remove_help"),
        )
    with col_u:
        uncertain_pressed = st.button(
            t("ep_btn_uncertain"),
            type="primary" if current_decision == "UNCERTAIN" else "secondary",
            use_container_width=True,
            key=f"btn_uncertain_{ep_id}",
            help=t("ep_btn_uncertain_help"),
        )

    if keep_pressed:
        st.session_state.user_verdicts[ep_id] = {
            "decision": "KEEP",
            "notes": current_notes,
        }
        st.success(t("ep_marked_keep", ep=ep_id))
        st.rerun()
    if remove_pressed:
        st.session_state.user_verdicts[ep_id] = {
            "decision": "REMOVE",
            "notes": current_notes,
        }
        st.error(t("ep_marked_remove", ep=ep_id))
        st.rerun()
    if uncertain_pressed:
        st.session_state.user_verdicts[ep_id] = {
            "decision": "UNCERTAIN",
            "notes": current_notes,
        }
        st.warning(t("ep_marked_uncertain", ep=ep_id))
        st.rerun()

    # Current decision status
    if current_decision:
        badge = {"KEEP": ("🟢", "#10b981", t("ep_badge_keep")),
                 "REMOVE": ("🔴", "#ef4444", t("ep_badge_remove")),
                 "UNCERTAIN": ("🟡", "#f59e0b", t("ep_badge_uncertain"))}.get(current_decision, ("⚪", "#9ca3af", ""))
        st.caption(f"{t('ep_current_decision')}<span style='color:{badge[1]}; font-weight:600;'>{badge[0]} {current_decision} · {badge[2]}</span>",
                   unsafe_allow_html=True)
    else:
        st.caption(t("ep_no_decision"))

    # Notes input
    notes_val = st.text_area(
        t("ep_notes_area"),
        value=current_notes,
        height=80,
        placeholder=t("ep_notes_area_placeholder"),
        key=f"notes_{ep_id}",
    )

    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if st.button(t("ep_save_notes"), key=f"save_notes_{ep_id}", use_container_width=True):
            existing = st.session_state.user_verdicts.get(ep_id, {"decision": None, "notes": ""})
            st.session_state.user_verdicts[ep_id] = {
                "decision": existing.get("decision"),
                "notes": notes_val,
            }
            st.success(t("ep_notes_saved"))
            st.rerun()
    with col_clear:
        if st.button(t("ep_clear_decision"), key=f"clear_verdict_{ep_id}", use_container_width=True):
            if ep_id in st.session_state.user_verdicts:
                del st.session_state.user_verdicts[ep_id]
            st.info(t("ep_decision_cleared"))
            st.rerun()

    # Close button
    if st.button(t("ep_close_detail"), key="close_detail"):
        st.session_state.selected_episode = None
        st.rerun()

@st.cache_data(show_spinner=t("ep_traj_loading"))
def _load_episode_data(dataset_path: str, episode_index: int):
    """Load raw data of a single episode (cached)."""
    from rda.io.lerobot_loader import iter_episodes

    for ep in iter_episodes(dataset_path):
        if ep.episode_index == episode_index:
            return ep
    return None

def _render_trajectory_visualization(ep_id, ep_result, ep_row):
    """Render trajectory visualization (State / Action time series with anomaly highlights)."""
    st.markdown(f"### {t('ep_traj_header')}")

    dataset_path = st.session_state.get("dataset_path")
    if not dataset_path:
        st.info(t("ep_traj_no_path"))
        return

    try:
        episode_data = _load_episode_data(str(dataset_path), int(ep_id))
    except Exception as e:
        st.warning(t("ep_traj_load_err", err=e))
        return

    if episode_data is None:
        st.warning(t("ep_traj_not_found", ep=ep_id))
        return

    has_state = bool(episode_data.observation)
    has_action = bool(episode_data.action)

    if not has_state and not has_action:
        st.info(t("ep_traj_no_data"))
        return

    # Display mode selection
    view_options = []
    if has_state:
        view_options.append("State")
    if has_action:
        view_options.append("Action")
    if has_state and has_action:
        view_options.append("Both")

    selected_view = st.selectbox(
        t("ep_traj_view_label"),
        options=view_options,
        index=len(view_options) - 1,  # default Both (if available)
        key=f"traj_view_{ep_id}",
    )

    # Timeline
    timestamps = episode_data.timestamps
    joint_limits = episode_data.meta.get("joint_limits") if episode_data.meta else None

    # Collect anomaly regions
    anomaly_regions = _collect_anomaly_regions(ep_result, episode_data.num_frames)

    show_state = selected_view in ("State", "Both") and has_state
    show_action = selected_view in ("Action", "Both") and has_action

    if show_state:
        _plot_timeseries(
            data_dict=episode_data.observation,
            timestamps=timestamps,
            title=t("ep_traj_state_title"),
            ep_id=ep_id,
            suffix="state",
            joint_limits=joint_limits,
            anomaly_regions=anomaly_regions,
        )

    if show_action:
        _plot_timeseries(
            data_dict=episode_data.action,
            timestamps=timestamps,
            title=t("ep_traj_action_title"),
            ep_id=ep_id,
            suffix="action",
            joint_limits=joint_limits,
            anomaly_regions=anomaly_regions,
        )

def _collect_anomaly_regions(ep_result, num_frames: int):
    """Extract anomaly regions from audit results for chart highlighting.

    Returns:
        List of (start_frame, end_frame, label) tuples.
    """
    regions = []

    for m_name, m in ep_result.metrics.items():
        if m.passed:
            continue

        # action_discontinuity — mark frames with large action jumps
        if "discontinuity" in m_name and m.measurement:
            spike_frames = m.measurement.get("spike_frames", [])
            window = max(1, num_frames // 100)  # ±1% window
            for frame_idx in spike_frames:
                start = max(0, int(frame_idx) - window)
                end = min(num_frames - 1, int(frame_idx) + window)
                regions.append((start, end, m_name))

        # spike-related metrics
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
    """Plot a time-series chart (State or Action) with joint-limit lines and anomaly highlights."""
    fig = go.Figure()

    # One curve per dimension
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

    # Joint-limit dashed lines
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

    # Highlight detected statistical-anomaly regions; they are not confirmed
    # error regions and are labeled accordingly in the chart.
    if anomaly_regions:
        for start, end, label in anomaly_regions:
            fig.add_vrect(
                x0=timestamps[start] if start < len(timestamps) else start,
                x1=timestamps[end] if end < len(timestamps) else end,
                fillcolor="red",
                opacity=0.12,
                line_width=0,
                annotation_text=f"Risk signal: {label}",
                annotation_position="top",
            )

    fig.update_layout(
        title=f"{title}  ·  Episode #{ep_id}",
        xaxis_title=t("ep_traj_time_axis"),
        yaxis_title=t("ep_traj_value_axis"),
        legend_title=t("ep_traj_legend"),
        hovermode="x unified",
        height=350,
        margin=dict(l=40, r=20, t=40, b=30),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"traj_{suffix}_{ep_id}")

def _format_measurement(measurement: dict) -> str:
    """Format a measurement dict into a readable string."""
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
    """Generate the three-layer diagnosis (WHAT / WHY / NEXT)."""
    pattern = ep_row["pattern_type"] if ep_row is not None else None
    sev_score = ep_row["_view_score"] if ep_row is not None else 0

    # Collect failed metrics
    failed_metrics = []
    for m_name, m in ep_result.metrics.items():
        if m.availability.value == "available" and not m.passed:
            failed_metrics.append(m_name)

    what_parts = []
    why_parts = []
    next_parts = []

    # Integrity issues
    critical_failed = [m for m in failed_metrics if m in {
        "missing_dropout", "invalid_values", "schema_consistency",
        "timestamp_validity", "joint_limit",
    }]
    if critical_failed:
        what_parts.append(t("diag_integrity_what", metrics=", ".join(critical_failed)))
        why_parts.append(t("diag_integrity_why"))
        next_parts.append(t("diag_integrity_next"))

    # Behavior issues
    if "action_discontinuity" in failed_metrics:
        what_parts.append(t("diag_disc_what"))
        why_parts.append(t("diag_disc_why"))
        next_parts.append(t("diag_disc_next"))

    if "idle_ratio" in failed_metrics:
        what_parts.append(t("diag_idle_what"))
        why_parts.append(t("diag_idle_why"))
        next_parts.append(t("diag_idle_next"))

    if "velocity_acceleration" in failed_metrics:
        what_parts.append(t("diag_vel_what"))
        why_parts.append(t("diag_vel_why"))
        next_parts.append(t("diag_vel_next"))

    if "distribution" in failed_metrics:
        what_parts.append(t("diag_dist_what"))
        why_parts.append(t("diag_dist_why"))
        next_parts.append(t("diag_dist_next"))

    if not what_parts:
        what_parts.append(t("diag_generic_what", score=sev_score))
        why_parts.append(t("diag_generic_why"))
        next_parts.append(t("diag_generic_next"))

    sep = "；" if get_lang() == "zh" else "; "
    return {
        "what": sep.join(what_parts),
        "why": sep.join(why_parts) if why_parts else t("diag_why_fallback"),
        "next": sep.join(next_parts) if next_parts else t("diag_next_fallback"),
    }



st.title(t("ep_title"))
st.caption(t("ep_caption"))

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
if st.session_state.audit_result is None:
    st.warning(t("not_audited"), icon="🔍")
    st.page_link("pages/2_Audit.py", label=t("go_audit"), icon="🔍")
    st.stop()

result = st.session_state.audit_result

# Make sure episodes_df exists (language-aware)
if st.session_state.episodes_df is None:
    st.session_state.episodes_df = build_episodes_dataframe(result, lang=get_lang())

df = st.session_state.episodes_df

# Make sure user_verdicts exists (defensive; initialized in app.py)
if "user_verdicts" not in st.session_state:
    st.session_state.user_verdicts = {}

# ---------------------------------------------------------------------------
# Score view switch (Portable / Platform-specific / Combined)
# ---------------------------------------------------------------------------
_has_platform = bool(df["has_platform_metrics"].any()) if not df.empty else False

_score_view_options = [t("ep_view_portable"), t("ep_view_combined")]
_score_view_keys = ["portable_score", "combined_score"]
if _has_platform:
    _score_view_options.insert(1, t("ep_view_platform"))
    _score_view_keys.insert(1, "platform_score")

score_view_label = st.radio(
    t("ep_score_view"),
    options=_score_view_options,
    index=0,  # default Portable Core
    horizontal=True,
    help=t("ep_score_view_help"),
)
_score_view_key = _score_view_keys[_score_view_options.index(score_view_label)]

# Dynamically compute deviation_score for the current view (sorting & display)
if not df.empty and df[_score_view_key].notna().any():
    df = df.copy()
    df["_view_score"] = df[_score_view_key].fillna(df["deviation_score"])
else:
    df = df.copy()
    df["_view_score"] = df["deviation_score"]

# ---------------------------------------------------------------------------
# Stats overview
# ---------------------------------------------------------------------------
total = len(df)
keep_count = len(df[df["behavior_verdict"] == "PASS"])
review_count = len(df[df["behavior_verdict"] == "REVIEW"])
remove_count = len(df[df["behavior_verdict"] == "EXCLUDE"])

# Decision stats
verdicts = st.session_state.user_verdicts
decided_count = sum(1 for v in verdicts.values() if v.get("decision"))
user_keep = sum(1 for v in verdicts.values() if v.get("decision") == "KEEP")
user_remove = sum(1 for v in verdicts.values() if v.get("decision") == "REMOVE")
user_uncertain = sum(1 for v in verdicts.values() if v.get("decision") == "UNCERTAIN")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(t("ep_metric_total"), total)
with col2:
    st.metric(t("ep_metric_keep_ai"), keep_count)
with col3:
    st.metric(t("ep_metric_review_ai"), review_count)
with col4:
    st.metric(t("ep_metric_remove_ai"), remove_count)

# Manual review progress
st.caption(
    t("ep_review_progress",
      decided=decided_count, need=review_count + remove_count,
      keep=user_keep, remove=user_remove, uncertain=user_uncertain)
)

# ---------------------------------------------------------------------------
# Review Queue tab
# ---------------------------------------------------------------------------
tab_queue, tab_all = st.tabs([t("ep_tab_queue"), t("ep_tab_all")])

with tab_queue:
    queue_df = df[df["behavior_verdict"].isin(["EXCLUDE", "REVIEW"])].copy()
    if queue_df.empty:
        st.success(t("ep_queue_empty"))
    else:
        st.info(
            t("ep_queue_info",
              n=len(queue_df),
              rm=len(queue_df[queue_df['behavior_verdict'] == 'EXCLUDE']),
              rv=len(queue_df[queue_df['behavior_verdict'] == 'REVIEW'])),
            icon="⚠️",
        )
        _render_review_queue(queue_df)

with tab_all:
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        verdict_filter = st.multiselect(
            t("ep_filter_verdict"),
            options=["PASS", "REVIEW", "EXCLUDE"],
            default=["PASS", "REVIEW", "EXCLUDE"],
            format_func=lambda x: {"PASS": "KEEP", "REVIEW": "REVIEW", "EXCLUDE": "REMOVE"}.get(x, x),
        )
    with col2:
        integrity_filter = st.multiselect(
            t("ep_filter_integrity"),
            options=["PASS", "FAIL"],
            default=["PASS", "FAIL"],
        )
    with col3:
        pattern_filter = st.multiselect(
            t("ep_filter_pattern"),
            options=[p for p in df["pattern_type"].unique() if p != "—"],
            default=[],
        )

    filtered = df[df["behavior_verdict"].isin(verdict_filter)]
    filtered = filtered[filtered["integrity_check"].isin(integrity_filter)]
    if pattern_filter:
        filtered = filtered[filtered["pattern_type"].isin(pattern_filter)]

    st.caption(t("ep_filter_showing", n=len(filtered), total=total))
    _render_episode_table(filtered, prefix="all")


# ---------------------------------------------------------------------------
# Episode detail panel (expander style)
# ---------------------------------------------------------------------------
if "selected_episode" in st.session_state and st.session_state.selected_episode is not None:
    ep_id = st.session_state.selected_episode
    _render_episode_detail(ep_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
