"""Page 2: Audit · run the audit (bilingual zh/en).

Calls DatasetAuditor.audit_dataset() to produce audit results with a
progress bar; on success it links to the Health Overview page.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rda.ui_app.i18n import get_lang, t  # noqa: E402
from rda.audit.execution_tier import ExecutionTier  # noqa: E402

from components.common import (  # noqa: E402
    build_episodes_dataframe,
    compute_dhi,
)

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _run_audit(execution_tier=None) -> None:
    """Run the audit with a progress bar."""
    from rda.audit.dataset_audit import DatasetAuditor, DatasetAuditResult
    from rda.io.lerobot_loader import iter_episodes

    st.session_state.audit_in_progress = True

    progress_bar = st.progress(0, text=t("audit_preparing"))
    status_text = st.empty()

    try:
        total = info.num_episodes
        auditor = DatasetAuditor(execution_tier=execution_tier)
        result = DatasetAuditResult(dataset_info=info)

        status_text.text(t("audit_loading_data"))

        # Audit episode by episode, updating progress
        for i, episode in enumerate(iter_episodes(info.path)):
            ep_result = auditor.episode_auditor.audit(episode)
            result.episodes[episode.episode_index] = ep_result

            progress = (i + 1) / total if total > 0 else 1.0
            progress_bar.progress(
                min(progress, 1.0),
                text=t("audit_progress", i=i + 1, total=total, pct=progress * 100),
            )

        # Compute verdict distribution
        result.compute_verdict_counts()

        # Store execution tier metadata
        result.execution_tier = execution_tier.value if execution_tier else "fast"
        result.video_quality_executed = execution_tier in (
            ExecutionTier.VIDEO_QUALITY, ExecutionTier.VIDEO_ONLY, ExecutionTier.FULL
        ) if execution_tier else False

        # Persist to session state
        st.session_state.audit_result = result

        # Precompute report data (language-aware)
        _lang = get_lang()
        st.session_state.dataset_report = compute_dhi(result, lang=_lang)
        st.session_state.episodes_df = build_episodes_dataframe(result, lang=_lang)
        st.session_state._report_lang = _lang

        # Save audit snapshot to history
        try:
            from rda.report.audit_history import save_audit_snapshot
            snapshot_path = save_audit_snapshot(result, info.path)
            st.session_state.dataset_path = info.path
        except (OSError, ValueError) as e:
            pass  # snapshot failure does not affect the main flow

        progress_bar.progress(1.0, text=t("audit_done_progress"))
        status_text.text(t("audit_done_status", n=len(result.episodes)))

        st.success(t("audit_done_success"), icon="🎉")
        st.page_link(
            "pages/3_Health_Overview.py",
            label=t("audit_view_health"),
            icon="💚",
        )

    except Exception as e:  # noqa: BLE001 - top-level UI catch-all
        st.error(t("audit_failed", err=e))
        import traceback
        import logging
        logging.getLogger(__name__).error("Audit failed: %s", e, exc_info=True)
        with st.expander(t("audit_detail_err")):
            st.code(traceback.format_exc())
    finally:
        st.session_state.audit_in_progress = False



st.title(t("audit_title"))
st.caption(t("audit_caption"))

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
if st.session_state.dataset_info is None:
    st.warning(t("audit_gate_upload"), icon="📤")
    st.page_link("pages/1_Upload.py", label=t("go_upload"), icon="📤")
    st.stop()

info = st.session_state.dataset_info
st.write(t("audit_dataset", path=info.path))
st.write(t("audit_summary_line", eps=info.num_episodes, frames=f"{info.total_frames:,}"))

# ---------------------------------------------------------------------------
# Audit configuration (read-only: show enabled check layers)
# ---------------------------------------------------------------------------
st.subheader(t("audit_layers_header"))

st.info(t("audit_layers_info"))

st.caption(t("audit_layers_note"))

# ---------------------------------------------------------------------------
# Execution tier selector
# ---------------------------------------------------------------------------
st.divider()
st.subheader(t("audit_tier_header"))

tier_options = {
    "fast": (ExecutionTier.FAST, t("audit_tier_fast"), t("audit_tier_fast_desc")),
    "video_quality": (ExecutionTier.VIDEO_QUALITY, t("audit_tier_video_quality"), t("audit_tier_video_quality_desc")),
    "no_video": (ExecutionTier.NO_VIDEO, t("audit_tier_no_video"), t("audit_tier_no_video_desc")),
    "video_only": (ExecutionTier.VIDEO_ONLY, t("audit_tier_video_only"), t("audit_tier_video_only_desc")),
    "full": (ExecutionTier.FULL, t("audit_tier_full"), t("audit_tier_full_desc")),
}

tier_keys = list(tier_options.keys())
default_index = tier_keys.index("fast")

selected_tier_key = st.radio(
    t("audit_tier_help"),
    options=tier_keys,
    index=default_index,
    format_func=lambda k: f"{tier_options[k][1]}  —  {tier_options[k][2]}",
    horizontal=False,
    label_visibility="collapsed",
)
st.session_state.selected_execution_tier = tier_options[selected_tier_key][0]

# ---------------------------------------------------------------------------
# Run audit
# ---------------------------------------------------------------------------
st.divider()

if st.session_state.audit_result is not None:
    st.info(t("audit_done_info", eps=info.num_episodes), icon="✅")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("audit_rerun_btn"), type="secondary"):
            st.session_state.audit_result = None
            st.session_state.dataset_report = None
            st.session_state.episodes_df = None
            st.rerun()
    with col2:
        st.page_link(
            "pages/3_Health_Overview.py",
            label=t("audit_view_health"),
            icon="💚",
        )

    # Quick result summary
    result = st.session_state.audit_result
    counts = {v.value: c for v, c in result.verdict_counts.items()}

    # Show execution tier info
    _exec_tier = getattr(result, "execution_tier", None)
    _vq_exec = getattr(result, "video_quality_executed", False)
    tier_labels = {
        "fast": t("audit_tier_fast"),
        "video_quality": t("audit_tier_video_quality"),
        "no_video": t("audit_tier_no_video"),
        "video_only": t("audit_tier_video_only"),
        "full": t("audit_tier_full"),
    }
    _tier_label = tier_labels.get(_exec_tier, t("audit_tier_fast"))
    st.info(f"**{t('audit_tier_result_label')}**: {_tier_label}  |  "
            f"{t('audit_tier_vq_included') if _vq_exec else t('audit_tier_vq_skipped')}")

    st.subheader(t("audit_result_header"))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 KEEP", counts.get("PASS", 0),
                  t("health_verdict_keep_pct", pct=counts.get('PASS', 0) / max(info.num_episodes, 1) * 100))
    with col2:
        st.metric("🟡 REVIEW", counts.get("REVIEW", 0),
                  t("health_verdict_keep_pct", pct=counts.get('REVIEW', 0) / max(info.num_episodes, 1) * 100))
    with col3:
        st.metric("🔴 REMOVE", counts.get("EXCLUDE", 0),
                  t("health_verdict_keep_pct", pct=counts.get('EXCLUDE', 0) / max(info.num_episodes, 1) * 100))

else:
    if st.button(t("audit_start_btn"), type="primary", disabled=st.session_state.audit_in_progress,
                 use_container_width=True):
        _tier = st.session_state.get("selected_execution_tier", ExecutionTier.FAST)
        _run_audit(execution_tier=_tier)


# ---------------------------------------------------------------------------
# Audit execution
# ---------------------------------------------------------------------------
