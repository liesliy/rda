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

from components.common import (  # noqa: E402
    build_episodes_dataframe,
    compute_dhi,
)

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _run_audit() -> None:
    """Run the audit with a progress bar."""
    from rda.audit.dataset_audit import DatasetAuditor, DatasetAuditResult
    from rda.io.lerobot_loader import iter_episodes

    st.session_state.audit_in_progress = True

    progress_bar = st.progress(0, text=t("audit_preparing"))
    status_text = st.empty()

    try:
        total = info.num_episodes
        auditor = DatasetAuditor()
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
        except Exception:
            pass  # snapshot failure does not affect the main flow

        progress_bar.progress(1.0, text=t("audit_done_progress"))
        status_text.text(t("audit_done_status", n=len(result.episodes)))

        st.success(t("audit_done_success"), icon="🎉")
        st.page_link(
            "pages/3_Health_Overview.py",
            label=t("audit_view_health"),
            icon="💚",
        )

    except Exception as e:
        st.error(t("audit_failed", err=e))
        import traceback
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
        _run_audit()


# ---------------------------------------------------------------------------
# Audit execution
# ---------------------------------------------------------------------------
