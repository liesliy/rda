"""Page 7: Recommend · optimization advice (bilingual zh/en).

Computes temporal-structure metrics locally, then sends only <1KB of
aggregated statistics to the RDA API for rule evaluation (point
RDA_API_URL at a private deployment if needed). The UI language is
forwarded so the rules engine returns matching-language text.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rda.ui_app.i18n import get_lang, t  # noqa: E402

st.title(t("rec_title"))
st.caption(t("rec_caption"))

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
info = st.session_state.get("dataset_info")
dataset_path = st.session_state.get("dataset_path")

if info is None:
    st.warning(t("not_uploaded"), icon="📤")
    st.page_link("pages/1_Upload.py", label=t("go_upload"), icon="📤")
    st.stop()

if not dataset_path:
    st.warning(t("rec_no_path"), icon="⚠️")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------
_lang = get_lang()
_policy_fw = t("rec_policy_fw")
_policy_tp = t("rec_policy_tp")

policy_label = st.radio(
    t("rec_policy_label"),
    options=[_policy_fw, _policy_tp],
    index=1,
    help=t("rec_policy_help"),
)

policy_name = policy_label.split(" — ")[0]

# REQ: policy_chunk_size input (v0.7.1 UI catch-up; CLI had it since v0.6.0).
_chunk_raw = st.text_input(
    t("rec_chunk_label"),
    value="",
    help=t("rec_chunk_help"),
    placeholder=t("rec_chunk_none"),
)
policy_chunk_size: Optional[int] = None
_chunk_val = (_chunk_raw or "").strip()
if _chunk_val:
    try:
        _cand = int(_chunk_val)
        if 1 <= _cand <= 128:
            policy_chunk_size = _cand
        else:
            st.warning(f"chunk size out of range (1-128): {_cand} — ignored")
    except ValueError:
        st.warning(f"invalid chunk size: {_chunk_val} — ignored")

st.info(t("rec_privacy"), icon="🔒")

# ---------------------------------------------------------------------------
# Run recommendation
# ---------------------------------------------------------------------------
if st.button(t("rec_run_btn"), type="primary", use_container_width=True):
    from rda.io.lerobot_loader import iter_episodes
    from rda.recommend.api_client import run_recommendation
    from rda.recommend.formatter import format_recommendation_text
    from rda.recommend.types import TargetPolicy

    target_policy = TargetPolicy.from_cli_name(policy_name)

    progress_bar = st.progress(0, text=t("rec_computing"))
    status_text = st.empty()

    try:
        def _progress(step, total, msg):
            ratio = step / total if total else 1.0
            progress_bar.progress(min(ratio, 1.0), text=f"{step}/{total}: {msg}")

        with st.spinner(t("rec_calling_api")):
            result = run_recommendation(
                iter_episodes(str(dataset_path)),
                target_policy=target_policy,
                total_episodes=info.num_episodes,
                total_frames=info.total_frames,
                progress_callback=_progress,
                lang=_lang,
                policy_chunk_size=policy_chunk_size,
            )

        progress_bar.progress(1.0, text=t("rec_done"))

        rules_ver = getattr(result, "rules_version", None)
        if rules_ver:
            st.caption(t("rec_rules_ver", v=rules_ver))

        # Text report
        st.subheader(t("rec_report_header"))
        st.code(format_recommendation_text(result, lang=_lang), language="text")

        # Structured recommendation cards
        st.subheader(t("rec_cards_header"))
        for rec in result.recommendations:
            action = rec.action
            action_name = getattr(action, "value", str(action)) if action else "?"
            confidence = rec.confidence
            conf_name = (
                getattr(confidence, "value", str(confidence)) if confidence else "?"
            )

            color = {
                "DO_NOT_PRUNE": "🔴",
                "DO_NOT_PRUNE_AGGRESSIVELY": "🟠",
                "TRIM_INITIAL": "🟡",
                "TRIM_IDLE_MILD": "🟢",
                # v0.7.x visual actions (REQ-4/engine v5)
                "VISUAL_REPAIR_FIRST": "🔴",
                "VISUAL_QUALITY_REVIEW": "🟡",
                # v0.6.0 advice actions
                "DISCARD_STATIC": "🔴",
                "SMOOTHING_REVIEW": "🟡",
                "CALIBRATION_CHECK": "🟡",
                "COVERAGE_SUGGESTION": "🟢",
            }.get(action_name, "⚪")

            with st.expander(
                f"{color} {rec.title}  ·  {action_name}  ·  {t('rec_confidence', v=conf_name)}"
            ):
                st.write(rec.summary)
                if rec.expected_impact:
                    st.markdown(t("rec_expected_impact", v=rec.expected_impact))
                if rec.details:
                    st.markdown(t("rec_details"))
                    for d in rec.details:
                        st.markdown(f"- {d}")
                if rec.caveats:
                    st.markdown(t("rec_caveats"))
                    for c in rec.caveats:
                        st.markdown(f"- ⚠️ {c}")

        # Raw JSON
        with st.expander(t("rec_raw_json")):
            st.json(result.to_dict())

    except Exception as e:  # noqa: BLE001
        progress_bar.empty()
        status_text.empty()
        st.error(t("rec_failed", err=e), icon="❌")
        st.caption(t("rec_failed_hint"))

st.divider()
st.caption(t("rec_disclaimer"))
