"""Page 5: Export · report export (bilingual zh/en).

Supports:
- JSON format (full audit result)
- CSV format (episode table)
- Scope: all / Review Queue only / anomalous only
- Cleaned export: strategy selection + preview + clean dataset export
"""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rda.ui_app.i18n import get_lang, t  # noqa: E402


def _get_tool_version() -> str:
    """Return the installed RDA tool version (lazy import)."""
    from rda import __version__

    return __version__

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _compute_episode_score(ep_result) -> float:
    """Compute the episode deviation score (0-100, higher = more anomalous)."""
    from rda.metrics.base import MetricAvailability
    import numpy as np

    scores = []
    for m_name, m in ep_result.metrics.items():
        if m.availability == MetricAvailability.AVAILABLE:
            scores.append(m.score)
    avg_score = float(np.mean(scores)) if scores else 1.0
    return round((1.0 - avg_score) * 100, 2)

def _get_pattern_type(ep_result) -> str:
    """Get the anomaly pattern type of an episode."""
    try:
        from components.common import _detect_pattern_type
        return _detect_pattern_type(ep_result) or ""
    except Exception:
        return ""

def _generate_json_export(episodes, scope: str) -> str:
    """Generate the JSON export."""
    from rda.report.json_report import generate_json_report
    from rda.report.aggregation import aggregate_dataset_metrics
    from rda.report.top_issues import compute_top_observations, compute_hero_metrics
    from rda.audit.dataset_audit import DatasetAuditResult
    from components.common import compute_dhi

    # Build a result containing only in-scope episodes (for computation)
    filtered_result = DatasetAuditResult(dataset_info=info)
    for ep in episodes:
        filtered_result.episodes[ep.episode_index] = ep
    filtered_result.compute_verdict_counts()

    # Full report structure
    full_report = generate_json_report(result)

    # Replace the episodes section
    episodes_data = []
    user_verdicts = st.session_state.get("user_verdicts", {})
    for ep in episodes:
        metrics_dict = {}
        for name, m in ep.metrics.items():
            metrics_dict[name] = {
                "availability": m.availability.value,
                "measurement": m.measurement,
                "assessment": m.assessment,
                "details": m.details,
                "message": m.message,
                "passed": m.passed,
                "score": m.score,
            }
        user_v = user_verdicts.get(ep.episode_index)
        # Behavior severity and pattern
        behavior_sev = getattr(ep, "behavior_severity", None)
        if behavior_sev is None:
            from rda.audit.rules import compute_behavior_severity
            behavior_sev = compute_behavior_severity(list(ep.metrics.values()))
        pattern = _get_pattern_type(ep)

        ep_entry = {
            "episode_index": ep.episode_index,
            "num_frames": ep.num_frames,
            "verdict": ep.verdict.value,
            "behavior_severity": round(behavior_sev, 2),
            "pattern_type": pattern,
            "metrics": metrics_dict,
        }
        if user_v:
            ep_entry["user_verdict"] = {
                "decision": user_v.get("decision"),
                "notes": user_v.get("notes", ""),
            }
        episodes_data.append(ep_entry)

    # Quality data within scope
    dhi_data = compute_dhi(filtered_result, lang=get_lang())
    dataset_metrics = aggregate_dataset_metrics(filtered_result)
    top_obs = compute_top_observations(filtered_result, dataset_metrics=dataset_metrics)
    hero = compute_hero_metrics(dataset_metrics)

    report = {
        "report_schema_version": "1.0",
        "tool_version": _get_tool_version(),
        "export_scope": scope,
        "exported_episodes": len(episodes),
        "dataset": full_report["dataset"],
        "summary": full_report["summary"],
        "quality": {
            "dhi": dhi_data["dhi"],
            "dhi_note": "Relative assessment based on structural integrity and behavioral consistency",
            "grade": dhi_data["grade"],
            "training_readiness": dhi_data["training_readiness"],
            "training_readiness_detail": dhi_data["training_readiness_detail"],
            "dimensions": dhi_data["dimensions"],
        },
        "three_layer_aggregates": {
            "layer1_integrity": dataset_metrics.get("integrity", {}),
            "layer2_temporal_motion": dataset_metrics.get("temporal_motion", {}),
            "layer3_dataset_utility": dataset_metrics.get("dataset_utility", {}),
        },
        "hero_metrics": hero,
        "top_observations": top_obs,
        "user_verdicts": {
            str(ep_id): v
            for ep_id, v in st.session_state.user_verdicts.items()
            if any(ep.episode_index == ep_id for ep in episodes)
        },
        "episodes": episodes_data,
    }

    return json.dumps(report, indent=2, ensure_ascii=False, default=str)

def _generate_csv_export(episodes) -> str:
    """Generate the CSV export."""
    from components.common import _detect_pattern_type
    from rda.audit.rules import CRITICAL_METRICS
    from rda.metrics.base import MetricAvailability
    import numpy as np

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "episode_id", "num_frames", "integrity_check",
        "behavior_verdict", "issue_severity_score", "pattern_type",
        "issue_count", "top_issues",
        "user_decision", "user_notes",
    ])

    user_verdicts = st.session_state.get("user_verdicts", {})

    for ep in sorted(episodes, key=lambda e: e.episode_index):
        # Integrity
        integrity_pass = True
        integrity_issues = []
        for m_name in CRITICAL_METRICS:
            m = ep.metrics.get(m_name)
            if m is None or m.availability != MetricAvailability.AVAILABLE:
                continue
            if not m.passed:
                integrity_pass = False
                integrity_issues.append(m_name.upper().replace("_", "-"))

        # Deviation score: combine metric-based + behavior severity
        scores = []
        for m_name, m in ep.metrics.items():
            if m.availability == MetricAvailability.AVAILABLE:
                scores.append(m.score)
        avg_score = float(np.mean(scores)) if scores else 1.0
        metric_deviation = round((1.0 - avg_score) * 100, 2)
        # Incorporate behavior severity (from episode audit or computed on the fly)
        behavior_sev = getattr(ep, "behavior_severity", None)
        if behavior_sev is None:
            from rda.audit.rules import compute_behavior_severity
            behavior_sev = compute_behavior_severity(list(ep.metrics.values()))
        deviation_score = round(max(metric_deviation, behavior_sev), 2)

        # Pattern type
        pattern = _detect_pattern_type(ep) or ""

        # Issues: combine failed metrics + pattern-based issues
        issues = []
        for m_name, m in ep.metrics.items():
            if m.availability != MetricAvailability.AVAILABLE:
                continue
            if not m.passed:
                issues.append(m_name.upper().replace("_", "-"))
        # Add pattern-based issues (behavioral signals not caught by metric pass/fail)
        if pattern:
            issues.append(f"PATTERN_{pattern.upper()}")

        # User verdict
        user_v = user_verdicts.get(ep.episode_index, {})
        user_decision = user_v.get("decision", "") or ""
        user_notes = user_v.get("notes", "") or ""

        writer.writerow([
            ep.episode_index,
            ep.num_frames,
            "PASS" if integrity_pass else "FAIL",
            ep.verdict.value,
            deviation_score,  # CSV column: issue_severity_score
            pattern,
            len(issues),
            "; ".join(issues),
            user_decision,
            user_notes,
        ])

    return output.getvalue()



st.title(t("export_title"))
st.caption(t("export_caption"))

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
if st.session_state.audit_result is None:
    st.warning(t("not_audited"), icon="🔍")
    st.page_link("pages/2_Audit.py", label=t("go_audit"), icon="🔍")
    st.stop()

result = st.session_state.audit_result
info = result.dataset_info

# ---------------------------------------------------------------------------
# Export configuration
# ---------------------------------------------------------------------------
st.subheader(t("export_config_header"))

_scope_all = t("export_scope_all")
_scope_queue = t("export_scope_queue")
_scope_anom = t("export_scope_anomalies")

col1, col2 = st.columns(2)

with col1:
    format_choice = st.radio(
        t("export_format_label"),
        options=["JSON", "CSV"],
        index=0,
        horizontal=True,
        help=t("export_format_help"),
    )

with col2:
    scope_choice = st.radio(
        t("export_scope_label"),
        options=[_scope_all, _scope_queue, _scope_anom],
        index=0,
        horizontal=True,
        help=t("export_scope_help"),
    )

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
st.subheader(t("export_preview_header"))

# Resolve scope
if scope_choice == _scope_all:
    scope_label = "all"
    episodes_to_export = list(result.episodes.values())
elif scope_choice == _scope_queue:
    scope_label = "review_queue"
    episodes_to_export = [
        ep for ep in result.episodes.values()
        if ep.verdict.value in ("REVIEW", "EXCLUDE")
    ]
else:  # anomalies only
    scope_label = "anomalies"
    episodes_to_export = [
        ep for ep in result.episodes.values()
        if ep.verdict.value == "EXCLUDE"
    ]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Episodes", len(episodes_to_export))
with col2:
    st.metric(t("export_metric_format"), format_choice)
with col3:
    st.metric(t("export_metric_scope"), scope_choice)

# Manual decision stats
user_verdicts = st.session_state.get("user_verdicts", {})
verdict_in_scope = {
    ep.episode_index: user_verdicts.get(ep.episode_index)
    for ep in episodes_to_export
    if user_verdicts.get(ep.episode_index)
}
if verdict_in_scope:
    keep_n = sum(1 for v in verdict_in_scope.values() if v.get("decision") == "KEEP")
    remove_n = sum(1 for v in verdict_in_scope.values() if v.get("decision") == "REMOVE")
    uncertain_n = sum(1 for v in verdict_in_scope.values() if v.get("decision") == "UNCERTAIN")
    st.info(
        t("export_verdicts_info", n=len(verdict_in_scope),
          keep=keep_n, remove=remove_n, uncertain=uncertain_n),
        icon="✅",
    )
else:
    st.caption(t("export_no_verdicts"))

st.divider()

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.subheader(t("export_download_header"))

if format_choice == "JSON":
    json_content = _generate_json_export(episodes_to_export, scope_label)
    file_name = f"rda_report_{scope_label}.json"

    st.download_button(
        label=t("export_download_json"),
        data=json_content,
        file_name=file_name,
        mime="application/json",
        use_container_width=True,
        type="primary",
    )

    with st.expander(t("export_preview_json")):
        preview_data = json.loads(json_content)
        # Show only the first 2 episodes
        if "episodes" in preview_data:
            preview_data["episodes"] = preview_data["episodes"][:2]
            preview_data["_note"] = t("export_preview_json_note", n=len(episodes_to_export))
        st.json(preview_data)

else:  # CSV
    csv_content = _generate_csv_export(episodes_to_export)
    file_name = f"rda_report_{scope_label}.csv"

    st.download_button(
        label=t("export_download_csv"),
        data=csv_content.encode("utf-8-sig"),  # BOM for Excel
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

    with st.expander(t("export_preview_csv")):
        lines = csv_content.split("\n")
        preview_lines = lines[:11]  # header + 10 rows
        st.code("\n".join(preview_lines), language="csv")
        if len(lines) > 11:
            st.caption(t("export_preview_csv_note", n=len(lines) - 1))

st.divider()

# ---------------------------------------------------------------------------
# Report docs
# ---------------------------------------------------------------------------
st.markdown(f"### {t('export_docs_header')}")

with st.expander(t("export_json_docs")):
    st.markdown(t("export_json_docs_body"))

with st.expander(t("export_csv_docs")):
    st.markdown(t("export_csv_docs_body"))


# ---------------------------------------------------------------------------
# 📦 Export cleaned dataset
# ---------------------------------------------------------------------------
st.divider()
st.header(t("export_clean_header"))
st.caption(t("export_clean_caption"))


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------
st.markdown(f"#### {t('export_strategy_header')}")

user_verdicts = st.session_state.get("user_verdicts", {})
total_episodes = len(result.episodes)

strategy_choice = st.radio(
    t("export_strategy_label"),
    options=["conservative", "aggressive", "custom"],
    format_func=lambda x: {
        "conservative": t("export_strategy_conservative"),
        "aggressive": t("export_strategy_aggressive"),
        "custom": t("export_strategy_custom"),
    }[x],
    index=0,
    key="clean_strategy_choice",
    label_visibility="collapsed",
)

# Custom mode: choose which AI verdicts to remove
custom_remove_verdicts = []
if strategy_choice == "custom":
    st.caption(t("export_custom_caption"))
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        remove_pass = st.checkbox(
            t("export_custom_pass"),
            value=False,
            key="custom_remove_pass",
            help=t("export_custom_pass_help"),
        )
    with col_c2:
        remove_review = st.checkbox(
            t("export_custom_review"),
            value=False,
            key="custom_remove_review",
            help=t("export_custom_review_help"),
        )
    with col_c3:
        remove_exclude = st.checkbox(
            t("export_custom_exclude"),
            value=True,
            key="custom_remove_exclude",
            help=t("export_custom_exclude_help"),
        )
    custom_remove_verdicts = []
    if remove_pass:
        custom_remove_verdicts.append("PASS")
    if remove_review:
        custom_remove_verdicts.append("REVIEW")
    if remove_exclude:
        custom_remove_verdicts.append("EXCLUDE")

# ---------------------------------------------------------------------------
# Episode classification
# ---------------------------------------------------------------------------
_keep_ids = []
_remove_ids = []       # user REMOVE + AI verdicts to remove
_uncertain_ids = []    # user UNCERTAIN
_undecided_ids = []    # undecided
_review_ids = []       # undecided with AI REVIEW

# Removal list for display (episode_id, pattern_type, score, ai_verdict)
_removal_info = []

for ep_id, ep_result in result.episodes.items():
    uv = user_verdicts.get(ep_id, {})
    decision = (uv.get("decision") or "").upper()
    ai_verdict = ep_result.verdict.value
    pattern = _get_pattern_type(ep_result)
    score = _compute_episode_score(ep_result)

    if decision == "KEEP":
        _keep_ids.append(ep_id)
    elif decision == "REMOVE":
        _remove_ids.append(ep_id)
        _removal_info.append((ep_id, pattern, score, "USER_REMOVE"))
    elif decision == "UNCERTAIN":
        _uncertain_ids.append(ep_id)
    else:
        # Undecided → follow AI verdict + strategy
        _undecided_ids.append(ep_id)
        if ai_verdict == "REVIEW":
            _review_ids.append(ep_id)

        # Apply strategy
        if strategy_choice == "conservative":
            should_remove = ai_verdict == "EXCLUDE"
        elif strategy_choice == "aggressive":
            should_remove = ai_verdict in ("EXCLUDE", "REVIEW")
        else:  # custom
            should_remove = ai_verdict in custom_remove_verdicts

        if should_remove:
            _remove_ids.append(ep_id)
            _removal_info.append((ep_id, pattern, score, ai_verdict))
        else:
            _keep_ids.append(ep_id)

# Undecided AI REVIEW count
_n_review_undecided = len(_review_ids)

# ---------------------------------------------------------------------------
# 📊 Cleaning preview panel
# ---------------------------------------------------------------------------
st.markdown(f"#### {t('export_preview_panel')}")

dataset_name = Path(result.dataset_info.path).name

# Category counts
n_keep = len(_keep_ids)
n_review = _n_review_undecided
n_remove = len(_remove_ids)

pct = lambda n: f"{n / total_episodes * 100:.1f}%" if total_episodes > 0 else "0.0%"

# Preview table
st.markdown(t("export_preview_dataset", name=dataset_name, n=total_episodes))

st.markdown(t(
    "export_preview_table",
    keep=n_keep, p_keep=pct(n_keep),
    review=n_review, p_review=pct(n_review),
    remove=n_remove, p_remove=pct(n_remove),
    undecided=len(_undecided_ids), p_undecided=pct(len(_undecided_ids)),
))

# Strategy execution summary
st.markdown(t("export_strategy_summary"))
summary_lines = [
    t("export_summary_remove", n=n_remove),
    t("export_summary_keep", n=n_keep),
    t("export_summary_usable", p=pct(n_keep)),
]
if _uncertain_ids:
    summary_lines.append(t("export_summary_uncertain", n=len(_uncertain_ids)))
st.markdown("\n".join(summary_lines))

# ---------------------------------------------------------------------------
# Removal list
# ---------------------------------------------------------------------------
if _removal_info:
    with st.expander(t("export_removal_list", n=len(_removal_info)), expanded=False):
        # Sort by score desc
        _removal_info_sorted = sorted(_removal_info, key=lambda x: x[2], reverse=True)

        _display_limit = 20
        _to_show = _removal_info_sorted[:_display_limit]

        # Build table
        _table_rows = (
            f"| {t('export_removal_col_id')} | {t('export_removal_col_pattern')} | "
            f"{t('export_removal_col_score')} | {t('export_removal_col_source')} |\n"
            f"|---:|:---|---:|:---|\n"
        )
        for _ep_id, _pattern, _score, _source in _to_show:
            _source_label = {
                "USER_REMOVE": t("export_source_user"),
                "EXCLUDE": "🤖 AI EXCLUDE",
                "REVIEW": "🤖 AI REVIEW",
                "PASS": "🤖 AI PASS",
            }.get(_source, _source)
            _table_rows += f"| {_ep_id} | {_pattern or '-'} | {_score} | {_source_label} |\n"

        st.markdown(_table_rows)

        _remaining = len(_removal_info_sorted) - _display_limit
        if _remaining > 0:
            st.caption(t("export_removal_remaining", n=_remaining))

# ---------------------------------------------------------------------------
# Output config & export button
# ---------------------------------------------------------------------------
st.markdown(f"#### {t('export_config_sub')}")

col_cfg1, col_cfg2 = st.columns([1, 2])
with col_cfg1:
    _keep_opt = t("export_uncertainty_keep")
    _remove_opt = t("export_uncertainty_remove")
    uncertainty_choice = st.radio(
        t("export_uncertainty_label"),
        options=[_keep_opt, _remove_opt],
        index=0,
        horizontal=True,
        help=t("export_uncertainty_help"),
        key="clean_uncertainty_strategy",
    )
    _uncertainty_strategy = "keep" if uncertainty_choice == _keep_opt else "remove"

with col_cfg2:
    default_output = str(_PROJECT_ROOT / "clean_export")
    output_path_str = st.text_input(
        t("export_output_path"),
        value=default_output,
        key="clean_output_path",
        help=t("export_output_path_help"),
    )

# Export button
if st.button(
    t("export_run_btn"),
    type="primary",
    use_container_width=True,
    key="btn_clean_export",
):
    from rda.export.clean_export import CleanDatasetExporter

    # AI verdicts
    _ai_verdicts = {
        ep_id: ep_result.verdict.value
        for ep_id, ep_result in result.episodes.items()
    }

    # User verdicts (int keys)
    _user_verdicts_int = {
        int(k): v for k, v in user_verdicts.items()
    }

    _output_path = Path(output_path_str)

    # cleaning_strategy parameter
    if strategy_choice == "conservative":
        _cleaning_strategy = "conservative"
    elif strategy_choice == "aggressive":
        _cleaning_strategy = "aggressive"
    else:
        _cleaning_strategy = custom_remove_verdicts  # list

    progress_bar = st.progress(0, text=t("export_running"))

    def _progress_cb(step, total, msg):
        progress_bar.progress(step / total, text=msg)

    try:
        exporter = CleanDatasetExporter(
            source_path=result.dataset_info.path,
            output_path=_output_path,
            user_verdicts=_user_verdicts_int,
            uncertainty_strategy=_uncertainty_strategy,
            ai_verdicts=_ai_verdicts,
            cleaning_strategy=_cleaning_strategy,
        )
        export_report = exporter.export(progress_callback=_progress_cb)
        progress_bar.progress(1.0, text=t("export_done"))
        st.success(t(
            "export_success",
            path=_output_path,
            kept=export_report.kept_episodes,
            total=export_report.total_episodes,
            removed=export_report.removed_episodes,
            frames=export_report.total_frames,
            strategy=export_report.cleaning_strategy,
        ))

        # Detailed stats
        with st.expander(t("export_stats")):
            st.json(export_report.to_dict())

    except Exception as e:
        progress_bar.empty()
        st.error(t("export_failed", err=e))


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------
