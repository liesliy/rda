"""Page 8: Acceptance Summary · one-page evidence for the data-acceptance party.

REQ-5 (v0.8.0). Folds four things the acceptance party otherwise has to
dig out of separate reports into a single page:

- verdict distribution (how much is usable / needs review / must go)
- dataset-relative P10 / P50 / P90 baselines (what "typical" means here)
- runtime outliers via the Tukey IQR fence [SLE]
- what was **not** checked — an absent dimension must never read as
  "checked and fine"

The page deliberately issues no accept/reject decision. RDA measures and
presents; the acceptance judgement belongs to the human reviewer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rda.ui_app.i18n import get_lang, t  # noqa: E402

from components.common import compute_dhi  # noqa: E402

st.title(t("acc_title"))
st.caption(t("acc_caption"))

# ---------------------------------------------------------------------------
# Preconditions — reuse the audit already in session state
# ---------------------------------------------------------------------------
if st.session_state.audit_result is None:
    st.warning(t("not_audited"), icon="🔍")
    st.page_link("pages/2_Audit.py", label=t("go_audit"), icon="🔍")
    st.stop()

result = st.session_state.audit_result

if st.session_state.dataset_report is None:
    st.session_state.dataset_report = compute_dhi(result, lang=get_lang())
report = st.session_state.dataset_report or {}

# Imported here so a missing symbol surfaces as a page error, not an
# import-time crash for the whole app.
from rda.report.acceptance import build_acceptance_summary  # noqa: E402
from rda.report.json_report import skipped_by_missing_dep  # noqa: E402

quality = {
    "dhi": report.get("dhi"),
    "grade": report.get("grade"),
    "training_readiness": report.get("training_readiness"),
}

summary = build_acceptance_summary(
    result,
    quality=quality,
    not_checked=skipped_by_missing_dep(result),
)

# ---------------------------------------------------------------------------
# 1. Dataset identity
# ---------------------------------------------------------------------------
st.subheader(t("acc_identity"))
ds = summary["dataset"]
c1, c2, c3, c4 = st.columns(4)
c1.metric(t("acc_episodes"), ds["num_episodes"])
c2.metric(t("acc_frames"), ds["total_frames"])
c3.metric(t("acc_fps"), ds.get("fps") or "—")
c4.metric(t("acc_robot"), ds.get("robot") or "—")
st.caption(f"{t('acc_path')}: `{ds['path']}`")

# ---------------------------------------------------------------------------
# 2. Verdict distribution
# ---------------------------------------------------------------------------
st.subheader(t("acc_verdict"))
vd = summary["verdict_distribution"]
c1, c2, c3, c4 = st.columns(4)
c1.metric(t("acc_verdict_pass"), vd["pass"])
c2.metric(t("acc_verdict_review"), vd["review"])
c3.metric(t("acc_verdict_exclude"), vd["exclude"])
c4.metric(
    t("acc_pass_rate"),
    f"{vd['pass_rate']:.1%}" if vd["pass_rate"] is not None else "—",
)

# ---------------------------------------------------------------------------
# 3. Quality (only when the DHI report produced one)
# ---------------------------------------------------------------------------
if quality.get("dhi") is not None:
    st.subheader(t("acc_quality"))
    c1, c2 = st.columns(2)
    c1.metric(t("acc_dhi"), quality["dhi"])
    c2.metric(t("acc_grade"), quality["grade"])
    if quality.get("training_readiness"):
        st.caption(f"{t('acc_readiness')}: {quality['training_readiness']}")

st.divider()

# ---------------------------------------------------------------------------
# 4. Dataset-relative baselines (P10 / P50 / P90)
# ---------------------------------------------------------------------------
st.subheader(t("acc_baseline"))
st.caption(t("acc_baseline_caption"))

rows = []
any_unavailable = False
for key, b in summary["baseline_percentiles"].items():
    n = b.get("available_episodes", 0)
    if n == 0:
        any_unavailable = True
        rows.append({
            t("acc_metric"): key,
            t("acc_p10"): "—",
            t("acc_p50"): "—",
            t("acc_p90"): "—",
            t("acc_min"): "—",
            t("acc_max"): "—",
            t("acc_episodes_n"): 0,
        })
        continue

    def _fmt(v: float) -> str:
        return f"{v:.4g}"

    rows.append({
        t("acc_metric"): key,
        t("acc_p10"): _fmt(b["p10"]),
        t("acc_p50"): _fmt(b["p50"]),
        t("acc_p90"): _fmt(b["p90"]),
        t("acc_min"): _fmt(b["min"]),
        t("acc_max"): _fmt(b["max"]),
        t("acc_episodes_n"): n,
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
if any_unavailable:
    st.caption(f"ℹ️ {t('acc_unavailable')}")

# ---------------------------------------------------------------------------
# 5. Runtime outliers (Tukey IQR fence)
# ---------------------------------------------------------------------------
st.subheader(t("acc_outliers"))
st.caption(t("acc_outliers_caption"))

out = summary["runtime_outliers"]

if out.get("available_episodes", 0) == 0:
    st.info(t("acc_outlier_na"), icon="ℹ️")
elif out.get("count", 0) == 0:
    st.success(t("acc_outlier_none"), icon="✅")
    if out.get("degenerate"):
        # Zero outliers because there is no spread at all — say so, or the
        # green tick reads as "checked and clean".
        st.warning(t("acc_degenerate"), icon="⚠️")
else:
    st.caption(
        t(
            "acc_fence",
            lo=f"{out['lower_fence']:.2f}",
            hi=f"{out['upper_fence']:.2f}",
            q1=f"{out['q1']:.2f}",
            q3=f"{out['q3']:.2f}",
            iqr=f"{out['iqr']:.2f}",
        )
    )
    st.write(
        t("acc_outlier_count", n=out["count"], total=out["available_episodes"])
    )
    if out.get("degenerate"):
        st.warning(t("acc_degenerate"), icon="⚠️")

    df = pd.DataFrame(out["episodes"])
    df["direction"] = df["direction"].map({
        "long": t("acc_dir_long"),
        "short": t("acc_dir_short"),
    })
    df.columns = [
        t("acc_col_episode"),
        t("acc_col_duration"),
        t("acc_col_direction"),
    ]
    st.dataframe(df, use_container_width=True, hide_index=True)

    if out.get("truncated"):
        st.caption(t("acc_truncated", n=len(out["episodes"])))

# ---------------------------------------------------------------------------
# 6. Not checked — the dimension that must never read as "fine"
# ---------------------------------------------------------------------------
st.subheader(t("acc_not_checked"))
not_checked = summary.get("not_checked") or {}
if not_checked:
    st.warning(t("acc_not_checked_hint"), icon="⚠️")
    st.json(not_checked)
else:
    st.caption(f"✅ {t('acc_not_checked_none')}")

# ---------------------------------------------------------------------------
# 7. Calibration context
# ---------------------------------------------------------------------------
with st.expander(t("acc_calibration")):
    st.caption(t("acc_calibration_caption"))
    cal = summary["calibration"]
    st.code(", ".join(cal["tier1_metrics"]), language="text")
    st.caption(f"ρ = {cal['tier1_rho_vs_full_set']}")

# ---------------------------------------------------------------------------
# 8. Export + disclaimer
# ---------------------------------------------------------------------------
st.divider()
st.download_button(
    t("acc_download"),
    data=json.dumps(summary, ensure_ascii=False, indent=2),
    file_name=t("acc_download_name"),
    mime="application/json",
    use_container_width=True,
)
st.caption(t("acc_disclaimer"))
