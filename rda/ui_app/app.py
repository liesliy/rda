"""RDA Streamlit UI — main app entry.

7-page workflow:
  1. Upload          load a dataset
  2. Audit           run the quality audit
  3. Health Overview dataset-level summary (landing page)
  4. Episode Explorer per-episode browsing
  5. Export          export reports / cleaned dataset
  6. History         audit trend tracking
  7. Recommend       optimization advice (rules API)

Navigation via st.navigation; audit state kept in st.session_state.
UI language (zh/en) is selectable in the sidebar; the selector lives
here because st.navigation re-runs this entrypoint on every interaction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 确保可以 import rda 以及 rda.ui_app.pages 下的 `from components.common import ...`
# __file__ 位于 site-packages/rda/ui_app/app.py，parents[0] = site-packages/rda
_PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# pages/ 子模块用 `from components.common import ...` 需要 rda/ui_app 在 sys.path 上
_UI_APP_DIR = Path(__file__).resolve().parents[0] / "ui_app" if Path(__file__).resolve().parents[0].name == "rda" else Path(__file__).resolve().parent
if str(_UI_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_UI_APP_DIR))

from rda.ui_app.i18n import get_lang, render_lang_selector, t  # noqa: E402

# ---------------------------------------------------------------------------
# Session State 初始化
# ---------------------------------------------------------------------------
if "rda_lang" not in st.session_state:
    st.session_state.rda_lang = "zh"
if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = None
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None
if "platform" not in st.session_state:
    st.session_state.platform = "custom"
if "audit_result" not in st.session_state:
    st.session_state.audit_result = None
if "dataset_report" not in st.session_state:
    st.session_state.dataset_report = None
if "episodes_df" not in st.session_state:
    st.session_state.episodes_df = None
if "audit_in_progress" not in st.session_state:
    st.session_state.audit_in_progress = False
if "user_verdicts" not in st.session_state:
    # 用户人工审核决策字典
    # key: episode_id (int)
    # value: {"decision": "KEEP"|"REMOVE"|"UNCERTAIN"|None,
    #         "notes": str,
    #         "reviewed_at": datetime|None}
    st.session_state.user_verdicts = {}
if "selected_episode" not in st.session_state:
    st.session_state.selected_episode = None

_lang = get_lang()

# 语言切换后，使依赖语言的缓存数据失效（审计报告 / episode 表）
if st.session_state.get("_report_lang") != _lang:
    if st.session_state.audit_result is not None:
        st.session_state.dataset_report = None
        st.session_state.episodes_df = None
    st.session_state._report_lang = _lang

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=t("page_title"),
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 导航定义
# ---------------------------------------------------------------------------
upload_page = st.Page("pages/1_Upload.py", title=t("nav_upload"), icon="📤")
audit_page = st.Page("pages/2_Audit.py", title=t("nav_audit"), icon="🔍")
health_page = st.Page("pages/3_Health_Overview.py", title=t("nav_health"), icon="💚")
episode_page = st.Page("pages/4_Episode_Explorer.py", title=t("nav_episode"), icon="📋")
export_page = st.Page("pages/5_Export.py", title=t("nav_export"), icon="📥")
history_page = st.Page("pages/6_History.py", title=t("nav_history"), icon="📈")
recommend_page = st.Page("pages/7_Recommend.py", title=t("nav_recommend"), icon="🧭")

pg = st.navigation({
    t("nav_group_flow"): [upload_page, audit_page],
    t("nav_group_results"): [health_page, episode_page, export_page],
    t("nav_group_recommend"): [recommend_page],
    t("nav_group_trend"): [history_page],
})

# 侧边栏头部
with st.sidebar:
    st.title("🤖 RDA")
    st.caption(t("app_subtitle"))
    render_lang_selector()
    st.divider()

    if st.session_state.dataset_info is not None:
        info = st.session_state.dataset_info
        st.metric("Episodes", info.num_episodes)
        st.metric(t("sidebar_total_frames"), f"{info.total_frames:,}")
        if hasattr(info, "modalities") and info.modalities:
            st.caption(t("sidebar_modalities") + f": {', '.join(info.modalities[:3])}" + ("..." if len(info.modalities) > 3 else ""))
    else:
        st.info(t("sidebar_no_dataset"), icon="📭")

pg.run()
