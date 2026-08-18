"""RDA Streamlit UI — 主应用入口。

5 页面流程：
  1. Upload          上传数据集
  2. Audit           运行审计
  3. Health Overview 数据集健康概览（首页）
  4. Episode Explorer 逐 episode 查看
  5. Export          导出报告

通过 st.navigation 实现导航，st.session_state 保存审计结果。
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

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RDA · 机器人数据质量审计",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 国际化（中文为主，Pattern Type 保留英文）
# ---------------------------------------------------------------------------
T = {
    # 通用
    "app_title": "RDA · 机器人数据质量审计",
    "app_subtitle": "Robot Data Assurance — 数据质量审计与治理工具",
    "not_uploaded": "⚠️ 请先上传数据集",
    "not_audited": "⚠️ 请先运行审计",
    "go_upload": "前往 Upload 页面 →",
    "go_audit": "前往 Audit 页面 →",

    # Training Readiness
    "training_readiness": "训练就绪度",
    "ready": "可直接用于训练",
    "conditionally_ready": "条件就绪",
    "not_ready": "不建议训练",

    # DHI
    "dhi": "数据健康指数 (DHI)",
    "dhi_note": "基于数据完整性与行为一致性的相对评估，非绝对评分",
    "grade_excellent": "优秀",
    "grade_good": "良好",
    "grade_fair": "一般",
    "grade_poor": "较差",

    # 维度
    "dim_integrity": "数据完整性 (Integrity)",
    "dim_temporal": "时间质量 (Temporal)",
    "dim_motion": "运动质量 (Motion)",
    "dim_consistency": "行为一致性 (Consistency)",

    # Verdict
    "verdict_pass": "建议保留",
    "verdict_review": "建议人工审核",
    "verdict_exclude": "建议排除",

    # Pattern Type (保留英文，附中文说明)
    "pattern_stuck": "Stuck · 卡死",
    "pattern_jittery": "Jittery · 抖动",
    "pattern_inefficient": "Inefficient · 低效",
    "pattern_frozen": "Frozen · 冻结",
    "pattern_unusual": "Unusual · 轨迹异常",

    # 问题等级
    "sev_critical": "严重 (Critical)",
    "sev_warning": "警告 (Warning)",
    "sev_info": "提示 (Info)",
}

# ---------------------------------------------------------------------------
# Session State 初始化
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 导航定义
# ---------------------------------------------------------------------------
upload_page = st.Page("pages/1_Upload.py", title="Upload · 上传数据集", icon="📤")
audit_page = st.Page("pages/2_Audit.py", title="Audit · 运行审计", icon="🔍")
health_page = st.Page("pages/3_Health_Overview.py", title="Health Overview · 健康概览", icon="💚")
episode_page = st.Page("pages/4_Episode_Explorer.py", title="Episode Explorer · 逐集查看", icon="📋")
export_page = st.Page("pages/5_Export.py", title="Export · 导出报告", icon="📥")
history_page = st.Page("pages/6_History.py", title="Audit History · 审计历史", icon="📈")

pg = st.navigation({
    "数据流程": [upload_page, audit_page],
    "审计结果": [health_page, episode_page, export_page],
    "趋势分析": [history_page],
})

# 侧边栏头部
with st.sidebar:
    st.title("🤖 RDA")
    st.caption(T["app_subtitle"])
    st.divider()

    if st.session_state.dataset_info is not None:
        info = st.session_state.dataset_info
        st.metric("Episodes", info.num_episodes)
        st.metric("总帧数", f"{info.total_frames:,}")
        if hasattr(info, "modalities") and info.modalities:
            st.caption(f"模态: {', '.join(info.modalities[:3])}" + ("..." if len(info.modalities) > 3 else ""))
    else:
        st.info("尚未加载数据集", icon="📭")

pg.run()
