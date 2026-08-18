"""Page 1: Upload · load a dataset.

Supports LeRobot-format datasets (local path or folder upload) and shows
a basic preview (episodes, frames, DOF, FPS).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rda.ui_app.i18n import t

# ---------------------------------------------------------------------------
# 辅助函数（必须在页面逻辑之前定义）
# ---------------------------------------------------------------------------

def _load_dataset(path_str: str) -> None:
    """从本地路径加载数据集。"""
    from rda.io.lerobot_loader import load_lerobot_dataset

    with st.spinner(t("upload_loading")):
        try:
            dataset_info = load_lerobot_dataset(path_str)
            st.session_state.dataset_info = dataset_info
            st.session_state.dataset_path = path_str
            # 清除旧的审计结果
            st.session_state.audit_result = None
            st.session_state.dataset_report = None
            st.session_state.episodes_df = None
            st.success(t("upload_loaded_ok", n=dataset_info.num_episodes))
        except Exception as e:
            st.error(t("upload_load_err", err=e))


def _handle_uploaded_files(files) -> None:
    """处理上传的文件，保存到临时目录并加载。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="rda_upload_"))
    for f in files:
        file_path = tmp_dir / f.name
        with open(file_path, "wb") as out:
            out.write(f.getbuffer())

    _load_dataset(str(tmp_dir))


def _estimate_dof(info) -> int:
    """估算动作维度 DOF。"""
    if not info.action_keys:
        return 0
    # 从 meta 中尝试获取
    if hasattr(info, "meta") and info.meta:
        dof = info.meta.get("action_dim") or info.meta.get("dof")
        if dof:
            return int(dof)
    # 简单按 key 数量估算
    return len(info.action_keys)


def _estimate_fps(info) -> float | None:
    """估算 FPS。仅在元数据明确提供时返回值，不做猜测。"""
    if hasattr(info, "meta") and info.meta:
        fps = info.meta.get("fps")
        if fps:
            return float(fps)
    # 不根据 episode 数和总帧数猜测 FPS（之前假设每 episode 10 秒的逻辑已移除）
    return None


st.title(t("upload_title"))
st.caption(t("upload_caption"))

# ---------------------------------------------------------------------------
# 平台选择
# ---------------------------------------------------------------------------
st.subheader(t("upload_source"))

platform = st.selectbox(
    t("upload_platform"),
    options=["armnetbench", "droid", "custom"],
    index=2 if st.session_state.get("platform") == "custom"
    else (0 if st.session_state.get("platform") == "armnetbench" else 1),
    help=t("upload_platform_help"),
)
st.session_state.platform = platform

# ---------------------------------------------------------------------------
# 数据集加载方式
# ---------------------------------------------------------------------------
col1, col2 = st.tabs([t("upload_tab_local"), t("upload_tab_files")])

with col1:
    st.markdown(t("upload_local_hint"))
    local_path = st.text_input(
        t("upload_path_label"),
        value=st.session_state.get("dataset_path") or "",
        placeholder="/path/to/lerobot/dataset",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button(t("upload_load_btn"), type="primary", key="load_local"):
            if not local_path:
                st.error(t("upload_path_empty_err"))
            else:
                _load_dataset(local_path)

with col2:
    st.markdown(t("upload_files_hint"))
    uploaded_files = st.file_uploader(
        t("upload_uploader_label"),
        accept_multiple_files=True,
        help=t("upload_uploader_help"),
    )

    if uploaded_files and st.button(t("upload_save_load_btn"), type="primary", key="load_uploaded"):
        _handle_uploaded_files(uploaded_files)


# ---------------------------------------------------------------------------
# 数据集预览
# ---------------------------------------------------------------------------
if st.session_state.dataset_info is not None:
    st.divider()
    st.subheader(t("upload_info_header"))

    info = st.session_state.dataset_info

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t("upload_metric_eps"), info.num_episodes)
    with col2:
        st.metric(t("upload_metric_frames"), f"{info.total_frames:,}")
    with col3:
        # 估算 DOF（action 的维度数）
        dof = _estimate_dof(info)
        st.metric(t("upload_metric_dof"), dof if dof else "—")
    with col4:
        # 估算 FPS（仅在元数据明确提供时显示，否则 Unknown）
        fps = _estimate_fps(info)
        st.metric(t("upload_metric_fps"), f"{fps:.1f}" if fps else "Unknown")

    # 模态与动作
    if info.modalities:
        with st.expander(t("upload_modalities", n=len(info.modalities))):
            st.write(", ".join(info.modalities))

    if info.action_keys:
        with st.expander(t("upload_actions", n=len(info.action_keys))):
            st.write(", ".join(info.action_keys))

    if info.meta:
        with st.expander(t("upload_meta")):
            st.json(info.meta)

    st.success(
        t("upload_success", eps=info.num_episodes, frames=f"{info.total_frames:,}"),
        icon="🎉",
    )

    st.page_link(
        "pages/2_Audit.py",
        label=t("upload_next_audit"),
        icon="🔍",
    )
