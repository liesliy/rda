"""Page 1: Upload · 上传数据集。

支持 LeRobot 格式数据集上传，预览基本信息（episode 数、frame 数、DOF、FPS）。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 辅助函数（必须在页面逻辑之前定义）
# ---------------------------------------------------------------------------

def _load_dataset(path_str: str) -> None:
    """从本地路径加载数据集。"""
    from rda.io.lerobot_loader import load_lerobot_dataset

    with st.spinner("正在加载数据集..."):
        try:
            dataset_info = load_lerobot_dataset(path_str)
            st.session_state.dataset_info = dataset_info
            st.session_state.dataset_path = path_str
            # 清除旧的审计结果
            st.session_state.audit_result = None
            st.session_state.dataset_report = None
            st.session_state.episodes_df = None
            st.success(f"成功加载 {dataset_info.num_episodes} 个 episodes")
        except Exception as e:
            st.error(f"加载失败：{e}")


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


st.title("📤 上传数据集")
st.caption("支持 LeRobot 格式数据集，上传后可进行质量审计")

# ---------------------------------------------------------------------------
# 平台选择
# ---------------------------------------------------------------------------
st.subheader("数据集来源")

platform = st.selectbox(
    "选择平台",
    options=["armnetbench", "droid", "custom"],
    index=2 if st.session_state.get("platform") == "custom"
    else (0 if st.session_state.get("platform") == "armnetbench" else 1),
    help="armnetbench / droid 为预置平台，custom 为自定义 LeRobot 格式数据集",
)
st.session_state.platform = platform

# ---------------------------------------------------------------------------
# 数据集加载方式
# ---------------------------------------------------------------------------
col1, col2 = st.tabs(["📁 本地路径", "📤 上传文件夹"])

with col1:
    st.markdown("输入本地 LeRobot 数据集目录路径")
    local_path = st.text_input(
        "数据集路径",
        value=st.session_state.get("dataset_path") or "",
        placeholder="/path/to/lerobot/dataset",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button("加载数据集", type="primary", key="load_local"):
            if not local_path:
                st.error("请输入数据集路径")
            else:
                _load_dataset(local_path)

with col2:
    st.markdown("上传数据集目录（多个文件）")
    uploaded_files = st.file_uploader(
        "选择数据集文件（支持 parquet / json / 等 LeRobot 格式文件）",
        accept_multiple_files=True,
        help="请选择数据集中的所有文件，将保存到临时目录",
    )

    if uploaded_files and st.button("保存并加载", type="primary", key="load_uploaded"):
        _handle_uploaded_files(uploaded_files)


# ---------------------------------------------------------------------------
# 数据集预览
# ---------------------------------------------------------------------------
if st.session_state.dataset_info is not None:
    st.divider()
    st.subheader("✅ 数据集信息")

    info = st.session_state.dataset_info

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Episode 数量", info.num_episodes)
    with col2:
        st.metric("总帧数", f"{info.total_frames:,}")
    with col3:
        # 估算 DOF（action 的维度数）
        dof = _estimate_dof(info)
        st.metric("动作维度 (DOF)", dof if dof else "—")
    with col4:
        # 估算 FPS（仅在元数据明确提供时显示，否则 Unknown）
        fps = _estimate_fps(info)
        st.metric("帧率 (FPS)", f"{fps:.1f}" if fps else "Unknown")

    # 模态与动作
    if info.modalities:
        with st.expander(f"📷 观测模态 ({len(info.modalities)})"):
            st.write(", ".join(info.modalities))

    if info.action_keys:
        with st.expander(f"🦾 动作空间 ({len(info.action_keys)})"):
            st.write(", ".join(info.action_keys))

    if info.meta:
        with st.expander("📝 元信息"):
            st.json(info.meta)

    st.success(
        f"数据集加载成功！共 {info.num_episodes} 个 episodes，"
        f"{info.total_frames:,} 帧。",
        icon="🎉",
    )

    st.page_link(
        "pages/2_Audit.py",
        label="下一步：运行审计 →",
        icon="🔍",
    )



