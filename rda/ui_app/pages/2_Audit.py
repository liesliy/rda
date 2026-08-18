"""Page 2: Audit · 运行审计。

调用 DatasetAuditor.audit_dataset() 生成审计结果，显示进度条，
完成后自动跳转/提示进入 Health Overview 页面。
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from components.common import (  # noqa: E402
    build_episodes_dataframe,
    compute_dhi,
)

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _run_audit() -> None:
    """执行审计，带进度条。"""
    from rda.audit.dataset_audit import DatasetAuditor, DatasetAuditResult
    from rda.io.lerobot_loader import iter_episodes

    st.session_state.audit_in_progress = True

    progress_bar = st.progress(0, text="准备中...")
    status_text = st.empty()

    try:
        total = info.num_episodes
        auditor = DatasetAuditor()
        result = DatasetAuditResult(dataset_info=info)

        status_text.text("正在加载数据集并执行审计...")

        # 逐 episode 审计，更新进度
        for i, episode in enumerate(iter_episodes(info.path)):
            ep_result = auditor.episode_auditor.audit(episode)
            result.episodes[episode.episode_index] = ep_result

            progress = (i + 1) / total if total > 0 else 1.0
            progress_bar.progress(
                min(progress, 1.0),
                text=f"审计中... Episode {i + 1}/{total} ({progress * 100:.0f}%)",
            )

        # 计算 verdict 分布
        result.compute_verdict_counts()

        # 保存到 session state
        st.session_state.audit_result = result

        # 预计算报告数据
        st.session_state.dataset_report = compute_dhi(result)
        st.session_state.episodes_df = build_episodes_dataframe(result)

        # 保存审计快照到历史记录
        try:
            from rda.report.audit_history import save_audit_snapshot
            snapshot_path = save_audit_snapshot(result, info.path)
            st.session_state.dataset_path = info.path
        except Exception:
            pass  # 快照保存失败不影响主流程

        progress_bar.progress(1.0, text="审计完成！")
        status_text.text(f"✅ 审计完成，共 {len(result.episodes)} 个 episodes")

        st.success("审计完成！快照已保存，可在 Audit History 查看趋势变化", icon="🎉")
        st.page_link(
            "pages/3_Health_Overview.py",
            label="查看健康概览 →",
            icon="💚",
        )

    except Exception as e:
        st.error(f"审计失败：{e}")
        import traceback
        with st.expander("详细错误信息"):
            st.code(traceback.format_exc())
    finally:
        st.session_state.audit_in_progress = False



st.title("🔍 运行审计")
st.caption("对数据集执行完整的质量审计，生成逐 episode 的检测结果")

# ---------------------------------------------------------------------------
# 检查前置条件
# ---------------------------------------------------------------------------
if st.session_state.dataset_info is None:
    st.warning("⚠️ 请先在 Upload 页面加载数据集", icon="📤")
    st.page_link("pages/1_Upload.py", label="前往 Upload 页面 →", icon="📤")
    st.stop()

info = st.session_state.dataset_info
st.write(f"**数据集**: `{info.path}`")
st.write(f"**共 {info.num_episodes} 个 episodes · {info.total_frames:,} 帧**")

# ---------------------------------------------------------------------------
# 审计配置（只读：展示已启用的检查层）
# ---------------------------------------------------------------------------
st.subheader("已启用的审计检查层")

st.info(
    "审计流程包含以下检查层，全部默认启用：\n\n"
    "**Layer 1A · 完整性检查** — 缺失帧、NaN/Inf、时间戳、格式一致性等确定性检查\n\n"
    "**Layer 1B · 行为质量检测** — 基于参考分布的统计异常检测（运动、时序、分布等）\n\n"
    "**Layer 2 · 诊断与归因** — Pattern Type 识别 + 原因分析 + 建议"
)

st.caption("V0.1 暂不支持 Layer 3 数据治理功能；审计时默认启用所有检查层。")

# ---------------------------------------------------------------------------
# 运行审计
# ---------------------------------------------------------------------------
st.divider()

if st.session_state.audit_result is not None:
    st.info(f"✅ 已完成审计，共 {info.num_episodes} 个 episodes", icon="✅")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重新运行审计", type="secondary"):
            st.session_state.audit_result = None
            st.session_state.dataset_report = None
            st.session_state.episodes_df = None
            st.rerun()
    with col2:
        st.page_link(
            "pages/3_Health_Overview.py",
            label="查看健康概览 →",
            icon="💚",
        )

    # 快速结果摘要
    result = st.session_state.audit_result
    counts = {v.value: c for v, c in result.verdict_counts.items()}

    st.subheader("审计结果摘要")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 KEEP", counts.get("PASS", 0),
                  f"{counts.get('PASS', 0) / max(info.num_episodes, 1) * 100:.1f}%")
    with col2:
        st.metric("🟡 REVIEW", counts.get("REVIEW", 0),
                  f"{counts.get('REVIEW', 0) / max(info.num_episodes, 1) * 100:.1f}%")
    with col3:
        st.metric("🔴 REMOVE", counts.get("EXCLUDE", 0),
                  f"{counts.get('EXCLUDE', 0) / max(info.num_episodes, 1) * 100:.1f}%")

else:
    if st.button("▶️ 开始审计", type="primary", disabled=st.session_state.audit_in_progress,
                 use_container_width=True):
        _run_audit()


# ---------------------------------------------------------------------------
# 审计执行
# ---------------------------------------------------------------------------

