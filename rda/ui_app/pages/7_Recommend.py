"""Page 7: Recommend · 优化建议。

基于已上传数据集的时间结构指标，生成数据优化建议。
本地计算指标 → 仅上传 <1KB 聚合统计到 RDA API 做规则评估
（可通过 RDA_API_URL 环境变量指向私有部署）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

st.title("🧭 Recommend · 数据优化建议")
st.caption(
    "基于数据集时间结构（idle 比例、active 段分布、有效窗口占比）"
    "生成保守的、带实验证据等级的优化建议。"
)

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
info = st.session_state.get("dataset_info")
dataset_path = st.session_state.get("dataset_path")

if info is None:
    st.warning("⚠️ 请先在 Upload 页面上传数据集", icon="📤")
    st.page_link("pages/1_Upload.py", label="前往 Upload 页面 →", icon="📤")
    st.stop()

if not dataset_path:
    st.warning("⚠️ 未找到数据集路径，请重新上传数据集。", icon="⚠️")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# 策略选择
# ---------------------------------------------------------------------------
policy_label = st.radio(
    "目标模型架构",
    options=[
        "frame-wise — MLP / BC（逐帧策略）",
        "temporal — ACT / Diffusion Policy / Transformer",
    ],
    index=1,
    help="不同架构对 idle 帧和有效窗口的容忍度不同，建议按实际训练模型选择。",
)

policy_name = policy_label.split(" — ")[0]

st.info(
    "🔒 隐私说明：原始 episode 数据不会上传。仅在本地计算指标后，"
    "将 <1KB 的聚合统计发送到 RDA API 进行规则评估。",
    icon="🔒",
)

# ---------------------------------------------------------------------------
# 运行推荐
# ---------------------------------------------------------------------------
if st.button("🚀 生成优化建议", type="primary", use_container_width=True):
    from rda.io.lerobot_loader import iter_episodes
    from rda.recommend.api_client import run_recommendation
    from rda.recommend.formatter import format_recommendation_text
    from rda.recommend.types import TargetPolicy

    target_policy = TargetPolicy.from_cli_name(policy_name)

    progress_bar = st.progress(0, text="正在本地计算时间结构指标...")
    status_text = st.empty()

    try:
        def _progress(step, total, msg):
            ratio = step / total if total else 1.0
            progress_bar.progress(min(ratio, 1.0), text=f"{step}/{total}: {msg}")

        with st.spinner("调用 RDA 规则引擎（rules API）..."):
            result = run_recommendation(
                iter_episodes(str(dataset_path)),
                target_policy=target_policy,
                total_episodes=info.num_episodes,
                total_frames=info.total_frames,
                progress_callback=_progress,
            )

        progress_bar.progress(1.0, text="完成！")

        rules_ver = getattr(result, "rules_version", None)
        if rules_ver:
            st.caption(f"规则版本：rules v{rules_ver}")

        # 文本报告
        st.subheader("建议报告")
        st.code(format_recommendation_text(result), language="text")

        # 结构化建议卡片
        st.subheader("逐条建议")
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
            }.get(action_name, "⚪")

            with st.expander(
                f"{color} {rec.title}  ·  {action_name}  ·  置信度: {conf_name}"
            ):
                st.write(rec.summary)
                if rec.expected_impact:
                    st.markdown(f"**预期影响：** {rec.expected_impact}")
                if rec.details:
                    st.markdown("**细节：**")
                    for d in rec.details:
                        st.markdown(f"- {d}")
                if rec.caveats:
                    st.markdown("**注意事项：**")
                    for c in rec.caveats:
                        st.markdown(f"- ⚠️ {c}")

        # 原始 JSON
        with st.expander("查看原始 JSON"):
            st.json(result.to_dict())

    except Exception as e:  # noqa: BLE001
        progress_bar.empty()
        status_text.empty()
        st.error(f"生成建议失败：{e}", icon="❌")
        st.caption(
            "提示：如果网络不可用，可检查 RDA_API_URL 环境变量；"
            "有本地缓存时会自动降级使用缓存结果。"
        )

st.divider()
st.caption(
    "RDA 是数据质量诊断 + 低风险优化建议工具，不保证成功率提升。"
    "任何裁剪操作都请在保留集上验证后再用于正式训练。"
)
