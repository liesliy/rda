"""Page 5: Export · 导出报告。

支持：
- JSON 格式（完整审计结果）
- CSV 格式（episode 列表）
- 导出范围：全量 / 仅 Review Queue / 仅异常 episodes
- 清洗导出：策略选择 + 预览 + 干净数据集导出
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

# ---------------------------------------------------------------------------
# Helper functions (moved to top for Streamlit compatibility)
# ---------------------------------------------------------------------------

def _compute_episode_score(ep_result) -> float:
    """计算 episode 的偏差得分 (0-100，越高越异常)。"""
    from rda.metrics.base import MetricAvailability
    import numpy as np

    scores = []
    for m_name, m in ep_result.metrics.items():
        if m.availability == MetricAvailability.AVAILABLE:
            scores.append(m.score)
    avg_score = float(np.mean(scores)) if scores else 1.0
    return round((1.0 - avg_score) * 100, 2)

def _get_pattern_type(ep_result) -> str:
    """获取 episode 的异常模式类型。"""
    try:
        from components.common import _detect_pattern_type
        return _detect_pattern_type(ep_result) or ""
    except Exception:
        return ""

def _generate_json_export(episodes, scope: str) -> str:
    """生成 JSON 格式导出。"""
    from rda.report.json_report import generate_json_report
    from rda.report.aggregation import aggregate_dataset_metrics
    from rda.report.top_issues import compute_top_observations, compute_hero_metrics
    from rda.audit.dataset_audit import DatasetAuditResult
    from components.common import compute_dhi

    # 构建一个仅包含范围内 episodes 的 result（用于计算）
    filtered_result = DatasetAuditResult(dataset_info=info)
    for ep in episodes:
        filtered_result.episodes[ep.episode_index] = ep
    filtered_result.compute_verdict_counts()

    # 获取完整报告结构
    full_report = generate_json_report(result)

    # 替换 episodes 部分
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

    # 计算范围内的质量数据
    dhi_data = compute_dhi(filtered_result)
    dataset_metrics = aggregate_dataset_metrics(filtered_result)
    top_obs = compute_top_observations(filtered_result, dataset_metrics=dataset_metrics)
    hero = compute_hero_metrics(dataset_metrics)

    report = {
        "version": "0.2.0",
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
    """生成 CSV 格式导出。"""
    from components.common import _detect_pattern_type
    from rda.audit.rules import CRITICAL_METRICS
    from rda.metrics.base import MetricAvailability
    import numpy as np

    output = io.StringIO()
    writer = csv.writer(output)

    # 表头
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
            deviation_score,  # CSV 列名: issue_severity_score
            pattern,
            len(issues),
            "; ".join(issues),
            user_decision,
            user_notes,
        ])

    return output.getvalue()



st.title("📥 导出报告")
st.caption("将审计结果导出为 JSON / CSV 格式，或导出排除坏 episodes 后的干净数据集")

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if st.session_state.audit_result is None:
    st.warning("⚠️ 请先运行审计", icon="🔍")
    st.page_link("pages/2_Audit.py", label="前往 Audit 页面 →", icon="🔍")
    st.stop()

result = st.session_state.audit_result
info = result.dataset_info

# ---------------------------------------------------------------------------
# 导出配置
# ---------------------------------------------------------------------------
st.subheader("导出配置")

col1, col2 = st.columns(2)

with col1:
    format_choice = st.radio(
        "导出格式",
        options=["JSON", "CSV"],
        index=0,
        horizontal=True,
        help="JSON 包含完整审计结果，CSV 仅包含 episode 汇总表",
    )

with col2:
    scope_choice = st.radio(
        "导出范围",
        options=["全量 episodes", "仅 Review Queue", "仅异常 episodes"],
        index=0,
        horizontal=True,
        help="Review Queue = REVIEW + EXCLUDE；仅异常 = 仅 EXCLUDE",
    )

# ---------------------------------------------------------------------------
# 预览
# ---------------------------------------------------------------------------
st.subheader("导出预览")

# 计算范围
if scope_choice == "全量 episodes":
    scope_label = "all"
    episodes_to_export = list(result.episodes.values())
elif scope_choice == "仅 Review Queue":
    scope_label = "review_queue"
    episodes_to_export = [
        ep for ep in result.episodes.values()
        if ep.verdict.value in ("REVIEW", "EXCLUDE")
    ]
else:  # 仅异常 episodes
    scope_label = "anomalies"
    episodes_to_export = [
        ep for ep in result.episodes.values()
        if ep.verdict.value == "EXCLUDE"
    ]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Episodes", len(episodes_to_export))
with col2:
    st.metric("格式", format_choice)
with col3:
    st.metric("范围", scope_choice)

# 人工审核决策统计
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
        f"📝 范围内已决策 {len(verdict_in_scope)} 个 episodes："
        f"🟢 KEEP {keep_n} · 🔴 REMOVE {remove_n} · 🟡 UNCERTAIN {uncertain_n}",
        icon="✅",
    )
else:
    st.caption("💡 尚未进行人工审核决策，导出将仅包含自动审计结果")

st.divider()

# ---------------------------------------------------------------------------
# 生成下载
# ---------------------------------------------------------------------------
st.subheader("下载")

if format_choice == "JSON":
    json_content = _generate_json_export(episodes_to_export, scope_label)
    file_name = f"rda_report_{scope_label}.json"

    st.download_button(
        label="📥 下载 JSON 报告",
        data=json_content,
        file_name=file_name,
        mime="application/json",
        use_container_width=True,
        type="primary",
    )

    with st.expander("预览 JSON 结构"):
        preview_data = json.loads(json_content)
        # 只显示前 2 个 episode 预览
        if "episodes" in preview_data:
            preview_data["episodes"] = preview_data["episodes"][:2]
            preview_data["_note"] = f"... 共 {len(episodes_to_export)} 个 episodes，此处仅预览前 2 个"
        st.json(preview_data)

else:  # CSV
    csv_content = _generate_csv_export(episodes_to_export)
    file_name = f"rda_report_{scope_label}.csv"

    st.download_button(
        label="📥 下载 CSV 报告",
        data=csv_content.encode("utf-8-sig"),  # BOM for Excel
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

    with st.expander("预览 CSV 内容"):
        lines = csv_content.split("\n")
        preview_lines = lines[:11]  # 表头 + 10 行
        st.code("\n".join(preview_lines), language="csv")
        if len(lines) > 11:
            st.caption(f"... 共 {len(lines) - 1} 行数据")

st.divider()

# ---------------------------------------------------------------------------
# 报告说明
# ---------------------------------------------------------------------------
st.markdown("### 📝 报告说明")

with st.expander("JSON 报告字段说明"):
    st.markdown("""
    **JSON 报告结构：**
    - `version`: RDA 版本号
    - `dataset`: 数据集元信息（路径、episode 数、总帧数、模态）
    - `summary`: 汇总统计（verdict 分布、通过率）
    - `quality`: 质量评估（DHI、等级、Training Readiness、四维得分）
    - `hero_metrics`: 关键指标摘要
    - `top_observations`: Top 问题观察
    - `user_verdicts`: 人工审核决策（episode_id → {decision, notes}）
    - `episodes`: 逐 episode 结果列表
      - `episode_index`: episode 编号
      - `num_frames`: 帧数
      - `verdict`: 判定结果 (PASS / REVIEW / EXCLUDE)
      - `metrics`: 各指标结果（availability / measurement / assessment）
    """)

with st.expander("CSV 报告字段说明"):
    st.markdown("""
    **CSV 包含字段：**
    - `episode_id`: Episode 编号
    - `num_frames`: 帧数
    - `integrity_check`: 完整性检查 (PASS / FAIL)
    - `behavior_verdict`: 行为判定 (PASS / REVIEW / EXCLUDE，对应 UI 显示 KEEP / REVIEW / REMOVE)
    - `issue_severity_score`: 问题严重度评分（0-100，越高越异常，原 deviation_score）
    - `pattern_type`: 异常模式类型
    - `issue_count`: 问题数量
    - `top_issues`: 主要问题列表
    - `user_decision`: 人工审核决策（KEEP / REMOVE / UNCERTAIN，空值表示未决策）
    - `user_notes`: 审核备注
    """)


# ---------------------------------------------------------------------------
# 📦 导出干净数据集
# ---------------------------------------------------------------------------
st.divider()
st.header("📦 导出干净数据集")
st.caption("排除不良 episodes，输出可直接用于训练的 LeRobot v3.0 数据集")


# ---------------------------------------------------------------------------
# Helper: 计算 episode deviation score
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Strategy 选择
# ---------------------------------------------------------------------------
st.markdown("#### 🧹 清洗策略")

user_verdicts = st.session_state.get("user_verdicts", {})
total_episodes = len(result.episodes)

strategy_choice = st.radio(
    "选择清洗策略",
    options=["conservative", "aggressive", "custom"],
    format_func=lambda x: {
        "conservative": "🛡️ Conservative（保守）— 仅移除 EXCLUDE，REVIEW 保留",
        "aggressive": "⚔️ Aggressive（激进）— 移除 EXCLUDE + REVIEW",
        "custom": "🔧 Custom（自定义）— 手动选择要移除的 AI 判定类型",
    }[x],
    index=0,
    key="clean_strategy_choice",
    label_visibility="collapsed",
)

# Custom 模式：让用户选择要移除的 AI verdict
custom_remove_verdicts = []
if strategy_choice == "custom":
    st.caption("选择要移除的 AI 判定类型：")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        remove_pass = st.checkbox(
            "❌ PASS（正常）",
            value=False,
            key="custom_remove_pass",
            help="移除 AI 判定为 PASS 的未决策 episodes（通常不建议）",
        )
    with col_c2:
        remove_review = st.checkbox(
            "⚠️ REVIEW（待审核）",
            value=False,
            key="custom_remove_review",
            help="移除 AI 判定为 REVIEW 的未决策 episodes",
        )
    with col_c3:
        remove_exclude = st.checkbox(
            "🚫 EXCLUDE（排除）",
            value=True,
            key="custom_remove_exclude",
            help="移除 AI 判定为 EXCLUDE 的未决策 episodes",
        )
    custom_remove_verdicts = []
    if remove_pass:
        custom_remove_verdicts.append("PASS")
    if remove_review:
        custom_remove_verdicts.append("REVIEW")
    if remove_exclude:
        custom_remove_verdicts.append("EXCLUDE")

# ---------------------------------------------------------------------------
# Episode 分类统计
# ---------------------------------------------------------------------------
# 根据 strategy 对每个 episode 进行分类
_keep_ids = []
_remove_ids = []       # 用户 REMOVE + AI 判定要移除的
_uncertain_ids = []    # 用户 UNCERTAIN
_undecided_ids = []    # 未决策
_review_ids = []       # 未决策中 AI 判定为 REVIEW 的

# 用于展示的移除列表（episode_id, pattern_type, score, ai_verdict）
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
        # 未决策 → 按 AI 判定 + strategy 处理
        _undecided_ids.append(ep_id)
        if ai_verdict == "REVIEW":
            _review_ids.append(ep_id)

        # 根据 strategy 决定
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

# 未决策中 AI 判定为 REVIEW 的数量
_n_review_undecided = len(_review_ids)

# ---------------------------------------------------------------------------
# 📊 清洗预览面板
# ---------------------------------------------------------------------------
st.markdown("#### 📊 清洗预览")

dataset_name = Path(result.dataset_info.path).name

# 统计各类别
n_keep = len(_keep_ids)
n_review = _n_review_undecided
n_remove = len(_remove_ids)
n_undecided_no_verdict = len(_undecided_ids) - _n_review_undecided  # 未决策且非 REVIEW
n_no_decision = len([
    ep_id for ep_id, ep_result in result.episodes.items()
    if not user_verdicts.get(ep_id, {}).get("decision")
    and ep_result.verdict.value == "PASS"
    and ep_id not in _remove_ids
])
# 未决策且最终保留（PASS 且未被移除）
n_undecided_kept = sum(
    1 for ep_id in _undecided_ids
    if ep_id not in _remove_ids
)
# "未决策"显示为最终保留中来自未决策的部分
# 按 UI 要求: ⚪ 未决策 = 没有用户决策且 AI=PASS 且被保留的
n_no_decision_display = sum(
    1 for ep_id in _undecided_ids
    if result.episodes[ep_id].verdict.value == "PASS"
    and ep_id not in _remove_ids
)

pct = lambda n: f"{n / total_episodes * 100:.1f}%" if total_episodes > 0 else "0.0%"

# 预览表格
st.markdown(f"**数据集:** `{dataset_name}` · **总 Episodes:** {total_episodes:,}")

preview_table = (
    f"| 状态 | 数量 | 占比 |\n"
    f"|:---|---:|---:|\n"
    f"| 🟢 保留 (KEEP) | {n_keep:,} | {pct(n_keep)} |\n"
    f"| 🟡 待审核 (REVIEW) | {n_review:,} | {pct(n_review)} |\n"
    f"| 🔴 排除 (REMOVE) | {n_remove:,} | {pct(n_remove)} |\n"
    f"| ⚪ 未决策 | {len(_undecided_ids):,} | {pct(len(_undecided_ids))} |"
)
st.markdown(preview_table)

# 策略执行摘要
st.markdown("🧹 **按当前策略将执行:**")
summary_lines = [
    f"- 移除 **{n_remove}** 个 episodes",
    f"- 保留 **{n_keep}** 个 episodes",
    f"- 预计可用数据: **{pct(n_keep)}**",
]
if _uncertain_ids:
    summary_lines.append(f"- ⚠️ {_uncertain_ids.__len__()} 个 UNCERTAIN episodes 按 uncertainty 策略处理")
st.markdown("\n".join(summary_lines))

# ---------------------------------------------------------------------------
# 待移除列表
# ---------------------------------------------------------------------------
if _removal_info:
    with st.expander(f"📋 待移除 Episodes 列表（{len(_removal_info)} 个）", expanded=False):
        # 按 score 降序排列
        _removal_info_sorted = sorted(_removal_info, key=lambda x: x[2], reverse=True)

        _display_limit = 20
        _to_show = _removal_info_sorted[:_display_limit]

        # 构建表格
        _table_rows = "| Episode ID | Pattern Type | Score | 来源 |\n|---:|:---|---:|:---|\n"
        for _ep_id, _pattern, _score, _source in _to_show:
            _source_label = {
                "USER_REMOVE": "👤 用户移除",
                "EXCLUDE": "🤖 AI EXCLUDE",
                "REVIEW": "🤖 AI REVIEW",
                "PASS": "🤖 AI PASS",
            }.get(_source, _source)
            _table_rows += f"| {_ep_id} | {_pattern or '-'} | {_score} | {_source_label} |\n"

        st.markdown(_table_rows)

        _remaining = len(_removal_info_sorted) - _display_limit
        if _remaining > 0:
            st.caption(f"... 还有 {_remaining} 个 episodes 未显示")

# ---------------------------------------------------------------------------
# 输出配置 & 导出按钮
# ---------------------------------------------------------------------------
st.markdown("#### ⚙️ 导出配置")

col_cfg1, col_cfg2 = st.columns([1, 2])
with col_cfg1:
    uncertainty_choice = st.radio(
        "UNCERTAIN 处理策略",
        options=["保留（默认）", "排除"],
        index=0,
        horizontal=True,
        help="用户标记为 UNCERTAIN 的 episodes 是保留还是排除",
        key="clean_uncertainty_strategy",
    )
    _uncertainty_strategy = "keep" if uncertainty_choice == "保留（默认）" else "remove"

with col_cfg2:
    default_output = str(_PROJECT_ROOT / "clean_export")
    output_path_str = st.text_input(
        "输出路径",
        value=default_output,
        key="clean_output_path",
        help="干净数据集的输出目录路径",
    )

# 导出按钮
if st.button(
    "📦 执行导出",
    type="primary",
    use_container_width=True,
    key="btn_clean_export",
):
    from rda.export.clean_export import CleanDatasetExporter

    # 构建 AI verdicts
    _ai_verdicts = {
        ep_id: ep_result.verdict.value
        for ep_id, ep_result in result.episodes.items()
    }

    # 构建用户 verdicts (int keys)
    _user_verdicts_int = {
        int(k): v for k, v in user_verdicts.items()
    }

    _output_path = Path(output_path_str)

    # 构建 cleaning_strategy 参数
    if strategy_choice == "conservative":
        _cleaning_strategy = "conservative"
    elif strategy_choice == "aggressive":
        _cleaning_strategy = "aggressive"
    else:
        _cleaning_strategy = custom_remove_verdicts  # list

    progress_bar = st.progress(0, text="正在导出...")

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
        progress_bar.progress(1.0, text="导出完成！")
        st.success(
            f"✅ 干净数据集导出成功！\n\n"
            f"- 输出路径：`{_output_path}`\n"
            f"- 保留 episodes：{export_report.kept_episodes} / {export_report.total_episodes}\n"
            f"- 排除 episodes：{export_report.removed_episodes}\n"
            f"- 总帧数：{export_report.total_frames}\n"
            f"- 清洗策略：{export_report.cleaning_strategy}\n"
            f"- 格式：LeRobot v3.0（parquet + meta）"
        )

        # 详细统计
        with st.expander("📊 导出详细统计"):
            st.json(export_report.to_dict())

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ 导出失败：{e}")


# ---------------------------------------------------------------------------
# 导出函数
# ---------------------------------------------------------------------------





