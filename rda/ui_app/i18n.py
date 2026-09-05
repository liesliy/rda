"""Bilingual (zh/en) string catalog and language selector for the RDA UI.

Language is stored in ``st.session_state["rda_lang"]`` (default "zh").
The selector is rendered in the app-level sidebar (app.py) and persists
across page navigation because st.navigation re-runs app.py on every
interaction.

Usage:
    from rda.ui_app.i18n import t, get_lang, render_lang_selector

    st.title(t("upload_title"))
"""
from __future__ import annotations

from typing import Dict

import streamlit as st

LANGS = ("zh", "en")
DEFAULT_LANG = "zh"


def get_lang() -> str:
    """Return the active UI language ('zh' or 'en')."""
    lang = st.session_state.get("rda_lang", DEFAULT_LANG)
    return lang if lang in LANGS else DEFAULT_LANG


def t(key: str, **kwargs) -> str:
    """Translate a key in the active language.

    Falls back to the key itself when missing (keeps pages usable and
    makes missing translations greppable).
    """
    table = STRINGS.get(get_lang(), STRINGS[DEFAULT_LANG])
    s = table.get(key)
    if s is None:
        s = STRINGS[DEFAULT_LANG].get(key, key)
    if kwargs:
        s = s.format(**kwargs)
    return s


def render_lang_selector() -> None:
    """Render the language switcher in the sidebar (call inside st.sidebar)."""
    lang = st.radio(
        "Language / 语言",
        options=list(LANGS),
        index=LANGS.index(get_lang()),
        format_func=lambda v: "中文" if v == "zh" else "English",
        key="rda_lang_selector",
        label_visibility="collapsed",
        horizontal=True,
    )
    st.session_state.rda_lang = lang


# ---------------------------------------------------------------------------
# String catalog
# ---------------------------------------------------------------------------

STRINGS: Dict[str, Dict[str, str]] = {
    "zh": {
        # ---- app shell / sidebar ----
        "app_title": "RDA · 机器人数据质量审计",
        "page_title": "RDA · 机器人数据质量审计",
        "app_subtitle": "Robot Data Assurance — 数据质量审计与治理工具",
        "nav_group_flow": "数据流程",
        "nav_group_results": "审计结果",
        "nav_group_recommend": "优化建议",
        "nav_group_trend": "趋势分析",
        "nav_upload": "Upload · 上传数据集",
        "nav_audit": "Audit · 运行审计",
        "nav_health": "Health Overview · 健康概览",
        "nav_episode": "Episode Explorer · 逐集查看",
        "nav_export": "Export · 导出报告",
        "nav_history": "Audit History · 审计历史",
        "nav_recommend": "Recommend · 优化建议",
        "sidebar_total_frames": "总帧数",
        "sidebar_modalities": "模态",
        "sidebar_no_dataset": "尚未加载数据集",

        # ---- common gates ----
        "not_uploaded": "⚠️ 请先在 Upload 页面上传数据集",
        "not_audited": "⚠️ 请先运行审计",
        "go_upload": "前往 Upload 页面 →",
        "go_audit": "前往 Audit 页面 →",

        # ---- page 1: upload ----
        "upload_title": "📤 上传数据集",
        "upload_caption": "支持 LeRobot 格式数据集，上传后可进行质量审计",
        "upload_source": "数据集来源",
        "upload_platform": "选择平台",
        "upload_platform_help": "armnetbench / droid 为预置平台，custom 为自定义 LeRobot 格式数据集",
        "upload_tab_local": "📁 本地路径",
        "upload_tab_files": "📤 上传文件夹",
        "upload_local_hint": "输入本地 LeRobot 数据集目录路径",
        "upload_path_label": "数据集路径",
        "upload_load_btn": "加载数据集",
        "upload_path_empty_err": "请输入数据集路径",
        "upload_files_hint": "上传数据集目录（多个文件）",
        "upload_uploader_label": "选择数据集文件（支持 parquet / json / 等 LeRobot 格式文件）",
        "upload_uploader_help": "请选择数据集中的所有文件，将保存到临时目录",
        "upload_save_load_btn": "保存并加载",
        "upload_loading": "正在加载数据集...",
        "upload_loaded_ok": "成功加载 {n} 个 episodes",
        "upload_load_err": "加载失败：{err}",
        "upload_info_header": "✅ 数据集信息",
        "upload_metric_eps": "Episode 数量",
        "upload_metric_frames": "总帧数",
        "upload_metric_dof": "动作维度 (DOF)",
        "upload_metric_fps": "帧率 (FPS)",
        "upload_modalities": "📷 观测模态 ({n})",
        "upload_actions": "🦾 动作空间 ({n})",
        "upload_meta": "📝 元信息",
        "upload_success": "数据集加载成功！共 {eps} 个 episodes，{frames} 帧。",
        "upload_next_audit": "下一步：运行审计 →",

        # ---- page 2: audit ----
        "audit_title": "🔍 运行审计",
        "audit_caption": "对数据集执行完整的质量审计，生成逐 episode 的检测结果",
        "audit_gate_upload": "⚠️ 请先在 Upload 页面加载数据集",
        "audit_dataset": "**数据集**: `{path}`",
        "audit_summary_line": "**共 {eps} 个 episodes · {frames} 帧**",
        "audit_layers_header": "已启用的审计检查层",
        "audit_layers_info": (
            "审计流程包含以下检查层，全部默认启用：\n\n"
            "**Layer 1A · 完整性检查** — 缺失帧、NaN/Inf、时间戳、格式一致性等确定性检查\n\n"
            "**Layer 1B · 行为质量检测** — 基于参考分布的统计异常检测（运动、时序、分布等）\n\n"
            "**Layer 2 · 诊断与归因** — Pattern Type 识别 + 原因分析 + 建议"
        ),
        "audit_layers_note": "V0.1 暂不支持 Layer 3 数据治理功能；审计时默认启用所有检查层。",
        "audit_done_info": "✅ 已完成审计，共 {eps} 个 episodes",
        "audit_rerun_btn": "🔄 重新运行审计",
        "audit_view_health": "查看健康概览 →",
        "audit_result_header": "审计结果摘要",
        "audit_preparing": "准备中...",
        "audit_loading_data": "正在加载数据集并执行审计...",
        "audit_progress": "审计中... Episode {i}/{total} ({pct:.0f}%)",
        "audit_done_progress": "审计完成！",
        "audit_done_status": "✅ 审计完成，共 {n} 个 episodes",
        "audit_done_success": "审计完成！快照已保存，可在 Audit History 查看趋势变化",
        "audit_failed": "审计失败：{err}",
        "audit_detail_err": "详细错误信息",
        "audit_start_btn": "▶️ 开始审计",

        # ---- page 3: health overview ----
        "health_title": "💚 数据集健康概览",
        "health_caption": "数据集整体质量评估 · 不展示算法细节，只展示结论",
        "health_readiness_label": "训练就绪度 · Training Readiness",
        "health_usable_metric": "当前 DHI",
        "health_usable_after_review": "Potentially usable after review",
        "health_usable_after_review_sub": "{n} / {total} episodes",
        "health_clear_keep": "Clear Keep (无疑问)",
        "health_clear_keep_sub": "{n} episodes",
        "health_usable_caption": "Potentially usable = KEEP + REVIEW（经人工审核后可保留的部分）；不包含明确 REMOVE 的 episodes",
        "health_dhi_header": "数据健康指数 (DHI)",
        "health_dhi_note": "基于数据完整性与行为一致性的相对评估<br/>Relative assessment, not absolute score",
        "health_grade_suffix": "级",
        "health_dims_header": "四维质量得分",
        "health_dim_integrity": "数据完整性 (Integrity)",
        "health_dim_temporal": "时间质量 (Temporal)",
        "health_dim_motion": "运动质量 (Motion)",
        "health_dim_consistency": "行为一致性 (Consistency)",
        "health_radar_header": "雷达图",
        "health_top_issues_header": "🔎 主要审计观察",
        "health_no_issues": "🎉 未发现确定性结构问题或明显风险信号",
        "health_metric_critical": "🔴 严重 (Critical)",
        "health_metric_warning": "🟡 风险信号 (Risk Signal)",
        "health_metric_types": "📋 观察类型数",
        "health_sev_critical": "严重",
        "health_sev_warning": "警告",
        "health_sev_info": "提示",
        "health_sev_unknown": "未知",
        "health_issue_expander": "{sev} **{idx}. {desc}**  `{code}`  · 影响 {count} 个 episodes ({pct:.1f}%)",
        "health_lbl_severity": "问题等级",
        "health_lbl_scope": "影响范围",
        "health_lbl_code": "问题编码",
        "health_recommendation": "**下一步（需人工确认）**",
        "health_view_episodes": "逐 episode 查看详情 →",
        "health_verdict_keep_pct": "{pct:.1f}% of episodes",

        # ---- page 4: episode explorer ----
        "ep_title": "📋 Episode Explorer · 逐集查看",
        "ep_caption": "浏览每个 episode 的审计结果、问题类型和诊断详情，并记录人工审核决策",
        "ep_score_view": "📊 行为评分视图",
        "ep_score_view_help": (
            "Portable Core：仅使用跨平台通用指标（duration / spike / effective_motion），"
            "适用于跨平台比较。\n"
            "Platform-specific：仅使用平台特有指标（velocity / path_length 等），"
            "仅在同一平台下有意义。\n"
            "Combined：所有指标综合，用于同平台深度分析。"
        ),
        "ep_view_portable": "Portable Core (跨平台)",
        "ep_view_platform": "Platform-specific (同平台)",
        "ep_view_combined": "Combined (深度分析)",
        "ep_metric_total": "总 Episodes",
        "ep_metric_keep_ai": "🟢 KEEP (AI)",
        "ep_metric_review_ai": "🟡 REVIEW (AI)",
        "ep_metric_remove_ai": "🔴 REMOVE (AI)",
        "ep_review_progress": "人工审核进度：**{decided} / {need}**（KEEP: {keep} · REMOVE: {remove} · UNCERTAIN: {uncertain}）",
        "ep_tab_queue": "🚨 Review Queue（优先处理）",
        "ep_tab_all": "📋 全部 Episodes",
        "ep_queue_empty": "🎉 没有需要人工确认的 episode",
        "ep_queue_info": "共 **{n}** 个 episode 需要关注（{rm} 个确定性排除建议，{rv} 个风险信号待确认）",
        "ep_queue_no_items": "没有需要审核的 episode",
        "ep_queue_undecided": "⚪ 未决策",
        "ep_frames": "**帧数**",
        "ep_main_issues": "**主要问题**",
        "ep_notes_label": "审核备注",
        "ep_notes_placeholder": "记录审核原因...",
        "ep_view_detail": "📖 查看详情",
        "ep_table_empty": "没有符合条件的 episode",
        "ep_col_episode": "Episode #",
        "ep_col_verdict": "Verdict",
        "ep_col_integrity": "完整性",
        "ep_col_behavior": "Behavior Score",
        "ep_col_pattern": "Pattern Type",
        "ep_col_frames": "帧数",
        "ep_col_issues": "主要问题",
        "ep_select_detail": "选择 episode 查看详情",
        "ep_detail_title": "🔍 Episode #{ep} 详情",
        "ep_not_found": "未找到 Episode #{ep}",
        "ep_integrity_header": "数据完整性 (Layer 1A)",
        "ep_integrity_pass": "PASS · 无硬检查问题",
        "ep_integrity_fail": "FAIL · 存在完整性问题",
        "ep_integrity_issues": "存在的问题：",
        "ep_behavior_header": "行为质量 (Layer 1B)",
        "ep_verdict_keep": "KEEP · 建议保留",
        "ep_verdict_review": "REVIEW · 建议人工审核",
        "ep_verdict_exclude": "REMOVE · 规则建议排除（需确认）",
        "ep_metrics_header": "📊 各项指标详情",
        "ep_col_metric": "指标",
        "ep_col_status": "状态",
        "ep_col_value": "测量值",
        "ep_col_note": "说明",
        "ep_na_reason": "数据不可用",
        "ep_error_reason": "计算出错",
        "ep_diag_header": "💡 诊断与建议",
        "ep_diag_normal": "该 episode 各项指标正常，无明显质量问题。",
        "ep_diag_what": "**WHAT · 问题描述**",
        "ep_diag_why": "**WHY · 证据解释**",
        "ep_diag_next": "**NEXT · 人工确认建议**",
        "ep_decision_header": "✍️ 人工审核决策 · Your Decision",
        "ep_btn_keep": "✅ KEEP · 保留",
        "ep_btn_keep_help": "确认该 episode 可用于训练",
        "ep_btn_remove": "🗑️ REMOVE · 排除",
        "ep_btn_remove_help": "确认该 episode 应从训练集中排除",
        "ep_btn_uncertain": "❓ UNCERTAIN · 存疑",
        "ep_btn_uncertain_help": "暂时标记为存疑，待进一步确认",
        "ep_marked_keep": "Episode #{ep} 已标记为 KEEP（保留）",
        "ep_marked_remove": "Episode #{ep} 已标记为 REMOVE（排除）",
        "ep_marked_uncertain": "Episode #{ep} 已标记为 UNCERTAIN（存疑）",
        "ep_badge_keep": "保留",
        "ep_badge_remove": "排除",
        "ep_badge_uncertain": "存疑",
        "ep_current_decision": "当前决策：",
        "ep_no_decision": "当前状态：未决策",
        "ep_notes_area": "审核备注（可选）",
        "ep_notes_area_placeholder": "记录审核原因、上下文或后续动作...",
        "ep_save_notes": "💾 保存备注",
        "ep_notes_saved": "备注已保存",
        "ep_clear_decision": "🗑️ 清除决策",
        "ep_decision_cleared": "决策已清除",
        "ep_close_detail": "关闭详情",
        "ep_traj_header": "📈 轨迹可视化",
        "ep_traj_no_path": "💡 数据集路径未设置，无法加载原始轨迹数据",
        "ep_traj_load_err": "⚠️ 加载轨迹数据失败：{err}",
        "ep_traj_not_found": "⚠️ 未找到 Episode #{ep} 的原始数据",
        "ep_traj_no_data": "该 episode 没有 state 或 action 数据，跳过可视化",
        "ep_traj_view_label": "显示模式",
        "ep_traj_state_title": "Joint States 时序图",
        "ep_traj_action_title": "Actions 时序图",
        "ep_traj_time_axis": "时间 (s)",
        "ep_traj_value_axis": "值",
        "ep_traj_legend": "维度",
        "ep_traj_loading": "正在加载 episode 原始数据...",
        "ep_filter_verdict": "Verdict 筛选",
        "ep_filter_integrity": "完整性筛选",
        "ep_filter_pattern": "Pattern Type 筛选",
        "ep_filter_showing": "显示 {n} / {total} 个 episodes",
        "ep_score_severity": "Issue Severity Score: {v}",

        # ---- page 5: export ----
        "export_title": "📥 导出报告",
        "export_caption": "将审计结果导出为 JSON / CSV 格式，或导出排除坏 episodes 后的干净数据集",
        "export_config_header": "导出配置",
        "export_format_label": "导出格式",
        "export_format_help": "JSON 包含完整审计结果，CSV 仅包含 episode 汇总表",
        "export_scope_label": "导出范围",
        "export_scope_all": "全量 episodes",
        "export_scope_queue": "仅 Review Queue",
        "export_scope_anomalies": "仅异常 episodes",
        "export_scope_help": "Review Queue = REVIEW + EXCLUDE；仅异常 = 仅 EXCLUDE",
        "export_preview_header": "导出预览",
        "export_metric_format": "格式",
        "export_metric_scope": "范围",
        "export_verdicts_info": "📝 范围内已决策 {n} 个 episodes：🟢 KEEP {keep} · 🔴 REMOVE {remove} · 🟡 UNCERTAIN {uncertain}",
        "export_no_verdicts": "💡 尚未进行人工审核决策，导出将仅包含自动审计结果",
        "export_download_header": "下载",
        "export_download_json": "📥 下载 JSON 报告",
        "export_download_csv": "📥 下载 CSV 报告",
        "export_preview_json": "预览 JSON 结构",
        "export_preview_json_note": "... 共 {n} 个 episodes，此处仅预览前 2 个",
        "export_preview_csv": "预览 CSV 内容",
        "export_preview_csv_note": "... 共 {n} 行数据",
        "export_docs_header": "📝 报告说明",
        "export_json_docs": "JSON 报告字段说明",
        "export_csv_docs": "CSV 报告字段说明",
        "export_json_docs_body": (
            "**JSON 报告结构：**\n"
            "- `version`: RDA 版本号\n"
            "- `dataset`: 数据集元信息（路径、episode 数、总帧数、模态）\n"
            "- `summary`: 汇总统计（verdict 分布、通过率）\n"
            "- `quality`: 质量评估（DHI、等级、Training Readiness、四维得分）\n"
            "- `hero_metrics`: 关键指标摘要\n"
            "- `top_observations`: Top 问题观察\n"
            "- `user_verdicts`: 人工审核决策（episode_id → {decision, notes}）\n"
            "- `episodes`: 逐 episode 结果列表\n"
            "  - `episode_index`: episode 编号\n"
            "  - `num_frames`: 帧数\n"
            "  - `verdict`: 判定结果 (PASS / REVIEW / EXCLUDE)\n"
            "  - `metrics`: 各指标结果（availability / measurement / assessment）"
        ),
        "export_csv_docs_body": (
            "**CSV 包含字段：**\n"
            "- `episode_id`: Episode 编号\n"
            "- `num_frames`: 帧数\n"
            "- `integrity_check`: 完整性检查 (PASS / FAIL)\n"
            "- `behavior_verdict`: 行为判定 (PASS / REVIEW / EXCLUDE，对应 UI 显示 KEEP / REVIEW / REMOVE)\n"
            "- `issue_severity_score`: 问题严重度评分（0-100，越高越异常，原 deviation_score）\n"
            "- `pattern_type`: 异常模式类型\n"
            "- `issue_count`: 问题数量\n"
            "- `top_issues`: 主要问题列表\n"
            "- `user_decision`: 人工审核决策（KEEP / REMOVE / UNCERTAIN，空值表示未决策）\n"
            "- `user_notes`: 审核备注"
        ),
        "export_clean_header": "📦 导出干净数据集",
        "export_clean_caption": "排除不良 episodes，输出可直接用于训练的 LeRobot v3.0 数据集",
        "export_strategy_header": "🧹 清洗策略",
        "export_strategy_label": "选择清洗策略",
        "export_strategy_conservative": "🛡️ Conservative（保守）— 仅移除 EXCLUDE，REVIEW 保留",
        "export_strategy_aggressive": "⚔️ Aggressive（激进）— 移除 EXCLUDE + REVIEW",
        "export_strategy_custom": "🔧 Custom（自定义）— 手动选择要移除的 AI 判定类型",
        "export_custom_caption": "选择要移除的 AI 判定类型：",
        "export_custom_pass": "❌ PASS（正常）",
        "export_custom_pass_help": "移除 AI 判定为 PASS 的未决策 episodes（通常不建议）",
        "export_custom_review": "⚠️ REVIEW（待审核）",
        "export_custom_review_help": "移除 AI 判定为 REVIEW 的未决策 episodes",
        "export_custom_exclude": "🚫 EXCLUDE（排除）",
        "export_custom_exclude_help": "移除 AI 判定为 EXCLUDE 的未决策 episodes",
        "export_preview_panel": "📊 清洗预览",
        "export_preview_dataset": "**数据集:** `{name}` · **总 Episodes:** {n:,}",
        "export_preview_table": (
            "| 状态 | 数量 | 占比 |\n|:---|---:|---:|\n"
            "| 🟢 保留 (KEEP) | {keep:,} | {p_keep} |\n"
            "| 🟡 待审核 (REVIEW) | {review:,} | {p_review} |\n"
            "| 🔴 排除 (REMOVE) | {remove:,} | {p_remove} |\n"
            "| ⚪ 未决策 | {undecided:,} | {p_undecided} |"
        ),
        "export_strategy_summary": "🧹 **按当前策略将执行:**",
        "export_summary_remove": "- 移除 **{n}** 个 episodes",
        "export_summary_keep": "- 保留 **{n}** 个 episodes",
        "export_summary_usable": "- 预计可用数据: **{p}**",
        "export_summary_uncertain": "- ⚠️ {n} 个 UNCERTAIN episodes 按 uncertainty 策略处理",
        "export_removal_list": "📋 待移除 Episodes 列表（{n} 个）",
        "export_removal_remaining": "... 还有 {n} 个 episodes 未显示",
        "export_removal_col_id": "Episode ID",
        "export_removal_col_pattern": "Pattern Type",
        "export_removal_col_score": "Score",
        "export_removal_col_source": "来源",
        "export_source_user": "👤 用户移除",
        "export_config_sub": "⚙️ 导出配置",
        "export_uncertainty_label": "UNCERTAIN 处理策略",
        "export_uncertainty_keep": "保留（默认）",
        "export_uncertainty_remove": "排除",
        "export_uncertainty_help": "用户标记为 UNCERTAIN 的 episodes 是保留还是排除",
        "export_output_path": "输出路径",
        "export_output_path_help": "干净数据集的输出目录路径",
        "export_run_btn": "📦 执行导出",
        "export_running": "正在导出...",
        "export_done": "导出完成！",
        "export_success": (
            "✅ 干净数据集导出成功！\n\n"
            "- 输出路径：`{path}`\n"
            "- 保留 episodes：{kept} / {total}\n"
            "- 排除 episodes：{removed}\n"
            "- 总帧数：{frames}\n"
            "- 清洗策略：{strategy}\n"
            "- 格式：LeRobot v3.0（parquet + meta）"
        ),
        "export_stats": "📊 导出详细统计",
        "export_failed": "❌ 导出失败：{err}",

        # ---- page 6: history ----
        "history_title": "📈 审计历史",
        "history_caption": "追踪数据质量变化趋势 · 基于多次审计结果对比分析",
        "history_empty": (
            "暂无审计历史记录。\n\n"
            "每次在 Audit 页面完成审计后，系统会自动保存一份快照。\n"
            "完成至少一次审计后即可在此查看历史趋势。"
        ),
        "history_filter_label": "📂 按数据集过滤",
        "history_filter_all": "全部",
        "history_overview": "📊 历史概览",
        "history_metric_runs": "审计次数",
        "history_metric_latest": "最近审计",
        "history_metric_pass": "最新 PASS 率",
        "history_need_two": "⚠️ 至少需要 2 次审计才能查看趋势变化",
        "history_trend_header": "📈 趋势分析",
        "history_verdict_chart": "**Verdict 分布变化**",
        "history_verdict_keep": "PASS · 保留",
        "history_verdict_review": "REVIEW · 审核",
        "history_verdict_exclude": "EXCLUDE · 排除",
        "history_axis_run": "审计次序",
        "history_axis_eps": "Episode 数量",
        "history_dev_chart": "**平均 Deviation Score 趋势**",
        "history_dev_mean": "平均值",
        "history_dev_median": "中位数",
        "history_excl_chart": "**EXCLUDE 数量趋势**",
        "history_excl_series": "EXCLUDE 数量",
        "history_excl_axis": "EXCLUDE Episode 数量",
        "history_integrity_chart": "**完整性通过率趋势**",
        "history_integrity_series": "完整性通过率",
        "history_rate_axis": "通过率 (%)",
        "history_list_header": "📋 审计记录",
        "history_integrity_pass_metric": "完整性通过率",
        "history_mean_dev": "平均 Deviation: {mean:.2f} · 中位数: {median:.2f} · RDA v{ver}",
        "history_top_patterns": "🔍 主要 Pattern: {patterns}",
        "history_sensor_sync": "传感器同步: {v:.1f}ms ({interp})",
        "history_sensor_sync_na": "传感器同步: N/A",
        "history_action_jitter": "动作抖动: {n} spikes",
        "history_space_coverage": "空间覆盖: {v:.1%} ({interp})",

        # ---- page 7: recommend ----
        "rec_title": "🧭 Recommend · 数据优化建议",
        "rec_caption": "基于数据集时间结构生成带实验证据等级的优化建议；这些建议与审计中的输入数据风险信号是两条不同证据链。",
        "rec_no_path": "⚠️ 未找到数据集路径，请重新上传数据集。",
        "rec_policy_label": "目标模型架构",
        "rec_policy_fw": "frame-wise — MLP / BC（逐帧策略）",
        "rec_policy_tp": "temporal — ACT / Diffusion Policy / Transformer",
        "rec_policy_help": "不同架构对 idle 帧和有效窗口的容忍度不同，建议按实际训练模型选择。",
        "rec_privacy": "🔒 隐私说明：原始 episode 数据不会上传。仅在本地计算指标后，将 <1KB 的聚合统计发送到 RDA API 进行规则评估。",
        "rec_chunk_label": "策略 chunk 大小（可选）",
        "rec_chunk_help": "你的策略每次执行的动作数（如 ACT/DROID 的 chunk=8/10）。声明后，idle 修剪建议将按该窗口长度评估——对齐 openpi/DROID 官方过滤语义。留空则按通用窗口档位评估。",
        "rec_chunk_none": "未声明（通用档位）",
        "rec_run_btn": "🚀 生成优化建议",
        "rec_computing": "正在本地计算时间结构指标...",
        "rec_calling_api": "调用 RDA 规则引擎（rules API）...",
        "rec_done": "完成！",
        "rec_rules_ver": "规则版本：rules v{v}",
        "rec_report_header": "建议报告",
        "rec_cards_header": "逐条建议",
        "rec_confidence": "置信度: {v}",
        "rec_expected_impact": "**预期影响：** {v}",
        "rec_details": "**细节：**",
        "rec_caveats": "**注意事项：**",
        "rec_raw_json": "查看原始 JSON",
        "rec_failed": "生成建议失败：{err}",
        "rec_failed_hint": "提示：如果网络不可用，可检查 RDA_API_URL 环境变量；有本地缓存时会自动降级使用缓存结果。",
        "rec_disclaimer": (
            "RDA 是数据质量诊断 + 低风险优化建议工具，不保证成功率提升。"
            "任何裁剪操作都请在保留集上验证后再用于正式训练。"
        ),

        # ---- misc (issue descriptions, recommendations, diagnosis) ----
        "issue_missing_dropout": "缺失帧 / 传感器断连",
        "issue_invalid_values": "NaN/Inf 异常值",
        "issue_schema_consistency": "数据格式不匹配",
        "issue_timestamp_validity": "时间戳异常（非单调/间隔不均）",
        "issue_joint_limit": "关节限位超限",
        "issue_sensor_synchronization": "多传感器同步偏差",
        "issue_sampling_jitter": "采样抖动",
        "issue_velocity_acceleration": "速度/加速度异常",
        "issue_action_discontinuity": "动作不连续（抖动）",
        "issue_idle_ratio": "有效运动比例过低",
        "issue_distribution": "轨迹分布离群",
        "issue_coverage": "状态空间覆盖不足",
        "rec_generic": "建议人工审核对应 episodes，确认具体原因。",
        "no_issues": "无",
        "radar_score": "得分",
        "dhi_empty": "数据集中无 episode",
        "dhi_ready": "数据可直接用于训练，共 {n} 个 episode",
        "dhi_cond": "审核 {n} 个 episode 后可用于训练",
        "dhi_not_ready": "存在显著质量问题，建议先处理数据",
        "diag_integrity_what": "完整性检查未通过（{metrics}）",
        "diag_integrity_why": "数据采集或存储过程中出现确定性错误",
        "diag_integrity_next": "检查数据采集管线，修复后重新采集",
        "diag_disc_what": "观察到动作不连续模式（Risk Signal）",
        "diag_disc_why": "统计信号可能来自控制器行为、传感器噪声或有效的快速动作，不能单独证明数据错误",
        "diag_disc_next": "结合轨迹上下文人工确认；如确有异常，再检查控制器参数或动作平滑处理",
        "diag_idle_what": "观察到有效运动比例偏低（Risk Signal）",
        "diag_idle_why": "可能是停顿、等待、任务特性或失败行为，仅凭该指标无法区分",
        "diag_idle_next": "先人工确认任务语义和轨迹，不要仅凭此信号排除或重采",
        "diag_vel_what": "速度/加速度超出正常范围",
        "diag_vel_why": "动作过于激进或轨迹异常",
        "diag_vel_next": "检查示教数据或控制器增益",
        "diag_dist_what": "轨迹分布偏离参考分布",
        "diag_dist_why": "可能是任务执行方式不同或异常行为",
        "diag_dist_next": "人工确认是否为有效但罕见的行为模式",
        "diag_generic_what": "Issue Severity Score {score:.2f}，略高于参考分布",
        "diag_generic_why": "多项指标轻微异常的累积效应",
        "diag_generic_next": "建议人工确认是否影响训练质量",
        "diag_why_fallback": "具体原因需结合数据上下文分析",
        "diag_next_fallback": "人工审核后决定保留或排除",
        "rec_missing_dropout": "检查采集系统的存储和传感器连接，确保录制过程中无丢包。",
        "rec_invalid_values": "检查数据采集管线的数值处理逻辑，排查 NaN/Inf 的产生源头。",
        "rec_schema_consistency": "检查数据格式版本一致性，确认所有 episode 的张量维度和 dtype 统一。",
        "rec_timestamp_validity": "检查采集系统的时间戳源，确保时钟同步和单调递增。",
        "rec_joint_limit": "检查机器人控制策略，避免关节打到机械限位。",
        "rec_sensor_synchronization": "校准多传感器的时间同步，或使用硬件触发同步。",
        "rec_sampling_jitter": "检查采集系统的采样时钟稳定性，可能需要优化调度优先级。",
        "rec_velocity_acceleration": "检查控制器参数，动作可能过于激进；或检查传感器噪声。",
        "rec_action_discontinuity": "这是统计风险信号。结合轨迹上下文人工确认后，再考虑调整控制器或加入动作滤波。",
        "rec_idle_ratio": "这是统计风险信号。先确认长时间停顿是否符合任务语义，不要仅凭该信号排除 episode。",
        "rec_distribution": "检查轨迹分布，离群 episode 可能对应失败或异常的任务执行。",
        "rec_coverage": "数据集状态空间覆盖度较低，建议增加多样化的采集场景。",
        # ---- visual metrics (v0.7.x UI catch-up, REQ-4) ----
        "rec_video_freeze": "相机流在机械臂运动期间出现冻结段——典型的相机掉线/录制卡顿特征。建议核对采集工控机与相机链路，重采受影响 span，不要直接用于需要视觉输入的训练。",
        "rec_video_timestamp_alignment": "视频流与 parquet 时间线跨度不一致——按帧索引训练会静默采样到错位时刻。建议核对录制起止时间戳与编码帧率。",
        "rec_video_stream_sync": "多相机流存在缺失或时长漂移——下游训练会静默丢失这些视角。建议检查相机接入与存储路径。",
        "rec_visual_quality": "视觉质量罚分来自模糊/暗帧/过曝（含高光削顶）测量，是观测信号不是否决——结合相机分辨率与场景光照人工确认后再决定是否重采。",
        # ---- visual audit section (Episode Explorer detail) ----
        "vis_header": "📷 视觉审计 (Layer 1C)",
        "vis_no_visual": "该数据集无视频流，视觉审计不适用。",
        "vis_dep_missing": "未安装 PyAV，视觉指标未执行——这是「未检测」而非「通过」。安装后重跑：pip install \"robot-data-audit[video]\"",
        "vis_va_a_header": "视觉流完整性（VA-A · 硬证据）",
        "vis_va_b_header": "视觉质量测量（VA-B · 观测）",
        "vis_vf_name": "视频冻结（运动中相机停帧）",
        "vis_vta_name": "视频-时间线对齐",
        "vis_vsync_name": "多相机同步",
        "vis_vq_name": "视觉质量（模糊/曝光/对比度）",
        "vis_freeze_regions": "冻结区域（前 {n} 个）：",
        "vis_freeze_none": "未检出运动中冻结段。",
        "vis_col_feature": "相机流",
        "vis_col_span": "span（parquet 帧）",
        "vis_col_duration": "时长 (s)",
        "vis_col_moving": "运动占比",
        "vis_vq_cameras": "按相机流罚分明细：",
        "vis_col_blur": "模糊方差(中位)",
        "vis_col_blur_pen": "模糊罚分",
        "vis_col_lum": "暗帧占比",
        "vis_col_bright": "过曝占比",
        "vis_col_contrast": "低对比占比",
        "vis_col_exposure_pen": "曝光罚分",
        "vis_col_penalty": "总罚分",
        "vis_col_dominant": "主要问题",
        "vis_samples_chart": "样本级质量曲线（越低越差；模糊方差轴已归一化）",
        "vis_blur_var_norm": "模糊方差（归一化）",
        "vis_mean_lum": "平均亮度",
        "vis_contrast": "对比度 (p5-p95)",
        "vis_clipped_frac": "高光削顶占比",
        "vis_thresholds": "阈值（SLE 先验）：模糊方差下限 {blur}、暗帧 {dark}、过曝均值 {bright}、对比度下限 {contrast}；高光削顶占比 ≥20% 计入过曝。",
        "vis_worst_frame": "最差帧 t={t}s",
        "vis_penalty_line": "质量罚分 **{pen}** · 主要问题 **{issue}** · 采样 {n} 帧",
        # ---- health overview: visual section ----
        "health_vis_header": "📷 视觉完整性概览",
        "health_vis_caption": "数据集级视觉指标统计（v0.7.0+ 视觉流审计）。检出为测量证据，不替代人工确认。",
        "health_vis_eps_checked": "已检 episodes",
        "health_vis_flagged": "视觉检出 episodes",
        "health_vis_not_checked": "未检 episodes",
        "health_vis_not_checked_hint": "视觉指标未执行（缺 PyAV 或无视频流）——未检测 ≠ 通过。",
        "health_vis_no_video": "该数据集未加载到视频流模态，视觉审计不适用。",
        "health_vis_all_ok": "视觉流完整性与质量均未发现问题。",
        "health_vis_flagged_list": "视觉检出明细（按指标）：",
    },

    "en": {
        # ---- app shell / sidebar ----
        "app_title": "RDA · Robot Data Quality Audit",
        "page_title": "RDA · Robot Data Quality Audit",
        "app_subtitle": "Robot Data Assurance — dataset quality auditing & governance",
        "nav_group_flow": "Workflow",
        "nav_group_results": "Audit Results",
        "nav_group_recommend": "Recommendations",
        "nav_group_trend": "Trends",
        "nav_upload": "Upload",
        "nav_audit": "Audit",
        "nav_health": "Health Overview",
        "nav_episode": "Episode Explorer",
        "nav_export": "Export",
        "nav_history": "Audit History",
        "nav_recommend": "Recommend",
        "sidebar_total_frames": "Total frames",
        "sidebar_modalities": "Modalities",
        "sidebar_no_dataset": "No dataset loaded yet",

        # ---- common gates ----
        "not_uploaded": "⚠️ Please upload a dataset first (Upload page)",
        "not_audited": "⚠️ Please run an audit first",
        "go_upload": "Go to Upload page →",
        "go_audit": "Go to Audit page →",

        # ---- page 1: upload ----
        "upload_title": "📤 Upload Dataset",
        "upload_caption": "Supports LeRobot-format datasets. Upload to run a quality audit",
        "upload_source": "Dataset source",
        "upload_platform": "Select platform",
        "upload_platform_help": "armnetbench / droid are preset platforms; custom = any LeRobot-format dataset",
        "upload_tab_local": "📁 Local path",
        "upload_tab_files": "📤 Upload folder",
        "upload_local_hint": "Enter a local LeRobot dataset directory path",
        "upload_path_label": "Dataset path",
        "upload_load_btn": "Load dataset",
        "upload_path_empty_err": "Please enter a dataset path",
        "upload_files_hint": "Upload dataset directory (multiple files)",
        "upload_uploader_label": "Choose dataset files (parquet / json / other LeRobot-format files)",
        "upload_uploader_help": "Select all files of the dataset; they will be saved to a temp directory",
        "upload_save_load_btn": "Save & load",
        "upload_loading": "Loading dataset...",
        "upload_loaded_ok": "Loaded {n} episodes successfully",
        "upload_load_err": "Failed to load: {err}",
        "upload_info_header": "✅ Dataset Info",
        "upload_metric_eps": "Episodes",
        "upload_metric_frames": "Total frames",
        "upload_metric_dof": "Action dims (DOF)",
        "upload_metric_fps": "Frame rate (FPS)",
        "upload_modalities": "📷 Observation modalities ({n})",
        "upload_actions": "🦾 Action space ({n})",
        "upload_meta": "📝 Metadata",
        "upload_success": "Dataset loaded! {eps} episodes, {frames} frames.",
        "upload_next_audit": "Next step: run audit →",

        # ---- page 2: audit ----
        "audit_title": "🔍 Run Audit",
        "audit_caption": "Run a full quality audit on the dataset with per-episode results",
        "audit_gate_upload": "⚠️ Please load a dataset first (Upload page)",
        "audit_dataset": "**Dataset**: `{path}`",
        "audit_summary_line": "**{eps} episodes · {frames} frames**",
        "audit_layers_header": "Enabled audit layers",
        "audit_layers_info": (
            "The audit pipeline includes the following layers, all enabled by default:\n\n"
            "**Layer 1A · Integrity checks** — missing frames, NaN/Inf, timestamps, "
            "schema consistency and other deterministic checks\n\n"
            "**Layer 1B · Behavioral quality** — reference-distribution-based statistical "
            "anomaly detection (motion, temporal, distribution)\n\n"
            "**Layer 2 · Diagnosis & attribution** — Pattern Type identification + "
            "root-cause analysis + suggestions"
        ),
        "audit_layers_note": "V0.1 does not yet include Layer 3 data governance; all check layers are enabled by default.",
        "audit_done_info": "✅ Audit completed — {eps} episodes",
        "audit_rerun_btn": "🔄 Re-run audit",
        "audit_view_health": "View health overview →",
        "audit_result_header": "Audit Result Summary",
        "audit_preparing": "Preparing...",
        "audit_loading_data": "Loading dataset and running audit...",
        "audit_progress": "Auditing... Episode {i}/{total} ({pct:.0f}%)",
        "audit_done_progress": "Audit complete!",
        "audit_done_status": "✅ Audit complete — {n} episodes",
        "audit_done_success": "Audit complete! Snapshot saved — see Audit History for trends",
        "audit_failed": "Audit failed: {err}",
        "audit_detail_err": "Detailed error",
        "audit_start_btn": "▶️ Start audit",

        # ---- page 3: health overview ----
        "health_title": "💚 Dataset Health Overview",
        "health_caption": "Overall dataset quality assessment · conclusions only, no algorithm details",
        "health_readiness_label": "Training Readiness",
        "health_usable_metric": "Current DHI",
        "health_usable_after_review": "Potentially usable after review",
        "health_usable_after_review_sub": "{n} / {total} episodes",
        "health_clear_keep": "Clear Keep",
        "health_clear_keep_sub": "{n} episodes",
        "health_usable_caption": "Potentially usable = KEEP + REVIEW (retained after manual review); excludes explicit REMOVE episodes",
        "health_dhi_header": "Data Health Index (DHI)",
        "health_dhi_note": "Relative assessment based on structural integrity<br/>and behavioral consistency, not an absolute score",
        "health_grade_suffix": "",
        "health_dims_header": "Four-dimension quality scores",
        "health_dim_integrity": "Integrity",
        "health_dim_temporal": "Temporal",
        "health_dim_motion": "Motion",
        "health_dim_consistency": "Consistency",
        "health_radar_header": "Radar chart",
        "health_top_issues_header": "🔎 Top Audit Observations",
        "health_no_issues": "🎉 No deterministic failures or elevated risk signals found",
        "health_metric_critical": "🔴 Critical",
        "health_metric_warning": "🟡 Risk Signals",
        "health_metric_types": "📋 Observation types",
        "health_sev_critical": "Critical",
        "health_sev_warning": "Warning",
        "health_sev_info": "Info",
        "health_sev_unknown": "Unknown",
        "health_issue_expander": "{sev} **{idx}. {desc}**  `{code}`  · affects {count} episodes ({pct:.1f}%)",
        "health_lbl_severity": "Severity",
        "health_lbl_scope": "Scope",
        "health_lbl_code": "Code",
        "health_recommendation": "**Next step (manual confirmation required)**",
        "health_view_episodes": "View per-episode details →",
        "health_verdict_keep_pct": "{pct:.1f}% of episodes",

        # ---- page 4: episode explorer ----
        "ep_title": "📋 Episode Explorer",
        "ep_caption": "Browse per-episode audit results, issue types and diagnosis, and record manual review decisions",
        "ep_score_view": "📊 Behavior score view",
        "ep_score_view_help": (
            "Portable Core: only cross-platform generic metrics (duration / spike / effective_motion), "
            "suitable for cross-platform comparison.\n"
            "Platform-specific: only platform-specific metrics (velocity / path_length etc.), "
            "meaningful within a single platform.\n"
            "Combined: all metrics combined, for deep same-platform analysis."
        ),
        "ep_view_portable": "Portable Core (cross-platform)",
        "ep_view_platform": "Platform-specific (same platform)",
        "ep_view_combined": "Combined (deep analysis)",
        "ep_metric_total": "Total episodes",
        "ep_metric_keep_ai": "🟢 KEEP (AI)",
        "ep_metric_review_ai": "🟡 REVIEW (AI)",
        "ep_metric_remove_ai": "🔴 REMOVE (AI)",
        "ep_review_progress": "Manual review progress: **{decided} / {need}** (KEEP: {keep} · REMOVE: {remove} · UNCERTAIN: {uncertain})",
        "ep_tab_queue": "🚨 Review Queue (priority)",
        "ep_tab_all": "📋 All Episodes",
        "ep_queue_empty": "🎉 No episodes require manual confirmation",
        "ep_queue_info": "**{n}** episodes need attention ({rm} deterministic exclusion suggestions, {rv} risk signals to review)",
        "ep_queue_no_items": "No episodes to review",
        "ep_queue_undecided": "⚪ Undecided",
        "ep_frames": "**Frames**",
        "ep_main_issues": "**Main issues**",
        "ep_notes_label": "Review notes",
        "ep_notes_placeholder": "Record review rationale...",
        "ep_view_detail": "📖 View details",
        "ep_table_empty": "No episodes match the filters",
        "ep_col_episode": "Episode #",
        "ep_col_verdict": "Verdict",
        "ep_col_integrity": "Integrity",
        "ep_col_behavior": "Behavior Score",
        "ep_col_pattern": "Pattern Type",
        "ep_col_frames": "Frames",
        "ep_col_issues": "Main issues",
        "ep_select_detail": "Select an episode to view details",
        "ep_detail_title": "🔍 Episode #{ep} Details",
        "ep_not_found": "Episode #{ep} not found",
        "ep_integrity_header": "Data Integrity (Layer 1A)",
        "ep_integrity_pass": "PASS · no hard-check issues",
        "ep_integrity_fail": "FAIL · integrity issues found",
        "ep_integrity_issues": "Issues found:",
        "ep_behavior_header": "Behavioral Quality (Layer 1B)",
        "ep_verdict_keep": "KEEP · recommended to keep",
        "ep_verdict_review": "REVIEW · needs manual review",
        "ep_verdict_exclude": "REMOVE · rule suggests exclusion (confirm manually)",
        "ep_metrics_header": "📊 Metric Details",
        "ep_col_metric": "Metric",
        "ep_col_status": "Status",
        "ep_col_value": "Measurement",
        "ep_col_note": "Notes",
        "ep_na_reason": "Data not available",
        "ep_error_reason": "Computation error",
        "ep_diag_header": "💡 Diagnosis & Suggestions",
        "ep_diag_normal": "All metrics normal — no obvious quality issues in this episode.",
        "ep_diag_what": "**WHAT · Issue**",
        "ep_diag_why": "**WHY · Evidence interpretation**",
        "ep_diag_next": "**NEXT · Manual confirmation**",
        "ep_decision_header": "✍️ Manual Review Decision",
        "ep_btn_keep": "✅ KEEP",
        "ep_btn_keep_help": "Confirm this episode is usable for training",
        "ep_btn_remove": "🗑️ REMOVE",
        "ep_btn_remove_help": "Confirm this episode should be excluded from the training set",
        "ep_btn_uncertain": "❓ UNCERTAIN",
        "ep_btn_uncertain_help": "Mark as uncertain for now, to be confirmed later",
        "ep_marked_keep": "Episode #{ep} marked as KEEP",
        "ep_marked_remove": "Episode #{ep} marked as REMOVE",
        "ep_marked_uncertain": "Episode #{ep} marked as UNCERTAIN",
        "ep_badge_keep": "Keep",
        "ep_badge_remove": "Remove",
        "ep_badge_uncertain": "Uncertain",
        "ep_current_decision": "Current decision: ",
        "ep_no_decision": "Current status: no decision",
        "ep_notes_area": "Review notes (optional)",
        "ep_notes_area_placeholder": "Record rationale, context, or follow-up actions...",
        "ep_save_notes": "💾 Save notes",
        "ep_notes_saved": "Notes saved",
        "ep_clear_decision": "🗑️ Clear decision",
        "ep_decision_cleared": "Decision cleared",
        "ep_close_detail": "Close details",
        "ep_traj_header": "📈 Trajectory Visualization",
        "ep_traj_no_path": "💡 Dataset path not set — cannot load raw trajectory data",
        "ep_traj_load_err": "⚠️ Failed to load trajectory data: {err}",
        "ep_traj_not_found": "⚠️ Raw data for Episode #{ep} not found",
        "ep_traj_no_data": "This episode has no state or action data — skipping visualization",
        "ep_traj_view_label": "Display mode",
        "ep_traj_state_title": "Joint States Time Series",
        "ep_traj_action_title": "Actions Time Series",
        "ep_traj_time_axis": "Time (s)",
        "ep_traj_value_axis": "Value",
        "ep_traj_legend": "Dimension",
        "ep_traj_loading": "Loading raw episode data...",
        "ep_filter_verdict": "Verdict filter",
        "ep_filter_integrity": "Integrity filter",
        "ep_filter_pattern": "Pattern Type filter",
        "ep_filter_showing": "Showing {n} / {total} episodes",
        "ep_score_severity": "Issue Severity Score: {v}",

        # ---- page 5: export ----
        "export_title": "📥 Export Report",
        "export_caption": "Export audit results as JSON / CSV, or export a cleaned dataset with bad episodes removed",
        "export_config_header": "Export configuration",
        "export_format_label": "Format",
        "export_format_help": "JSON contains the full audit result; CSV contains only the per-episode summary table",
        "export_scope_label": "Scope",
        "export_scope_all": "All episodes",
        "export_scope_queue": "Review Queue only",
        "export_scope_anomalies": "Anomalous episodes only",
        "export_scope_help": "Review Queue = REVIEW + EXCLUDE; anomalies only = EXCLUDE",
        "export_preview_header": "Export preview",
        "export_metric_format": "Format",
        "export_metric_scope": "Scope",
        "export_verdicts_info": "📝 {n} episodes decided in scope: 🟢 KEEP {keep} · 🔴 REMOVE {remove} · 🟡 UNCERTAIN {uncertain}",
        "export_no_verdicts": "💡 No manual review decisions yet — export will contain automated audit results only",
        "export_download_header": "Download",
        "export_download_json": "📥 Download JSON report",
        "export_download_csv": "📥 Download CSV report",
        "export_preview_json": "Preview JSON structure",
        "export_preview_json_note": "... {n} episodes total — showing first 2 only",
        "export_preview_csv": "Preview CSV content",
        "export_preview_csv_note": "... {n} rows of data",
        "export_docs_header": "📝 Report Notes",
        "export_json_docs": "JSON report fields",
        "export_csv_docs": "CSV report fields",
        "export_json_docs_body": (
            "**JSON report structure:**\n"
            "- `version`: RDA version\n"
            "- `dataset`: dataset metadata (path, episode count, total frames, modalities)\n"
            "- `summary`: summary statistics (verdict distribution, pass rate)\n"
            "- `quality`: quality assessment (DHI, grade, Training Readiness, 4-dimension scores)\n"
            "- `hero_metrics`: key metric digest\n"
            "- `top_observations`: top observations/issues\n"
            "- `user_verdicts`: manual review decisions (episode_id → {decision, notes})\n"
            "- `episodes`: per-episode result list\n"
            "  - `episode_index`: episode number\n"
            "  - `num_frames`: frame count\n"
            "  - `verdict`: verdict (PASS / REVIEW / EXCLUDE)\n"
            "  - `metrics`: per-metric results (availability / measurement / assessment)"
        ),
        "export_csv_docs_body": (
            "**CSV fields:**\n"
            "- `episode_id`: episode number\n"
            "- `num_frames`: frame count\n"
            "- `integrity_check`: integrity check (PASS / FAIL)\n"
            "- `behavior_verdict`: behavioral verdict (PASS / REVIEW / EXCLUDE; shown as KEEP / REVIEW / REMOVE in the UI)\n"
            "- `issue_severity_score`: issue severity score (0-100, higher = more anomalous; formerly deviation_score)\n"
            "- `pattern_type`: anomaly pattern type\n"
            "- `issue_count`: number of issues\n"
            "- `top_issues`: main issue list\n"
            "- `user_decision`: manual review decision (KEEP / REMOVE / UNCERTAIN; empty = undecided)\n"
            "- `user_notes`: review notes"
        ),
        "export_clean_header": "📦 Export Cleaned Dataset",
        "export_clean_caption": "Remove bad episodes and output a training-ready LeRobot v3.0 dataset",
        "export_strategy_header": "🧹 Cleaning strategy",
        "export_strategy_label": "Select cleaning strategy",
        "export_strategy_conservative": "🛡️ Conservative — remove EXCLUDE only, keep REVIEW",
        "export_strategy_aggressive": "⚔️ Aggressive — remove EXCLUDE + REVIEW",
        "export_strategy_custom": "🔧 Custom — manually choose which AI verdicts to remove",
        "export_custom_caption": "Choose AI verdict types to remove:",
        "export_custom_pass": "❌ PASS (normal)",
        "export_custom_pass_help": "Remove undecided episodes judged PASS by the AI (usually not recommended)",
        "export_custom_review": "⚠️ REVIEW (needs review)",
        "export_custom_review_help": "Remove undecided episodes judged REVIEW by the AI",
        "export_custom_exclude": "🚫 EXCLUDE (remove)",
        "export_custom_exclude_help": "Remove undecided episodes judged EXCLUDE by the AI",
        "export_preview_panel": "📊 Cleaning preview",
        "export_preview_dataset": "**Dataset:** `{name}` · **Total episodes:** {n:,}",
        "export_preview_table": (
            "| Status | Count | Share |\n|:---|---:|---:|\n"
            "| 🟢 Keep (KEEP) | {keep:,} | {p_keep} |\n"
            "| 🟡 Needs review (REVIEW) | {review:,} | {p_review} |\n"
            "| 🔴 Remove (REMOVE) | {remove:,} | {p_remove} |\n"
            "| ⚪ Undecided | {undecided:,} | {p_undecided} |"
        ),
        "export_strategy_summary": "🧹 **With the current strategy:**",
        "export_summary_remove": "- Remove **{n}** episodes",
        "export_summary_keep": "- Keep **{n}** episodes",
        "export_summary_usable": "- Estimated usable data: **{p}**",
        "export_summary_uncertain": "- ⚠️ {n} UNCERTAIN episodes handled by the uncertainty strategy",
        "export_removal_list": "📋 Episodes to remove ({n})",
        "export_removal_remaining": "... {n} more episodes not shown",
        "export_removal_col_id": "Episode ID",
        "export_removal_col_pattern": "Pattern Type",
        "export_removal_col_score": "Score",
        "export_removal_col_source": "Source",
        "export_source_user": "👤 User removal",
        "export_config_sub": "⚙️ Export configuration",
        "export_uncertainty_label": "UNCERTAIN handling",
        "export_uncertainty_keep": "Keep (default)",
        "export_uncertainty_remove": "Remove",
        "export_uncertainty_help": "Whether episodes marked UNCERTAIN by the user are kept or removed",
        "export_output_path": "Output path",
        "export_output_path_help": "Output directory for the cleaned dataset",
        "export_run_btn": "📦 Run export",
        "export_running": "Exporting...",
        "export_done": "Export complete!",
        "export_success": (
            "✅ Cleaned dataset exported successfully!\n\n"
            "- Output path: `{path}`\n"
            "- Kept episodes: {kept} / {total}\n"
            "- Removed episodes: {removed}\n"
            "- Total frames: {frames}\n"
            "- Cleaning strategy: {strategy}\n"
            "- Format: LeRobot v3.0 (parquet + meta)"
        ),
        "export_stats": "📊 Export statistics",
        "export_failed": "❌ Export failed: {err}",

        # ---- page 6: history ----
        "history_title": "📈 Audit History",
        "history_caption": "Track data quality trends · comparative analysis across audits",
        "history_empty": (
            "No audit history yet.\n\n"
            "A snapshot is saved automatically after each audit on the Audit page.\n"
            "Complete at least one audit to view trends here."
        ),
        "history_filter_label": "📂 Filter by dataset",
        "history_filter_all": "All",
        "history_overview": "📊 History Overview",
        "history_metric_runs": "Audits",
        "history_metric_latest": "Latest audit",
        "history_metric_pass": "Latest PASS rate",
        "history_need_two": "⚠️ At least 2 audits are needed to show trends",
        "history_trend_header": "📈 Trend Analysis",
        "history_verdict_chart": "**Verdict distribution over time**",
        "history_verdict_keep": "PASS · keep",
        "history_verdict_review": "REVIEW · review",
        "history_verdict_exclude": "EXCLUDE · remove",
        "history_axis_run": "Audit #",
        "history_axis_eps": "Episode count",
        "history_dev_chart": "**Mean Deviation Score trend**",
        "history_dev_mean": "Mean",
        "history_dev_median": "Median",
        "history_excl_chart": "**EXCLUDE count trend**",
        "history_excl_series": "EXCLUDE count",
        "history_excl_axis": "EXCLUDE episode count",
        "history_integrity_chart": "**Integrity pass rate trend**",
        "history_integrity_series": "Integrity pass rate",
        "history_rate_axis": "Pass rate (%)",
        "history_list_header": "📋 Audit Records",
        "history_integrity_pass_metric": "Integrity pass rate",
        "history_mean_dev": "Mean deviation: {mean:.2f} · median: {median:.2f} · RDA v{ver}",
        "history_top_patterns": "🔍 Top patterns: {patterns}",
        "history_sensor_sync": "Sensor sync: {v:.1f}ms ({interp})",
        "history_sensor_sync_na": "Sensor sync: N/A",
        "history_action_jitter": "Action jitter: {n} spikes",
        "history_space_coverage": "Space coverage: {v:.1%} ({interp})",

        # ---- page 7: recommend ----
        "rec_title": "🧭 Recommend · Data Optimization Advice",
        "rec_caption": "Evidence-graded optimization advice based on temporal structure; recommendation evidence is separate from input-data risk signals in the audit.",
        "rec_no_path": "⚠️ Dataset path not found — please re-upload the dataset.",
        "rec_policy_label": "Target model architecture",
        "rec_policy_fw": "frame-wise — MLP / BC (per-frame policy)",
        "rec_policy_tp": "temporal — ACT / Diffusion Policy / Transformer",
        "rec_policy_help": "Architectures differ in tolerance for idle frames and valid windows — pick the one you actually train.",
        "rec_privacy": "🔒 Privacy: raw episode data is never uploaded. Metrics are computed locally; only aggregated statistics (<1KB) are sent to the RDA API for rule evaluation.",
        "rec_chunk_label": "Policy chunk size (optional)",
        "rec_chunk_help": "How many actions your policy executes per chunk (e.g. ACT/DROID chunk=8/10). When declared, idle-trim advice is evaluated against this window length — aligned with the openpi/DROID filtering semantics. Leave empty for generic window tiers.",
        "rec_chunk_none": "Not declared (generic tiers)",
        "rec_run_btn": "🚀 Generate recommendations",
        "rec_computing": "Computing temporal-structure metrics locally...",
        "rec_calling_api": "Calling RDA rules engine (rules API)...",
        "rec_done": "Done!",
        "rec_rules_ver": "Rules version: rules v{v}",
        "rec_report_header": "Recommendation Report",
        "rec_cards_header": "Recommendations",
        "rec_confidence": "Confidence: {v}",
        "rec_expected_impact": "**Expected impact:** {v}",
        "rec_details": "**Details:**",
        "rec_caveats": "**Caveats:**",
        "rec_raw_json": "View raw JSON",
        "rec_failed": "Failed to generate recommendations: {err}",
        "rec_failed_hint": "Tip: if the network is unavailable, check the RDA_API_URL environment variable; cached results are used as fallback when available.",
        "rec_disclaimer": (
            "RDA provides data quality diagnostics and low-risk optimization advice — "
            "success-rate improvements are not guaranteed. Validate any pruning on a held-out set "
            "before using it for production training."
        ),

        # ---- misc ----
        "issue_missing_dropout": "Missing frames / sensor dropout",
        "issue_invalid_values": "NaN/Inf invalid values",
        "issue_schema_consistency": "Schema mismatch",
        "issue_timestamp_validity": "Timestamp anomalies (non-monotonic / uneven intervals)",
        "issue_joint_limit": "Joint limit violation",
        "issue_sensor_synchronization": "Multi-sensor sync deviation",
        "issue_sampling_jitter": "Sampling jitter",
        "issue_velocity_acceleration": "Velocity/acceleration anomalies",
        "issue_action_discontinuity": "Action discontinuity (jitter)",
        "issue_idle_ratio": "Low effective-motion ratio",
        "issue_distribution": "Trajectory distribution outlier",
        "issue_coverage": "Insufficient state-space coverage",
        "rec_generic": "Manually review the affected episodes to identify the cause.",
        "no_issues": "None",
        "radar_score": "Score",
        "dhi_empty": "No episodes in dataset",
        "dhi_ready": "Data is ready for training — {n} episodes",
        "dhi_cond": "Usable for training after reviewing {n} episodes",
        "dhi_not_ready": "Significant quality issues found — clean the data first",
        "diag_integrity_what": "Integrity checks failed ({metrics})",
        "diag_integrity_why": "Deterministic errors occurred during data collection or storage",
        "diag_integrity_next": "Check the data collection pipeline; fix and re-record",
        "diag_disc_what": "Observed an action-discontinuity pattern (Risk Signal)",
        "diag_disc_why": "The signal may reflect controller behavior, sensor noise, or a valid fast maneuver; it does not prove corruption",
        "diag_disc_next": "Confirm against trajectory context before checking controller parameters or applying smoothing",
        "diag_idle_what": "Observed a low effective-motion ratio (Risk Signal)",
        "diag_idle_why": "The pattern may be waiting, task-specific behavior, or failure; the metric alone cannot distinguish them",
        "diag_idle_next": "Confirm task semantics and trajectory first; do not exclude or re-record from this signal alone",
        "diag_vel_what": "Velocity/acceleration out of normal range",
        "diag_vel_why": "Motions may be too aggressive or trajectories abnormal",
        "diag_vel_next": "Check teleoperation data or controller gains",
        "diag_dist_what": "Trajectory distribution deviates from reference",
        "diag_dist_why": "Possibly a different task execution style or abnormal behavior",
        "diag_dist_next": "Manually confirm whether this is valid but rare behavior",
        "diag_generic_what": "Issue Severity Score {score:.2f}, slightly above the reference distribution",
        "diag_generic_why": "Cumulative effect of mild anomalies across multiple metrics",
        "diag_generic_next": "Manually confirm whether training quality is affected",
        "diag_why_fallback": "Root cause requires analysis in data context",
        "diag_next_fallback": "Decide keep/remove after manual review",
        "rec_missing_dropout": "Check storage and sensor connections of the collection system to avoid packet loss during recording.",
        "rec_invalid_values": "Check numeric handling in the data pipeline and trace the source of NaN/Inf values.",
        "rec_schema_consistency": "Check format version consistency; confirm tensor shapes and dtypes are uniform across episodes.",
        "rec_timestamp_validity": "Check the timestamp source of the collection system; ensure clock sync and monotonic increase.",
        "rec_joint_limit": "Check the robot control policy to avoid driving joints to mechanical limits.",
        "rec_sensor_synchronization": "Calibrate multi-sensor time sync, or use hardware-triggered sync.",
        "rec_sampling_jitter": "Check sampling clock stability of the collection system; scheduler priority tuning may be needed.",
        "rec_velocity_acceleration": "Check controller parameters — actions may be too aggressive; or check sensor noise.",
        "rec_action_discontinuity": "This is a statistical risk signal. Confirm it against trajectory context before tuning the controller or applying action filtering.",
        "rec_idle_ratio": "This is a statistical risk signal. Confirm whether long pauses fit the task semantics; do not exclude episodes from this signal alone.",
        "rec_distribution": "Check trajectory distribution; outlier episodes may correspond to failed or abnormal task executions.",
        "rec_coverage": "State-space coverage is low — consider adding diverse collection scenarios.",
        # ---- visual metrics (v0.7.x UI catch-up, REQ-4) ----
        "rec_video_freeze": "A camera stream froze while the arm was moving — the classic camera drop-out / recorder stall signature. Check the capture host and camera link, re-record the affected spans, and do not feed them to vision-dependent training.",
        "rec_video_timestamp_alignment": "Video and parquet timeline spans disagree — frame-index training would silently sample misaligned moments. Verify recording start/end timestamps and encoding fps.",
        "rec_video_stream_sync": "Camera streams are missing or their spans drift — downstream training silently loses those views. Check camera wiring and storage paths.",
        "rec_visual_quality": "The visual-quality penalty comes from blur / dark / blown-out (incl. highlight clipping) measurements. It is observational, not a veto — confirm with camera resolution and scene lighting before re-recording.",
        # ---- visual audit section (Episode Explorer detail) ----
        "vis_header": "📷 Visual Audit (Layer 1C)",
        "vis_no_visual": "This dataset has no video streams; the visual audit does not apply.",
        "vis_dep_missing": "PyAV is not installed, so visual metrics were NOT executed — this is \"not audited\", never \"pass\". Install and re-run: pip install \"robot-data-audit[video]\"",
        "vis_va_a_header": "Visual-Stream Integrity (VA-A · hard evidence)",
        "vis_va_b_header": "Visual Quality Measurement (VA-B · observational)",
        "vis_vf_name": "Video freeze (camera stalled while moving)",
        "vis_vta_name": "Video–timeline alignment",
        "vis_vsync_name": "Multi-camera sync",
        "vis_vq_name": "Visual quality (blur / exposure / contrast)",
        "vis_freeze_regions": "Freeze regions (first {n}):",
        "vis_freeze_none": "No frozen-while-moving spans detected.",
        "vis_col_feature": "Camera stream",
        "vis_col_span": "span (parquet frames)",
        "vis_col_duration": "Duration (s)",
        "vis_col_moving": "Moving ratio",
        "vis_vq_cameras": "Per-camera penalty breakdown:",
        "vis_col_blur": "Blur var (median)",
        "vis_col_blur_pen": "Blur penalty",
        "vis_col_lum": "Dark frac",
        "vis_col_bright": "Blown frac",
        "vis_col_contrast": "Low-contrast frac",
        "vis_col_exposure_pen": "Exposure penalty",
        "vis_col_penalty": "Penalty",
        "vis_col_dominant": "Dominant issue",
        "vis_samples_chart": "Per-sample quality curves (lower is worse; blur variance axis normalized)",
        "vis_blur_var_norm": "Blur variance (normalized)",
        "vis_mean_lum": "Mean luminance",
        "vis_contrast": "Contrast (p5-p95)",
        "vis_clipped_frac": "Clipped-highlight frac",
        "vis_thresholds": "Thresholds (SLE priors): blur var floor {blur}, dark mean {dark}, bright mean {bright}, contrast floor {contrast}; clipped-highlight frac ≥20% counts as exposure.",
        "vis_worst_frame": "Worst frame t={t}s",
        "vis_penalty_line": "Quality penalty **{pen}** · dominant issue **{issue}** · {n} frames sampled",
        # ---- health overview: visual section ----
        "health_vis_header": "📷 Visual Integrity Overview",
        "health_vis_caption": "Dataset-level visual-metric statistics (v0.7.0+ visual-stream audit). Findings are measurement evidence, not a replacement for human review.",
        "health_vis_eps_checked": "Episodes checked",
        "health_vis_flagged": "Episodes flagged",
        "health_vis_not_checked": "Not checked",
        "health_vis_not_checked_hint": "Visual metrics did not run (PyAV missing or no video streams) — not audited ≠ pass.",
        "health_vis_no_video": "No video modality loaded for this dataset; the visual audit does not apply.",
        "health_vis_all_ok": "Visual-stream integrity and quality found no issues.",
        "health_vis_flagged_list": "Visual findings by metric:",
    },
}
