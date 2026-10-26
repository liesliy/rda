"""conform 报告生成：markdown 全文 + JSON 的 ``conformance`` 节点。

报告纪律
--------
* 抽样规模必须在正文里写出来——抽了 20 个 episode 就写 20，不许含糊成
  "对数据集全域进行扫描"。
* UNDETERMINED 必须在正文里单列，不许混进 PASS 区域。
* coverage 门禁未实现就写"未实现"，不许写成"已通过"。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from rda.conform.engine import ConformanceResult
from rda.conform.standards.gbz218_1 import (
    ORIGIN_DECLARED,
    ORIGIN_ENGINEERING,
    ORIGIN_LABEL,
    ORIGIN_STANDARD,
)
from rda.conform.verdict import Verdict

_CONFORMANCE_NODE_VERSION = "1.0"

_ORIGIN_LEGEND = {
    ORIGIN_STANDARD: " —— GB/Z 218.1-2026 正文给出的阈值，照抄。",
    ORIGIN_ENGINEERING: (
        " —— 标准未给数值，本工具为可判定而自行选定的口径，**不是国标要求值**，"
        "复核时须单独审视。"
    ),
    ORIGIN_DECLARED: (
        " —— 标准把该值交给数据提供方声明，工具不得代为取值；未声明即判"
        " UNDETERMINED。"
    ),
}


def _declared_ws(prof: dict, result) -> bool:
    """Whether the feasible-space max distance was declared for this run."""
    for it in result.items:
        if it.metric == "trajectory_hausdorff_ratio":
            return it.details.get("workspace_max_distance") is not None
    return False

# 四态的中文标签
_VERDICT_LABEL = {
    Verdict.PASS.value: "符合",
    Verdict.FAIL.value: "不符合",
    Verdict.NOT_APPLICABLE.value: "不适用",
    Verdict.UNDETERMINED.value: "无法裁定",
}

_ORDER = [Verdict.FAIL.value, Verdict.UNDETERMINED.value,
          Verdict.PASS.value, Verdict.NOT_APPLICABLE.value]


def build_conformance_node(result: ConformanceResult) -> Dict[str, Any]:
    """Build the ``conformance`` node to be merged into the JSON report.

    这是一个**新增的顶层键**，不改动既有 report 结构；既有消费者不读它
    就不会受影响。
    """
    node = result.to_dict()
    node["conformance_node_version"] = _CONFORMANCE_NODE_VERSION
    return node


def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, dict):
        return "；".join(f"{k}={_fmt_value(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "、".join(str(x) for x in v)
    return str(v)


def render_conformance_markdown(result: ConformanceResult,
                                generated_at: Optional[str] = None) -> str:
    """Render the full conformance report as markdown."""
    from rda import __version__ as rda_version

    s = result.summary
    lines: list[str] = []
    add = lines.append

    add(f"# RDA 符合性测评报告 — {result.standard}")
    add("")
    add(f"- 数据集：`{result.dataset}`")
    add(f"- 标准：{result.standard}")
    add(f"- RDA 版本：{rda_version}")
    if generated_at:
        add(f"- 生成时间：{generated_at}")
    add(f"- 总体结论：**{result.overall}（{_VERDICT_LABEL.get(result.overall, '')}）**")
    add("")

    # --- 汇总 -------------------------------------------------------------
    add("## 1. 结论汇总")
    add("")
    add("| 结论 | 数量 |")
    add("|---|---|")
    add(f"| 符合 PASS | {s['pass']} |")
    add(f"| 不符合 FAIL | {s['fail']} |")
    add(f"| 不适用 NOT_APPLICABLE | {s['not_applicable']} |")
    add(f"| 无法裁定 UNDETERMINED | {s['undetermined']} |")
    add(f"| **合计** | **{s['total']}** |")
    add("")
    add(f"- 强制性条款（应）FAIL：**{s['mandatory_fail']}**")
    add(f"- 建议性条款（宜）FAIL：**{s['advisory_fail']}**（按标准要求与"
        f"应类 FAIL 分列统计；二者都会使 CLI 退出码为 1）")
    add(f"- 信息性输出（—）：{s['informational']}")
    add(f"- 可裁定覆盖率（PASS+FAIL）/（全部判定项）：{s['coverage']:.4f}")
    if s["undetermined_metrics"]:
        add(f"- 未裁定的判定项：**{', '.join(s['undetermined_metrics'])}**")
    add("")

    # --- 判定项明细 --------------------------------------------------------
    add("## 2. 判定项明细")
    add("")
    add("| 条款 | 判定项 | binding | 实测值 | 阈值 | 结论 | 说明 |")
    add("|---|---|---|---|---|---|---|")
    items = sorted(result.items,
                   key=lambda it: (_ORDER.index(it.verdict.value), it.clause))
    for it in items:
        note = it.note.replace("\n", " ").replace("|", "/")
        if len(note) > 220:
            note = note[:220] + "…"
        add(f"| {it.clause} | `{it.metric}` | {it.binding} | {_fmt_value(it.value)} "
            f"| {_fmt_value(it.threshold)} | **{it.verdict.value}** | {note} |")
    add("")

    # --- 判定方法（可复算性）----------------------------------------------
    add("## 3. 判定方法与抽样口径")
    add("")
    for it in sorted(result.items, key=lambda x: x.clause):
        add(f"- **{it.clause} `{it.metric}`**：{it.method}")
    add("")

    # --- 观测项（附录 B）---------------------------------------------------
    add("## 4. 观测项（附录 B，标准未设阈值，不做判定）")
    add("")
    obs = result.observations or {}
    hd = obs.get("trajectory_hausdorff", {})
    if hd.get("mean") is not None:
        add(f"- **轨迹 Hausdorff 均值**：{hd['mean']:.6f} ± {hd['std']:.6f}"
            f"（min {hd['min']:.4f} / max {hd['max']:.4f}）")
        add(f"  - 抽样：{hd['episodes_sampled']} 个 episode，{hd['n_pairs']} 对，"
            f"重采样 T={hd['resample_T']}，seed={hd['seed']}，{hd['normalization']}")
    else:
        add(f"- 轨迹 Hausdorff 均值：{hd.get('note', '未计算')}")
    dt = obs.get("dtw", {})
    if dt.get("mean_dtw_over_L") is not None:
        add(f"- **DTW/L**：{dt['mean_dtw_over_L']:.6f} ± {dt['std_dtw_over_L']:.6f}"
            f"（字段 {dt['field']}，{dt['dim']} 维，{dt['n_pairs']} 对）")
        add(f"  - 抽样：{dt['episodes_sampled']} 个 episode（上限 {dt['max_pairs']} 对），"
            f"seed={dt['seed']}；{dt['formula']}")
        add(f"  - 重采样点数 resample_T={dt.get('resample_T')}"
            f"（须与 Hausdorff 的 resample_T 同口径才可比）")
    else:
        add(f"- DTW/L：{dt.get('note', '未计算')}")
    ac = obs.get("autocorrelation", {})
    if ac.get("per_episode"):
        add(f"- **自相关 ACF(1)（action）**：跨 {ac['episodes_sampled']} 个 episode 的"
            f"各维均值再平均 = {ac['acf_lag1_mean_over_episodes']:.6f}")
        add(f"  - {ac['formula']}")
        vzc = ac.get("velocity_first_zero_crossing_median_over_episodes")
        if vzc is not None:
            add(f"  - 速度层（一阶差分）首个过零 lag 的中位数（跨 episode）≈ {vzc:.0f} 帧")
        for e in ac["per_episode"]:
            add(f"  - episode {e['episode']}：{e['frames']} 帧，ACF(1) 均值 "
                f"{e['acf_lag1_mean']:.6f}，速度层首个过零中位数 "
                f"{e['velocity_first_zero_crossing_lag_median']}")
    else:
        add(f"- 自相关：{ac.get('note', '未计算')}")
    add("")

    # --- 生效阈值 ----------------------------------------------------------
    add("## 5. 本次判定生效的阈值")
    add("")
    prof = result.profile or {}
    ths = prof.get("thresholds", {})
    add("| key | 条款 | binding | 值 | 比较 | 量 | 口径来源 |")
    add("|---|---|---|---|---|---|---|")
    for key in sorted(ths):
        t = ths[key]
        origin = t.get("origin", ORIGIN_STANDARD)
        label = ORIGIN_LABEL.get(origin, origin)
        add(f"| `{key}` | {t['clause']} | {t['binding']} | {_fmt_value(t['value'])} "
            f"| {t['comparator']} | {t['quantity']} | {label} |")
    add("")
    add("口径来源图例：")
    for k in (ORIGIN_STANDARD, ORIGIN_ENGINEERING, ORIGIN_DECLARED):
        add(f"- `{k}` = {ORIGIN_LABEL[k]}" + _ORIGIN_LEGEND.get(k, ""))
    add("")
    add(f"- profile 来源：{prof.get('source') or '内置默认（标准原文）'}")
    add(f"- 丢帧识别系数 frame_loss_factor：{prof.get('frame_loss_factor')}"
        f"（**工程口径**，标准只给了 1% 限值、未规定识别系数，可由 profile 覆盖）")
    add("- 委托方声明值：可行解空间最大距离"
        f"（{'已声明' if _declared_ws(prof, result) else '未声明'}，"
        "未声明时 6.2.3.4 a 判 UNDETERMINED，不得由实测值反推）")
    add("")

    # --- 数据集事实 --------------------------------------------------------
    add("## 6. 数据集事实（用于复核）")
    add("")
    for k, v in (result.dataset_info or {}).items():
        add(f"- {k}: {_fmt_value(v)}")
    add("")

    # --- 门禁与遗留 --------------------------------------------------------
    add("## 7. 门禁状态与遗留项")
    add("")
    gate = result.coverage_gate or {}
    add(f"- coverage 门禁：enabled={gate.get('enabled')}，"
        f"status=**{gate.get('status')}**，coverage={gate.get('coverage')}")
    add(f"  - {gate.get('note', '')}")
    add("")

    return "\n".join(lines) + "\n"
