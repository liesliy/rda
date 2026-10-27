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

# 附录 A 观测项的计算状态 → 中文说明。观测项没有四态结论，用"能不能算出来"
# 表达，不冒充 PASS/FAIL。
_STATUS_LABEL = {
    "MEASURED": "本次已实测",
    "NOT_APPLICABLE": "条款对该数据集不适用",
    "NOT_MEASURED": "数据存在但本次算不出来（不等于通过）",
    "NO_VIDEO_FEATURES": "数据集未声明视频流特征",
    "NO_PAYLOAD": "本地副本缺少视频载荷文件",
    "NO_DATA": "已读到视频文件但没解出有效帧对",
    "DEPENDENCY_MISSING": "本地环境缺少 av / opencv",
    "OFF": "已按 --video=off 关闭视频层",
}


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
    obs = result.observations or {}   # 附录 A / B 两节共用同一份观测结果
    add("## 4. 观测项（附录 B，标准未设阈值，不做判定）")
    add("")
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

    # --- 观测项（附录 A 动态性测量）---------------------------------------
    add("## 5. 观测项（附录 A 动态性测量，标准未设数据阈值，不做判定）")
    add("")
    add("- 视频层档位：`--video={}`".format(
        (obs.get("appendix_a_optical_flow") or {}).get("mode", "未记录")))
    add("")

    # A.1 光流
    flow = obs.get("appendix_a_optical_flow") or {}
    add("### 附录 A.1 帧间光流位移（`—` 信息性输出）")
    add("")
    add(f"- 计算状态：`{flow.get('status', '未计算')}`"
        f"（{_STATUS_LABEL.get(flow.get('status', ''), '')}）")
    sp = flow.get("sampling") or {}
    if sp:
        add(f"- 抽样：{len(sp.get('episodes_sampled') or [])} / "
            f"{sp.get('total_episodes') or '?'} 个 episode"
            f"（episode {sp.get('episodes_sampled')}），"
            f"每 episode 每路相机 {sp.get('video_frame_pairs')} 个相邻帧对，"
            f"seed={sp.get('seed')}")
    feat = flow.get("features") or []
    if feat:
        add("- 按**特征级**的载荷覆盖情况（不看 videos/ 目录在不在）：")
        for f in feat:
            add(f"  - `{f['feature']}`：本地 payload 文件数 {f['file_count']}"
                f"（{'有' if f['file_count'] else '无'}）")
    cams = flow.get("cameras") or []
    if cams:
        add("")
        # max 列特意标 ⚠：它是单点极值，单独引用会误导复核（见下方脚注）
        add("| 相机 | 分辨率 | fps | p50 (px) | p95 (px) | p99.9 (px) | "
            "max (px) ⚠ | n>P99 | LK 残差 err p50/p95/该max点 | "
            "有效点/总点 | 帧对(用/剔) | 解码帧数 |")
        add("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cams:
            d = c.get("displacement_px") or {}
            te = c.get("track_err") or {}
            add(f"| `{c['feature']}` | {c.get('frame_shape') or '—'} | "
                f"{_fmt_value(c.get('fps'))} | "
                f"{_fmt_value(d.get('p50'))} | {_fmt_value(d.get('p95'))} | "
                f"{_fmt_value(d.get('p99_9'))} | {_fmt_value(d.get('max'))} | "
                f"{d.get('n_above_p99', 0)} | "
                f"{_fmt_value(te.get('p50'))} / {_fmt_value(te.get('p95'))} / "
                f"{_fmt_value(te.get('at_max'))} | "
                f"{c['n_valid_points']}/{c['n_total_points']} | "
                f"{c['n_pairs_used']}/{c['n_pairs_rejected']} | {c['n_frames_decoded']} |")
        add("")
        add("**列口径说明（第三方请以这些为准阅读上表）**：")
        add("")
        add("- **max (px) ⚠ 是单点极值，不是稳健统计量**：它取全部有效跟踪点里"
            "的**单一最大值**，一个跟丢的点就能把它抬到整帧真实运动的好几倍，"
            "**不得单独引用**。")
        add("- 判定用的是 **p95**（保守口径），复核的稳健上界请看 **p99.9**，"
            "尾部点的可信程度请看 **LK 残差 err** —— `err` 是 "
            "`cv2.calcOpticalFlowPyrLK` 对每个点返回的 patch SSD 残差，本工具"
            "**只用其披露、不用其剔除**（没有依据给 err 划一条'多大算坏'的线）。"
            "最后一列给出产生 max 的那个点自身的残差，数值远高于 p50 时即可"
            "判断该 max 是跟踪野值。")
        add("- `n>P99` = 超过该相机 p99 的点数，用于判断尾部到底是'一小撮野值'"
            "还是'确实存在一个运动的子集'。")
        add("")
        # 剔除原因必须逐条披露：静默丢点会让 statistic 失去可复算性
        for c in cams:
            if c.get("decode_errors") or c.get("notes"):
                add(f"- `{c['feature']}`：")
                for e in c.get("decode_errors", []):
                    add(f"  - 解码失败：{e}")
                for n in c.get("notes", []):
                    add(f"  - {n}")
            eps = c.get("episodes") or []
            if eps:
                add("")
                add(f"  `{c['feature']}` 逐 episode 明细（抽样到第几个 episode "
                    f"必须可核对，不能只给一个聚合值；max 列同上方口径，为单点极值）：")
                add("")
                add("  | episode | 载荷文件 | 窗口起点(s) | 解码帧数 | 帧对 | "
                    "有效点/总点 | p50 | p95 | p99.9 | max ⚠ | err p50 | err@max |")
                add("  |---|---|---|---|---|---|---|---|---|---|---|")
                for e in eps:
                    add(f"  | {e['episode']} | {e['file']} | "
                        f"{_fmt_value(e.get('start_sec'))} | {e['n_frames_decoded']} | "
                        f"{e['n_pairs_used']} | {e['n_valid_points']}/{e['n_total_points']} | "
                        f"{_fmt_value(e['p50'])} | {_fmt_value(e['p95'])} | "
                        f"{_fmt_value(e.get('p99_9'))} | {_fmt_value(e['max'])} | "
                        f"{_fmt_value(e.get('track_err_p50'))} | "
                        f"{_fmt_value(e.get('track_err_at_max'))} |")
                add("")
    else:
        add("- 每路相机的位移统计：**无**（见上方状态与 note）")
    for n in flow.get("notes") or []:
        add(f"- {n}")
    if flow.get("dependency_error"):
        add(f"- 依赖缺失：{flow['dependency_error']}"
            f"（**本地环境未安装，非数据集缺陷**）")
    add(f"- 临时文件：{flow.get('temp_files', '未记录')}")
    add(f"- 剔除规则：{flow.get('reject_rule', '')}")
    add(f"- 百分位插值方法：{flow.get('percentile_method', '')}")
    add("")

    # A.2 频谱流量
    sf = obs.get("appendix_a_spectral_flux") or {}
    add("### 附录 A.2 频谱流量 SF(t)")
    add("")
    add(f"- 公式：`{sf.get('formula', '')}`")
    add(f"- 适用状态：`{sf.get('status', '未计算')}`"
        f"（{_STATUS_LABEL.get(sf.get('status', ''), '')}）")
    stft = sf.get("stft") or {}
    if sf.get("status") == "MEASURED":
        add(f"- 特征：`{sf.get('feature')}`"
            f"（{len(sf.get('per_episode') or [])} 个 episode，"
            f"共 {sf.get('n_frames')} 个 STFT 帧）")
        add(f"- SF(t) 统计量：mean {_fmt_value(sf.get('mean'))} / "
            f"p50 {_fmt_value(sf.get('p50'))} / p95 {_fmt_value(sf.get('p95'))} / "
            f"max {_fmt_value(sf.get('max'))}")
        add(f"- STFT 参数：窗 {stft.get('window')}，nperseg={stft.get('nperseg')}，"
            f"hop={stft.get('hop')}（{stft.get('note', '')}）")
    if sf.get("note"):
        add(f"- {sf['note']}")
    add("")

    # A.3 点云动态性
    pc = obs.get("appendix_a_pointcloud_dynamics") or {}
    add("### 附录 A.3 点云运动动态性 D(t)")
    add("")
    add(f"- 公式：`{pc.get('formula', '')}`")
    add(f"- 适用状态：`{pc.get('status', '未计算')}`"
        f"（{_STATUS_LABEL.get(pc.get('status', ''), '')}）")
    if pc.get("status") == "MEASURED":
        add(f"- 特征：`{pc.get('feature')}`"
            f"（{len(pc.get('per_episode') or [])} 个 episode，"
            f"共 {pc.get('n_frames')} 帧）")
        add(f"- D(t) 统计量：mean {_fmt_value(pc.get('mean'))} / "
            f"p50 {_fmt_value(pc.get('p50'))} / p95 {_fmt_value(pc.get('p95'))} / "
            f"max {_fmt_value(pc.get('max'))}")
    if pc.get("note"):
        add(f"- {pc['note']}")
    add("")

    # --- 生效阈值 ----------------------------------------------------------
    add("## 6. 本次判定生效的阈值")
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
    alpha = prof.get("exposure_fraction")
    add("- 委托方声明值：曝光分数 α（曝光时间/帧周期）"
        + (f"**已声明 = {alpha:g}**" if alpha is not None else "**未声明**")
        + "，未声明时 6.2.5.2 a) 1) 只做单边判定：光流位移 p95 ≤ 5 px 判 PASS，"
          "> 5 px 判 UNDETERMINED（绝不判 FAIL，因为没有 α 就排除不掉 L 仍 ≤ 5 px）")
    add("")

    # --- 数据集事实 --------------------------------------------------------
    add("## 7. 数据集事实（用于复核）")
    add("")
    for k, v in (result.dataset_info or {}).items():
        add(f"- {k}: {_fmt_value(v)}")
    add("")

    # --- 门禁与遗留 --------------------------------------------------------
    add("## 8. 门禁状态与遗留项")
    add("")
    gate = result.coverage_gate or {}
    add(f"- coverage 门禁：enabled={gate.get('enabled')}，"
        f"status=**{gate.get('status')}**，coverage={gate.get('coverage')}")
    add(f"  - {gate.get('note', '')}")
    add("")

    return "\n".join(lines) + "\n"
