"""GB/Z 218.1-2026 判定项实现。每项一个函数，返回 ConformanceItem。

判定纪律（本文件的红线）
------------------------
1. **"应"类条款违反 → FAIL；"宜"类条款违反 → FAIL**，但在 note 注明
   "宜类条款（建议性）"，并由汇总函数单列 ``advisory_fail``。
2. **数据/输入不足 → UNDETERMINED**，note 必须写明缺什么输入。
3. **条款不适用 → NOT_APPLICABLE**，note 说明原因。
4. 严禁把 UNDETERMINED 写成 PASS；严禁在没有标准阈值的地方编造阈值；
   严禁用代理指标冒充标准指标。
5. 6.2.3.4 a 的"可行解空间最大距离"标准未定义，未声明时**必须**
   UNDETERMINED，不许用绝对 Hausdorff 值反推。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rda.conform.profile import ConformProfile
from rda.conform.standards.gbz218_1 import (
    APPENDIX_A_RECOMMENDED,
    BINDING_ADVISORY,
    VIDEO_PIXEL_LAYER_CHECKS,
    VIDEO_PIXEL_LAYER_CLAUSES,
    VIDEO_PIXEL_LAYER_NOTE,
)
from rda.conform.timeseries import (
    autocorrelation,
    dtw_normalized,
    hausdorff_distance,
    resample_sequence,
)
from rda.conform.verdict import ConformanceItem, Verdict, compare

# 观测项（附录 B）计算结果的进程内缓存：Hausdorff/DTW 很贵，
# 判定项与观测节共用一次计算结果，避免重复跑分钟级运算。
_OBS_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _ok(item_clause: str, metric: str, value: Any, threshold, binding: str,
        method: str, note: str = "", details: Optional[dict] = None) -> ConformanceItem:
    return ConformanceItem(
        clause=item_clause, metric=metric, value=value, threshold=threshold,
        binding=binding, verdict=Verdict.PASS, method=method, note=note,
        details=details or {},
    )


def _fail(item_clause: str, metric: str, value: Any, threshold, binding: str,
          method: str, note: str = "", details: Optional[dict] = None) -> ConformanceItem:
    note = note or ""
    if binding == BINDING_ADVISORY:
        note = f"宜类条款（建议性），违反仍记 FAIL，汇总单列 advisory_fail。{note}"
    return ConformanceItem(
        clause=item_clause, metric=metric, value=value, threshold=threshold,
        binding=binding, verdict=Verdict.FAIL, method=method, note=note,
        details=details or {},
    )


def _undetermined(item_clause: str, metric: str, threshold, binding: str,
                  method: str, note: str, details: Optional[dict] = None,
                  value: Any = None) -> ConformanceItem:
    return ConformanceItem(
        clause=item_clause, metric=metric, value=value, threshold=threshold,
        binding=binding, verdict=Verdict.UNDETERMINED, method=method, note=note,
        details=details or {},
    )


def _na(item_clause: str, metric: str, threshold, binding: str,
        method: str, note: str, details: Optional[dict] = None,
        value: Any = None) -> ConformanceItem:
    return ConformanceItem(
        clause=item_clause, metric=metric, value=value, threshold=threshold,
        binding=binding, verdict=Verdict.NOT_APPLICABLE, method=method, note=note,
        details=details or {},
    )


def _fmt_rate(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.6f}%"


# ---------------------------------------------------------------------------
# 6.2.2.1 a  核心字段缺失率
# ---------------------------------------------------------------------------


def check_core_field_missing_rate(view, profile: ConformProfile,
                                  options) -> ConformanceItem:
    """6.2.2.1 a：核心字段（timestamp/frame_index/action/observation.state）缺失率 < 5%。"""
    th = profile.threshold("core_field_missing_rate")
    metric = "core_field_missing_rate"
    columns = view.parquet_columns
    if not columns:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "读取 data/chunk-*/file-*.parquet 的列结构。",
            "缺失输入：data 目录下未找到任何 parquet 文件，无法统计字段缺失率。",
        )

    core = list(profile.core_fields)
    per_field: Dict[str, Any] = {}
    total_null = 0
    total_rows = 0
    nulls = view.column_null_counts()
    specs = view.feature_specs()
    unresolvable: List[str] = []
    for field in core:
        if field in nulls:
            n, t = nulls[field]
            per_field[field] = {"missing": n, "total": t, "rate": n / t if t else None}
            total_null += n
            total_rows += t
        elif _payload_state(view, field, specs.get(field, {})) == _PAYLOAD_UNDOWNLOADED:
            # 核心字段被声明成视频/图像类且本机没下载 —— 与 6.2.2.1 b 同一类：
            # 不可核验，不许按全帧缺失计入（那会冤枉被测对象）。
            per_field[field] = {"missing": None, "total": None, "rate": None,
                                "note": "载荷应在 videos/ 下但本地副本不存在，不可核验"}
            unresolvable.append(field)
        else:
            # 列根本不存在 = 该核心字段在所有帧上全部缺失
            rows = max((t for _, t in nulls.values()), default=0)
            per_field[field] = {"missing": rows, "total": rows, "rate": 1.0,
                                "note": "列不存在于 data parquet"}
            total_null += rows
            total_rows += rows

    if unresolvable:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "按列统计 null 计数（parquet row group statistics）。",
            f"视频层未下载/不可用，无法判定字段缺失：核心字段 "
            f"{', '.join(unresolvable)} 被声明为视频/图像类，其载荷位于 videos/ "
            f"目录，但本数据集副本的 videos/ 不存在对应文件。"
            f"这不是数据集本身的字段缺失，故本项无法裁定。"
            f"请提供含 videos/ 的完整数据集后复测。",
            {"per_field": per_field, "core_fields": core,
             "videos_dir_present": view.videos_dir.exists()},
        )

    if total_rows == 0:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "按列统计 null 计数（parquet row group statistics）。",
            "缺失输入：data parquet 中没有任何可用于统计的行。",
        )

    rate = total_null / total_rows
    method = (
        "全量统计：对每个核心字段，取 data/*.parquet 全量行的 null 计数，"
        "rate = Σnull / Σrows（不抽样）。列不存在的字段按全缺失计入。"
    )
    ok = compare(rate, th.value, th.comparator)
    details = {"per_field": per_field, "total_missing": total_null,
               "total_cells": total_rows, "core_fields": core}
    if ok is None:
        return _undetermined(th.clause, metric, th.describe(), th.binding, method,
                             "缺失输入：无法计算核心字段缺失率。", details, rate)
    if ok:
        return _ok(th.clause, metric, rate, th.describe(), th.binding, method,
                   f"核心字段缺失率 {_fmt_rate(rate)}，满足 {th.describe()}。", details)
    return _fail(th.clause, metric, rate, th.describe(), th.binding, method,
                 f"核心字段缺失率 {_fmt_rate(rate)}，不满足 {th.describe()}。", details)


# ---------------------------------------------------------------------------
# 6.2.2.1 b  一般字段缺失率
# ---------------------------------------------------------------------------


_VIDEO_DTYPES = ("video", "image")

# 载荷状态取值
_PAYLOAD_PARQUET = "parquet"          # 载荷在 data parquet 里，可直接统计 null
_PAYLOAD_AVAILABLE = "available"      # 外部载荷（视频/图像）确实存在
_PAYLOAD_UNDOWNLOADED = "undownloaded"  # 外部载荷应在但本地副本没有
_PAYLOAD_ABSENT = "absent"            # 非外部载荷类型，却在任何地方都找不到


def _payload_state(view, feature: str, spec: dict) -> str:
    """判定一个声明特征的载荷处于哪种**可核验**状态。

    这里有一条必须守住的边界：``undownloaded``（本地副本没下载视频）与
    ``absent``（数据集自己没这个字段）是两件完全不同的事。前者是被测环境
    的不足，后者才可能是数据集的缺陷。把前者当成后者去判 FAIL，就是拿
    "我没拿到数据"去指控"你的数据不合格"——这份报告一旦落到第三方手里，
    这种误判会直接击穿整份报告的可信度。
    """
    if feature in view.parquet_columns:
        return _PAYLOAD_PARQUET
    if str(spec.get("dtype", "")).lower() in _VIDEO_DTYPES:
        if view.video_payload_files(feature):
            return _PAYLOAD_AVAILABLE
        return _PAYLOAD_UNDOWNLOADED
    return _PAYLOAD_ABSENT


def check_general_field_missing_rate(view, profile: ConformProfile,
                                     options) -> ConformanceItem:
    """6.2.2.1 b：一般字段缺失率 < 20%。

    **视频层未下载时不得判 FAIL。** 数据集声明了视觉流而本地副本的
    ``videos/`` 不存在（或其中没有对应文件），说明是我们手上这份副本不完整，
    不是数据集缺字段。这类特征既不能计成"全帧缺失"（会冤枉被测对象），
    也不能计成"零缺失"（会凭空放行），只能让本项判 UNDETERMINED 并把
    可核验部分的对照值一并给出。
    """
    th = profile.threshold("general_field_missing_rate")
    metric = "general_field_missing_rate"
    columns = view.parquet_columns
    if not columns:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "读取 data parquet 的列结构。",
            "缺失输入：data 目录下未找到任何 parquet 文件。",
        )

    specs = view.feature_specs()
    core = set(profile.core_fields)
    # 一般字段 = 声明的特征 ∪ parquet 实际列，去掉核心字段
    general = sorted((set(specs) | set(columns)) - core)
    if not general:
        return _na(th.clause, metric, th.describe(), th.binding,
                   "一般字段集合为空。",
                   "数据集未声明除核心字段外的任何一般字段，本条无从判定。")

    nulls = view.column_null_counts()
    rows_total = max((t for _, t in nulls.values()), default=0)
    per_field: Dict[str, Any] = {}
    countable_missing = 0
    countable_cells = 0
    parquet_missing = 0
    parquet_cells = 0
    undownloaded: List[str] = []
    absent: List[str] = []

    for field in general:
        spec = specs.get(field, {})
        if field in nulls:
            n, t = nulls[field]
            per_field[field] = {"missing": n, "total": t, "rate": n / t if t else None,
                                "residence": _PAYLOAD_PARQUET}
            countable_missing += n
            countable_cells += t
            parquet_missing += n
            parquet_cells += t
            continue

        state = _payload_state(view, field, spec)
        if state == _PAYLOAD_AVAILABLE:
            # 外部载荷已存在，本版本不解码其内容，故按"已采集"计入，缺失 0。
            # 这是存在性判断，不是内容合格性判断——报告里必须写明。
            per_field[field] = {"missing": 0, "total": rows_total, "rate": 0.0,
                                "residence": _PAYLOAD_AVAILABLE,
                                "dtype": spec.get("dtype")}
            countable_missing += 0
            countable_cells += rows_total
            continue
        if state == _PAYLOAD_UNDOWNLOADED:
            # 视频层未下载 —— 不可核验，绝不计入分子分母。
            per_field[field] = {"missing": None, "total": rows_total, "rate": None,
                                "residence": _PAYLOAD_UNDOWNLOADED,
                                "dtype": spec.get("dtype"),
                                "note": "载荷应在 videos/ 下但本地副本不存在该文件"}
            undownloaded.append(field)
            continue
        # 既不在 parquet 里、也不是视频/图像类的外部载荷 → 数据集确实没存
        per_field[field] = {"missing": rows_total, "total": rows_total, "rate": 1.0,
                            "residence": _PAYLOAD_ABSENT,
                            "dtype": spec.get("dtype"),
                            "note": "声明了特征但 parquet 中无该列，且非外部载荷类型"}
        countable_missing += rows_total
        countable_cells += rows_total
        absent.append(field)

    rate = countable_missing / countable_cells if countable_cells else None
    parquet_rate = parquet_missing / parquet_cells if parquet_cells else None

    method = (
        "全量统计：一般字段 = 声明特征 ∪ data parquet 实际列 − 核心字段；"
        "有 parquet 列的按 null 计数；video/image 类特征看 videos/ 下是否有"
        "对应文件——**文件不存在时该字段不可核验，不计入缺失率分子分母"
        "（视频层未下载 ≠ 数据集缺字段）**；非外部载荷类型却找不到列的，"
        "按全帧缺失计入。"
    )
    details = {
        "per_field": per_field,
        "general_fields": general,
        "rows_per_field": rows_total,
        "rate_countable_only": rate,
        "rate_parquet_columns_only": parquet_rate,
        "undownloaded_payload_features": undownloaded,
        "absent_features": absent,
        "videos_dir_present": view.videos_dir.exists(),
        "videos_dir_file_count": view.videos_dir_file_count(),
        "accounting_note": (
            "rate_countable_only 为可在本次副本上核验的那部分字段的缺失率；"
            "rate_parquet_columns_only 为只统计有 parquet 列字段时的对照口径。"
            "存在 undownloaded_payload_features 时，二者都只是参考值，不构成判定。"
        ),
    }

    if undownloaded:
        # 核心修正：视频层未下载 → UNDETERMINED，不许判 FAIL。
        note = (
            f"视频层未下载/不可用，无法判定这些字段是否缺失：数据集声明了 "
            f"{len(undownloaded)} 个视觉/图像类特征（{', '.join(undownloaded)}），"
            f"其载荷位于 videos/ 目录；本数据集副本"
            + ("的 videos/ 目录不存在" if not view.videos_dir.exists()
               else f"的 videos/ 目录下没有这些特征的文件（videos/ 内文件总数 "
                    f"{view.videos_dir_file_count()}）")
            + "。这是本地副本不完整，**不是数据集本身的字段缺失**，因此这些特征"
              "既不计入缺失率分子也不计入分母，本项无法裁定。"
        )
        if rate is not None:
            note += (
                f"参考值（不参与判定）：仅按本次可核验的 {len(general) - len(undownloaded)}"
                f"/{len(general)} 个一般字段统计的缺失率为 {_fmt_rate(rate)}；"
                f"仅按 parquet 列统计为 {_fmt_rate(parquet_rate)}。"
            )
        note += "请提供含 videos/ 的完整数据集后复测。"
        return _undetermined(th.clause, metric, th.describe(), th.binding,
                             method, note, details, value=None)

    if rate is None:
        return _undetermined(th.clause, metric, th.describe(), th.binding, method,
                             "缺失输入：没有可核验的一般字段单元，无法计算缺失率。",
                             details, value=None)
    ok = compare(rate, th.value, th.comparator)
    extra = ""
    if absent:
        extra = (
            f"其中 {len(absent)} 个特征已声明但在 parquet 与非视频载荷位置均找不到"
            f"（{', '.join(absent)}），按全帧缺失计入；"
            f"若不计这些，仅按 parquet 列统计的缺失率为 {_fmt_rate(parquet_rate)}。"
        )
    if ok is None:
        return _undetermined(th.clause, metric, th.describe(), th.binding, method,
                             "缺失输入：无法计算一般字段缺失率。", details, rate)
    if ok:
        return _ok(th.clause, metric, rate, th.describe(), th.binding, method,
                   f"一般字段缺失率 {_fmt_rate(rate)}，满足 {th.describe()}。{extra}", details)
    return _fail(th.clause, metric, rate, th.describe(), th.binding, method,
                 f"一般字段缺失率 {_fmt_rate(rate)}，不满足 {th.describe()}。{extra}", details)


# ---------------------------------------------------------------------------
# 6.2.2.1 c  超阈值数据处理方式说明
# ---------------------------------------------------------------------------

_HANDLING_KEYWORDS = (
    # 中文
    "超阈值", "超出阈值", "越限", "越界", "异常值", "离群", "缺失值",
    "处理方式", "清洗", "裁剪", "截断",
    # 英文
    "out-of-range", "out of range", "outlier", "clipping", "clip",
    "imputation", "impute", "missing value", "sanitiz", "sanitis",
)


def check_out_of_threshold_handling(view, profile: ConformProfile,
                                    options) -> ConformanceItem:
    """6.2.2.1 c：应提供"超阈值数据的处理方式说明"。"""
    th = profile.threshold("out_of_threshold_handling_declared")
    metric = "out_of_threshold_handling_declared"
    card = view.dataset_card_text()
    info = view.info

    carriers: List[str] = []
    if card:
        carriers.append("根目录 README*/dataset_card*.md")
    # info.json 里任何字符串值都可能是说明载体
    info_text = " ".join(
        str(v) for v in info.values() if isinstance(v, (str, int, float))
    )
    if info_text.strip():
        carriers.append("meta/info.json 的标量字段")

    if not carriers:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "扫描数据集级说明载体并检索超阈值处理相关关键词。",
            "缺失输入：数据集根目录下未找到 README/dataset card 说明文档，"
            "meta/info.json 亦无可读标量字段，无法确认是否提供了"
            "超阈值数据的处理方式说明。",
        )

    haystack = f"{card or ''}\n{info_text}".lower()
    hits = sorted({kw for kw in _HANDLING_KEYWORDS if kw.lower() in haystack})
    method = (
        "检索数据集级说明载体（根目录 README*/dataset_card*.md 与 meta/info.json "
        "标量字段）中是否出现超阈值/异常值处理相关关键词："
        + "、".join(_HANDLING_KEYWORDS[:6]) + " 等。"
    )
    details = {"carriers": carriers, "keyword_hits": hits,
               "card_chars": len(card) if card else 0}
    if hits:
        return _ok(th.clause, metric, True, th.describe(), th.binding, method,
                   f"在说明载体中检出超阈值/异常处理相关表述：{', '.join(hits)}。",
                   details)
    return _fail(th.clause, metric, False, th.describe(), th.binding, method,
                 "已找到说明载体，但未在其中检出任何超阈值/异常值数据处理方式的"
                 "表述，不满足「应提供」要求。", details)


# ---------------------------------------------------------------------------
# 6.2.2.3  时间戳抖动
# ---------------------------------------------------------------------------


def _episode_diffs(view):
    """Concatenated per-episode timestamp differences (sorted within episode)."""
    out = []
    for _ep, ts in view.timestamps_by_episode().items():
        if ts.size >= 2:
            out.append(np.diff(ts))
    if not out:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(out)


def check_timestamp_jitter(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.2.2.3：时间戳抖动标准差 ≤ 采样周期 × 10%。"""
    th = profile.threshold("timestamp_jitter_std_ratio")
    metric = "timestamp_jitter_std"
    fps = view.fps
    if fps is None or fps <= 0:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "std(diff(timestamp))，限值 = 0.1 / fps。",
            "缺失输入：meta/info.json 未声明 fps，无法确定采样周期，"
            "也就无从得到抖动限值。",
        )
    dt = _episode_diffs(view)
    if dt.size < 2:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "std(diff(timestamp))，限值 = 0.1 / fps。",
            "缺失输入：可用于计算的时间戳间隔不足（<2）。",
        )
    std_s = float(np.std(dt))
    period = 1.0 / fps
    limit = th.value * period
    method = (
        f"全量计算：每个 episode 内 timestamp 排序后求一阶差分，拼接后取标准差；"
        f"采样周期 = 1/fps = {period:.9f} s，限值 = {th.value} × 周期 = {limit:.9f} s。"
    )
    details = {"std_seconds": std_s, "sampling_period_seconds": period,
               "limit_seconds": limit, "n_intervals": int(dt.size),
               "margin_ratio": (limit / std_s) if std_s > 0 else None,
               "mean_interval_seconds": float(np.mean(dt)),
               "max_interval_seconds": float(np.max(dt))}
    # 阈值写成"周期 × 10% = 具体秒数"，避免读者把 10% 当成绝对比例
    shown = f"{th.comparator} 采样周期 × {th.value:g} = {limit:.6e} s"
    ok = compare(std_s, limit, th.comparator)
    if ok is None:
        return _undetermined(th.clause, metric, shown, th.binding, method,
                             "缺失输入：无法计算时间戳抖动。", details, std_s)
    if ok:
        margin = f"，裕度 {(limit / std_s):.1f}×" if std_s > 0 else ""
        return _ok(th.clause, metric, std_s, shown, th.binding, method,
                   f"时间戳抖动标准差 {std_s:.6e} s ≤ 限值 {limit:.6e} s{margin}。",
                   details)
    return _fail(th.clause, metric, std_s, shown, th.binding, method,
                 f"时间戳抖动标准差 {std_s:.6e} s > 限值 {limit:.6e} s。", details)


# ---------------------------------------------------------------------------
# 6.2.2.3  连续丢帧率
# ---------------------------------------------------------------------------


def check_frame_loss_rate(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.2.2.3：连续丢帧率 ≤ 1%。"""
    th = profile.threshold("consecutive_frame_loss_rate")
    metric = "consecutive_frame_loss_rate"
    fps = view.fps
    if fps is None or fps <= 0:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "帧间隔 > frame_loss_factor × 采样周期 的间隔占比。",
            "缺失输入：meta/info.json 未声明 fps，无法识别丢帧。",
        )
    dt = _episode_diffs(view)
    if dt.size == 0:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "帧间隔 > frame_loss_factor × 采样周期 的间隔占比。",
            "缺失输入：没有可用的帧间隔。",
        )
    period = 1.0 / fps
    cutoff = profile.frame_loss_factor * period
    lost = int(np.sum(dt > cutoff))
    rate = lost / dt.size
    method = (
        f"全量计算：帧间隔 > {profile.frame_loss_factor} × 采样周期"
        f"（{cutoff:.6f} s）计为丢帧；丢帧率 = 丢帧间隔数 / 全部帧间隔数。"
        f"注：标准未规定丢帧识别系数，{profile.frame_loss_factor} 为可由 profile "
        f"覆盖的工程口径。"
    )
    details = {"lost_intervals": lost, "total_intervals": int(dt.size),
               "cutoff_seconds": cutoff,
               "frame_loss_factor": profile.frame_loss_factor,
               "max_interval_seconds": float(np.max(dt))}
    ok = compare(rate, th.value, th.comparator)
    if ok is None:
        return _undetermined(th.clause, metric, th.describe(), th.binding, method,
                             "缺失输入：无法计算丢帧率。", details, rate)
    if ok:
        return _ok(th.clause, metric, rate, th.describe(), th.binding, method,
                   f"连续丢帧率 {_fmt_rate(rate)}（{lost}/{dt.size}），满足 {th.describe()}。",
                   details)
    return _fail(th.clause, metric, rate, th.describe(), th.binding, method,
                 f"连续丢帧率 {_fmt_rate(rate)}（{lost}/{dt.size}），不满足 {th.describe()}。",
                 details)


# ---------------------------------------------------------------------------
# 6.1.1 f  多模态统一时间基准
# ---------------------------------------------------------------------------


def check_multimodal_time_base(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.1.1 f：帧层与视频层须同源、跨视角可比。

    **可比量是「时长」(to − from)，不是 from/to 的绝对值。**

    LeRobot v3 的 ``videos/<key>/from_timestamp`` / ``to_timestamp`` 是**视频
    文件内的相对偏移**，不是 episode 的绝对时间。实测证据（lerobot_folding）：

      * 每个视频文件的首个 episode，``from`` 恒为 0（240/240、155/155、
        153/153 个文件全部如此）；
      * 同一文件内相邻 episode 首尾严格相接（``from_{i+1} == to_i``，差 0）；
      * 各相机的分块数不同（240 / 155 / 153），同一个 episode 在不同相机里
        落在不同的文件偏移上；
      * 而 ``to − from`` 恒等于 ``length / fps``（1200/1200 个 episode 精确相等）。

    因此跨视角直接对 ``from``/``to`` 做差得到的只是**分块差异**，不是时间基准
    差异；旧实现据此得出的「跨视角偏差 1033 s」是一个纯属口径错误的伪值，
    会把一个时间基准完全统一的数据集判成违反"应"类条款。本函数只比较时长。
    """
    th = profile.threshold("multimodal_time_base_unified")
    tol = profile.threshold("time_base_tolerance_s")
    metric = "multimodal_time_base_unified"
    windows = view.video_windows()

    if not windows:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "对比各视频流的时长（to − from）与帧层时间戳跨度。",
            "缺失输入：meta/episodes/*.parquet 中没有 "
            "videos/<key>/from_timestamp、to_timestamp 列，无法核查"
            "帧层与视频层是否同源。",
        )

    cameras = sorted(windows.keys())

    # 各相机各 episode 的时长；from/to 缺失则记为 None（不可核验）
    durations: Dict[str, Dict[int, Optional[float]]] = {}
    for cam in cameras:
        slot: Dict[int, Optional[float]] = {}
        for ep, (fr, to, _fi) in windows[cam].items():
            slot[ep] = None if (fr is None or to is None) else float(to) - float(fr)
        durations[cam] = slot

    # 证据 A：同一 episode 内，各视角时长的最大偏差（跨视角可比性）
    cross_dur = 0.0
    cross_worst: Optional[int] = None
    cross_n = 0
    if len(cameras) >= 2:
        common_eps = set.intersection(*[set(durations[c]) for c in cameras])
        for ep in sorted(common_eps):
            dvals = [durations[c][ep] for c in cameras]
            if any(v is None for v in dvals):
                continue
            cross_n += 1
            dev = max(dvals) - min(dvals)
            if dev > cross_dur:
                cross_dur = dev
                cross_worst = ep

    # 证据 B：视频层时长 vs 帧层跨度。N 帧的帧层跨度是 (N−1) 个采样周期，
    # 而视频时长是 N 个采样周期，故可比量 = 帧层跨度 + 1 个标称周期。
    frame_span = {ep: (float(ts.min()), float(ts.max()))
                  for ep, ts in view.timestamps_by_episode().items() if ts.size}
    if not frame_span:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "对比各视频流的时长（to − from）与帧层时间戳跨度。",
            "缺失输入：data parquet 中没有可用的帧层 timestamp，"
            "无法核查帧层与视频层是否同源。",
        )

    fps = view.fps
    period = 1.0 / fps if (fps and fps > 0) else None
    frame_dev = 0.0
    frame_worst: Optional[Tuple[int, str]] = None
    frame_n = 0
    for cam in cameras:
        for ep, dur in durations[cam].items():
            if dur is None or period is None:
                continue
            span = frame_span.get(ep)
            if span is None:
                continue
            frame_n += 1
            dev = abs(dur - ((span[1] - span[0]) + period))
            if dev > frame_dev:
                frame_dev = dev
                frame_worst = (ep, cam)

    if cross_n == 0 and frame_n == 0:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "对比各视频流的时长（to − from）与帧层时间戳跨度。",
            "缺失输入：没有可比的时长样本"
            + ("（meta/info.json 未声明 fps，无法把帧层跨度换算成时长）"
               if period is None else "")
            + "。",
            {"cameras": cameras, "videos_dir_present": view.videos_dir.exists()},
        )

    worst = max(cross_dur, frame_dev)
    method = (
        f"取 meta/episodes 中各视频流的时长 duration = to − from："
        f"(a) 同一 episode 跨视角 duration 的最大偏差（{cross_n} 个 episode）；"
        f"(b) 视频层 duration 与「帧层 timestamp 跨度 + 1 个标称采样周期"
        + (f"（={period:.6f} s）" if period else "")
        + f"」的最大偏差（{frame_n} 个样本；N 帧的帧层跨度是 N−1 个周期，"
        f"视频时长是 N 个周期，故需补 1 个周期才可比）。"
        f"两者最大值 > {tol.value} s 即判不同源。"
        f"**注意：from/to 是视频文件内相对偏移，跨视角直接相减只反映分块差异，"
        f"本实现不采用该口径。**"
        f"（{tol.value} s 容差本身是工程口径，沿用 {tol.clause}，见 profile 标注。）"
    )
    details = {
        "cameras": cameras,
        # 可比量（参与判定）
        "cross_camera_max_duration_deviation_s": cross_dur,
        "cross_camera_worst_episode": cross_worst,
        "cross_camera_episodes_compared": cross_n,
        "frame_vs_video_max_duration_deviation_s": frame_dev,
        "frame_vs_video_worst": frame_worst,
        "frame_vs_video_samples_compared": frame_n,
        "nominal_period_s": period,
        # 仅供参考：不可比，写出来是为了让复核者看到"为什么它不是证据"
        "cross_camera_offset_max_deviation_s_NOT_A_METRIC": _offset_spread(windows),
        "offset_note": (
            "from/to 是视频文件内相对偏移，各相机分块数不同导致同一 episode "
            "的偏移量不同；上值仅为分块差异，不参与判定。"
        ),
        "same_file_index_episodes": _same_file_index_count(windows),
        "tolerance_s": tol.value,
        "tolerance_origin": tol.origin,
        "videos_dir_present": view.videos_dir.exists(),
    }
    ok = compare(worst <= tol.value, True, "is_true")
    if ok is None:
        return _undetermined(th.clause, metric, th.describe(), th.binding, method,
                             "缺失输入：无法量化帧层/视频层时间偏差。", details)
    if ok:
        return _ok(
            th.clause, metric, True, th.describe(), th.binding, method,
            f"帧层与视频层时间基准一致：跨视角时长最大偏差 {cross_dur:.6e} s"
            f"（{cross_n} 个 episode），视频层时长与帧层跨度最大偏差 "
            f"{frame_dev:.6e} s（{frame_n} 个样本），均 ≤ {tol.value} s。"
            f"局限：本判定基于 meta/episodes 元数据与帧层 timestamp，"
            + ("videos/ 未下载，未核验视频实际帧数与像素级（恒定相位）对齐。"
               if not view.videos_dir.exists() else "")
            + f"（{tol.value} s 容差为工程口径，非 {th.clause} 标准值。）",
            details)
    return _fail(
        th.clause, metric, False, th.describe(), th.binding, method,
        f"帧层与视频层未统一时间基准：跨视角时长最大偏差 {cross_dur:.6f} s"
        f"（最差 episode {cross_worst}），视频层时长与帧层跨度最大偏差 "
        f"{frame_dev:.6f} s（最差 {frame_worst}），最大值超过 {tol.value} s 容差。",
        details)


def _offset_spread(windows) -> Optional[float]:
    """跨视角 from/to 绝对偏移的最大差值。**仅供参考，不参与判定。**"""
    cameras = sorted(windows.keys())
    if len(cameras) < 2:
        return None
    common = set.intersection(*[set(windows[c]) for c in cameras])
    best = 0.0
    for ep in sorted(common):
        fr = [windows[c][ep][0] for c in cameras]
        to = [windows[c][ep][1] for c in cameras]
        if any(v is None for v in fr + to):
            continue
        best = max(best, max(fr) - min(fr), max(to) - min(to))
    return best


def _same_file_index_count(windows) -> Optional[int]:
    """同一 episode 各视角 file_index 全部相同的 episode 数（参考信息）。"""
    cameras = sorted(windows.keys())
    if len(cameras) < 2:
        return None
    common = set.intersection(*[set(windows[c]) for c in cameras])
    n = 0
    for ep in common:
        fis = [windows[c][ep][2] for c in cameras]
        if None in fis:
            continue
        if len(set(fis)) == 1:
            n += 1
    return n


# ---------------------------------------------------------------------------
# 6.2.5.2 c) 4)  多传感器时间戳对齐误差
# ---------------------------------------------------------------------------


def check_sensor_alignment(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.2.5.2 c) 4)：多传感器时间戳对齐误差 ≤ 1 ms。"""
    th = profile.threshold("sensor_alignment_error_s")
    metric = "sensor_alignment_error_s"
    candidates = [
        c for c in view.parquet_columns
        if c.lower().endswith("timestamp")
        or "stream_timestamp" in c.lower()
        or c.lower().endswith("_ts")
    ]
    method = (
        "需要 ≥2 路传感器的独立逐帧时间戳序列，按 6.2.5.2 c) 4) 计算"
        "跨传感器对齐误差的最大值（≤ 1 ms）。"
    )
    if len(candidates) < 2:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding, method,
            f"缺失输入：需要 ≥2 路传感器的独立逐帧时间戳序列；本数据集 "
            f"data/*.parquet 中命中的时间戳列只有 {len(candidates)} 个"
            f"（{candidates or '无'}）"
            + ("，且 videos/ 目录不存在，无法从视频层取得各相机逐帧时间戳。"
               if not view.videos_dir.exists() else "。"),
            {"timestamp_columns": candidates,
             "videos_dir_present": view.videos_dir.exists()},
        )
    return _undetermined(
        th.clause, metric, th.describe(), th.binding, method,
        f"已找到 {len(candidates)} 个候选时间戳列（{', '.join(candidates)}），"
        f"但本版本尚未实现跨传感器逐帧对齐误差的估计（需要对齐策略与"
        f"基准传感器的明确定义），为避免用代理指标冒充标准指标，此处不下结论。",
        {"timestamp_columns": candidates},
    )


# ---------------------------------------------------------------------------
# 6.2.3.2  任务/技能类型种类数
# ---------------------------------------------------------------------------


def check_task_type_diversity(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.2.3.2：任务/技能类型种类数 ≥ 2。"""
    th = profile.threshold("task_type_min_kinds")
    metric = "task_type_kinds"
    ti = view.task_indices()
    declared = view.info.get("total_tasks")
    try:
        declared = int(declared) if declared is not None else None
    except (TypeError, ValueError):
        declared = None

    if ti is None or ti.size == 0:
        if declared is None:
            return _undetermined(
                th.clause, metric, th.describe(), th.binding,
                "统计 task_index 的取值种类数，并与 meta/tasks.parquet 交叉核对。",
                "缺失输入：data parquet 中没有 task_index 列，"
                "meta/info.json 也没有 total_tasks 字段，无法统计任务种类数。",
            )
        kinds = declared
        method = "取自 meta/info.json 的 total_tasks（data 中无 task_index 列）。"
        details = {"task_kinds": kinds, "source": "info.json.total_tasks"}
    else:
        uniq, cnt = np.unique(ti, return_counts=True)
        kinds = int(uniq.size)
        p = cnt / cnt.sum()
        entropy = float(-np.sum(p * np.log2(p))) if kinds > 1 else 0.0
        method = (
            "全量统计：对 data/*.parquet 的 task_index 列取唯一值个数，"
            "并与 meta/info.json.total_tasks、meta/tasks.parquet 交叉核对。"
        )
        details = {
            "task_kinds": kinds,
            "task_index_values": [int(x) for x in uniq.tolist()],
            "counts": [int(x) for x in cnt.tolist()],
            "entropy_bits": entropy,
            "declared_total_tasks": declared,
            "source": "data.parquet task_index",
        }

    ok = compare(kinds, th.value, th.comparator)
    if ok is None:
        return _undetermined(th.clause, metric, th.describe(), th.binding, method,
                             "缺失输入：无法统计任务种类数。", details, kinds)
    if ok:
        return _ok(th.clause, metric, kinds, th.describe(), th.binding, method,
                   f"任务/技能类型 {kinds} 种，满足 {th.describe()}。", details)
    return _fail(th.clause, metric, kinds, th.describe(), th.binding, method,
                 f"任务/技能类型仅 {kinds} 种，不满足 {th.describe()}。", details)


# ---------------------------------------------------------------------------
# 6.2.3.5  传感器模态覆盖矩阵
# ---------------------------------------------------------------------------

_MODALITY_RULES: Sequence[Tuple[str, Tuple[str, ...]]] = (
    # 顺序有意义：深度/点云要在视觉之前匹配，否则 depth image 会被算成视觉
    ("三维感知", ("depth", "pointcloud", "point_cloud", "pcd", "lidar", "tof", "rgbd")),
    ("声学", ("audio", "mic", "sound", "speech")),
    ("力学", ("force", "torque", "wrench", "tactile", "touch", "effort")),
    ("惯性", ("imu", "accel", "gyro", "angular_velocity", "linear_acceleration")),
    ("本体感知", ("state", "joint", "qpos", "qvel", "proprio", "pose", "eef")),
    ("视觉", ("image", "video", "rgb", "camera", "cam", "view", "wrist", "base")),
)

MODALITY_CLASSES = ("视觉", "声学", "力学", "惯性", "三维感知", "本体感知")


def classify_modality(feature: str, dtype: str = "") -> Optional[str]:
    """把一个特征名归入六类传感器模态之一，无法归类时返回 ``None``。

    短关键词（<4 字符，如 ``imu`` / ``cam`` / ``eef``）只在**完整词条**上匹配。
    纯子串匹配会把 ``left_wrist`` 里的 ``ft_`` 误判成力学传感器——这类误判
    会直接污染覆盖矩阵，必须挡住。
    """
    low = feature.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    for cls, rule_tokens in _MODALITY_RULES:
        for tok in rule_tokens:
            if len(tok) >= 4:
                if tok in low:
                    return cls
            elif tok in tokens:
                return cls
    return None


def check_modality_coverage(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.2.3.5：输出六类传感器模态覆盖矩阵（标准未设阈值，仅信息性输出）。"""
    th = profile.threshold("modality_coverage")
    metric = "modality_coverage_matrix"
    specs = view.feature_specs()
    if not specs:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding,
            "按特征名与 dtype 把声明特征归入六类模态。",
            "缺失输入：meta/info.json 中没有 features 段，无法输出覆盖矩阵。",
        )

    matrix: Dict[str, List[str]] = {c: [] for c in MODALITY_CLASSES}
    unclassified: List[str] = []
    for name, spec in specs.items():
        cls = classify_modality(name, str(spec.get("dtype", "")))
        if cls is None:
            unclassified.append(name)
        else:
            matrix[cls].append(name)

    method = (
        "遍历 meta/info.json.features，按特征名关键词（depth/point → 三维感知，"
        "audio/mic → 声学，force/tactile → 力学，imu/accel → 惯性，"
        "state/joint → 本体感知，image/video → 视觉）归类，输出覆盖矩阵。"
    )
    details = {
        "coverage_matrix": matrix,
        "counts": {c: len(v) for c, v in matrix.items()},
        "unclassified_features": unclassified,
        "declared_feature_count": len(specs),
    }
    total_covered = sum(len(v) for v in matrix.values())
    return _na(
        th.clause, metric, "六类传感器模态覆盖矩阵（标准未设阈值）", th.binding, method,
        f"6.2.3.5 只要求输出六类传感器模态覆盖矩阵，未设符合性阈值；"
        f"本项为信息性输出，不构成 PASS/FAIL 结论（已归类 {total_covered} 个特征，"
        f"覆盖 {sum(1 for v in matrix.values() if v)}/6 类）。",
        details, value=details["counts"],
    )


# ---------------------------------------------------------------------------
# 6.2.7 f  DOI/URI 全链路唯一标识
# ---------------------------------------------------------------------------

_ID_KEY_TOKENS = ("doi", "uri", "url", "identifier", "permalink", "homepage", "paper")
_PLACEHOLDERS = {
    "", "-", "n/a", "na", "none", "null", "tbd", "todo", "unknown",
    "more information needed", "[more information needed]", "n.a.",
}
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>)]+", re.IGNORECASE)
_URI_RE = re.compile(r"\bhttps?://[^\s\"'<>)\]]+", re.IGNORECASE)


def _is_placeholder(text: str) -> bool:
    return text.strip().lower() in _PLACEHOLDERS


def check_doi_uri(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.2.7 f：应存在 DOI/URI 全链路唯一标识。"""
    th = profile.threshold("doi_uri_present")
    metric = "doi_uri_present"
    method = (
        "在 (1) meta/info.json 中键名含 doi/uri/url/identifier/permalink/homepage "
        "的字段值、(2) README 的 YAML front-matter 同名键、(3) README 正文中的 "
        "**Homepage:** / **Paper:** 行，检索 DOI（10.xxxx/...）或 http(s) URI；"
        "剔除 'More Information Needed' 等占位符。"
    )

    found: List[Dict[str, str]] = []
    scanned: List[str] = []
    placeholder_hits: List[str] = []

    info = view.info
    for key, val in info.items():
        kl = str(key).lower()
        if not any(tok in kl for tok in _ID_KEY_TOKENS):
            continue
        scanned.append(f"info.json:{key}")
        sval = str(val).strip()
        if _is_placeholder(sval):
            placeholder_hits.append(f"info.json:{key}={sval}")
            continue
        if _DOI_RE.search(sval):
            found.append({"source": f"info.json:{key}", "kind": "DOI",
                          "value": _DOI_RE.search(sval).group(0)})
        elif _URI_RE.search(sval):
            found.append({"source": f"info.json:{key}", "kind": "URI",
                          "value": _URI_RE.search(sval).group(0)})

    card = view.dataset_card_text()
    if card:
        scanned.append("README*.md")
        for key in _ID_KEY_TOKENS:
            m = re.search(rf"^\s*{key}\s*:\s*(.+)$", card, re.IGNORECASE | re.MULTILINE)
            if not m:
                continue
            raw = m.group(1).strip()
            if _is_placeholder(raw):
                placeholder_hits.append(f"README:{key}={raw}")
                continue
            if _DOI_RE.search(raw):
                found.append({"source": f"README:{key}", "kind": "DOI",
                              "value": _DOI_RE.search(raw).group(0)})
            elif _URI_RE.search(raw):
                found.append({"source": f"README:{key}", "kind": "URI",
                              "value": _URI_RE.search(raw).group(0)})
        # README 正文里的 **Homepage:** / **Paper:** 行
        for label in ("Homepage", "Paper", "DOI", "URI"):
            m = re.search(rf"\*\*{label}:\*\*\s*(.+)", card, re.IGNORECASE)
            if not m:
                continue
            raw = m.group(1).strip()
            if _is_placeholder(raw):
                placeholder_hits.append(f"README:{label}={raw}")
                continue
            if _DOI_RE.search(raw):
                found.append({"source": f"README:{label}", "kind": "DOI",
                              "value": _DOI_RE.search(raw).group(0)})
            elif _URI_RE.search(raw):
                found.append({"source": f"README:{label}", "kind": "URI",
                              "value": _URI_RE.search(raw).group(0)})

    if not scanned:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding, method,
            "缺失输入：既没有可读的 meta/info.json，也没有根目录 README/"
            "dataset card，无法核查是否提供了 DOI/URI 唯一标识。",
        )

    details = {"found": found, "scanned": scanned,
               "placeholders_rejected": placeholder_hits}
    if found:
        return _ok(th.clause, metric, True, th.describe(), th.binding, method,
                   "检出数据集级唯一标识：" +
                   "；".join(f"{f['kind']} {f['value']}（{f['source']}）" for f in found),
                   details)
    return _fail(th.clause, metric, False, th.describe(), th.binding, method,
                 "未检出数据集级 DOI/URI 唯一标识"
                 + (f"；检出的标识字段均为占位符：{', '.join(placeholder_hits)}"
                    if placeholder_hits else "")
                 + "。注：README 正文中指向工具仓库的链接（如 LeRobot 项目地址）"
                   "不算数据集自身的唯一标识，未予采信。", details)


# ---------------------------------------------------------------------------
# 6.2.3.4 a  轨迹 Hausdorff
# ---------------------------------------------------------------------------


def _sample_episode_ids(view, n: int, seed: int) -> List[int]:
    all_ids = view.episode_ids()
    if not all_ids:
        return []
    n = min(n, len(all_ids))
    if n == len(all_ids):
        return sorted(all_ids)
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(np.array(all_ids), size=n, replace=False))


def _pairs(ids: List[int], max_pairs: int, seed: int) -> List[Tuple[int, int]]:
    out = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    if len(out) > max_pairs:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(out), size=max_pairs, replace=False))
        out = [out[i] for i in sel]
    return out


def compute_observations(view, profile: ConformProfile, options) -> Dict[str, Any]:
    """计算附录 B 的三项观测（Hausdorff / DTW / 自相关）。

    结果按 (数据集, 抽样参数) 缓存；判定项 6.2.3.4 a 复用同一份结果，
    不重复跑分钟级的 DTW。
    """
    s = profile.sampling
    key = "|".join([
        str(view.root), str(s.hausdorff_episodes), str(s.hausdorff_resample_T),
        str(s.dtw_episodes), str(s.dtw_max_pairs), str(s.acf_episodes),
        str(s.acf_max_lag), str(s.seed),
    ])
    if key in _OBS_CACHE:
        return _OBS_CACHE[key]

    obs: Dict[str, Any] = {"note": "观测项：标准未设阈值，仅输出，不做符合性判定。"}
    state_col = "observation.state"
    action_col = "action"

    # --- Hausdorff ---
    h_ids = _sample_episode_ids(view, s.hausdorff_episodes, s.seed)
    h_values: List[float] = []
    if h_ids and state_col in view.parquet_columns:
        arrays = view.load_episode_arrays([state_col, "timestamp"], h_ids)
        resampled = {e: resample_sequence(d[state_col], s.hausdorff_resample_T)
                     for e, d in arrays.items() if state_col in d}
        keys = sorted(resampled)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                h = hausdorff_distance(resampled[keys[i]], resampled[keys[j]])
                if np.isfinite(h):
                    h_values.append(h)
    if h_values:
        arr = np.asarray(h_values)
        obs["trajectory_hausdorff"] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()),
            "n_pairs": int(arr.size),
            "episodes_sampled": len(h_ids),
            "resample_T": s.hausdorff_resample_T,
            "seed": s.seed,
            "normalization": "未做位姿归一化（关节空间原始值，含各关节量纲差异）",
            "field": state_col,
        }
    else:
        obs["trajectory_hausdorff"] = {
            "mean": None, "episodes_sampled": len(h_ids),
            "note": "无法计算：缺少 observation.state 或抽样为空。",
        }

    # --- DTW ---
    d_ids = _sample_episode_ids(view, s.dtw_episodes, s.seed)
    ratios: List[float] = []
    lengths: List[int] = []
    raws: List[float] = []
    if d_ids and state_col in view.parquet_columns:
        arrays = view.load_episode_arrays([state_col, "timestamp"], d_ids)
        seqs = {e: d[state_col] for e, d in arrays.items() if state_col in d}
        # 与 Hausdorff 同口径重采样。实测：不重采样（用原始 ~2700 帧序列）时
        # DTW/L = 73.32，重采样到 T=64 后 = 82.32，两者不可比——同一份报告里
        # 两个附录 B 观测项必须同口径，否则读者无法判断哪个是"真的"。
        dtw_T = s.dtw_resample_T
        if dtw_T:
            seqs = {e: resample_sequence(v, dtw_T) for e, v in seqs.items()}
        pairs = _pairs(sorted(seqs), s.dtw_max_pairs, s.seed)
        dim = None
        for a, b in pairs:
            raw, norm, L = dtw_normalized(seqs[a], seqs[b])
            dim = dim or (seqs[a].shape[1] if seqs[a].ndim > 1 else 1)
            if L and np.isfinite(norm):
                raws.append(raw)
                ratios.append(norm)
                lengths.append(L)
    if ratios:
        r = np.asarray(ratios)
        obs["dtw"] = {
            "mean_dtw_over_L": float(r.mean()), "std_dtw_over_L": float(r.std()),
            "mean_dtw_raw": float(np.mean(raws)),
            "mean_path_length": float(np.mean(lengths)),
            "n_pairs": int(r.size),
            "episodes_sampled": len(d_ids),
            "max_pairs": s.dtw_max_pairs,
            "resample_T": dtw_T,
            "dim": dim, "seed": s.seed, "field": state_col,
            "formula": "附录 B.3 标准递推，L 为最优路径实际步数",
            "caliber_note": (
                "重采样点数必须与 Hausdorff 的 resample_T 一致才可比；"
                "本报告两处取值见各自 resample_T 字段。"
                + ("" if dtw_T else "本次未重采样（dtw_resample_T=null），"
                                    "结果与重采样口径不可比。")
            ),
        }
    else:
        obs["dtw"] = {"mean_dtw_over_L": None, "episodes_sampled": len(d_ids),
                      "note": "无法计算：缺少 observation.state 或抽样为空。"}

    # --- 自相关 ---
    a_ids = _sample_episode_ids(view, s.acf_episodes, s.seed)
    acf_items: List[Dict[str, Any]] = []
    if a_ids and action_col in view.parquet_columns:
        arrays = view.load_episode_arrays([action_col, "timestamp"], a_ids)
        for ep in sorted(arrays):
            act = np.asarray(arrays[ep].get(action_col), dtype=np.float64)
            if act.size == 0 or act.ndim != 2:
                continue
            dim = act.shape[1]
            # 短 episode 上 k >= T 的滞后阶无定义（返回 nan），钳制掉，
            # 免得把 nan 当成"过零"或"相关"。
            kmax = max(1, min(s.acf_max_lag, act.shape[0] - 1))
            ac1 = np.array([autocorrelation(act[:, d], 1)[0] for d in range(dim)])
            # 速度层（一阶差分）首个过零
            zeros = []
            for d in range(dim):
                v = np.diff(act[:, d])
                ac = autocorrelation(v, kmax)
                idx = np.flatnonzero(ac <= 0)
                zeros.append(int(idx[0]) + 1 if idx.size else None)
            valid = [z for z in zeros if z is not None]
            acf_items.append({
                "episode": ep,
                "frames": int(act.shape[0]),
                "acf_lag1_mean": float(ac1.mean()),
                "acf_lag1_min": float(ac1.min()),
                "acf_lag1_max": float(ac1.max()),
                "velocity_first_zero_crossing_lag_median": (
                    float(np.median(valid)) if valid else None),
                "velocity_first_zero_crossing_lag_per_dim": zeros,
            })
    if acf_items:
        obs["autocorrelation"] = {
            "formula": "附录 B.1，分母为全长度 T 项求和（非 T-k）",
            "episodes_sampled": len(acf_items),
            "max_lag": s.acf_max_lag,
            "per_episode": acf_items,
            "acf_lag1_mean_over_episodes": float(
                np.mean([x["acf_lag1_mean"] for x in acf_items])),
            "velocity_first_zero_crossing_median_over_episodes": (
                float(np.median([x["velocity_first_zero_crossing_lag_median"]
                                 for x in acf_items
                                 if x["velocity_first_zero_crossing_lag_median"]]))
                if any(x["velocity_first_zero_crossing_lag_median"] for x in acf_items)
                else None),
        }
    else:
        obs["autocorrelation"] = {"note": "无法计算：缺少 action 或抽样为空。"}

    _OBS_CACHE[key] = obs
    return obs


def check_trajectory_hausdorff(view, profile: ConformProfile, options) -> ConformanceItem:
    """6.2.3.4 a：轨迹 Hausdorff 均值 ≥ 可行解空间最大距离 × 30%。"""
    th = profile.threshold("trajectory_hausdorff_ratio")
    metric = "trajectory_hausdorff_ratio"
    max_dist = options.resolved_max_distance()
    obs = compute_observations(view, profile, options)
    hd = obs.get("trajectory_hausdorff", {})
    h_mean = hd.get("mean")
    method = (
        f"每条 episode 的 observation.state 线性重采样到 T={profile.sampling.hausdorff_resample_T}，"
        f"两两求对称 Hausdorff 距离后取均值（{hd.get('n_pairs', 0)} 对，"
        f"抽样 {hd.get('episodes_sampled', 0)}/{view.num_episodes or '?'} episode，"
        f"seed={profile.sampling.seed}，未做位姿归一化）。"
    )
    if max_dist is None:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding, method,
            "缺失输入：6.2.3.4 a 的分母「可行解空间最大距离」标准未定义，"
            "需由数据提供方声明（--workspace-bounds 或 profile.workspace_max_distance）。"
            "未声明时不得用绝对 Hausdorff 值反推阈值，故判 UNDETERMINED。"
            + (f"实测 Hausdorff 均值 = {h_mean:.6f}（仅供参考，不参与判定）。"
               if h_mean is not None else ""),
            {"observed_hausdorff_mean": h_mean,
             "sampling": hd}, value=None,
        )
    if h_mean is None:
        return _undetermined(
            th.clause, metric, th.describe(), th.binding, method,
            "缺失输入：无法计算轨迹 Hausdorff 均值（缺少 observation.state 或抽样为空）。",
            {"observed_hausdorff_mean": None, "sampling": hd},
        )
    ratio = h_mean / max_dist
    limit = th.value
    shown = f"{th.comparator} 可行解空间最大距离 × {limit:g}"
    details = {"hausdorff_mean": h_mean, "workspace_max_distance": max_dist,
               "ratio": ratio, "sampling": hd}
    ok = compare(ratio, limit, th.comparator)
    if ok is None:
        return _undetermined(th.clause, metric, shown, th.binding, method,
                             "缺失输入：无法计算 Hausdorff 比值。", details, ratio)
    # 单位一致性无法由程序核验：若 state 是关节角/编码器值而声明的是笛卡尔
    # 米，这个比值就是量纲错配的垃圾。必须写在报告里，不能默默给出一个数。
    unit_caveat = (
        "注意：该比值只有在「可行解空间声明的量纲」与 observation.state 的量纲"
        "一致时才有意义；本实现无法核验单位，若 state 为关节角/编码器值而声明"
        "的是笛卡尔坐标，此比值不可用。"
    )
    if ok:
        return _ok(th.clause, metric, ratio, shown, th.binding, method,
                   f"Hausdorff 均值 / 可行解空间最大距离 = {ratio:.4f} ≥ {limit}。"
                   f"{unit_caveat}", details)
    return _fail(th.clause, metric, ratio, shown, th.binding, method,
                 f"Hausdorff 均值 / 可行解空间最大距离 = {ratio:.4f} < {limit}。"
                 f"{unit_caveat}", details)


# ---------------------------------------------------------------------------
# 视频像素层判定（本版本未实现）
# ---------------------------------------------------------------------------


def check_video_pixel_layer(view, profile: ConformProfile, options) -> ConformanceItem:
    """视频像素层判定：本版本一律 NOT_APPLICABLE。

    P0 不引入 opencv / av / torch，无法解码视频像素，因此这些判定项
    不做、也不许说做了。条款号已按 GB/Z 218.1-2026 正式版补录
    （6.2.5.2 a) 1) 与附录 A.1~A.3）；仍未查到原文的两项继续留 ``—``
    并标注"待补录"，不编造条款号。
    """
    clause = "6.2.5.2 a) 1) / 附录 A.1–A.3（另 2 项条款号待补录）"
    return _na(
        clause,
        "video_pixel_layer_checks",
        None, "—",
        "需解码视频像素后逐帧计算。",
        f"{VIDEO_PIXEL_LAYER_NOTE}：{', '.join(VIDEO_PIXEL_LAYER_CHECKS)}。",
        {
            "checks": list(VIDEO_PIXEL_LAYER_CHECKS),
            "clauses": list(VIDEO_PIXEL_LAYER_CLAUSES),
            "supplementary_methods": [
                {"clause": t.clause, "binding": t.binding, "quantity": t.quantity,
                 "value": t.value, "note": t.note}
                for t in APPENDIX_A_RECOMMENDED
            ],
            "videos_dir_present": view.videos_dir.exists(),
        },
        value=None,
    )


ALL_CHECKS = [
    check_core_field_missing_rate,
    check_general_field_missing_rate,
    check_out_of_threshold_handling,
    check_timestamp_jitter,
    check_frame_loss_rate,
    check_multimodal_time_base,
    check_sensor_alignment,
    check_task_type_diversity,
    check_modality_coverage,
    check_doi_uri,
    check_trajectory_hausdorff,
    check_video_pixel_layer,
]
