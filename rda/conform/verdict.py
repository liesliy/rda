"""Four-state conformance verdicts for RDA conform.

为什么是四态而不是两态
----------------------
第三方采信一份符合性报告时，最危险的不是 FAIL，而是**把"没查"说成
"查过了没问题"**。因此本模块把判定结果强制拆成四态：

  ``PASS``           实测值满足标准阈值。
  ``FAIL``           实测值不满足标准阈值。
  ``NOT_APPLICABLE`` 该条款对本数据集不适用，或标准只要求输出而不设阈值。
  ``UNDETERMINED``   想判但判不了——**缺少输入**。

UNDETERMINED 与 PASS 的边界是本模块的红线：
  * 缺输入永远写 UNDETERMINED，note 必须写明缺什么；
  * 标准没给阈值的地方不许拿代理指标凑，一律 UNDETERMINED；
  * 任何"值缺失 → 当成 0 → 判 PASS"的写法都是禁止的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from rda.conform.standards.gbz218_1 import (
    BINDING_ADVISORY,
    BINDING_MANDATORY,
)


class Verdict(str, Enum):
    """四态结论。

    Attributes:
        PASS: 满足标准阈值。
        FAIL: 不满足标准阈值（"应"类与"宜"类都判 FAIL，靠 binding 区分严重性）。
        NOT_APPLICABLE: 条款不适用 / 标准仅要求信息性输出。
        UNDETERMINED: 输入不足，无法裁定。绝不可与 PASS 混用。
    """

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNDETERMINED = "UNDETERMINED"

    @property
    def is_decisive(self) -> bool:
        """True when the verdict actually judges compliance (PASS or FAIL)."""
        return self in (Verdict.PASS, Verdict.FAIL)


@dataclass
class ConformanceItem:
    """单条判定项的结果。

    Attributes:
        clause: 标准条款号。
        metric: 机器可读的判定项名（与 profile 的 threshold key 同名）。
        value: 实测值。无法实测时为 ``None``——此时 verdict 必须是
            UNDETERMINED 或 NOT_APPLICABLE，绝不可以是 PASS/FAIL。
        threshold: 判定时实际使用的阈值（含 comparator 的可读描述）。
        binding: ``"应"`` / ``"宜"`` / ``"—"``。
        verdict: 四态结论。
        method: 该值是怎么算出来的（抽样规模、口径、公式编号等），
            供第三方复算。
        note: 结论说明。UNDETERMINED 时必须写明缺什么输入。
        details: 补充证据（样本数、分布、文件路径等）。不参与判定，
            但必须能把 method 里的说法坐实。
    """

    clause: str
    metric: str
    value: Any
    threshold: Any
    binding: str
    verdict: Verdict
    method: str = ""
    note: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_advisory_fail(self) -> bool:
        """True when this is a FAIL on an advisory (宜) clause."""
        return self.verdict == Verdict.FAIL and self.binding == BINDING_ADVISORY

    @property
    def is_mandatory_fail(self) -> bool:
        """True when this is a FAIL on a mandatory (应) clause."""
        return self.verdict == Verdict.FAIL and self.binding == BINDING_MANDATORY

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "clause": self.clause,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "binding": self.binding,
            "verdict": self.verdict.value,
            "method": self.method,
            "note": self.note,
            "details": self.details,
        }


def summarize(items: List[ConformanceItem]) -> Dict[str, Any]:
    """Aggregate conformance items into counts used for gating.

    Args:
        items: 判定项列表。

    Returns:
        Dict with:
          ``total`` / ``pass`` / ``fail`` / ``not_applicable`` / ``undetermined``
          ``mandatory_fail``  违反"应"类条款的数量（阻断性）
          ``advisory_fail``   违反"宜"类条款的数量（建议性，单列）
          ``informational``   "—"类（仅信息性输出）的数量
          ``undetermined_metrics``  UNDETERMINED 项的 metric 名列表，
              让"没查出来"在报告里无处可藏。
    """
    counts = {
        "total": len(items),
        "pass": 0,
        "fail": 0,
        "not_applicable": 0,
        "undetermined": 0,
        "mandatory_fail": 0,
        "advisory_fail": 0,
        "informational": 0,
    }
    undetermined_metrics: List[str] = []
    for it in items:
        key = it.verdict.value.lower()
        counts[key] = counts.get(key, 0) + 1
        if it.is_mandatory_fail:
            counts["mandatory_fail"] += 1
        elif it.is_advisory_fail:
            counts["advisory_fail"] += 1
        if it.binding == "—":
            counts["informational"] += 1
        if it.verdict == Verdict.UNDETERMINED:
            undetermined_metrics.append(it.metric)
    counts["undetermined_metrics"] = undetermined_metrics
    # 可裁定覆盖率：真正下了结论（PASS/FAIL）的项占比。UNDETERMINED 不计入
    # 分子——"没查"不能算"查过了"。
    decisive = counts["pass"] + counts["fail"]
    counts["coverage"] = round(decisive / len(items), 6) if items else 0.0
    return counts


def overall_verdict(items: List[ConformanceItem]) -> Verdict:
    """Derive a single headline verdict from a set of items.

    Precedence: any FAIL → FAIL; else any UNDETERMINED → UNDETERMINED;
    else any PASS → PASS; else NOT_APPLICABLE.

    FAIL 优先于 UNDETERMINED：已经查实的不符合不能被"还有别的没查"
    掩盖。反过来，UNDETERMINED 优先于 PASS——只要有一项没查实，整份
    报告就不能宣称"符合"。
    """
    if not items:
        return Verdict.UNDETERMINED
    verdicts = {it.verdict for it in items}
    for v in (Verdict.FAIL, Verdict.UNDETERMINED, Verdict.PASS):
        if v in verdicts:
            return v
    return Verdict.NOT_APPLICABLE


def compare(value: Optional[float], threshold: Any, comparator: str) -> Optional[bool]:
    """Compare ``value`` against ``threshold`` using a standard comparator.

    Args:
        value: 实测值。``None`` 表示没有实测到。
        threshold: 阈值。
        comparator: ``"<"`` / ``"<="`` / ``">="`` / ``">"`` / ``"=="`` /
            ``"is_true"``。

    Returns:
        ``True`` / ``False``，或 ``None``（当 value 为 None 或 comparator
        无法识别时）。**调用方在拿到 None 时必须走 UNDETERMINED，
        不得回落成 PASS。**
    """
    if value is None:
        return None
    try:
        if comparator == "<":
            return bool(value < threshold)
        if comparator == "<=":
            return bool(value <= threshold)
        if comparator == ">=":
            return bool(value >= threshold)
        if comparator == ">":
            return bool(value > threshold)
        if comparator == "==":
            return bool(value == threshold)
        if comparator == "is_true":
            return bool(value) is True
    except TypeError:
        return None
    return None
