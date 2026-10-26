"""RDA conform — 国家标准符合性测评（GB/Z 218.1-2026）。

这个子包回答的是与 ``rda audit`` 不同的问题：

  * ``rda audit``   —— 这份数据"好不好"？（质量画像、可疑 episode 定位）
  * ``rda conform`` —— 这份数据"符不符合标准"？（对照 GB/Z 218.1-2026 的
    数字阈值，逐条出具四态结论，产出可被第三方采信的测评报告）

核心 API::

    from rda.conform import run_conformance, load_profile, ConformOptions

    result = run_conformance("path/to/dataset")
    result.summary["mandatory_fail"]   # "应"类条款违反数（阻断性）
    result.summary["advisory_fail"]    # "宜"类条款违反数（建议性，单列）

四态结论见 :class:`rda.conform.verdict.Verdict`；阈值表见
:mod:`rda.conform.standards.gbz218_1`——每个数字都带条款号与 binding。
"""
from __future__ import annotations

from rda.conform.engine import (
    ConformOptions,
    ConformanceResult,
    DatasetView,
    run_conformance,
)
from rda.conform.profile import (
    ConformProfile,
    SamplingSpec,
    ThresholdSpec,
    default_profile,
    load_profile,
    profile_to_dict,
)
from rda.conform.report import (
    build_conformance_node,
    render_conformance_markdown,
)
from rda.conform.timeseries import (
    autocorrelation,
    dtw,
    dtw_normalized,
    fourier_transform,
    hausdorff_distance,
    resample_sequence,
)
from rda.conform.verdict import (
    ConformanceItem,
    Verdict,
    compare,
    overall_verdict,
    summarize,
)

__all__ = [
    # engine
    "run_conformance",
    "ConformanceResult",
    "ConformOptions",
    "DatasetView",
    # profile
    "ConformProfile",
    "SamplingSpec",
    "ThresholdSpec",
    "default_profile",
    "load_profile",
    "profile_to_dict",
    # verdict
    "Verdict",
    "ConformanceItem",
    "summarize",
    "overall_verdict",
    "compare",
    # timeseries (附录 B)
    "autocorrelation",
    "fourier_transform",
    "dtw",
    "dtw_normalized",
    "hausdorff_distance",
    "resample_sequence",
    # report
    "build_conformance_node",
    "render_conformance_markdown",
]
