# timestamp_validity

> **Layer**: Layer 1 - Data Integrity (hard check)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — timestamp_validity

Monotonicity and sampling-integrity check on the `timestamp` column: rejects duplicate and negative-delta timestamps, reports median/p95/p99/max of consecutive deltas (dt). Any non-monotonic step fails the check (EXCLUDE, reason code INVALID).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

