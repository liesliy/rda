# invalid_values

> **Layer**: Layer 1 - Data Integrity (hard check)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — invalid_values

Elementwise NaN/Inf scan over all numeric state/action cells (isfinite). Reports nan_count and inf_count per feature and in total. Severity: any NaN/Inf fails the check (EXCLUDE) and maps to reason code INVALID in the recommendation gate (v0.5.9+).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

