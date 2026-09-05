# sampling_jitter

> **Layer**: Layer 2 - Temporal & Motion Anomaly (observational)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — sampling_jitter

Sampling-interval regularity: per-episode coefficient of variation of consecutive dt (std/mean of timestamp deltas). Episode-scoped baseline; RISK_SIGNAL only (never EXCLUDE).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

