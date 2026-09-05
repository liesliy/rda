# action_discontinuity

> **Layer**: Layer 2 - Temporal & Motion Anomaly (observational)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — action_discontinuity

Action-space step anomaly detection: first/second differences of the action stream are converted to robust z-scores via the MAD (median absolute deviation); steps beyond 5 sigma are spikes. Baseline is episode-scoped; affected joints are reported. Flags only (RISK_SIGNAL) - never auto-deleted.

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

