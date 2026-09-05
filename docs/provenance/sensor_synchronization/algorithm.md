# sensor_synchronization

> **Layer**: Layer 2 - Temporal & Motion Anomaly (observational)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — sensor_synchronization

Pairwise cross-correlation alignment over per-stream timestamps: for each stream pair, the offset that maximizes cross-correlation of resampled signals is taken as the estimated skew; p95/max of |skew| over pairs are reported. N/A when per-stream timestamps are not provided.

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

