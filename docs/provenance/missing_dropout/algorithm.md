# missing_dropout

> **Layer**: Layer 1 - Data Integrity (hard check)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — missing_dropout

Per-feature frame-count comparison against the expected length. Expected frames come from info.json / episode metadata; each feature column is counted per frame index. Missing frames are gaps in the row sequence; dropouts are runs of missing samples within a feature (>= 1 consecutive frame absent). Severity: any missing frame or dropout fails the check (EXCLUDE).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

