# temporal_sufficiency

> **Layer**: Layer 2 - Temporal & Motion Anomaly (observational; policy-profiled)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — temporal_sufficiency

Idle/active segmentation of episodes from a motion threshold: idle_ratio (total and prefix), active-run length percentiles (p50/p90/max), idle->active transition counts, and valid_window_ratio over sliding windows (5/10/20 frames; policy_chunk_size-aligned in v0.6.0+). Thresholds are POLICY-PROFILED (calibrated against openpi/DROID published filter parameters, chunk 8 / min_idle_len 7 / min_non_idle_len 16) - they are not a universal idle definition.

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

