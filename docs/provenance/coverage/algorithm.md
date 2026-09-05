# coverage

> **Layer**: Layer 3 - Dataset Utility (episode-level semantic: state_space_occupancy)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — coverage

State-space occupancy: the concatenated state space is binned into a 10x10x10 grid over the dataset's observed bounding box; occupancy = fraction of non-empty cells per episode (observationally weighted). Low occupancy signals limited pose/scene diversity for采集方 feedback. Named `coverage` internally; reported as coverage_proxy pending REQ-9 calibration (design doc v1.1: renamed from bare 'coverage' to avoid over-claiming).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

