# joint_limit

> **Layer**: Layer 1 - Data Integrity (hard check, NA without limits)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — joint_limit

Per-joint min/max limit violation check: each joint value is compared against provided soft/hard limits; violations are counted per joint. If no limits are provided the metric reports N/A (joint_limits_not_provided). Boundary subtlety: URDF soft/hard limit definitions differ between exporters, so borderline values may reflect limit-definition mismatch rather than true violations (design doc v1.1).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

