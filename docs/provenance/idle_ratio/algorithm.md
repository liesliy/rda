# idle_ratio

> **Layer**: Layer 3 - Dataset Utility
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — idle_ratio

Dataset-level idle statistics: median idle ratio and effective motion ratio (1 - idle) across episodes, reusing the temporal_sufficiency segmentation. RISK_SIGNAL only.

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

