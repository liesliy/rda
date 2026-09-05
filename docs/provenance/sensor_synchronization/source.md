# sensor_synchronization

> **Layer**: Layer 2 - Temporal & Motion Anomaly (observational)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 来源依据 — sensor_synchronization

思想来源与公开先例（仅参考思想，未复制任何代码）：

- Cross-correlation time-offset estimation (classic signal processing; Knapp & Zhang 1976, GCC for time-delay estimation).
- Internal MVP Spec v0.2.0, Tier-1 calibration layer.

## 语义边界声明（v1.1 设计文档 §1.1）

- RDA 的指标是 **evidence-oriented measurement framework**，不是"数据质量标准"。
- 与训练收益的关系不在本指标承诺范围内（收益属于 curation 方法）。
- 涉及 policy 对齐的阈值一律标注 **policy-profiled metric**，不称"标准定义"。

