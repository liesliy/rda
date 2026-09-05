# timestamp_validity

> **Layer**: Layer 1 - Data Integrity (hard check)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 来源依据 — timestamp_validity

思想来源与公开先例（仅参考思想，未复制任何代码）：

- Standard event-timestamp validity practice in robotics logging (monotonic non-decreasing clock).
- Internal MVP Spec v0.2.0, Layer 1 definition.

## 语义边界声明（v1.1 设计文档 §1.1）

- RDA 的指标是 **evidence-oriented measurement framework**，不是"数据质量标准"。
- 与训练收益的关系不在本指标承诺范围内（收益属于 curation 方法）。
- 涉及 policy 对齐的阈值一律标注 **policy-profiled metric**，不称"标准定义"。

