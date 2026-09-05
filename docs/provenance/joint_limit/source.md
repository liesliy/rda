# joint_limit

> **Layer**: Layer 1 - Data Integrity (hard check, NA without limits)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 来源依据 — joint_limit

思想来源与公开先例（仅参考思想，未复制任何代码）：

- URDF joint <safety_controller> / limit tag semantics: http://wiki.ros.org/urdf/XML/joint
- LingBot-VLA 2.0 data pipeline (URDF-based consistency checks): https://github.com/lingbot-ai
- Internal MVP Spec v0.2.0.

## 语义边界声明（v1.1 设计文档 §1.1）

- RDA 的指标是 **evidence-oriented measurement framework**，不是"数据质量标准"。
- 与训练收益的关系不在本指标承诺范围内（收益属于 curation 方法）。
- 涉及 policy 对齐的阈值一律标注 **policy-profiled metric**，不称"标准定义"。

