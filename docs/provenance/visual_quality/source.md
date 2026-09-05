# visual_quality

> **Layer**: Layer 2 - Temporal & Motion Anomaly（观测型，REVIEW 级，永不否决）
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 来源依据 — visual_quality

思想来源与公开先例（仅参考思想，未复制任何代码）：

- score_lerobot_episodes（Apache-2.0）：Laplacian 方差判模糊、10 帧均匀采样、max(blur, exposure) 聚合、暗帧/曝光带思想。
- HF 博客 GIGO 原文：语义级好坏是开放问题 → RDA 只测量呈现，不裁决。这正是 VA-B 为何停留在 REVIEW 级的定位依据。

## 语义边界声明（v1.1 设计文档 §1.1）

- **与 score_lerobot 的差异化**：它拿语义/质量分删数据；RDA 只给测量+复核建议，删除决定留给人。
- 不做 VLM 语义判断、不做遮挡语义判断（VA-C 明确不做）。

