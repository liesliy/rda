# video_freeze

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 来源依据 — video_freeze

思想来源与公开先例（仅参考思想，未复制任何代码）：

- LingBot-VLA 2.0 数据清洗管线（官方技术稿转述）：视觉-状态一致性检查、静态/异常样本过滤。
- RDA 已有 `missing_dropout`（传感器掉线）的视觉版推论：掉线的模态"时间轴继续、数值冻结"。
- 阈值未采用任何外部数值；自适应 epsilon 由本仓库在 libero_10 上诊断得出。

## 语义边界声明（v1.1 设计文档 §1.1）

- 本指标只断言"视觉模态在这些时间跨度内不存在"，不判断画面语义好坏。
- URDF 投影逐帧比对（VA-A+）明确不在本指标范围内（opt-in 深度校验，未来版本）。

