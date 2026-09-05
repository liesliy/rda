# video_stream_sync

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 来源依据 — video_stream_sync

- LingBot-VLA 2.0（技术稿转述）：多视角错位过滤。
- 多模态训练的常识约束：缺失视角会被训练脚本以零/垃圾填充。

## 语义边界声明（v1.1 设计文档 §1.1）

- 本指标检查"流的存在性与跨度一致"，不做画面内容级同步校验（VA-A+ 范围）。

