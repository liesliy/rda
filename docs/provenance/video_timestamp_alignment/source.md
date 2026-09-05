# video_timestamp_alignment

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 来源依据 — video_timestamp_alignment

- LeRobot v3.0 chunk 视频布局的 from/to_timestamp 语义（本仓库 loader 已注入）。
- LingBot-VLA 2.0（技术稿转述）：视频-状态一致性是量产管线必选项。
- 容差数值（2%/10%）为本仓库自定，无外部出处。

## 语义边界声明（v1.1 设计文档 §1.1）

- 本指标只测量两条时间轴的相对偏差，不判断"哪个时刻画面该是什么"（那是 VA-A+ URDF 投影的范围）。

