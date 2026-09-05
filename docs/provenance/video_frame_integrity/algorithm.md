# video_frame_integrity

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — video_frame_integrity

Video-stream presence and decodability check: every declared image/video feature must resolve to a readable file whose frame count is consistent with the episode length. Unreadable or missing streams are reported per feature. Full-frame decoding is NOT performed in the default path (performance budget: metadata + sampling only).

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

