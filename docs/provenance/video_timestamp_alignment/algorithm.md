# video_timestamp_alignment

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 算法说明 — video_timestamp_alignment

VA-A：视频时间跨度 vs parquet 时间轴一致性。用 chunk 元数据 `(to_timestamp − from_timestamp)` 与 parquet 首末 timestamp 差求相对偏差；>10% 判 EXCLUDE（系统错位，按帧索引训练会静默采错时刻），2%–10% 记 REVIEW，以内 PASS。与 `video_frame_integrity`（帧数交叉核对）互补：后者抓"帧数对不上"，本指标抓"时间轴漂移但帧数恰好匹配"的情形。

