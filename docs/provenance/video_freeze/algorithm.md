# video_freeze

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 算法说明 — video_freeze

VA-A 视觉流完整性：检测"手臂在动而视频冻结"的相机掉线签名。将 episode 对应的视频段解码为 64×64 灰度帧，计算相邻帧平均绝对差分；冻结判定用**自适应阈值** `max(0.10, 0.25 × p10(帧差分))`——校准依据（libero_10 诊断，2026-09-05）：正常运动帧差分 ~1.0–3.0，慢动作 0.3–1.0，真停帧（codec 逐帧相同）< ~0.1。短停帧（≥0.5s 且手臂在动）记 REVIEW；多次停帧（≥3 次）或单次 ≥3s 或冻结占比 ≥30% 判 EXCLUDE。

## 输出

measurement 含 checked_features / freeze_region_count；details.freeze_regions 逐段列出 feature、视频/parquet 帧区间、时长、moving_ratio。

