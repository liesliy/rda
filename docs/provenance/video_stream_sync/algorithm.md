# video_stream_sync

> **Layer**: Layer 1 - Data Integrity (hard check, NA without video)
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 算法说明 — video_stream_sync

VA-A：多相机流存在性与同步性。episode 声明的每个相机 feature 必须能解析到可读文件（缺失 = 静默丢模态，EXCLUDE）；≥2 路时比较各路时间跨度与中位数的偏差（>10% = 视角不同步，EXCLUDE）。单相机数据集记 N/A。

