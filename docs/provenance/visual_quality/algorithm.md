# visual_quality

> **Layer**: Layer 2 - Temporal & Motion Anomaly（观测型，REVIEW 级，永不否决）
> **Status**: 自研实现（无第三方代码） · RDA v0.7.0 provenance 记录


# 算法说明 — visual_quality

VA-B 视觉质量测量：每 episode 均匀采样 10 帧（128×128 灰度），逐帧测量 ① 清晰度 = 中心裁剪的 Laplacian 方差（经典模糊代理）② 曝光 = 平均亮度 + 暗帧(<50)/过曝(>230)占比 ③ 对比度 = P95−P5 亮度。episode 级聚合用 `max(blur_penalty, exposure_penalty)`，≥0.5 记 REVIEW（人工复核信号，附最差帧时间戳可定位回看），否则 PASS。**测量级指标，永不产生 EXCLUDE**。

**阈值声明**：`blur_var_floor=80` 等参考 score_lerobot_episodes 的公开参数语义作为初始先例，但按 v1.1 排期风险注记，**未直接照抄**——上线前已在真实数据集（libero_10）上校准（健康数据 penalty=0.00），后续数据集扩展时需按相机分辨率/焦距重校。

