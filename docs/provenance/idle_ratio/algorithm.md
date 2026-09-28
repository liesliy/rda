# idle_ratio

> **Layer**: Layer 3 - Dataset Utility
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — idle_ratio

Dataset-level idle statistics: median idle ratio and effective motion ratio (1 - idle) across episodes, reusing the temporal_sufficiency segmentation. RISK_SIGNAL only.

## 阈值校准 (v0.9.17)

**idle_ratio 本身是纯观测指标**，不影响 verdict。但 v0.9.17 新增了
frozen episode 检测，基于 `effective_motion_ratio` 的极端低值。

**Frozen episode 检测阈值**: `FROZEN_EPISODE_EMR_THRESHOLD = 0.02`
- 当 `effective_motion_ratio < 0.02`（即 < 2% 有效运动）时，verdict 从 PASS 升级为 REVIEW
- 校准依据:
  - 盲测冻结 episode: EMR = 0%（注入 motion=0% 的 episode，10/10 被捕获）
  - ArmnetBench 真实数据最低 EMR: 0.023 (success), 0.026 (failure)
  - 14 个 benchmark 数据集最低中位 EMR: ~5.3% (imperialcollege_sawyer)
  - 阈值 0.02 在冻结 episode (EMR=0) 和真实数据 (EMR≥0.023) 之间提供了清晰的分割

**ArmnetBench 统计** (N=2499, SO-101):
- 成功组 EMR: mean=0.239, median=0.238
- 失败组 EMR: mean=0.361, median=0.355
- ROC-AUC = 0.8115 (中等判别力)
- Cliff's δ = +0.6231 (large effect)
- 注意: 失败组 EMR 反而更高（失败 episode 运动更多/更不稳定）

**回归修复说明**:
- v0.5.4: idle_ratio 在 REVIEW_METRICS 中，冻结 episode 会被标记为 REVIEW
- v0.9: idle_ratio 移至 DIAGNOSTIC_METRICS，冻结 episode 不再被捕获（已知回归）
- v0.9.17: 新增 `check_frozen_episode()` 函数，专门恢复冻结 episode 检测

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。
