# action_discontinuity

> **Layer**: Layer 2 - Temporal & Motion Anomaly (observational)
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录

# 算法说明 — action_discontinuity

Action-space step anomaly detection: first/second differences of the action stream are converted to robust z-scores via the MAD (median absolute deviation); steps beyond 5 sigma are spikes. Baseline is episode-scoped; affected joints are reported. Flags only (RISK_SIGNAL) - never auto-deleted.

## 阈值校准 (v0.9.17)

**MAD z-score spike detection threshold**: `spike_threshold = 5.0`（保持不变）
- 这是判定单个 timestep 是否为 spike 的 z-score 阈值
- 来源: Iglewicz & Hoaglin (1993), 0.6745-sigma consistency constant

**行为严重度分级阈值**（`compute_behavior_severity` 中使用）:
- `spike_count > 37`: severity 30 — 极端抖动（ArmnetBench failure P75）
- `spike_count > 27`: severity 20 — 高抖动（ArmnetBench failure median）
- `spike_count > 16`: severity 10 — 中等抖动（Youden's J 最优截断点）

**校准数据来源**:
- 数据集: ArmnetBench (armnet/armnetbench_v01_lerobot_so101), 2499 episodes, SO-101 robot
- 成功组 (N=915): mean=10.5, median=8, P25=4, P75=13
- 失败组 (N=1584): mean=29.9, median=27, P25=18, P75=37
- ROC-AUC = 0.8614 (强判别力)
- Cliff's δ = +0.7227 (large effect)
- Youden's J 最优截断点 = 15.5 (J=0.5918, sensitivity=0.793, specificity=0.799)
- 7 种 policy 上全部显著 (within-policy p<0.05)
- 校准脚本: `scripts/calibrate_thresholds.py`

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。
