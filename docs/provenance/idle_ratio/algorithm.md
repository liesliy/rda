# idle_ratio

> **Layer**: Layer 3 - Dataset Utility
> **Status**: 自研实现（无第三方代码） · RDA v0.6.0 provenance 记录
> **最后修订**: v0.12.3（2026-11-05，回退阈值修复）

# 算法说明 — idle_ratio

Dataset-level idle statistics: median idle ratio and effective motion ratio (1 - idle) across episodes, reusing the temporal_sufficiency segmentation. RISK_SIGNAL only.

## 阈值选择（v0.12.3）

每集的空闲阈值由 `rda.metrics.motion.select_idle_threshold()` **单点决定**，
`IdleRatioMetric`（`rda audit`）与 `rda.recommend.temporal_metrics.compute_idle_mask`
（`rda recommend`）共用同一实现 —— v0.12.3 之前两处各有一份拷贝，存在分叉风险。

两级 + 兜底：

| 方法 | 触发条件 | 阈值 |
|---|---|---|
| `bimodal_gap` | 运动量直方图存在真实静止模态分离（低簇 ≥10%，谷底 bin <1%，其后回升 >2%） | 谷底 bin 上界 |
| `median_ratio` | **未检出静止模态** | `IDLE_FALLBACK_RATIO × median(逐帧运动量)`，默认 0.10 |
| `floor` | 上一步阈值低于 `1e-6`（如整集冻结，median=0） | `1e-6` |

`motion` 定义为 action 逐帧差分的跨关节 L2 范数。

## v0.12.3 修复：`3 × MAD` 回退把连续运动判成"大部分时间没动"

### 缺陷

v0.12.2 及之前，未检出双峰时回退到 `3 × MAD`。MAD 是**离散度**统计量
（平滑单峰分布下 ≈0.54σ），`3 × MAD` 落在约 1.6σ 处 —— **高于中位数**。
于是任何连续运动轨迹都会被判成"多数帧空闲"，**由构造决定，与数据质量无关**。

### 实测证据（同一套默认参数，仅换回退规则）

| 数据集 | 集数 | 修复前中位空闲率 | 修复后 | 走回退的集数 |
|---|---:|---:|---:|---:|
| `lerobot/aloha_sim_transfer_cube_human` | 50 | 71.2% | **0.6%** | 43/50 |
| `lerobot/droid_100` | 100 | 70.7% | **8.6%** | 89/100 |
| OpenLET / Kuavo5W `box_depalletizing` | 1 | 74.5% | **3.5%** | 1/1 |
| `lerobot/xarm_lift_medium` | 800 | 20.8% | 20.8% | 3/800 |
| `lerobot/pusht` | 206 | 81.7% | 81.1% | 70/206 |

两条基线几乎不动，是因为它们**真的**有静止模态（`bimodal_gap` 路径 136/206 与
797/800），修复只影响"根本没有静止模态"的那些集。

反证：LET 样本的逐帧运动量中位数 3.905、p95 11.90、最大 16.78（度/帧），
14 关节单集最大位移 9~150 度 —— 数据明显在动，`3 × MAD = 6.38` 却高于中位数，
判出 74.5% 空闲。按 `0.1 × median = 0.39` 只切出 5 个真正冻结的帧（3.5%）。

### 系数选择

`IDLE_FALLBACK_RATIO = 0.10` 的含义是"一帧的位移不到该集典型位移的十分之一"，
即接近静止。敏感性实测（中位空闲率）：

| 数据集 | c=0.05 | c=0.10 | c=0.15 | c=0.20 | c=0.30 |
|---|---:|---:|---:|---:|---:|
| pusht | 81.1% | 81.1% | 81.1% | 81.1% | 81.1% |
| xarm_lift | 20.8% | 20.8% | 20.8% | 20.8% | 20.8% |
| aloha_sim_transfer_cube | 0.0% | 0.6% | 1.8% | 4.5% | 12.0% |
| droid_100 | 8.6% | 8.6% | 8.8% | 9.4% | 12.4% |
| LET 样本 | 3.5% | 3.5% | 3.5% | 3.5% | 8.5% |

已校准的两个基准在 c 的整个区间内不动（它们走双峰路径），剩下三个在 c=0.05~0.20
区间稳定。取 0.10 居中以保留余量。

### 为什么不额外加"簇分离"守卫

评估过在 `bimodal_gap` 路径上加"低簇中位运动 / 高簇中位运动 < 0.2"的守卫，
实测会把 `xarm_lift_medium` 的 47% 合法双峰集判为无效（该数据集的分离比中位
0.51）。双峰路径是既有校准行为，本次不动，避免把测量口径的争议混进一个 bug 修复。

## 阈值校准 (v0.9.17 遗留，仍然有效)

**idle_ratio 本身是纯观测指标**，不影响 verdict。但 v0.9.17 新增了
frozen episode 检测，基于 `effective_motion_ratio` 的极端低值。

**Frozen episode 检测阈值**: `FROZEN_EPISODE_EMR_THRESHOLD = 0.02`
- 当 `effective_motion_ratio < 0.02`（即 < 2% 有效运动）时，verdict 从 PASS 升级为 REVIEW
- 校准依据:
  - 盲测冻结 episode: EMR = 0%（注入 motion=0% 的 episode，10/10 被捕获）
  - ArmnetBench 真实数据最低 EMR: 0.023 (success), 0.026 (failure)
  - 14 个 benchmark 数据集最低中位 EMR: ~5.3% (imperialcollege_sawyer)
  - 阈值 0.02 在冻结 episode (EMR=0) 和真实数据 (EMR≥0.023) 之间提供了清晰的分割

> 注意：上面 ArmnetBench 的 EMR 统计（成功组中位 0.238 / 失败组 0.355）是在
> v0.12.2 的阈值口径下得到的。回退规则变更后这些数字需要复测，其
> ROC-AUC 0.8115 的判别力结论在复测前不作数。

**回归修复说明**:
- v0.5.4: idle_ratio 在 REVIEW_METRICS 中，冻结 episode 会被标记为 REVIEW
- v0.9: idle_ratio 移至 DIAGNOSTIC_METRICS，冻结 episode 不再被捕获（已知回归）
- v0.9.17: 新增 `check_frozen_episode()` 函数，专门恢复冻结 episode 检测

## 输出

该指标的 measurement/assessment 结构见 `rda/audit` 与 `rda/report` 中
对应 Metric 类的实现，字段语义以代码为准。

`details.threshold_method` 取值 `bimodal_gap` / `median_ratio` / `floor`
（v0.12.3 起 `mad` 不再出现）。
