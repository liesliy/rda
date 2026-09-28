# RDA 核心问题 FAQ（v0.9.16）

> 整理时间：2026-09-24

---

## Q1. 为什么做 RDA？

具身智能 / 机器人数据当前缺乏统一的"体检"工具。数据质量问题（丢帧、静止片段、传感器失同步、动作跳变）往往**在训练后才被发现**，浪费大量算力。RDA 的目标：

- **训练前**做数据体检，提前发现脏数据，节省 GPU 时间
- 把"数据质量"这个黑盒问题，拆成可量化、可复现的指标
- 不绑定具体算法/模型，做通用数据集层审计工具
- 提供 PASS / REVIEW / EXCLUDE 判定，下游可直接过滤

---

## Q2. RDA 整体架构？

三层分离设计：

```
RDA 审计报告
├── Integrity Gate         → 硬判定层（PASS / REVIEW / EXCLUDE / N/A）
│   ├── Structural（schema、timestamp、missing、invalid、joint_limit）
│   └── Video Integrity（freeze、stream 同步、帧数一致性）
├── Trajectory Diagnostics → 软诊断层（measurement + finding，不自动否决）
│   └── sensor_sync、jitter、velocity/accel、action_discontinuity、idle_ratio、visual_quality
└── Dataset Summary        → 数据集级画像（跨 episode 分布）
    └── idle_ratio 分布、sync 分布、visual_quality 分布
```

关键原则：
- **判定与诊断分离**：Integrity Gate 决定"数据能不能用"，Diagnostics 描述"数据长什么样"
- **N/A ≠ PASS**：测不到就明说，不假装通过
- **阈值透明可配置**：所有 heuristic 阈值都标为可配置，并提供 strict/default/lenient 三套 profile

---

## Q3. 数据质量怎么定义？

RDA 不追求"通用好"，而是定义"**可训练性**"——数据是否适合直接喂给下游模型。从三个维度切：

| 维度 | 含义 | 代表指标 |
|---|---|---|
| **完整性** | 数据有没有丢 | missing_dropout、video_frame_integrity、schema_consistency |
| **时序合理性** | 时间戳是否靠谱 | timestamp_validity、sampling_jitter、sensor_synchronization |
| **信号质量** | 物理信号是否可信 | action_discontinuity、velocity_acceleration、visual_quality |

一个"好数据"= 没丢帧 + 时间戳单调合理 + 多流同步稳 + 动作/画面没有跳变。

---

## Q4. 如何做完整性检查？

多个指标分层交叉验证，**不是单一指标拍板**：

1. **missing_dropout**：检测 action/state 的丢帧率（基于时间戳间隔 gap）
2. **video_frame_integrity**：MP4 实际帧数 vs parquet 期望帧数，判断截断或计数错配
3. **schema_consistency**：episode 内所有字段是否存在、shape 是否对齐
4. **invalid_values**：NaN/Inf/极端离群值检测
5. **video_stream_presence / span_consistency**：多相机流是否齐全、起止跨度是否一致

任一 Integrity Gate 命中 EXCLUDE 条件 → 整个 episode 被标记剔除；命中 REVIEW → 人工复核。

---

## Q5. 如何判断时间同步？

**最近邻匹配法**（nearest-neighbour offset）：

1. 对任意两路流 A、B，对 A 的每一帧时间戳 `ts_a_i` 在 B 中二分查找最近邻
2. 算逐帧偏移 `offset_i = ts_a_i − nearest(ts_b)`
3. 统计每对流的 `|offset|` 的 median / p95 / p99 / max / signed_median
4. 所有流对取最差的 `worst_p95_offset_ms` 作为整体指标

**两条路径**：
- `sensor_synchronization`：通用多传感器（IMU/力/相机…）
- `video_stream_temporal_offset` + `video_stream_temporal_drift`：视频专用，后者额外拟合 `offset(t)=a+b·t` 抓时钟漂移

**没数据 = N/A**：没有 `stream_timestamps` 或只有 1 路流时，**明确标 N/A**，绝不假装通过。这也是 v0.9 评审时修的头号 bug。

---

## Q6. Action discontinuity 怎么算？

检测动作轨迹中的**突变（jerk）**，用二阶差分 + MAD 鲁棒异常检测：

1. 取主动作数组（优先 `joint_pos` / `position` / `action`）
2. 一阶差分：`Δa[t] = a[t+1] − a[t]`，取 L2 范数
3. 二阶差分：`Δ²a[t] = Δa[t+1] − Δa[t]`，取 L2 范数 ← 关键
4. MAD z-score：`z = 0.6745 × |Δ²a − median| / MAD(Δ²a)`
5. `spike_count = sum(z > 5.0)`
6. 额外输出逐关节 spike 计数、Δ/Δ² 的分位数统计

**为什么看二阶差分而不是一阶？** 一阶差分大 = 运动快（正常）；二阶差分大 = "变化的变化"突变（不正常）。能区分"正常快速摆动"和"数据真的断裂了"。

---

## Q7. Threshold 怎么确定？

**所有阈值都是可配置的**，提供 `strict / default / lenient` 三套 profile。核心原则（INV-008）：

- 统一采用 **reference + MAD z-score**，禁止引入无来源的固定 σ 或绝对值阈值
- MAD 对异常值自身不敏感（中位数 + 绝对中位差），不会被少数毛刺拉偏基准线

**具体取值**：
- `spike_threshold = 5.0`（对应正态分布下约 1/170 万的概率）
- 选 5 不选 3 的原因：机器人动作数据尾部厚、正常快速动作本身就比正态分布"胖"；3.0 会误报，5.0 只抓极端跳变
- 该指标是纯观测性的（observational），永远 pass、不否决数据——宁可漏报轻微异常，也不要假阳性噪音干扰用户

**其他阈值示例**：
- gap_threshold = median_dt × 2.5（基于数据自适应，不拍脑袋）
- video_freeze 阈值 = `max(0.10, 0.25 × p10(frame diffs))`（自适应）

---

## Q8. 为什么某个指标能说明数据质量？

每个指标都对应一个**可证伪的物理假设**，且与下游任务有直接因果：

| 指标 | 它抓的问题 | 下游后果 |
|---|---|---|
| missing_dropout | 时间轴有空洞 | 训练时状态转移错误 |
| timestamp_validity | 时间戳非单调 / 重复 / 跳变 | 时序模型学不出因果关系 |
| action_discontinuity | 动作跳变 / jerk 异常 | 模型预测动作不可执行 |
| sensor_synchronization | 多模态错位 | 视听融合信号污染 |
| video_freeze | 相机画面卡死 | 视觉特征失效 |
| idle_ratio | 数据里有多少"无效静止" | 浪费训练算力、引入偏置 |
| visual_quality | 模糊 / 过曝 / 欠曝 | 视觉 backbone 学不到真特征 |

这些不是"看起来像问题"的启发式，而是**可量化、可复现、可被下游任务验证**的信号。

---

## Q9. RDA 有哪些误报 / 局限？

**已知局限**：

1. **没有视频硬时间戳** → `video_stream_temporal_offset/drift` 直接 N/A，测不到
2. **只读 parquet 级信息**，不解码视频像素 → 画面语义问题（标注错误、目标穿模）检测不到
3. **阈值 profile 仍需人工选**：strict/default/lenient 没有自动选择逻辑，跨任务差异大
4. **MAD 假设数据单峰**：双峰分布（如两段不同动作拼接）会让 z-score 失真
5. **纯观测指标默认通过**：action_discontinuity、sensor_sync 永远 PASS，需要用户主动看 measurement
6. **没有跨 episode 的 ranking 自适应**：每个 episode 独立判断，dataset summary 只是事后汇总
7. **LIBERO 这类只有帧号、没有物理时间戳的数据集**：timestamp_validity 走帧号模式、sensor_sync 直接 N/A

**已修复的误报**（v0.9.15 的 D-24 全部 14 个偏差）：
- Pusht 仿真数据误报 blown_out（visual_quality 已降为 opt-in）
- Severity 接线缺陷导致误判 verdict

---

## Q10. 如果接入真实机器人生产 pipeline 怎么改？

**最小改动路径**（零侵入）：

1. **作为训练前 filter**：在 dataloader 上游加一步 `rda audit`，EXCLUDE 的 episode 不进训练集
2. **输出 JSON 直接喂 MLOps**：RDA 的 measurement dict 可以接入 W&B / MLflow，按 `spike_count`、`worst_p95_offset_ms` 做数据集健康度看板
3. **CLI 接口稳定**：`rda audit --profile strict <path>` 直接嵌 CI/CD

**中度集成**（扩展能力）：

4. **自定义适配器**：继承 `rda.io.schema.EpisodeData`，把 ROS bag / HDF5 / 自研格式转进来
5. **阈值按任务定制 profile**：为抓取/导航/双臂等不同任务写专用 profile
6. **和训练 reward 挂钩**：把 RDA measurement 作为数据权重，spike 多的 episode 训练时降权

**重度集成**（深度改造）：

7. **在线审计**：从离线 batch 改为 streaming 模式，数据录制时实时跑 RDA 指标
8. **与仿真平台对接**：MuJoCo / Isaac 的 ground truth 可以直接用来校准阈值

---

## Q11. 多模态传感器怎么做时间 / 空间对齐？

### 时间对齐（已实现）

见 Q5。**最近邻匹配 + offset 统计** 是核心方法。

对于**视频流**，额外依赖 PTS（presentation timestamp）或 parquet 的逐帧 timestamp。如果只有帧号没有物理时间戳 → **N/A，并建议硬件触发或启用 PTS 重录**。

`video_stream_temporal_drift` 还拟合**时钟漂移率** `offset(t) = a + b·t`，单位 ms/min，抓"两个时钟越走越不齐"的长期问题。

### 空间对齐（目前只部分实现）

当前 RDA 的"空间一致性"主要通过：

- **schema_consistency**：所有 episode 的字段 shape、传感器数量是否一致
- **video_stream_presence / span_consistency**：多相机是否齐全、覆盖范围是否一致
- **video_frame_integrity**：MP4 帧数与期望是否匹配

**当前没做的**：
- 相机外参标定验证（extrinsic calibration 检查）
- 传感器坐标系一致性（如 TF 树 / 手眼标定）
- 空间几何关系验证（如双目左右相机基线长度）

这些属于**物理层 / 机器人系统层**的校验，超出 RDA "数据层审计" 的定位。如果需要做，建议拆出单独的 `CalibrationCheck` 模块，读取 URDF / camera_info / TF 数据。

---

## 总结一句话

> **RDA 是一个"数据层体检工具"，用一组可量化、可复现、可配置的指标，把"数据能不能训练"从玄学变成工程问题。**
