# RDA (Robot Data Audit) — 机器人数据独立验收审计

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/robot-data-audit)](https://pepy.tech/project/robot-data-audit)
[![Tests](https://github.com/liesliy/rda/actions/workflows/ci.yml/badge.svg)](https://github.com/liesliy/rda/actions/workflows/ci.yml)

[![audit: lerobot/pusht](https://raw.githubusercontent.com/liesliy/rda/main/docs/examples/rda_badge_pusht.svg)](https://liesliy.github.io/rda/examples/rda_report_pusht.html) [![audit: AgiBotWorld2026 RL](https://raw.githubusercontent.com/liesliy/rda/main/docs/examples/rda_badge_agibot_rl.svg)](https://liesliy.github.io/rda/examples/rda_report_agibot_rl_hgdagger.html)

> **机器人数据的独立质量评估。** 本地运行——数据不出你的机器。
>
> RDA 是诊断工具，不保证训练成功率提升。

RDA 审计机器人操作数据集（LeRobot 格式），对每一集给出三档判定——
**PASS / REVIEW / EXCLUDE**——并附测量诊断。适合在验收供应商数据、开训、
或发布 benchmark 之前做一次独立检查。

**当前版本：v0.9.2** —— `pip install robot-data-audit`。

## 四层审计架构

每一集数据依次经过四层。核心设计原则：**只有 L1 硬性完整性检查能把一集判为
EXCLUDE，诊断性测量永远不会。** 这把"数据坏了"和"数据看着不寻常"分开，避免
观测性信号被误当成致命缺陷。

| 层级 | 职责 | 指标数 | 能否决定判定 |
|---|---|---|---|
| **L1 — 完整性门禁** | 确定性硬检查（缺失/NaN/限位/视频流） | 9 | ✅ PASS → EXCLUDE |
| **L2 — 轨迹诊断** | 观测性运动与视频异常 | 8 | ❌ 仅出 finding |
| **L3 — 数据集画像** | 训练数据效率与覆盖率 | 4 | ❌ 仅出 finding |
| **L4 — 数据集汇总** | 数据集级 P10/P50/P90 聚合 | — | 📊 仅报告 |

一集只有在 L1 硬检查失败时才 **EXCLUDE**；REVIEW/PASS 由 L1 状态决定，
L2/L3 只提供测量与 finding 供人工复核。RDA 负责测量与呈现——接受/拒收的
决定权在你。

## 安装

```bash
pip install robot-data-audit
```

可选依赖：

| 层级 | Extra | 解锁 |
|---|---|---|
| 核心（默认） | — | parquet 审计：完整性 + 时序/运动 + 数据集效用指标 |
| 视觉 | `[video]` 或 `pip install av` | 视频视觉指标（冻结/时间戳对齐/流跨度·偏移·漂移/质量） |
| LeRobot | `[lerobot]` | 通过 lerobot 包加载 `.parquet` 数据集 |
| 网页面板 | `[ui]` | Streamlit 网页面板 |
| 全量 | `[all]` | 以上全部 |

**未安装 PyAV 时，视觉指标报"未审计"，绝不报"通过"**：JSON 报告顶层带
`skipped_by_missing_dep` 字段，CLI 会打印被跳过的检查清单，网页面板同样显示
"未审计 ≠ 通过"提示条。

## 快速开始

```bash
# 1. 审计数据集 —— 四层共 21 个指标，三档判定
rda audit /path/to/lerobot/dataset

# 2. 按目标模型类型给出优化建议
rda recommend /path/to/dataset --policy temporal   # 或 frame-wise

# 3. 可选的网页面板
rda ui
```

`rda audit` 完全离线，输出结构化 JSON 报告：逐集判定、每个指标的
measurement/finding，以及数据集级 `acceptance_summary`（P10/P50/P90 基线、
运行时离群值、未检查清单）。`rda recommend` 在本地完成全部指标计算，只把聚合
统计（<1 KB）发给规则 API——结果本地缓存可离线复用，私有部署可用
`RDA_API_URL` 指向自己的服务器。

### 代码调用

```python
import numpy as np
from rda.io.schema import EpisodeData
from rda.audit.episode_audit import EpisodeAuditor

episode = EpisodeData(
    episode_index=0,
    num_frames=n_frames,
    timestamps=np.array(timestamps),          # 秒
    observation={"state": state_array},      # 形状 (T, DoF)
    action={"joint_pos": action_array},      # 形状 (T, DoF)
    meta={"fps": 10, "source": "my/dataset"},
)
result = EpisodeAuditor().audit(episode)
print(result.verdict)                        # PASS / REVIEW / EXCLUDE
for name, metric in result.metrics.items():
    if metric.has_finding:
        print(name, metric.measurement)
```

## 21 个指标

**L1 — 完整性门禁（硬检查，可判 EXCLUDE）**
`missing_dropout` · `invalid_values`（NaN/Inf）· `schema_consistency` ·
`temporal_validity` · `joint_limit`（三级 PASS/REVIEW/EXCLUDE，参数
`approach_threshold` / `consecutive_frames` / `jump_multiplier` 可配）·
`video_frame_integrity` · `video_freeze` · `video_timestamp_alignment` ·
`video_stream_presence`

**L2 — 轨迹诊断（观测性，不判 EXCLUDE）**
`sensor_sync` · `sampling_jitter` · `velocity_acceleration` ·
`action_discontinuity`（基于 MAD 的突变检测）· `visual_quality` ·
`video_stream_span_consistency` · `video_stream_temporal_offset` ·
`video_stream_temporal_drift`

**L3 — 数据集画像（效率与覆盖）**
`idle_ratio`（三级回退：30-bin 谷底 → 3×MAD → 1e-6 下限）·
`distribution` · `coverage` · `temporal_structure`

指标还按**跨平台可移植性**分类（MVP Spec v0.2.0 §1.5）：Tier-1 通用
（`duration_sec`、`spike_count`、`effective_motion_ratio`，任意机器人可直接
比较）；Tier-2 可归一化（速度/加速度/加加速度/路径长度，需平台缩放）；
Tier-3 平台专属（关节限位、工作空间、力矩/力/触觉）。

## 真实数据集验证

**12 个本地数据集、4,959 集、同一套默认阈值、零调参**——完整表格见
[docs/benchmark.md](docs/benchmark.md)。

**四个热门 LeRobot 数据集实测（v0.9.2）**——我们用 RDA 跑了
`lerobot/pusht`、`aloha_sim_transfer_cube_human`、`xarm_lift_medium` 和
`droid_100`（共 1,156 集，覆盖 2-DOF 仿真、14-DOF 双臂仿真、4-DOF 单臂和
7-DOF Franka）。全部通过 L1 完整性；但数据集画像差异显著——中位空闲帧占比
从 xArm 的 **20.8%** 到 PushT 的 **81.7%**，双臂 ALOHA 仿真每集中位突变
30.5 次。所有数字都可直接用 PyPI 包复现（`pip install robot-data-audit`）。

**lerobot/libero_10（v3.0）全量审计**——379 集、101,469 帧：所有适用的
完整性检查通过、0 个硬伤。**[查看报告 →](docs/benchmark_libero10.md)**

**盲测**——往 `lerobot/pusht` 里注入 50 个缺陷集（5 类缺陷，seed=42），
另留 156 集作对照。宽口径下 50 个全部命中，严格口径下 precision **1.000**
（对照组零误伤）。
**[查看盲测报告 →](https://liesliy.github.io/rda/examples/rda_report_pusht.html)**

**AgiBotWorld2026 实测**——对智元第三期数据集的第三方审计：仿真 5 个任务
全量 + 1 个真机 RL 包，共 1,112 集；4,448 项完整性检查 0 失败；RDA 动作
跳变信号与官方「人工接管」标记交叉验证，接管边界处 **3.1 倍富集**。零适配
直接跑通。**[查看案例报告 →](docs/examples/agibotworld2026.md)**

每次审计都可以渲染成可分享的单文件 HTML 报告和 README 徽章：

```bash
python tools/rda_render.py rda_report.json --html report.html --badge badge.svg
```

更多：[CLI 参考与指标表](docs/cli.md) · [实验记录](experiments/) · [真实使用反馈表](https://github.com/liesliy/rda/issues/new?template=real-world-feedback.yml)

## 治理保障

RDA 的指标 I/O 由核心参数规范锁定，每次变更都有语义不变量测试在 CI 中守护。
7 个不变量守护测试（INV-003 … INV-009）保护核心架构——例如"L2 诊断不得判
EXCLUDE"、"不得把推断当成测量上报"、"报告必须携带工具版本号"。完整测试套件
在每次推送时跨 Python 3.10–3.12 运行。

## 引用

```bibtex
@software{robot_data_audit,
  title = {Robot Data Audit: Quality Auditing for Robot Manipulation Datasets},
  author = {Niu Su Tech},
  year = {2026},
  url = {https://github.com/liesliy/rda}
}
```

MIT License.
