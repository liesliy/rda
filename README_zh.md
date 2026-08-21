# RDA (Robot Data Audit) — 机器人数据集体检工具

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/robot-data-audit)](https://pepy.tech/project/robot-data-audit)
[![Tests](https://github.com/liesliy/rda/actions/workflows/ci.yml/badge.svg)](https://github.com/liesliy/rda/actions/workflows/ci.yml)

> **机器人数据集质量审计 + 优化建议。**
> 先诊断数据问题，再给带置信度的修复建议。
> RDA 是**诊断工具**——不保证训练成功率提升。

RDA 审计机器人操作数据集（LeRobot 格式）的完整性、时序一致性、运动质量和分布覆盖，然后基于你的目标模型架构生成优化建议。

<img width="2549" height="1403" alt="rda首页" src="https://github.com/user-attachments/assets/35173b6a-275c-466c-985f-9674f7a59a1d" />

<img width="2549" height="1403" alt="RDA建议页" src="https://github.com/user-attachments/assets/e366cdf1-b169-46b0-b020-5327493a9124" />

## 功能特性

- **13 个质量指标**，分布在 4 个层级：完整性、时序、运动、分布
- **`rda recommend`** — 按目标模型类型校准的数据优化建议
  - 逐帧模型（MLP/BC）：温和的空闲帧裁剪建议
  - 时序模型（ACT/DP/Transformer）：保守的"不要裁剪"指引
  - 每条建议都带置信度（HIGH / EXPERIMENTAL / NOT_RECOMMENDED）
- **LeRobot v2.1 + v3.0** 双格式自动识别
- **三档判定**：PASS / REVIEW / EXCLUDE
- **证据分层报告**：分别展示 HARD_FAIL / RISK_SIGNAL / UNVERIFIABLE；统计信号不会被表述为已确认的数据错误
- **盲审支持**：可生成不提前暴露 RDA 结论的人工标注样本，并在外部 QC 标签回填后进行对照
- **CLI 优先设计**：JSON + 文本输出，管道友好
- **时序充分性分析**：空闲检测、有效动作段分布、有效窗口比例

## 在公开数据上的实测

RDA 已使用同一套默认阈值、未针对单个数据集调参，审计了 **12 个本地 LeRobot 格式数据集，共 4,959 集**。几个发现：

- 一个 xArm 数据集是真干净（767/800 集 PASS，中位空闲 20.8%）；而**同一个机器人平台的另一个数据集** 83.3% 的集都在空转——不是采得糙，是任务本身简单
- 一个 SO-100 社区数据集，**100% 的集都有动作尖峰**，中位空闲 86.7%
- 一份本地 LIBERO 副本有 773 个 episode 在已下载的数据文件中缺失。RDA 0.5.3 修复了 LeRobot v3.0 元数据与数据文件定位不一致的问题，能区分本地文件缺失和实际存在的 episode

完整表格、每数据集数字、五个反复出现的 pattern：**[docs/benchmark.md](docs/benchmark.md)**

方法论与其他实验记录（i18n 烟雾测试、服务器部署验证、spike/verdict bug 回归测试钉子、wheel 泄露守卫）：
**[experiments/](experiments/)**

## 安装

```bash
pip install robot-data-audit
```

带 LeRobot 依赖（用于 `.parquet` 数据集加载）：

```bash
pip install robot-data-audit[lerobot]
```

带 Streamlit 网页 UI：

```bash
pip install robot-data-audit[ui]
```

## 分享真实使用反馈

如果你在真实数据流程、数据集或现有 QC 流程中运行了 RDA，欢迎通过[真实使用反馈表](https://github.com/liesliy/rda/issues/new?template=real-world-feedback.yml)分享观察结果。
不需要上传数据，只需说明版本、数据格式、运行命令和客观现象；如果有独立 QC 结果或人工复核，也请说明是否一致。请删除原始数据、私有路径、客户信息和凭据。公开 Issue 不适合提交机密信息。

## 快速开始

### 1. 审计一个数据集

```bash
rda audit /path/to/lerobot/dataset
```

跑完 13 个指标，打印文本摘要。JSON 报告存到 `<dataset>/rda_report.json`。

### 2. 获取优化建议

```bash
# 逐帧模型（MLP、BC 等）
rda recommend /path/to/dataset --policy frame-wise

# 时序模型（ACT、Diffusion Policy、Transformer）
rda recommend /path/to/dataset --policy temporal

# JSON 输出，方便脚本处理
rda recommend /path/to/dataset --policy frame-wise --format json

# 英文输出（默认中文）
rda recommend /path/to/dataset --policy temporal --lang en
```

**`recommend` 告诉你什么：**
- 数据集是否有过量空闲帧
- 是否建议裁剪（以及裁剪多激进）
- 针对模型类型的警告（例如"时序模型绝对不要裁剪"）
- 每条建议的置信度和实验性说明

**隐私（v0.5.0 起）：** `rda recommend` 在**本地**完成全部指标计算，只把聚合统计（<1KB，不含任何原始 episode 数据）发给 RDA 规则 API（`https://rda.niusu2026.cn`）做评估。结果在本地缓存，离线可复用。`rda audit` 始终完全离线。私有/隔离部署可设置 `RDA_API_URL` 指向你自己的服务器。

### 3. 启动网页 UI

```bash
pip install robot-data-audit[ui]
rda ui
```

在 `http://localhost:8501` 打开 Streamlit 面板——上传数据集、实时进度审计、逐 episode 结果浏览、生成建议（走同一套隐私保护 API 路径）、导出报告。

**双语 UI（v0.5.2 起）：** 整个面板——包括后端建议文案和导出报告——通过侧边栏语言选择器在**英文**和**中文**之间干净切换，不混文字。

### 4. JSON 输出与管道

```bash
# JSON 输出到 stdout
rda audit /path/to/dataset --format json

# 生成可对外分享的盲审报告（路径会哈希脱敏）
rda audit /path/to/dataset --blind --format json

# 报告保存到自定义路径
rda audit /path/to/dataset -o /tmp/my_report.json

# 详细模式 + 平台信息
rda audit /path/to/dataset --platform so101 -v
```

### 5. Python API

```python
from rda.audit.dataset_audit import DatasetAuditor
from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset

dataset_info = load_lerobot_dataset("/path/to/dataset")
auditor = DatasetAuditor()
result = auditor.audit_dataset(dataset_info, iter_episodes("/path/to/dataset"))
print(f"DHI: {result.quality['dhi']} / 100")
```

## CLI 参考

### `rda audit`

```bash
rda audit [OPTIONS] PATH
```

| 选项 | 说明 |
|--------|-------------|
| `-o, --output FILE` | 保存 JSON 报告（默认：`<path>/rda_report.json`） |
| `--format [json\|text]` | 输出格式（默认：`text`） |
| `--platform TEXT` | 机器人平台（如 `so101`、`droid`），用于 Tier 3 指标 |
| `-v, --verbose` | 详细输出 |
| `--blind` | 对外分享报告时脱敏路径等识别信息 |

### `rda recommend`

```bash
rda recommend [OPTIONS] PATH
```

| 选项 | 说明 |
|--------|-------------|
| `--policy [frame-wise\|temporal]` | 目标模型架构类型（必填） |
| `-o, --output FILE` | 保存 JSON 建议报告 |
| `--format [json\|text]` | 输出格式（默认：`text`） |
| `--lang [zh\|en]` | 建议文案语言（默认：`zh`） |
| `-v, --verbose` | 详细输出 |

### `rda example`

显示用法示例和样例数据集路径。

### 退出码

| 退出码 | 含义 |
|------|---------|
| `0` | 完成，无 EXCLUDE 判定 |
| `1` | 出错（路径无效、加载失败等） |
| `2` | 完成，至少有一条 EXCLUDE 判定 |

## 理解建议

RDA 的建议遵循**保守、带证据分级**的方法：

| 置信度 | 含义 |
|-----------|---------|
| **HIGH** | 有优化实验支撑，风险低 |
| **EXPERIMENTAL** | 方向一致但未在你的设置上验证 |
| **NOT_RECOMMENDED** | 对你的模型类型可能有害，谨慎 |

**关键原则：**
- 所有建议都是**假设**，不是保证
- 效果因任务领域和模型架构而异
- 应用到训练数据前一定要在留出集上验证
- 时序模型（ACT、DP）通常对数据裁剪更敏感

## 指标总览

| 层级 | 指标 | 检测什么 |
|------|--------|----------------|
| L1 | 时间戳单调性 | 时钟重置、重复时间戳 |
| L1 | 帧间隔一致性 | 抖动或不规则采样 |
| L1 | 模式合规性 | 缺失/多余字段、类型不匹配 |
| L2 | 时序缺口检测 | 大段时间不连续 |
| L2 | 传感器同步 | 跨传感器时间戳漂移 |
| L2 | **时序充分性** | 空闲/活动结构、有效窗口分析 |
| L3 | 关节限位违反 | 执行器被驱动到安全范围外 |
| L3 | 速度尖峰 | 突发的不可信跳变 |
| L3 | 运动不连续 | 不平滑轨迹段 |
| L3 | 空闲帧检测 | 静止/暂停段 |
| L4 | 时长异常值 | 相比同批过短/过长的 episode |
| L4 | 尖峰数异常值 | jerk 曲线异常的 episode |
| L4 | 有效运动比 | 活动度过低的 episode |

## 项目结构

```
rda/
├── cli/          # Click CLI 入口
├── io/           # 数据加载与模式定义（LeRobot v2.1/v3.0）
├── metrics/      # 13 个审计指标实现
├── recommend/    # 优化建议引擎
├── audit/        # 数据集与 episode 级审计编排
└── report/       # 报告生成与摘要
```

## 开发与测试

```bash
git clone https://github.com/liesliy/rda.git
cd robot-data-audit
pip install -e ".[dev]"
pytest
```

### 给审计门本身做体检

`tests/` 目录的存在有一个值得讲的故事。0.4.x 时期有一个 bug：行为层正确计算了动作尖峰和空闲比，但判定聚合器完全无视这些信号——异常 episode 直接拿了 PASS 徽章。**审计门没在消费自己的证据。**

`tests/test_negative_control.py` 专门钉住这个失败模式：指标结果通过所有规则但含已知异常（150 个尖峰、一条冻结的机械臂）**必须**返回 REVIEW——而严重损坏必须保持 EXCLUDE（审计门在任何方向都不能 fail open）。其余测试覆盖 i18n 目录（zh/en key 对齐）和两个核心行为指标的边界行为。

CI 在每次 push 时跑测试，并额外验证闭源建议层的代码永远不会泄进发布的 wheel 包。

## 引用

```bibtex
@software{robot_data_audit,
  title = {Robot Data Audit: Quality Auditing for Robot Manipulation Datasets},
  author = {Niu Su Tech},
  year = {2026},
  url = {https://github.com/liesliy/rda}
}
```

## License

MIT
