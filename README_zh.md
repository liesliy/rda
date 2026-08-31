# RDA (Robot Data Audit) — 机器人数据独立验收审计

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/robot-data-audit)](https://pepy.tech/project/robot-data-audit)
[![Tests](https://github.com/liesliy/rda/actions/workflows/ci.yml/badge.svg)](https://github.com/liesliy/rda/actions/workflows/ci.yml)
[![audit: lerobot/pusht](https://raw.githubusercontent.com/liesliy/rda/main/docs/examples/rda_badge_pusht.svg)](https://liesliy.github.io/rda/examples/rda_report_pusht.html)

> **机器人数据的独立质量评估。** 本地运行——数据不出你的机器。
>
> RDA 是诊断工具，不保证训练成功率提升。

RDA 审计机器人操作数据集（LeRobot 格式）的完整性、时序一致性、运动质量和分布覆盖，再按目标模型架构给出优化建议。适合在验收供应商数据、开训、或发布 benchmark 之前做一次独立检查。

<p align="center">
  <img width="2549" alt="RDA 审计总览" src="https://github.com/user-attachments/assets/35173b6a-275c-466c-985f-9674f7a59a1d" />
</p>

## 安装

```bash
pip install robot-data-audit
```

`[lerobot]` 提供 `.parquet` 数据集加载，`[ui]` 提供网页面板。

## 快速开始

```bash
# 1. 审计数据集 —— 13 个指标，三档判定（PASS / REVIEW / EXCLUDE）
rda audit /path/to/lerobot/dataset

# 2. 按目标模型类型给出优化建议
rda recommend /path/to/dataset --policy temporal   # 或 frame-wise

# 3. 可选的网页面板
rda ui
```

`rda audit` 完全离线。`rda recommend` 在本地完成全部指标计算，只把聚合统计（<1KB）发给规则 API——结果本地缓存可离线复用，私有部署可用 `RDA_API_URL` 指向自己的服务器。

<p align="center">
  <img width="2549" alt="RDA 建议页" src="https://github.com/user-attachments/assets/e366cdf1-b169-46b0-b020-5327493a9124" />
</p>

## 为什么是 RDA

**12 个本地数据集、4,959 集、同一套默认阈值、零调参**——完整表格见 [docs/benchmark.md](docs/benchmark.md)。

**盲测**——我们往 `lerobot/pusht` 里注入了 50 个缺陷集（5 类缺陷，seed=42），另留 156 集作对照。宽口径下 50 个全部命中，严格口径下 precision **1.000**（对照组零误伤）。**[查看盲测报告 →](https://liesliy.github.io/rda/examples/rda_report_pusht.html)**

每次审计都可以渲染成可分享的单文件 HTML 报告和 README 徽章：

```bash
python tools/rda_render.py rda_report.json --html report.html --badge badge.svg
```

更多：[CLI 参考与指标表](docs/cli.md) · [实验记录](experiments/) · [真实使用反馈表](https://github.com/liesliy/rda/issues/new?template=real-world-feedback.yml)

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
