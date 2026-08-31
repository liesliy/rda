# Robot Data Audit (RDA)

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/robot-data-audit)](https://pepy.tech/project/robot-data-audit)
[![Tests](https://github.com/liesliy/rda/actions/workflows/ci.yml/badge.svg)](https://github.com/liesliy/rda/actions/workflows/ci.yml)
[![audit: lerobot/pusht](https://raw.githubusercontent.com/liesliy/rda/main/docs/examples/rda_badge_pusht.svg)](https://liesliy.github.io/rda/examples/rda_report_pusht.html)

> **Independent quality assessment for robot data.** Runs locally — your data never leaves your machine.
>
> RDA is a diagnostic tool. It does not guarantee training success-rate improvements.

RDA audits robot manipulation datasets (LeRobot format) for integrity, temporal consistency, motion quality and distribution coverage, then generates optimization recommendations calibrated to your target model architecture. Use it as an independent check before you accept a vendor dataset, train a policy, or publish a benchmark.

<p align="center">
  <img width="2549" alt="RDA audit overview" src="https://github.com/user-attachments/assets/35173b6a-275c-466c-985f-9674f7a59a1d" />
</p>

## Install

```bash
pip install robot-data-audit
```

`[lerobot]` adds `.parquet` dataset loading, `[ui]` adds the web dashboard.

## Quick start

```bash
# 1. Audit a dataset — 13 metrics, three-tier verdicts (PASS / REVIEW / EXCLUDE)
rda audit /path/to/lerobot/dataset

# 2. Recommendations calibrated to your model type
rda recommend /path/to/dataset --policy temporal   # or frame-wise

# 3. Optional web dashboard
rda ui
```

`rda audit` is fully offline. `rda recommend` computes all metrics locally and sends only aggregated statistics (<1KB) to the rules API — cached for offline reuse, and `RDA_API_URL` can point to your own server for private deployments.

<p align="center">
  <img width="2549" alt="RDA recommendations" src="https://github.com/user-attachments/assets/e366cdf1-b169-46b0-b020-5327493a9124" />
</p>

## Why RDA

**12 local datasets, 4,959 episodes, one set of default thresholds, zero per-dataset tuning** — full table in [docs/benchmark.md](docs/benchmark.md).

**Blind test** — we injected 50 defective episodes (5 defect classes, seed=42) into `lerobot/pusht` and kept 156 as controls. RDA caught all 50 under the broad criterion, precision **1.000** (zero false alarms on controls) under the strict one. **[Read the blind-test report →](https://liesliy.github.io/rda/examples/rda_report_pusht.html)**

Every audit can also render into a shareable single-file HTML report and a README badge:

```bash
python tools/rda_render.py rda_report.json --html report.html --badge badge.svg
```

More: [CLI reference & metrics table](docs/benchmark.md) · [experiments](experiments/) · [real-world feedback form](https://github.com/liesliy/rda/issues/new?template=real-world-feedback.yml)

## Citation

```bibtex
@software{robot_data_audit,
  title = {Robot Data Audit: Quality Auditing for Robot Manipulation Datasets},
  author = {Niu Su Tech},
  year = {2026},
  url = {https://github.com/liesliy/rda}
}
```

MIT License.
