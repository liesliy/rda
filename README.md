# Robot Data Audit (RDA)

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/robot-data-audit)](https://pepy.tech/project/robot-data-audit)

> **Quality auditing + optimization recommendations for robot datasets.**
> Diagnose data quality issues. Get actionable, confidence-graded suggestions.
> RDA is a **diagnostic tool** — it does NOT guarantee training success rate improvements.

RDA audits robot manipulation datasets (LeRobot format) for integrity, temporal consistency, motion quality, and distribution coverage. It then generates optimization recommendations calibrated to your target model architecture.

![Demo](https://raw.githubusercontent.com/liesliy/rda/main/assets/demo.svg)

## Features

- **13 quality metrics** across 3 tiers: integrity, temporal, motion, and distribution
- **`rda recommend`** — Data optimization suggestions calibrated to your model type
  - Frame-wise models (MLP/BC): mild idle trimming suggestions
  - Temporal models (ACT/DP/Transformer): conservative "do not prune" guidance
  - All suggestions include confidence levels (HIGH / EXPERIMENTAL / NOT_RECOMMENDED)
- **LeRobot v2.1 + v3.0** dual-format auto-detection
- **Three-tier verdicts**: PASS / REVIEW / EXCLUDE
- **CLI-first design**: JSON + text output, pipe-friendly
- **Temporal sufficiency analysis**: idle detection, active run distribution, valid window ratios

## Installation

```bash
pip install robot-data-audit
```

With LeRobot dependency (for `.parquet` dataset loading):

```bash
pip install robot-data-audit[lerobot]
```

With the Streamlit web UI:

```bash
pip install robot-data-audit[ui]
```

## Quick Start

### 1. Audit a dataset

```bash
rda audit /path/to/lerobot/dataset
```

Runs all 13 metrics, prints a text summary. JSON report saved to `<dataset>/rda_report.json`.

### 2. Get optimization recommendations

```bash
# For frame-wise models (MLP, BC, etc.)
rda recommend /path/to/dataset --policy frame-wise

# For temporal models (ACT, Diffusion Policy, Transformer)
rda recommend /path/to/dataset --policy temporal

# JSON output for scripting
rda recommend /path/to/dataset --policy frame-wise --format json

# English output (default is Chinese)
rda recommend /path/to/dataset --policy temporal --lang en
```

**What `recommend` tells you:**
- Whether your dataset has excessive idle frames
- Whether trimming is advisable (and how aggressively)
- Model-specific warnings (e.g., "DO NOT prune for temporal models")
- Confidence levels and experimental caveats for every suggestion

**Privacy (since v0.5.0):** `rda recommend` computes all metrics **locally** and sends
only aggregated statistics (<1KB, no raw episode data) to the RDA rules API
(`https://rda.niusu2026.cn`) for evaluation. Results are cached locally for offline
reuse. `rda audit` remains fully offline. For private/air-gapped deployments, point
`RDA_API_URL` at your own server.

### 3. Launch the web UI

```bash
pip install robot-data-audit[ui]
rda ui
```

Opens a Streamlit dashboard at `http://localhost:8501` — upload a dataset, run the
audit with live progress, explore per-episode results, generate recommendations
(via the same privacy-preserving API path), and export reports.

**Bilingual UI (since v0.5.2):** the entire dashboard — including backend
recommendation copy and exported reports — switches cleanly between
**English** and **中文** via the language selector in the sidebar. No mixed text.


### 4. JSON output & piping

```bash
# JSON to stdout
rda audit /path/to/dataset --format json

# Save report to custom path
rda audit /path/to/dataset -o /tmp/my_report.json

# Verbose mode with platform info
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

## CLI Reference

### `rda audit`

```bash
rda audit [OPTIONS] PATH
```

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Save JSON report (default: `<path>/rda_report.json`) |
| `--format [json\|text]` | Output format (default: `text`) |
| `--platform TEXT` | Robot platform (e.g. `so101`, `droid`) for Tier 3 metrics |
| `-v, --verbose` | Verbose output |

### `rda recommend`

```bash
rda recommend [OPTIONS] PATH
```

| Option | Description |
|--------|-------------|
| `--policy [frame-wise\|temporal]` | Target model architecture type (required) |
| `-o, --output FILE` | Save JSON recommendation report |
| `--format [json\|text]` | Output format (default: `text`) |
| `--lang [zh\|en]` | Language of the recommendation text (default: `zh`) |
| `-v, --verbose` | Verbose output |

### `rda example`

Show example usage and sample dataset paths.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Completed, no EXCLUDE verdicts |
| `1` | Error (invalid path, load failure, etc.) |
| `2` | Completed, at least one EXCLUDE verdict |

## Understanding Recommendations

RDA recommendations follow a **conservative, evidence-graded** approach:

| Confidence | Meaning |
|-----------|---------|
| **HIGH** | Well-supported by optimization experiments; low risk |
| **EXPERIMENTAL** | Directionally consistent but not yet validated for your setup |
| **NOT_RECOMMENDED** | Likely harmful for your model type; proceed with caution |

**Key principles:**
- All suggestions are **hypotheses**, not guarantees
- Effects vary by task domain and model architecture
- Always validate on a held-out set before applying to training data
- Temporal models (ACT, DP) are generally more sensitive to data trimming

## Metrics Overview

| Tier | Metric | What it detects |
|------|--------|----------------|
| L1 | Timestamp monotonicity | Clock resets, duplicate timestamps |
| L1 | Frame interval consistency | Jittery or irregular sampling |
| L1 | Schema compliance | Missing/extra fields, type mismatches |
| L2 | Temporal gap detection | Large time discontinuities |
| L2 | Sensor synchronization | Cross-sensor timestamp drift |
| L2 | **Temporal sufficiency** | Idle/active structure, valid window analysis |
| L3 | Joint limit violations | Actuators driven beyond safe range |
| L3 | Velocity spikes | Sudden implausible jumps |
| L3 | Motion discontinuities | Non-smooth trajectory segments |
| L3 | Idle frame detection | Stationary/paused segments |
| L4 | Duration outliers | Episodes too short/long vs. cohort |
| L4 | Spike count outliers | Episodes with unusual jerk profiles |
| L4 | Effective motion ratio | Low-activity episodes |

## Project Structure

```
rda/
├── cli/          # Click CLI entry points
├── io/           # Data loading and schema definitions (LeRobot v2.1/v3.0)
├── metrics/      # 13 audit metric implementations
├── recommend/    # Optimization recommendation engine
├── audit/        # Dataset and episode-level audit orchestration
└── report/       # Report generation and summary
```

## Development

```bash
git clone https://github.com/liesliy/rda.git
cd robot-data-audit
pip install -e ".[dev]"
pytest
```

## Citation

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
