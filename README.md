# Robot Data Audit (RDA)

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/robot-data-audit)](https://pepy.tech/project/robot-data-audit)
[![Tests](https://github.com/liesliy/rda/actions/workflows/ci.yml/badge.svg)](https://github.com/liesliy/rda/actions/workflows/ci.yml)

> **Quality auditing + optimization recommendations for robot datasets.**
> Diagnose data quality issues. Get actionable, confidence-graded suggestions.
> RDA is a **diagnostic tool** — it does NOT guarantee training success rate improvements.

RDA audits robot manipulation datasets (LeRobot format) for integrity, temporal consistency, motion quality, and distribution coverage. It then generates optimization recommendations calibrated to your target model architecture.

<img width="2549" height="1403" alt="rda-firstpage" src="https://github.com/user-attachments/assets/35173b6a-275c-466c-985f-9674f7a59a1d" />

<img width="2549" height="1403" alt="RDA_RECOMMAND" src="https://github.com/user-attachments/assets/e366cdf1-b169-46b0-b020-5327493a9124" />

## Features

- **13 quality metrics** across 3 tiers: integrity, temporal, motion, and distribution
- **`rda recommend`** — Data optimization suggestions calibrated to your model type
  - Frame-wise models (MLP/BC): mild idle trimming suggestions
  - Temporal models (ACT/DP/Transformer): conservative "do not prune" guidance
  - All suggestions include confidence levels (HIGH / EXPERIMENTAL / NOT_RECOMMENDED)
- **LeRobot v2.1 + v3.0** dual-format auto-detection
- **Three-tier verdicts**: PASS / REVIEW / EXCLUDE
- **Evidence-aware reporting**: HARD_FAIL / RISK_SIGNAL / UNVERIFIABLE are shown separately; statistical signals are not presented as confirmed corruption
- **Blind-validation support**: generate annotator-safe samples and compare against external QC labels without exposing RDA verdicts before review
- **CLI-first design**: JSON + text output, pipe-friendly
- **Temporal sufficiency analysis**: idle detection, active run distribution, valid window ratios

## Tested on public data

RDA has been run over **12 local LeRobot-format datasets — 4,959 episodes** — with the same default thresholds everywhere, zero per-dataset tuning. A few things it found:

- One xArm dataset is genuinely clean (767/800 episodes PASS, 20.8% median idle); another from the *same robot platform* runs 83.3% idle — task difficulty, not collection sloppiness
- A community SO-100 dataset has action spikes in **100% of episodes** and 86.7% median idle
- A local LIBERO copy had 773 episodes missing from the downloaded data files. RDA 0.5.3 fixes a LeRobot v3.0 metadata/data-file mapping issue so missing local files are distinguished from episodes that are actually present

Full table, per-dataset numbers, and the five recurring patterns: **[docs/benchmark.md](docs/benchmark.md)**

Methodology and other experiment write-ups (i18n smoke test, server deploy
verify, the spike/verdict bug regression pin, wheel leak guard):
**[experiments/](experiments/)**

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

## Share real-world feedback

If you run RDA on a real workflow, dataset, or existing QC process, please share
what you observed through the [real-world feedback form](https://github.com/liesliy/rda/issues/new?template=real-world-feedback.yml).
You do not need to upload data. Describe the version, format, command, and
observable result; an independent QC result or human review is especially useful.
Please remove raw data, private paths, customer information, and credentials.
A public issue is not a substitute for confidential disclosure.

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

# Blind report for external sharing (paths are hashed)
rda audit /path/to/dataset --blind --format json

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
| `--blind` | Redact identifying paths for externally shareable reports |

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

### Testing the gate itself

The suite in `tests/` exists for a reason worth explaining. During the
0.4.x era, a bug let anomalous episodes walk away with a PASS badge: the
behavior layer correctly computed action spikes and idle ratios, but the
verdict aggregator ignored those signals entirely. The gate wasn't
consuming its own evidence.

`tests/test_negative_control.py` pins that exact failure mode: metric
results that pass every rule but contain known anomalies (150 spikes, a
frozen arm) **must** come back REVIEW — and hard corruption must stay
EXCLUDE (the gate can't fail open in either direction). The rest of the
suite covers the i18n catalog (zh/en key alignment) and boundary
behavior of the two core behavioral metrics.

CI runs the tests on every push and additionally verifies that the
closed-source recommendation layer never leaks into the published wheel.

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
