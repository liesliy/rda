# CLI Reference & Metrics

## `rda audit`

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

```bash
# JSON to stdout
rda audit /path/to/dataset --format json

# Blind report for external sharing (paths are hashed)
rda audit /path/to/dataset --blind --format json

# Save report to custom path
rda audit /path/to/dataset -o /tmp/my_report.json
```

## `rda recommend`

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

**Privacy (since v0.5.0):** all metrics are computed **locally**; only aggregated statistics (<1KB, no raw episode data) are sent to the RDA rules API for evaluation. Results are cached locally for offline reuse. For private/air-gapped deployments, point `RDA_API_URL` at your own server.

## `rda ui`

```bash
pip install robot-data-audit[ui]
rda ui
```

Streamlit dashboard at `http://localhost:8501` — upload a dataset, run the audit with live progress, explore per-episode results, export reports. Bilingual (EN/中文) since v0.5.2.

## `rda example`

Show example usage and sample dataset paths.

## Report rendering (`tools/rda_render.py`)

```bash
python tools/rda_render.py rda_report.json --html report.html --badge badge.svg [--title "dataset · context"]
```

Renders any audit JSON report into a self-contained single-file HTML report (verdict cards, three-layer metric table, per-episode verdict strip) and a shields-style SVG badge (red if any EXCLUDE, amber if any REVIEW, green when all PASS). Zero dependencies.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Completed, no EXCLUDE verdicts |
| `1` | Error (invalid path, load failure, etc.) |
| `2` | Completed, at least one EXCLUDE verdict |

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

## Understanding Recommendations

RDA recommendations follow a conservative, evidence-graded approach:

| Confidence | Meaning |
|-----------|---------|
| **HIGH** | Well-supported by optimization experiments; low risk |
| **EXPERIMENTAL** | Directionally consistent but not yet validated for your setup |
| **NOT_RECOMMENDED** | Likely harmful for your model type; proceed with caution |

All suggestions are hypotheses, not guarantees. Effects vary by task domain and model architecture — always validate on a held-out set before applying to training data. Temporal models (ACT, Diffusion Policy) are generally more sensitive to data trimming.
