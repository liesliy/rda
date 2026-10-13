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
| `--video-quality` | Include visual quality analysis (default: Fast Audit skips it) |
| `--no-video` | Skip all video-related metrics (9 metrics) |
| `--video-only` | Run only video-related metrics |
| `--full` | Full audit — all metrics including visual quality |
| `-v, --verbose` | Verbose output |
| `--blind` | Redact identifying paths **and all operator-authored free text** (task descriptions become stable hash stubs) for externally shareable reports |
| `--pii-video` | Extend the privacy scan to sampled video frames (face detection; needs `pip install robot-data-audit[pii]`) |
| `--pii-strict` | Treat privacy findings as an acceptance gate instead of a warning (see below) |

> **Execution Tiers (v0.9.4+):** By default RDA runs in **Fast Audit** mode, which
> skips `visual_quality` (requires frame decoding). Use `--video-quality` or `--full`
> to include it. The four flags above are mutually exclusive.

```bash
# JSON to stdout
rda audit /path/to/dataset --format json

# Blind report for external sharing (paths are hashed)
rda audit /path/to/dataset --blind --format json

# Save report to custom path
rda audit /path/to/dataset -o /tmp/my_report.json

# Include visual quality analysis
rda audit /path/to/dataset --video-quality

# Full audit (all metrics including visual quality)
rda audit /path/to/dataset --full

# Skip all video metrics (fastest for non-video datasets)
rda audit /path/to/dataset --no-video

# Privacy / PII exposure scan (v0.9.22+)
rda audit /path/to/dataset                       # text surfaces only, WARNING-level
rda audit /path/to/dataset --pii-video           # + faces in sampled video frames
rda audit /path/to/dataset --pii-strict          # promote findings to EXCLUDE
```

## Privacy / PII scan (v0.9.22+)

`pii_scan` covers YD/T 6771-2026's **安全性** dimension — the only P0 admission
gate among the eight quality dimensions. It detects **objective, reproducible
evidence of exposure** and nothing more:

| Surface | Detected |
|---|---|
| Text (episode meta, task text, feature names) | Chinese mobile numbers, Chinese resident ID numbers (checksum-validated per GB 11643-1999), emails, MAC addresses, public IPv4 addresses (RFC1918/loopback/link-local are classified as *not* exposure) |
| Numeric features | Precise lat/lon recorded as a *pair* (≥95% of samples inside valid range) |
| Video frames (`--pii-video`) | Human faces via Haar cascade, ≥2 hit frames to report |

**What it does NOT claim:** faces beyond the sampled frames, OCR (licence plates,
door numbers, badges), voice biometrics, consent/licensing review, or legal
compliance. Findings are **items a human must review**, not an acceptance verdict.

### Verdict semantics

Findings are **WARNING-level by default** — `has_finding=True` while
`assessment.status="pass"`, so the verdict is untouched. A false-positive face
detection must never silently discard usable demonstrations. `--pii-strict`
promotes `pii_scan` into the critical set **for that run only**, so findings then
drive EXCLUDE. The gate is the operator's decision, never the detector's.

### The report is not a leak channel

Identifiers are never written raw. Every finding carries a masked snippet plus a
SHA-256 fingerprint (first 12 hex chars) for cross-episode correlation, and every
free-text field the report echoes back (per-episode `task_description`, the
dataset-level `tasks` map) goes through the same detector table first:

```
"operator reachable at 13812345678"
  → "operator reachable at [REDACTED:cn_mobile]"
```

`task_description_redacted` marks that a field was rewritten; text beyond the
100k-character scan budget is dropped with an explicit marker rather than echoed
un-scanned. `--blind` goes further and replaces every task string with a stable
hash stub, because no regex recognises arbitrary names or home addresses.


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
| `3` | Completed, but **no effective audit** (v0.9.21+) — zero episodes, or every episode had no auditable content. Checked before `2`, so "the audit never ran" cannot be masked by "the data looks bad" |

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
