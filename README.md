# Robot Data Audit (RDA)

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Audit robot dataset quality. Local only.**

RDA checks LeRobot-format datasets for data integrity, temporal consistency, and motion anomalies. It flags problematic episodes and reports what's wrong. RDA is a **diagnostic tool only** — it does not guarantee training success rate improvements.

---

## Quick Start

```bash
pip install robot-data-audit           # install
rda audit /path/to/dataset             # audit → text + JSON report in <dataset>/rda_report.json
```

---

## Sample Output

```
$ rda audit ~/datasets/my_robot_data

  ── Verdict ──
  PASS:    180 (90.0%)
  REVIEW:   15 ( 7.5%)
  EXCLUDE:   5 ( 2.5%)

  ── Top Issues ──
  1. [HIGH ★] Action discontinuity: 3421 spikes across 195 episodes
  2. [MEDIUM] High idle ratio: median 72% idle, 28% effective motion
  3. [LOW] Extreme acceleration spikes: 2847 across 180 episodes
```

Output options: `--format json` for scripting, `-o FILE` to save elsewhere.

---

## Real Audit Results

Audited on [ArmnetBench](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101) (SO-101 arm, 2,499 episodes) and [DROID](https://droid-dataset.github.io/) (100 episodes):

| Metric | ArmnetBench (200 ep subset) | DROID (100 ep) |
|--------|---------------------------|----------------|
| Action spikes detected | 5,229 | 1,428 |
| Median idle ratio | 68.6% | 70.7% |
| Median episode duration | 22.4s | 15.0s |
| Verdict | All PASS | All PASS |

Both are curated benchmark datasets — all episodes pass. RDA also runs on noisier, real-world collections where REVIEW/EXCLUDE verdicts appear more frequently.

Full calibration analysis (ArmnetBench, comparing successful vs failure episodes by label): [`docs/ARMNETBENCH_CALIBRATION_REPORT.md`](docs/ARMNETBENCH_CALIBRATION_REPORT.md)

---

## Design

- **Local only** — No data leaves your machine. Runs entirely offline.
- **Diagnostic, not predictive** — RDA identifies data issues. Whether fixing them improves training is a separate question and depends on your task, model, and setup.
- **Statistical anomaly detection** — Uses MAD on reference distributions instead of fixed thresholds (no hardcoded 3σ rules). Adapts to each dataset's characteristics.
- **Universal core metrics** — Primary ranking uses 3 platform-independent metrics (duration, spike_count, effective_motion_ratio). Platform-specific signals (velocity, path_length) are optional diagnostics.

See [`docs/MVP_PRODUCT_SPEC.md`](docs/MVP_PRODUCT_SPEC.md) for full metric definitions.

---

## Metrics

| Tier | Metric | Detects | Cross-platform? |
|------|--------|---------|:---:|
| L1 | Timestamp monotonicity | Clock resets, duplicates | ✅ |
| L1 | Frame interval consistency | Irregular sampling | ✅ |
| L1 | Schema compliance | Missing/extra fields | ✅ |
| L1 | Numeric sanity | Values exceeding loose physical bounds (angle/force/torque/velocity) | ✅ |
| L2 | Temporal gap detection | Time discontinuities | ✅ |
| L2 | Sensor synchronization | Multi-sensor drift | ⚠️ |
| L2 | Temporal sufficiency | Idle vs active structure | ✅ |
| L3 | Velocity spikes | Implausible jumps | ️ |
| L3 | Motion discontinuities | Jerky trajectories | ⚠️ |
| L3 | Idle frame detection | Paused segments | ✅ |
| L4 | Duration outliers | Too short / too long | ✅ |
| L4 | Spike count outliers | Unusual jerk profiles | ✅ |
| L4 | Effective motion ratio | Low-activity episodes | ✅ |
| SP | Privacy/PII exposure scan (`pii_scan`) | Identifiers in text surfaces, precise geolocation features, faces in sampled frames (`--pii-video`) | ✅ |

> `pii_scan` (v0.9.22+) covers YD/T 6771-2026's **安全性** dimension — the only
> P0 admission gate among the eight quality dimensions. It reports **evidence a
> human must review**, not compliance: WARNING-level by default (never changes
> the verdict), promoted to a gate only with `--pii-strict`. No OCR (licence
> plates, badges), no voice biometrics, no consent review. Findings carry a
> masked snippet plus a SHA-256 fingerprint — raw identifiers never reach the
> report.

---

## CLI Reference

### `rda audit`

```bash
rda audit /path/to/dataset [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Save JSON report (default: `<path>/rda_report.json`) |
| `--format [json\|text]` | Output format (default: `text`) |
| `--platform TEXT` | Robot platform name for Tier 3 normalization |
| `--blind` | Redact identifying paths and operator-authored free text for externally shareable reports |
| `--pii-video` | Extend the privacy scan to sampled video frames (needs `[pii]` extra) |
| `--pii-strict` | Treat privacy findings as an acceptance gate instead of a warning |
| `--continue-on-unverifiable` | Keep exit 0 when episodes are UNVERIFIABLE instead of failing; records `verifiability_override: true` in the report |
| `-v, --verbose` | Verbose output |

Exit codes: `0` = no EXCLUDE, `1` = error, `2` = at least one EXCLUDE,
`3` = audit completed but **nothing was effectively audited** (zero episodes, or
every episode lacked auditable content) — never report this as a pass,
`4` = at least one episode was **UNVERIFIABLE**: a critical check applies to it
but its input is absent from this copy (typically videos never downloaded).
Absence is not a pass — re-run on a complete copy, or pass
`--continue-on-unverifiable` to record the override and keep exit 0.

---

## Python API

```python
from rda.audit.dataset_audit import DatasetAuditor
from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset

dataset_info = load_lerobot_dataset("/path/to/dataset")
auditor = DatasetAuditor()
result = auditor.audit_dataset(dataset_info, iter_episodes("/path/to/dataset"))
print(f"PASS: {result.verdict_counts['PASS']}")
```

---

## Development

```bash
git clone https://github.com/liesliy/rda.git
cd rda
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
@software{robot_data_audit,
  title     = {Robot Data Audit: Quality Auditing for Robot Manipulation Datasets},
  author    = {Niu Su Tech},
  year      = {2026},
  url       = {https://github.com/liesliy/rda}
}
```

## License

MIT
