# Robot Data Audit (RDA)

[![PyPI](https://img.shields.io/pypi/v/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![Python](https://img.shields.io/pypi/pyversions/robot-data-audit)](https://pypi.org/project/robot-data-audit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/robot-data-audit)](https://pepy.tech/project/robot-data-audit)
[![Tests](https://github.com/liesliy/rda/actions/workflows/ci.yml/badge.svg)](https://github.com/liesliy/rda/actions/workflows/ci.yml)

[![audit: lerobot/pusht](https://raw.githubusercontent.com/liesliy/rda/main/docs/examples/rda_badge_pusht.svg)](https://liesliy.github.io/rda/examples/rda_report_pusht.html) [![audit: AgiBotWorld2026 RL](https://raw.githubusercontent.com/liesliy/rda/main/docs/examples/rda_badge_agibot_rl.svg)](https://liesliy.github.io/rda/examples/rda_report_agibot_rl_hgdagger.html)

> **Independent quality assessment for robot data.** Runs locally — your data never leaves your machine.
>
> RDA is a diagnostic tool. It does not guarantee training success-rate improvements.

RDA audits robot manipulation datasets (LeRobot format) and reports a
three-tier verdict per episode — **PASS / REVIEW / EXCLUDE** — together with
measured diagnostics. Use it as an independent check before you accept a
vendor dataset, train a policy, or publish a benchmark.

**Current release: v0.9.2** — `pip install robot-data-audit`.

## The four-layer audit

RDA runs every episode through four sequential layers. The key design rule:
**only hard integrity checks can flip an episode to EXCLUDE; diagnostic
measurements never do.** This separates "this data is broken" from "this data
looks unusual", so observational signals never masquerade as fatal defects.

| Layer | Role | # Metrics | Can set verdict? |
|---|---|---|---|
| **L1 — Integrity Gate** | Deterministic hard checks (missing / NaN / limit / video-stream) | 9 | ✅ PASS → EXCLUDE |
| **L2 — Trajectory Diagnostics** | Observational motion & video anomalies | 8 | ❌ findings only |
| **L3 — Dataset Profile** | Training-data efficiency & coverage | 4 | ❌ findings only |
| **L4 — Dataset Summary** | Dataset-level P10/P50/P90 aggregation | — | 📊 report only |

An episode is **EXCLUDE** only if an L1 hard check fails; **REVIEW/PASS** are
driven by L1 state, while L2/L3 surface measurements and findings for the
human reviewer. RDA measures and presents — the accept/reject decision stays
with you.

## Install

```bash
pip install robot-data-audit
```

Optional dependency tiers:

| Tier | Extra | Unlocks |
|---|---|---|
| Core (default) | — | parquet audits: integrity + temporal/motion + dataset-utility metrics |
| Visual | `pip install robot-data-audit[video]` or `pip install av` | the video visual metrics (freeze / timestamp-alignment / stream-span/offset/drift / quality) |
| Lerobot | `[lerobot]` | `.parquet` dataset loading via the lerobot package |
| UI | `[ui]` | the web dashboard |
| Everything | `[all]` | all of the above |

**Visual metrics without PyAV are reported as "not audited", never as
"pass"**: the JSON report carries a top-level `skipped_by_missing_dep` field
and the CLI prints a warning listing the skipped checks. The web dashboard
shows the same guarantee — a dep-missing dataset renders a "not audited ≠
pass" banner.

## Quick start

```bash
# 1. Audit a dataset — 21 metrics across four layers, three-tier verdicts
rda audit /path/to/lerobot/dataset

# 2. Recommendations calibrated to your model type
rda recommend /path/to/dataset --policy temporal   # or frame-wise

# 3. Optional web dashboard
rda ui
```

`rda audit` is fully offline and emits a structured JSON report: per-episode
verdicts, every metric's measurement/findings, plus a dataset-level
`acceptance_summary` (P10/P50/P90 baselines, runtime outliers, and the
not-checked inventory). `rda recommend` computes all metrics locally and
sends only aggregated statistics (<1 KB) to the rules API — cached for
offline reuse, and `RDA_API_URL` can point to your own server for private
deployments.

### Programmatic use

```python
import numpy as np
from rda.io.schema import EpisodeData
from rda.audit.episode_audit import EpisodeAuditor

episode = EpisodeData(
    episode_index=0,
    num_frames=n_frames,
    timestamps=np.array(timestamps),          # seconds
    observation={"state": state_array},      # shape (T, DoF)
    action={"joint_pos": action_array},      # shape (T, DoF)
    meta={"fps": 10, "source": "my/dataset"},
)
result = EpisodeAuditor().audit(episode)
print(result.verdict)                        # PASS / REVIEW / EXCLUDE
for name, metric in result.metrics.items():
    if metric.has_finding:
        print(name, metric.measurement)
```

## The 21 metrics

**L1 — Integrity Gate (hard checks, can EXCLUDE)**
`missing_dropout` · `invalid_values` (NaN/Inf) · `schema_consistency` ·
`temporal_validity` · `joint_limit` (three-level PASS/REVIEW/EXCLUDE with
configurable `approach_threshold` / `consecutive_frames` / `jump_multiplier`)
· `video_frame_integrity` · `video_freeze` · `video_timestamp_alignment` ·
`video_stream_presence`

**L2 — Trajectory Diagnostics (observational, never EXCLUDE)**
`sensor_sync` · `sampling_jitter` · `velocity_acceleration` ·
`action_discontinuity` (MAD-based spike detection) · `visual_quality` ·
`video_stream_span_consistency` · `video_stream_temporal_offset` ·
`video_stream_temporal_drift`

**L3 — Dataset Profile (efficiency & coverage)**
`idle_ratio` (three-tier fallback: 30-bin valley → 3×MAD → 1e-6 floor) ·
`distribution` · `coverage` · `temporal_structure`

Metrics are also classified by **cross-platform portability** (MVP Spec
v0.2.0 §1.5): Tier-1 universal (`duration_sec`, `spike_count`,
`effective_motion_ratio` — comparable across any robot), Tier-2 normalizable
(velocity / acceleration / jerk / path-length — need platform scaling),
Tier-3 platform-specific (joint limits, workspace, torque/force/tactile).

## Validated on real datasets

**12 local datasets, 4,959 episodes, one set of default thresholds, zero
per-dataset tuning** — full table in [docs/benchmark.md](docs/benchmark.md).

**Four popular LeRobot datasets audited (v0.9.2)** — we ran RDA against
`lerobot/pusht`, `aloha_sim_transfer_cube_human`, `xarm_lift_medium` and
`droid_100` (1,156 episodes across a 2-DOF sim, a 14-DOF bimanual sim, a
4-DOF arm and a 7-DOF Franka). All episodes pass L1 integrity; the dataset
profiles differ sharply — median idle frames range from **20.8%** (xArm) to
**81.7%** (PushT), and the bimanual ALOHA sim shows a median of 30.5
discontinuity spikes per episode. Every figure is reproducible straight from
the PyPI package (`pip install robot-data-audit`).

**Full audit of lerobot/libero_10 (v3.0)** — 379 episodes, 101,469 frames:
all applicable integrity checks clean, 0 hard defects.
**[Read the report →](docs/benchmark_libero10.md)**

**Blind test** — we injected 50 defective episodes (5 defect classes,
seed=42) into `lerobot/pusht` and kept 156 as controls. RDA caught all 50
under the broad criterion, precision **1.000** (zero false alarms on
controls) under the strict one.
**[Read the blind-test report →](https://liesliy.github.io/rda/examples/rda_report_pusht.html)**

**Validated on AgiBotWorld2026** — third-party audit of AgiBot's Phase 3
dataset: all 5 simulation tasks + a real-robot RL package, 1,112 episodes,
4,448 integrity checks with 0 failures, and a **3.1× enrichment** of RDA's
discontinuity spikes at official human-takeover boundaries. Zero adaptation
needed. **[Read the case study →](docs/examples/agibotworld2026.md)**

Every audit can also render into a shareable single-file HTML report and a
README badge:

```bash
python tools/rda_render.py rda_report.json --html report.html --badge badge.svg
```

More: [CLI reference & metrics table](docs/cli.md) · [experiments](experiments/) · [real-world feedback form](https://github.com/liesliy/rda/issues/new?template=real-world-feedback.yml)

## Governance

RDA's metric I/O is pinned by a core-parameter spec, and every change is
guarded by semantic-invariant tests run in CI. Seven invariant guard tests
(INV-003 … INV-009) protect the core architecture — e.g. "L2 diagnostics
never set an EXCLUDE verdict", "no inference is reported as a measurement",
"the report always carries a tool version". The full suite runs on every
push across Python 3.10–3.12.

## Metric provenance

Every metric ships with a four-file provenance record
(`docs/provenance/<metric>/`): **algorithm.md** (how it works),
**source.md** (public precedents consulted — ideas only),
**implementation_origin.md** (original implementation, zero third-party
code), **license.md** (compliance notes). Index:
[docs/provenance/](docs/provenance/).

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
