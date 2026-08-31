# Blind Test: RDA 0.5.4 on Controlled-Defect pusht

**Date**: 2026-08-31 · **RDA version**: 0.5.4 (PyPI) · **Dataset**: [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) (LeRobot v3.0, 206 episodes, 25,650 frames) · **Seed**: 42

## Why this experiment

Every number in our [benchmark](benchmark.md) is an *observation* on datasets
whose ground truth we don't control. To claim detection ability, we need the
opposite: a dataset whose ground truth we **manufacture**. So we took a
well-known public dataset, injected 50 defective episodes with known
locations, shuffled nothing else, and asked RDA to find them — default
thresholds, zero tuning.

## Method

1. Downloaded `lerobot/pusht` verbatim (all 16 files, 7.7 MB) and verified
   integrity: 25,650 frames = sum of episode lengths = video frame count.
2. **Baseline audit** of the untouched copy: 43 PASS / 163 REVIEW / 0 EXCLUDE.
   (pusht is a low-motion task; the REVIEWs come from `idle_ratio` — see
   [benchmark.md](benchmark.md). No integrity-layer findings.)
3. Injected 5 defect classes × 10 episodes each (seed=42), leaving the other
   **156 episodes untouched as controls**:

| Class | Operation | Expected verdict |
|---|---|---|
| `empty` | all rows deleted from data parquet; `meta/episodes` left stale (length > 0) | EXCLUDE (regression probe: the v0.4.12 zero-frame silent-PASS bug) |
| `nan_state` | 15 random frames per episode, `observation.state` set to NaN (both dims, 300 cells) | EXCLUDE via `invalid_values` |
| `timestamp_reverse` | second half of the episode's timestamps reversed → 692 negative deltas total | EXCLUDE via `timestamp_validity` |
| `frozen` | entire episode `observation.state` + `action` frozen at first-frame values | REVIEW via `idle_ratio` (effective motion 0%) |
| `duplicate_frames` | 5 random frames per episode duplicated once, copies appended at episode end → 2–4 negative deltas per episode | honest probe; detected via `timestamp_validity` only because duplicates land at the episode tail |

4. Ran `rda audit` (0.5.4, defaults, no tuning) and compared every verdict
   against the manifest.

## Results

**Confusion matrix — strict criterion (EXCLUDE = flagged):**

| | flagged | not flagged |
|---|---:|---:|
| **defective (50)** | TP = 40 | FN = 10 |
| **control (156)** | FP = 0 | TN = 156 |

- **Precision 1.000 (40/40) · Recall 0.800 (40/50)**
- **0 false positives on 156 untouched controls** — every control episode's
  verdict is identical to its clean-baseline verdict.
- **Broad criterion (EXCLUDE + REVIEW = flagged): TP = 50, FN = 0.**

Per class (strict / broad):

| Class | EXCLUDE | REVIEW | Caught (strict) | Caught (broad) | Detector |
|---|---:|---:|---:|---:|---|
| empty | 10/10 | 0 | 10/10 | 10/10 | `_zero_frame_guard` (v0.4.12 P0 fix, regression-confirmed) |
| nan_state | 10/10 | 0 | 10/10 | 10/10 | `invalid_values` (30 NaN/episode) |
| timestamp_reverse | 10/10 | 0 | 10/10 | 10/10 | `timestamp_validity` (non-monotonic) |
| frozen | 0 | 10/10 | 0/10 | 10/10 | `idle_ratio` — effective motion 0.0%, RISK_SIGNAL → REVIEW |
| duplicate_frames | 10/10 | 0 | 10/10 | 10/10 | `timestamp_validity` (2–4 negative deltas/episode) |

Runtime: **2 seconds** for the full 206-episode / 24,488-frame audit on a
consumer laptop (Windows, Python 3.13).

## Honest caveats

- **Frozen episodes are caught as REVIEW, not EXCLUDE.** RDA treats
  statistical anomalies as review signals, not hard failures — by design.
  If you need frozen arms to hard-fail, that's a one-line policy change and
  we'd rather discuss it in an issue than hardcode it.
- **`duplicate_frames` detection is an artifact of placement.** Our copies
  landed at the episode tail, creating negative deltas. A mid-stream
  duplicate with monotone timestamps would *not* be flagged by 0.5.4.
  We publish this precisely because the class name sounds more alarming
  than what was measured.
- **pusht is an unusually easy payload**: 2-D state, 96×96 video, no
  multi-stream timestamps. `sensor_synchronization` and `joint_limit`
  returned N/A throughout — they are not exercised by this dataset.
- One control-class observation worth knowing: 121/156 clean controls get
  REVIEW from `idle_ratio` because pusht is a low-motion task (median idle
  82.1%). This is exactly why the benchmark reports idle distributions
  instead of bare pass rates.

## Reproduce

The injection script and ground-truth manifest are reproducible from the
seed; report JSONs are preserved locally (baseline + blind test, 6.7 MB).
The injection tool will be released as `rda blindtest` in a future version —
until then, the manifest schema in this document is the reference.

```bash
pip install robot-data-audit==0.5.4
# download lerobot/pusht (v3.0), then apply the 5x10 injection (seed=42)
rda audit <blindtest_dataset> --format json
```
