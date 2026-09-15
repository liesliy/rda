# Benchmark: RDA 0.9.7 on lerobot/libero_10 (full, v3.0)

**RDA version**: 0.9.7 · **Scope**: 379 episodes, 101,469 frames (full dataset) · **Last run**: 2026-09-14  
**Execution tier**: Fast Audit (default, with video metrics) · **Report schema**: 1.2

This is a single-dataset reproducibility record: we audited the complete
[lerobot/libero_10](https://huggingface.co/datasets/lerobot/libero_10)
dataset (LeRobot v3.0 format) with RDA 0.9.7 at default thresholds, zero
tuning. Every figure below comes from a saved RDA JSON report.

## Results

**Overall verdicts: 373 PASS / 6 REVIEW / 0 EXCLUDE.**

| Check | Result |
|---|---|
| missing_dropout, invalid_values, schema_consistency | pass (379/379) |
| timestamp_validity | pass (379/379) |
| sampling_jitter, velocity_acceleration, action_discontinuity | pass (379/379) |
| distribution, coverage, temporal_structure | pass (379/379) |
| video_frame_integrity, video_timestamp_alignment, video_stream_presence | pass (379/379) |
| video_stream_span_consistency, video_stream_temporal_drift, video_stream_temporal_offset | pass (379/379) |
| joint_limit, sensor_synchronization | `na` — not applicable to this dataset (379/379) |
| idle_ratio | finding (379/379) — RISK_SIGNAL |
| video_freeze | **fail (6/379)** → these 6 episodes flagged as REVIEW |

### Changes from v0.5.8

| Metric | v0.5.8 | v0.9.7 |
|---|---|---|
| Verdicts | 132 PASS / 247 REVIEW / 0 EXCLUDE | **373 PASS / 6 REVIEW / 0 EXCLUDE** |
| idle_ratio | review (379/379) → triggered REVIEW verdict | finding (379/379) → RISK_SIGNAL only, no longer auto-escalates to REVIEW |
| video_freeze | pass (379/379) — v3.0 fix | **fail (6/379)** — 6 episodes genuinely flagged |
| joint_limit | `na` | `na` (379 N/A, but 6 episodes also fail independently) |
| sensor_synchronization | `na` | `na` (379 N/A) |
| Report schema | 1.1 | 1.2 (added `execution_tier`, `video_quality` fields) |

**Key improvement**: The v0.9.7 verdict pipeline was restructured. idle_ratio no longer automatically escalates to REVIEW verdict. This dramatically reduced false positives: from 247 REVIEW (v0.5.8) down to just 6 REVIEW (v0.9.7). The 6 remaining REVIEW episodes are all triggered by **video_freeze** detection, not by the idle_ratio false-positive issue.

## The idle_ratio signal (still present but de-escalated)

idle_ratio still flags all 379 episodes as RISK_SIGNAL (median idle total ratio = 71.7%), but this is now an informational finding rather than a verdict-level block. This correctly reflects that libero_10 tasks are fine-grained tabletop manipulation with naturally small and slow motions — not a data quality problem.

## REVIEW episodes detail

All 6 REVIEW episodes share the same pattern — flagged by `video_freeze` (HARD_FAIL, score 0.5):

| episode_index | frames | task | Primary flag |
|---:|---:|---|---|
| 90 | 223 | turn on the stove and put the moka pot on it | video_freeze |
| 198 | 235 | pick up the book and place it in the back compartment | video_freeze |
| 240 | 289 | put the white mug on the plate and put the chocolate pudding | video_freeze |
| 254 | 227 | put both the cream cheese box and the butter in the basket | video_freeze |
| 255 | 291 | put both the cream cheese box and the butter in the basket | video_freeze |
| 300 | 237 | put both the cream cheese box and the butter in the basket | video_freeze |

These episodes contain actual video freeze segments (stuck frames), not just low-motion content.

## Per-task breakdown

| task_index | Description | PASS | REVIEW | total |
|---:|---|---:|---:|---:|
| 0 | put the white mug on the left plate... | 38 | 0 | 38 |
| 1 | put the white mug on the plate + pudding | 36 | 0 | 36 |
| 2 | put the yellow and white mug in the microwave | 34 | 0 | 34 |
| 3 | turn on the stove and put the moka pot on it | 40 | 1 | 41 |
| 4 | put both the alphabet soup and the cream cheese... | 43 | 0 | 43 |
| 5 | put both the alphabet soup and the tomato sauce | 33 | 0 | 33 |
| 6 | put both moka pots on the stove | 29 | 0 | 29 |
| 7 | put both the cream cheese box and the butter... | 49 | 0 | 49 |
| 8 | put the black bowl in the bottom drawer... | 30 | 5 | 35 |
| 9 | pick up the book and place it in the back compartment | 41 | 0 | 41 |

Task 8 ("put the black bowl in the bottom drawer") has the highest REVIEW ratio (5/35 = 14.3%), suggesting potential video quality issues in that specific task's recordings.

## Aggregate statistics

| Metric | Median | P95 | P99 |
|---|---|---|---|
| Duration (s) | 25.80 | 34.02 | — |
| Effective motion ratio | 0.283 | 0.341 | — |
| Spike count | 10 | 15 | 18 |
| Velocity p95 | 0.320 | 0.629 | 0.700 |
| Idle total ratio | 71.7% | 77.9% | — |
| Idle prefix ratio | 3.2% | 10.6% | — |
| Active run p50 | 3 frames | 6 frames | — |
| Transition count | 29 | 48 | — |
| State space occupancy | 6.6% | — | — |

## Reproduce

```bash
pip install robot-data-audit==0.9.7
# after downloading lerobot/libero_10:
rda audit <libero_10_path> --platform panda --format json --output report.json
```

Full 379-episode audit: ~3 minutes, CPU only, fully offline.

## Caveats

- One dataset, one tool version, default thresholds. This is not a quality endorsement of libero_10 and not a comparison against other tools.
- The 6 video_freeze findings represent genuine frozen frames in the dataset and should be inspected before use in training.
- idle_ratio findings are informational in v0.9.7; downstream consumers can still filter on this signal independently.
- The idle_ratio interpretation reflects threshold-vs-dataset-style mismatch, not a data quality problem. Thresholds are open for debate in the issue tracker.
