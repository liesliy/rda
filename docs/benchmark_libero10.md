# Benchmark: RDA 0.5.8 on lerobot/libero_10 (full, v3.0)

**RDA version**: 0.5.8 · **Scope**: 379 episodes, 101,469 frames (full dataset) · **Last run**: 2026-09-03

This is a single-dataset reproducibility record: we audited the complete
[lerobot/libero_10](https://huggingface.co/datasets/lerobot/libero_10)
dataset (LeRobot v3.0 format) with RDA 0.5.8 at default thresholds, zero
tuning. Every figure below comes from a saved RDA JSON report. It also
serves as the release regression record for the v3.0 fixes shipped in
0.5.7–0.5.8 (task identity, chunked-video frame validation).

## Results

**Overall verdicts: 132 PASS / 247 REVIEW / 0 EXCLUDE.**

| Check | Result |
|---|---|
| missing_dropout, invalid_values, schema_consistency | pass (379/379) |
| timestamp_validity, video_frame_integrity | pass (379/379) |
| sampling_jitter, velocity_acceleration, action_discontinuity | pass (379/379) |
| temporal_sufficiency, distribution, coverage | pass (379/379) |
| joint_limit, sensor_synchronization | `na` — not applicable to this dataset |
| idle_ratio (low-motion heuristic) | review (379/379) |

Zero hard defects: no corruption, no frame misalignment, no timestamp
inversions. The integrity layers are clean across the full dataset.

## The one open question: idle_ratio

The 247 REVIEW verdicts (65%) come from exactly one rule: episodes whose
**effective motion ratio** (active frames / total frames) falls below the
default 0.30 heuristic. Across libero_10 the ratio spans 2%–41% with a
median of 28.3% — the whole dataset sits near the threshold.

We believe this reflects threshold-vs-dataset-style mismatch rather than a
data quality problem, for three reasons:

1. libero_10 tasks are fine-grained tabletop manipulation; motion is
   naturally small and slow, so fewer frames register as "active";
2. the same episodes pass timestamp, action-continuity, and
   velocity/acceleration checks — a genuinely stuck demonstration would
   likely trip those too;
3. the threshold is a general-purpose heuristic and is adjustable.

REVIEW means "look before you train," not "discard." This is the same
pattern already visible in the [multi-dataset benchmark](benchmark.md):
median idle varies 20.8%–93.3% across datasets, and task style — not
collection sloppiness — explains most of it.

## Per-task breakdown

| task_index | PASS | REVIEW | total |
|---:|---:|---:|---:|
| 0 | 8 | 30 | 38 |
| 1 | 22 | 14 | 36 |
| 2 | 9 | 25 | 34 |
| 3 | 18 | 23 | 41 |
| 4 | 19 | 24 | 43 |
| 5 | 13 | 20 | 33 |
| 6 | 8 | 21 | 29 |
| 7 | 14 | 35 | 49 |
| 8 | 8 | 27 | 35 |
| 9 | 13 | 28 | 41 |

Flagged share varies from 19% (task 1) to 90% (task 7) — the tasks with the
smallest manipulation footprint flag most, consistent with the
style-mismatch interpretation above. As of 0.5.8, every episode record in
the JSON report carries `task_index` / `task_description`, so this slicing
is one `groupby` away in any consumer.

## Regression notes (0.5.7 → 0.5.8)

- 0.5.7 fixed LeRobot v3.0 task identity loss (dataset-level `num_tasks` /
  `tasks` mapping; loader reads `tasks.parquet`).
- 0.5.8 fixed the chunked-video false flag in `video_frame_integrity`
  (whole-file frame counts vs. per-episode spans) and surfaced
  `task_index` / `task_description` on every episode result.
- On this run: `video_frame_integrity` pass = 379/379 (0 false flags),
  episode `task_index` missing = 0, distinct task_index = {0..9},
  dataset `num_tasks` = 10. Verdicts are identical between a
  site-packages install and a source checkout.

## Reproduce

```bash
pip install robot-data-audit
# after downloading lerobot/libero_10:
rda audit <libero_10_path> --format json --output report.json
```

Full 379-episode audit: ~3 minutes, CPU only, fully offline.

## Caveats

- One dataset, one tool version, default thresholds. This is not a quality
  endorsement of libero_10 and not a comparison against other tools.
- The idle_ratio interpretation above is our analysis of a heuristic
  signal, not a ground-truth judgment. Thresholds are open for debate in
  the issue tracker.
