# Benchmark: RDA 0.9.8 on unitreerobotics/G1_WBT_Brainco_Pickup_Pillow

**RDA version**: 0.9.8 · **Scope**: 300 episodes, 177,811 frames (full dataset) · **Last run**: 2026-09-15  
**Execution tier**: Full Audit (with video metrics) · **Report schema**: 1.2  
**Source**: [HuggingFace: unitreerobotics/G1_WBT_Brainco_Pickup_Pillow](https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Pickup_Pillow)

This is a single-dataset reproducibility record for the Unitree G1 humanoid robot
teleoperation dataset. Dataset was audited with RDA 0.9.8 at default thresholds,
zero tuning. All data is real, downloaded from HuggingFace Hub.

**v0.9.8 key change**: Added `_resolve_state_array()` to support LeRobot v3.0+
hierarchical state fields (e.g. `state.robot_q_current`, `state.ee_state`,
`state.hand_state`), which this dataset uses. Without this fix, velocity,
coverage, and joint_limit metrics would return N/A.

## Dataset Overview

| Property | Value |
|---|---|
| Robot | Unitree G1 (humanoid) + Brainco hands |
| Task | Pick up the pillow and place it on the sofa |
| Episodes | 300 |
| Total frames | 177,811 |
| FPS | 30 |
| Median duration | 18.87 s |
| Duration range | 12.3 – 37.2 s |
| State dims | robot_q_current (36D), ee_state (12D), hand_state (12D) |
| Video streams | 4× (head_stereo_left, head_stereo_right, wrist_left, wrist_right) |
| Format | LeRobot v3.0+ hierarchical field naming |

## Results

**Overall verdicts: 22 PASS / 0 REVIEW / 278 EXCLUDE.**

> **Note**: 278 EXCLUDE episodes are due to **video_stream_presence** only.
> Only the first chunk (chunk-000, ~22 episodes) of each video stream was
> downloaded for this audit. Episodes 22–299 have all tabular data intact but
> lack video files locally. Data quality for state/action modalities is
> unaffected across all 300 episodes.

| Check | Result |
|---|---|
| missing_dropout | pass (300/300) |
| invalid_values | pass (300/300) |
| schema_consistency | pass (300/300) |
| timestamp_validity | pass (300/300) |
| video_frame_integrity | pass (44/44 available) |
| video_freeze | pass (44/44 available) — **0 frozen regions** |
| video_timestamp_alignment | pass (300/300) |
| video_stream_span_consistency | pass (available episodes) |
| video_stream_temporal_offset | pass — 0 ms offset across all stream pairs |
| video_stream_temporal_drift | pass — 0 ms/min drift |
| video_stream_presence | **fail (278/300)** — episodes 22–299 have no local video |
| joint_limit | `na` — dataset metadata does not include joint limits (300/300) |
| sensor_synchronization | `na` — no stream timestamps in metadata (300/300) |

### Layer 2: Temporal & Motion

| Metric | Value | Note |
|---|---|---|
| frame_interval median_dt | 33.33 ms | Consistent with 30 FPS |
| velocity_acceleration | p95 median = 3.83 | Available for all 300 episodes ✅ |
| action_discontinuity | 3,340 spikes across 299 episodes | Median 10/ep, P95 23, P99 32 |
| idle_ratio | median 65.6% | Effective motion 34.4% |
| temporal_structure | stable | idle prefix ratio median 0.78% |

### Layer 3: Dataset Utility

| Metric | Value | Note |
|---|---|---|
| state_space_occupancy | median 6.5% | Low coverage (36D space) |
| coverage | Available ✅ | v0.9.8 fix enabled this |
| distribution | stable | Duration p10=15.0s, p50=18.9s, p90=25.4s |
| path_length | available | p50 = 32.57 |

## Joint Limit Analysis (Manual Injection)

The dataset metadata does not include joint limits. We manually injected the
G1 29-DOF official joint limits (from [Unitree developer docs](https://support.unitree.com/home/en/G1_developer/joint_limit))
padded to 36 dimensions with ±π defaults for the 7 extra DOFs.

**Result: All 300 episodes fail (EXCLUDE).**

| Joint (index) | Violation Rate | Actual Range | Configured Limit |
|---|---|---|---|
| WAIST_ROLL (13) | **24.31%** | [-1.70, 0.17] | [-0.52, 0.52] |
| R_LEG_ANKLE_PITCH (10) | **21.52%** | [-0.09, 2.23] | [-0.87, 0.52] |
| L_LEG_ANKLE_ROLL (5) | 8.26% | [-0.59, 0.23] | [-0.26, 0.26] |
| R_LEG_ANKLE_ROLL (11) | 7.62% | [-0.59, 0.11] | [-0.26, 0.26] |
| R_LEG_KNEE (9) | 4.16% | [-0.26, 0.03] | [-0.09, 2.88] |
| L_LEG_HIP_ROLL (1) | 2.25% | [-0.18, 0.02] | [-0.52, 2.97] |
| L_LEG_HIP_PITCH (0) | 0.59% | [-0.09, 0.75] | [-2.53, 2.88] |

**Interpretation**: The 36D `robot_q_current` field likely uses a different joint
ordering than the official 29-DOF docs, or includes hand/finger DOFs interleaved
with body joints. The violations may reflect a mapping mismatch rather than
actual hardware limit breaches. A GitHub issue has been opened on
`unitreerobotics/unitree_lerobot` requesting clarification.

## Video Quality Summary (22 Episodes)

| Episode | Visual Quality | Freeze Regions | Worst Stream |
|---|---|---|---|
| 0–16 | 1.0 (pass) | 0 | head_stereo_left |
| 17 | 0.9 (pass) | 0 | wrist_left |
| **18** | **0.5 (review)** | 0 | wrist_left (t≈6.0s quality drop) |
| 19–21 | 1.0 (pass) | 0 | head_stereo_left |

Key observations:
- **Zero frozen frames** across all 44 video stream segments (22 eps × 4 cameras)
- Episode 18's wrist_left stream has a visual quality dip (score 0.5) but no
  frozen frames — likely a lighting change or brief camera occlusion
- Multi-stream synchronization is excellent: 0ms offset, 0ms/min drift

## Comparison with Other Benchmarks

| Dataset | Type | Episodes | P/R/E | Spikes | Idle med | Occ |
|---|---|---:|---|---:|---:|---:|
| **G1_WBT (this)** | **real G1 / teleop** | **300** | **22/0/278*** | **3,340 (299)** | **65.6%** | **6.5%** |
| jaco_play | real Jaco | 1,085 | 1085/0/0 | 11,958 (837) | 74.1% | 2.8% |
| libero_10 | sim Panda / teleop | 379 | 373/6/0 | 3,622 (379) | 71.7% | 6.6% |
| droid_100 | real Franka | 100 | 100/0/0 | 1,428 (99) | 70.7% | 4.8% |
| pusht | sim 2-DOF | 206 | 206/0/0 | 1,148 (199) | 81.7% | 39.0% |

\* 278 EXCLUDE due to missing video files only; data integrity passes for all 300.

## Reproducing

```bash
pip install robot-data-audit==0.9.8
# Download dataset from HuggingFace
rda audit G1_WBT_Brainco_Pickup_Pillow --full --platform g1_wbt -o report.json
```

JSON report: `rda_report_v098_full.json` (available on request).
