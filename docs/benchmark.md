# Benchmark: RDA v0.9.7 on 14 Public LeRobot Datasets + Blind Test
**RDA version**: 0.9.7 · **Scope**: 5,110 episodes (observation) + 206 episodes (blind test) · **Last run**: 2026-09-18

We ran `rda audit` across 14 public LeRobot-format datasets from HuggingFace Hub —
sim and real, scripted and human teleop, research arms and hobby hardware —
with zero tuning per dataset. The same default thresholds, everywhere.
All data is real, downloaded from HuggingFace Hub, no fabrication.

## Full results

| Dataset | Type | Episodes | Verdicts (P/R/E) | Action spikes | Idle median | State occupancy |
|---|---|---:|---|---:|---:|---:|
| jaco_play | real Jaco | 1,085 | 1085 / 0 / 0 | 11,958 (837 eps) | 74.1% | 2.8% |
| xarm_lift_medium | real xArm | 800 | 800 / 0 / 0 | 6 (5 eps) | **20.8%** | 2.4% |
| xarm_push_medium | real xArm | 800 | 800 / 0 / 0 | 845 (500 eps) | 83.3% | 1.7% |
| utokyo_pr2_tabletop | sim PR2 / RLDS | 240 | 240 / 0 / 0 | 681 (202 eps) | 83.6% | 5.0% |
| imperialcollege_sawyer_wrist_cam | real Sawyer / RLDS | 170 | 42 / 128 / 0 | 1,018 (170 eps) | **94.7%** | 20.0% |
| cmu_stretch | real Stretch | 135 | 135 / 0 / 0 | 3,855 (135 eps) | 66.7% | 1.9% |
| droid_100 | real Franka | 100 | 100 / 0 / 0 | 1,428 (99 eps) | 70.7% | 4.8% |
| **libero_10** | sim Panda / teleop | 379 | **373 / 6 / 0** | 3,622 (379 eps) | 71.7% | 6.6% |
| **pusht** | sim 2-DOF | 206 | **206 / 0 / 0** | 1,148 (199 eps) | 81.7% | **39.0%** |
| aloha_sim_transfer_cube_human | sim ALOHA / teleop | 50 | 50 / 0 / 0 | 1,535 (50 eps) | 71.2% | 3.8% |
| aloha_sim_insertion_human | sim ALOHA / teleop | 50 | 50 / 0 / 0 | 1,338 (50 eps) | 70.7% | 3.8% |
| aloha_sim_insertion_scripted | sim ALOHA / scripted | 50 | 50 / 0 / 0 | 1,633 (50 eps) | 63.7% | 4.2% |
| aloha_sim_transfer_cube_scripted | sim ALOHA / scripted | 50 | 50 / 0 / 0 | 2,489 (50 eps) | 64.0% | 5.0% |
| svla_so101_pickplace | real SO-100 | 50 | 50 / 0 / 0 | 260 (50 eps) | 86.7% | 4.5% |

### Addition (2026-09-18): imperialcollege_sawyer_wrist_cam

Real Sawyer teleop (RLDS port): 170 episodes, 17 household tasks, 5 fps, 64x64
video, 1-D state + 8-D action. First benchmark dataset where Layer-1
`video_timestamp_alignment` drives the verdicts: video/parquet spans differ by
one frame (0.2 s) on 128/170 episodes - a 2.4% median drift ratio, above the 2%
soft tolerance but inside the 10% hard tolerance - putting all 128 in REVIEW.
Median idle is 94.7%, the new benchmark high (idle_ratio flags all 170 episodes
but no longer escalates verdicts under v0.9.7). State occupancy (20.0%) is
measured on a 1-D state grid and is not directly comparable with multi-DOF
datasets.

### v0.9.7 verdict pipeline change

The v0.9.7 verdict pipeline was restructured with a three-layer aggregate model.
Findings (RISK_SIGNAL, RISK_LEVEL) no longer automatically escalate episode verdicts
to REVIEW. This significantly reduces false-positive REVIEWs on low-motion datasets:

- **libero_10**: v0.5.8 → 132/247/0 → v0.9.7 → **373/6/0** (6 REVIEW from video_freeze only)
- **pusht**: v0.5.4 → 43/163/0 → v0.9.7 → **206/0/0**
- **All other datasets**: 100% PASS under v0.9.7 (previously had REVIEW counts under v0.8.0)

### Layer 1 integrity

13 of 14 datasets pass all Layer 1 (data integrity) checks:
no missing data, no NaN values, no schema inconsistencies, no timestamp anomalies.
Exception: `imperialcollege_sawyer_wrist_cam` - video/parquet span drift above the
2% soft tolerance on 128/170 episodes (REVIEW, not EXCLUDE; see Addition note).

## Blind test (2026-09-14, v0.9.7)

5 defect classes × 10 episodes injected into `lerobot/pusht` (seed=42), 156
episodes left as controls, default thresholds, zero tuning.

| Class (10 eps each) | Caught strict (EXCLUDE) | Caught broad (+REVIEW) | Detector |
|---|---:|---:|---|
| empty episode (stale meta) | 10/10 | 10/10 | `_zero_frame_guard` |
| NaN in state (300 cells/class) | 10/10 | 10/10 | `invalid_values` |
| reversed timestamps (692 neg. deltas) | 10/10 | 10/10 | `timestamp_validity` |
| frozen episode (motion = 0%) | **0/10** | **0/10** | ⚠️ idle_ratio finding but not escalated |
| duplicate frames (tail-placed) | 10/10 | 10/10 | `timestamp_validity` |

**Strict confusion matrix: TP=40, FN=10, FP=0, TN=156 → precision 1.000,
recall 0.800. Broad: precision 1.000, recall 0.800.** All 156 control
verdicts identical to the clean baseline.

**Known regression**: Frozen episodes are no longer caught. In v0.5.4,
broad recall was 1.000 because idle_ratio triggered REVIEW verdicts. In
v0.9.7, idle_ratio findings are reported but don't change verdicts. This
should be addressed in a future release.

## Excluded datasets

| Dataset | Reason |
|---|---|
| aloha_corrupted | Synthetic corruption fixture, not a public HF dataset |
| bridge_sample | Only available in legacy format (not LeRobot v3.0), RDA incompatible |
| panda_pick_place_can | Repository not found on HuggingFace (404), even with auth token |

## Five patterns worth knowing before you train

**1. Median idle runs 20.8%–94.7%, and all but one dataset sit above 63%.**
Loss functions trained on a 75%-idle distribution are structurally biased
toward predicting "do nothing" unless you weight or curriculum around it.

**2. Same robot, same lab, four-fold idle difference.**
`xarm_lift_medium`: 20.8% median idle, all 800 episodes PASS.
`xarm_push_medium`: 83.3% median idle, 500/800 episodes with action spikes.
Same xArm platform — the difference is task difficulty, not collection sloppiness.

**3. Action discontinuity tracks the controller, not the dataset's reputation.**
ALOHA sim and SO-100 spike in 100% of episodes; xarm_lift has 6 spikes
across 800 episodes. If your policy uses smoothness regularization, this
number decides your curriculum.

**4. Clean integrity ≠ good training data.**
All 14 datasets pass Layer 1 hard checks. The behavior layer still flags
many episodes with risk signals. Both layers matter; most pipelines check neither.

**5. State space occupancy is uniformly low.**
Median occupancy ranges from 1.7% to 39%, with most datasets in the 2%–5%
range. Episodes overlap heavily in state space, suggesting limited exploration
diversity within individual datasets.

## Reproduce

```bash
pip install robot-data-audit==0.9.7
# download any LeRobot-format dataset from HuggingFace, then:
rda audit <dataset_path> --no-video -v
# with video quality checks (requires PyAV):
rda audit <dataset_path> --platform panda -v
```

All JSON reports saved in `rda_benchmark_v097/` directory.

## Caveats

- RDA flags statistical anomalies, not ground-truth errors. REVIEW means
  "look before you train," not "discard."
- These are default thresholds with zero per-dataset tuning.
- All 14 datasets audited at v0.9.7; verdict numbers reflect the new
  pipeline (fewer REVIEWs for low-motion datasets compared to v0.8.0).
- Blind test frozen episode detection is a known regression in v0.9.7.
