# Benchmark: RDA on 12 Local Robot Datasets

**RDA version**: 0.5.3 · **Scope**: 4,959 episodes · **Last run**: 2026-08-20

We ran `rda audit` across 12 local LeRobot-format datasets — sim and real,
scripted and human teleop, research arms and hobby hardware — with zero tuning
per dataset. The same default thresholds, everywhere. This page is the
reproducibility record; every figure below comes from saved RDA JSON reports.
The corrected rerun includes the LeRobot v3.0 loader fix released in 0.5.3.

## Full results

| Dataset | Type | Episodes | Verdicts (P/R/E) | Action spikes | Idle median |
|---|---|---:|---|---:|---:|
| aloha_corrupted | synthetic corruption fixture | 50 | 6 / 42 / 2 | 1,273 (48 eps) | 70.7% |
| aloha_insertion | sim / teleop | 50 | 5 / 45 / 0 | 1,338 (50 eps) | 70.7% |
| aloha_insertion_scripted | sim / scripted | 50 | 1 / 49 / 0 | 1,633 (50 eps) | 63.7% |
| aloha_transfer_scripted | sim / scripted | 50 | 5 / 45 / 0 | 2,489 (50 eps) | 64.0% |
| bridge_sample | real WidowX / sampled | 25 | 4 / 21 / 0 | 91 (21 eps) | **93.3%** |
| droid_100 | real Franka | 100 | 34 / 66 / 0 | 1,428 (99 eps) | 70.7% |
| jaco_play | real Jaco | 1,085 | 390 / 695 / 0 | 11,958 (837 eps) | 74.1% |
| libero (local copy) | sim / incomplete local copy | 1,693 | 213 / 707 / 773 | 6,276 (918 eps) | 73.7% (920 eps) |
| pusht | sim | 206 | 43 / 163 / 0 | 1,148 (199 eps) | 81.7% |
| svla_so101_pickplace | real SO-100 | 50 | 5 / 45 / 0 | 260 (50 eps) | 86.7% |
| xarm_lift_medium | real xArm | 800 | **767** / 33 / 0 | 6 (5 eps) | **20.8%** |
| xarm_push_medium | real xArm | 800 | 238 / 562 / 0 | 845 (500 eps) | 83.3% |

Notes:

- All integrity layers (NaN/Inf, timestamp validity, missing frames, schema)
  came back clean on the 11 non-LIBERO datasets in this rerun.
- **libero local copy**: metadata lists 1,693 episodes; 920 were found in the
  downloaded parquet files and 773 were absent from the local copy. The local
  copy is incomplete; this is not evidence that the complete upstream dataset
  contains 773 empty episodes. RDA 0.5.3 falls back to the actual parquet
  location when metadata `file_index` values are stale.
- **bridge**: sampled 25 episodes (the full set is 53k); the other datasets
  were audited in full.
- **aloha_sim_transfer_cube_human** was audited earlier at v0.5.1 with
  matching results (1/49/0, 1,535 spikes, 71.2% idle). It has since become a
  gated repo, so we could not re-pull it for the 0.5.2 run.

## Five patterns worth knowing before you train

**1. Median idle runs 20.8%–93.3%, and 10 of 12 datasets sit above 65%.**
Loss functions trained on a 75%-idle distribution are structurally biased
toward predicting "do nothing" unless you weight or curriculum around it.
Bridge data pushes it to 93%. Measure yours before the GPU bill, not after.

**2. Same robot, same lab, four-fold idle difference.**
`xarm_lift_medium`: 20.8% median idle, 767/800 episodes PASS.
`xarm_push_medium`: 83.3% median idle, 562/800 REVIEW. Same xArm platform —
the difference is task difficulty (lifting vs. pushing a flat object), not
collection sloppiness. High idle isn't always a bug; it's a property you need
to know and design around. RDA flags both sides of this honestly.

**3. Action discontinuity tracks the controller, not the dataset's reputation.**
Sim ALOHA and the SO-100 hobby setup spike in literally 100% of episodes;
xArm lift data has 6 spikes across 800 episodes. If your policy uses
smoothness regularization or you're doing sim-to-real action statistics,
this number decides your curriculum.

**4. Clean integrity ≠ good training data.**
Across the non-LIBERO datasets in this rerun, the integrity layers were clean
except for the intentional `aloha_corrupted` fixture. The behavior layer still
flagged many episodes for review in most datasets. Both layers matter; most
pipelines check neither.

**5. Benchmark scope matters.**
The bridge result is a 25-episode sample, and `aloha_corrupted` is an intentional
corruption fixture rather than a production dataset. Treat these numbers as
observed signals in this local rerun, not as universal claims about a robot or
upstream dataset.

## Reproduce

Any of these datasets can be re-audited in minutes:

```bash
pip install robot-data-audit
# download any LeRobot-format dataset, then:
rda audit <dataset_path> -v
```

Most of the datasets above are public on the HuggingFace Hub under
`lerobot` / physical-intelligence / HuggingFaceVLA orgs. The audit runs
fully offline; nothing leaves your machine.

## How RDA tests itself

The verdict pipeline has its own negative-control suite
(`tests/test_negative_control.py`): metric results that pass every rule but
contain known anomalies (spikes, frozen arms) must still get REVIEW — the
gate has to prove it consumes its own signals. See the repo README for the
story behind it.

## Caveats

- RDA flags statistical anomalies, not ground-truth errors. REVIEW means
  "look before you train," not "discard."
- These are default thresholds with zero per-dataset tuning — that is the
  point of an audit tool, but it also means edge cases exist. All thresholds
  are open for debate in the issue tracker.
