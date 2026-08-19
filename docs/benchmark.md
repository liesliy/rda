# Benchmark: RDA on 11 Public Robot Datasets

**RDA version**: 0.5.2 · **Scope**: 4,909 episodes · **Last run**: 2026-08

We ran `rda audit` across 11 public LeRobot-format datasets — sim and real,
scripted and human teleop, research arms and $100 hobby hardware — with zero
tuning per dataset. The same default thresholds, everywhere. This page is the
evidence behind the numbers; every figure below comes from a saved
`rda_report.json`.

## Full results

| Dataset | Type | Episodes | Verdicts (P/R/E) | Action spikes | Idle median | Idle p5 |
|---|---|---|---|---|---|---|
| aloha_sim_insertion_human | sim, human | 50 | 5 / 45 / 0 | 1,338 (100% eps) | 70.7% | 65.1% |
| aloha_sim_transfer_cube_scripted | sim, scripted | 50 | 5 / 45 / 0 | 2,489 (100% eps) | 64.0% | 44.2% |
| aloha_sim_insertion_scripted | sim, scripted | 50 | 1 / 49 / 0 | 1,633 (100% eps) | 63.7% | 53.7% |
| droid_100 | real Franka | 100 | 34 / 66 / 0 | 1,428 (99% eps) | 70.7% | 55.7% |
| pusht | sim | 206 | 43 / 163 / 0 | 1,148 (97% eps) | 81.7% | 59.5% |
| HuggingFaceVLA/libero | sim | 1,693 | 0 / 3 / 1,690 | 34 (3 eps) | 76.5% (3 eps) | 74.6% |
| bridge_orig_lerobot (sampled) | real WidowX | 25 | 4 / 21 / 0 | 91 (84% eps) | **93.3%** | 20.5% |
| xarm_lift_medium | real xArm | 800 | **767** / 33 / 0 | 6 (1% eps) | **20.8%** | 12.5% |
| xarm_push_medium | real xArm | 800 | 238 / 562 / 0 | 845 (62% eps) | 83.3% | 16.7% |
| svla_so101_pickplace | real SO-100 | 50 | 5 / 45 / 0 | 260 (100% eps) | 86.7% | 59.5% |
| jaco_play | real Jaco | 1,085 | 390 / 695 / 0 | 11,958 (77% eps) | 74.1% | 52.7% |

Notes:

- All integrity layers (NaN/Inf, timestamp validity, missing frames, schema)
  came back clean on all 11 datasets.
- **libero**: 1,690 of 1,693 episodes read 0 frames — a dataset-side
  meta/layout mismatch. RDA flags those EXCLUDE instead of silently passing,
  which is exactly the failure mode you want a gate to catch.
- **bridge**: sampled 25 episodes (the full set is 53k); the other 10 datasets
  were audited in full.
- **aloha_sim_transfer_cube_human** was audited earlier at v0.5.1 with
  matching results (1/49/0, 1,535 spikes, 71.2% idle). It has since become a
  gated repo, so we could not re-pull it for the 0.5.2 run.

## Five patterns worth knowing before you train

**1. Median idle runs 20.8%–93.3%, and 8 of 11 datasets sit above 65%.**
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
Integrity passed 11/11 — zero NaNs, zero timestamp reversals, zero missing
frames anywhere. The behavior layer still flagged 45–98% of episodes for
review in most datasets. Both layers matter; most pipelines check neither.

**5. Cheap hardware produces the most expensive data.**
The community SO-100 pick-place set: 86.7% median idle plus action spikes in
every episode. If you're fine-tuning on hobby-robot uploads, this is what
you're inheriting.

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
