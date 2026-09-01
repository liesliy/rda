# Case study: AgiBotWorld2026 (third-party audit, Sep 2026)

> Dataset: [agibot-world/AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) Phase 3 · Tool: RDA v0.5.5 · Date: 2026-09-01
>
> Full audit discussion on the dataset page: [HF Discussion #9](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026/discussions/9)

AgiBotWorld2026 Phase 3 ships real-robot reinforcement-learning trajectories alongside simulation tasks, all in LeRobot v2.1 format. We treated it as a third-party test: can RDA audit a fresh, large, externally-produced dataset with zero adaptation?

**Answer: yes — zero code changes, and the audit produced an independently verifiable signal.**

## Scope (said honestly)

- **Simulation**: all 5 tasks, 1,102 episodes (~500 MB) — 100% of the simulation split
- **Real robot**: 1 of 50 RL packages — `ReinforcementLearning/Home/task_12192/HG-DAgger/20857245_20857259.tar.gz` (3.57 GB, the smallest), 10 episodes / 23,658 frames
- Not tested: the remaining 49 real-robot packages (~3.9 TB), ImitationLearning (4.4 TB), RichInteraction (2.3 TB). Every sampled `info.json` (6/6) uses LeRobot v2.1 with the same pipeline, so format-level conclusions generalize.
- RDA does not read video files; the `videos/` tarballs were never downloaded.

## Results

| Dataset | Episodes | PASS | REVIEW | EXCLUDE | FAIL | idle median | action spikes |
|---|---|---|---|---|---|---|---|
| sim/tidy_up_food_in_freezer | 200 | 0 | 200 | 0 | 0 | 73.1% | 24,448 |
| sim/scoop_popcorn_to_bucket | 211 | 28 | 183 | 0 | 0 | 67.5% | 6,323 |
| sim/take_bagged_food_to_cart | 201 | 2 | 199 | 0 | 0 | 75.7% | 2,893 |
| sim/take_cup_to_cart | 283 | 1 | 282 | 0 | 0 | 79.4% | 4,785 |
| sim/take_drink_to_cart | 207 | 0 | 207 | 0 | 0 | 76.7% | 10,318 |
| real RL · HG-DAgger (task_12192) | 10 | 0 | 10 | 0 | 0 | 70.2% | 1,712 |
| **Total** | **1,112** | **31** | **1,081** | **0** | **0** | — | **50,479** |

- **4,448 integrity checks** (missing frames / invalid values / schema / timestamps) across all 1,112 episodes: **0 failures**. Constant 30 fps.
- All 31 PASS episodes come from simulation tasks — the three-tier verdict layering works on external data, it is not a blanket REVIEW.

## Cross-validation against official takeover labels

The real-robot HG-DAgger package ships an official `intervened` column (frame-by-frame human-takeover labels). That allowed a test no synthetic benchmark offers: **do RDA's action-discontinuity spikes mark anything real?**

Using RDA's reported `spike_indices` (1,712 spikes across 10 episodes):

- **226 spikes (13.2%)** fall within **±1 frame of a takeover transition**; random baseline ≈ **4.3%** → **~3.1× enrichment** (1.6×–5.2× per episode)
- Spikes inside takeover intervals (away from transitions) are *not* enriched vs. random — most spikes are normal high-dynamics teleoperation

Reading: RDA flags discontinuity as an *observational* risk signal, not confirmed corruption — and the enrichment at takeover boundaries shows those flags align with real business events. This is third-party evidence that the metric measures something, not just noise.

## Reproduce

```bash
pip install robot-data-audit
# simulation (a few hundred MB is enough; videos/ tarballs not needed)
rda audit ./AgiBotWorld2026/simulation/<task> --format json -o report.json
# real robot: download one ReinforcementLearning/*/HG-DAgger/*.tar.gz and extract
rda audit ./rl_hgdagger/data --format json -o rl_report.json
```

## Sample reports

- [Real-robot RL audit (10 episodes)](https://liesliy.github.io/rda/examples/rda_report_agibot_rl_hgdagger.html)
- [Blind-test baseline on lerobot/pusht](https://liesliy.github.io/rda/examples/rda_report_pusht.html)

Full per-episode JSON reports (6 files) are available on request — see the [HF discussion](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026/discussions/9) or open an issue.
