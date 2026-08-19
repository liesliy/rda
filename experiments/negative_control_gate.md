# The spike/verdict decoupling bug — and the test that pins it

**When discovered**: tracked across v0.4.9 → v0.4.11.
**When fixed**: v0.4.11 era (behavior signals now consumed by the verdict
aggregator).
**When pinned by a test**: v0.5.2 (the test was added *after* the dev.to
write-up — see honesty note below).

## The bug, in one sentence

The behavior layer (`compute_behavior_severity`) correctly computed
action-discontinuity spikes and idle ratios, but the verdict aggregator
(`upgrade_verdict_by_behavior`) ignored those signals entirely — an episode
with 150 action spikes and 95% idle frames could still walk away with a
PASS badge because no code path escalated it.

This is the classic "the sensor reported it but the alarm didn't ring"
failure. For a data-quality gate it's the worst kind of bug: the tool looks
like it's working (it found the anomaly), but the *verdict* — the thing
users actually gate on — silently says "fine."

## The fix

`upgrade_verdict_by_behavior()` now consumes the behavior severity and
upgrades PASS → REVIEW when the severity crosses threshold. Critically, it
**never** softens an EXCLUDE back to REVIEW — the gate can't fail open in
either direction.

## The pin (regression test)

[`tests/test_negative_control.py`](../tests/test_negative_control.py) —
7 assertions, all passing in CI. The test feeds the classifier metrics
that *pass every rule* but contain known anomalies, then asserts the
verdict **must** come back REVIEW, not PASS:

- 150 injected spikes across an episode → REVIEW (not PASS)
- 95% idle frames → REVIEW
- Combined anomalies escalate severity
- Clean episode keeps its PASS (negative control's negative control)
- Severity = exactly 20 boundary → REVIEW
- NaN hard-damage → stays EXCLUDE (behavior layer cannot soften it)
- Missing action channel → N/A (not PASS, not EXCLUDE)

The test lives right at the `classify_episode → upgrade_verdict_by_behavior`
call site — the exact joint where the original bug lived.

## Honesty note

The dev.to launch post for v0.5.2 said "I wrote a negative-control test" in
present tense, as if it already existed. It didn't — not in the repo at
that point. The test was added afterwards, in the same session the
discrepancy was noticed, and pushed as commit `e8e9efa`. This write-up
records that gap so the timeline is honest: the fix predates the test, the
test predates the public claim, but only by hours, and the claim was made
before the file existed. Lesson logged: **a claim in a launch post must
point to a committed file, or it doesn't count.**

## What this does NOT prove

- It proves the wiring is correct — behavior signals reach the verdict.
  It does **not** prove the *thresholds* are well-calibrated (is "150 spikes
  = REVIEW" the right cutoff?). Threshold tuning against labelled data is
  roadmap L2.
- It covers the one bug we know about. Other decoupling joints in the
  pipeline (e.g. distribution metrics → verdict) are not yet pinned by
  negative-control tests. Adding them is in the L2 benchmark plan.

## Backing artifact

- Test: [`tests/test_negative_control.py`](../tests/test_negative_control.py)
  (7/7 pass, runs in CI).
- Fix site: [`rda/audit/rules.py`](../rda/audit/rules.py) —
  `upgrade_verdict_by_behavior()` and `compute_behavior_severity()`.
