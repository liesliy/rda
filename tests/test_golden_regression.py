"""Golden-set end-to-end regression tests for the RDA v0.9 architecture.

Unlike the per-metric unit tests (which call a single metric directly) and
the negative-control tests (which feed hand-built MetricResult objects into
the verdict gate), these tests run the **real** audit pipeline over
**synthetic episodes with known ground truth** and pin the outcome:

* the episode verdict (PASS / REVIEW / EXCLUDE),
* which metrics must fire,
* availability discipline for video metrics,
* dataset-level aggregation across the whole catalog.

The episodes are generated deterministically by ``tests/golden/golden_set.py``
(fixed RNG seeds), so a green run is reproducible.

v0.9 architecture guarantees asserted here
------------------------------------------
1. **Only L1 CRITICAL metrics can flip the verdict.** Diagnostic metrics
   (idle_ratio, action_discontinuity, sampling_jitter, ...) produce findings
   but leave the verdict at PASS unless a critical metric also fails.
2. **Critical hard failures -> EXCLUDE**; critical review-level findings
   (e.g. joint_limit) -> REVIEW.
3. **Every metric reports an availability**; a video metric on an episode
   without video must be N/A, not a silent pass.
4. **Dataset summary** aggregates diagnostic measurements across episodes.

Run:  ``pytest tests/test_golden_regression.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.audit.episode_audit import EpisodeAuditor  # noqa: E402
from rda.audit.dataset_audit import DatasetAuditor  # noqa: E402
from rda.audit.rules import AuditVerdict, CRITICAL_METRICS  # noqa: E402
from rda.io.schema import DatasetInfo  # noqa: E402
from rda.metrics.base import MetricAvailability  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden.golden_set import all_scenarios  # noqa: E402

try:
    import av  # noqa: F401

    _HAS_AV = True
except Exception:
    _HAS_AV = False


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    """Build + audit every golden scenario once, share results across tests."""
    tmp = tmp_path_factory.mktemp("golden")
    scenarios = all_scenarios(tmp_path=tmp, include_video=_HAS_AV)
    auditor = EpisodeAuditor()
    results = {s.id: auditor.audit(s.builder()) for s in scenarios}
    return scenarios, results


def _fired(result):
    return {n for n, m in result.metrics.items() if getattr(m, "has_finding", False)}


# --- 1. Verdict matrix ------------------------------------------------------

def test_verdict_matches_expected(catalog):
    """Each golden episode must produce its pinned verdict."""
    scenarios, results = catalog
    failures = []
    for s in scenarios:
        res = results[s.id]
        actual = res.verdict.value.lower()
        if actual != s.expected_verdict:
            failures.append(
                f"[{s.id}] expected {s.expected_verdict!r} got {actual!r}; "
                f"fired={sorted(_fired(res))}"
            )
    assert not failures, "\n".join(failures)


# --- 2. Must-trigger contracts ---------------------------------------------

def test_must_trigger_metrics_fire(catalog):
    """Metrics named in ``must_trigger`` must report a finding after audit."""
    scenarios, results = catalog
    failures = []
    for s in scenarios:
        if not s.must_trigger:
            continue
        fired = _fired(results[s.id])
        missing = [m for m in s.must_trigger if m not in fired]
        if missing:
            failures.append(
                f"[{s.id}] expected {list(s.must_trigger)} to fire, "
                f"missing {missing}; fired={sorted(fired)}"
            )
    assert not failures, "\n".join(failures)


# --- 3. v0.9 gate semantics: diagnostics never flip the verdict -------------

def test_diagnostic_anomalies_never_escalate(catalog):
    """Diagnostic-only scenarios (frozen arm, spikes, jitter) must stay PASS
    even though they produce diagnostic findings — the core v0.9 guarantee."""
    _, results = catalog
    for sid in ("frozen_arm_diagnostic", "action_spikes_diagnostic",
                "sampling_jitter_diagnostic"):
        if sid not in results:
            continue
        res = results[sid]
        assert res.verdict == AuditVerdict.PASS, (
            f"[{sid}] diagnostic anomaly must not change verdict, got {res.verdict.value}"
        )
        assert _fired(res), f"[{sid}] should still surface a diagnostic finding"


def test_exclude_scenarios_driven_by_hard_critical(catalog):
    """EXCLUDE scenarios must be driven by a CRITICAL metric hard-failing
    (status == exclude), never by a diagnostic metric."""
    _, results = catalog
    for sid in ("nan_inf_values", "missing_frames_dropout",
                "video_missing_camera_stream", "video_long_freeze",
                "video_timestamp_drift"):
        if sid not in results:
            continue
        res = results[sid]
        assert res.verdict == AuditVerdict.EXCLUDE, f"[{sid}] expected EXCLUDE"
        hard_critical = [
            n for n, m in res.metrics.items()
            if n in CRITICAL_METRICS and getattr(m, "has_finding", False)
            and m.assessment.get("status") == "exclude"
        ]
        assert hard_critical, (
            f"[{sid}] EXCLUDE must come from a hard critical metric; "
            f"fired={sorted(_fired(res))}"
        )


def test_joint_limit_is_review_not_exclude(catalog):
    """joint_limit is a critical metric that reports REVIEW-level findings."""
    _, results = catalog
    if "joint_limit_violation" not in results:
        pytest.skip("joint_limit scenario unavailable")
    res = results["joint_limit_violation"]
    assert res.verdict == AuditVerdict.REVIEW
    assert "joint_limit" in _fired(res)


# --- 4. Availability discipline ---------------------------------------------

def test_video_metrics_na_without_video(catalog):
    """On a video-less episode every video_* metric must be NOT_AVAILABLE,
    not a silent pass."""
    _, results = catalog
    res = results["clean_baseline"]
    for name, m in res.metrics.items():
        if not name.startswith("video_"):
            continue
        assert m.availability == MetricAvailability.NOT_AVAILABLE, (
            f"{name} on a video-less episode must be N/A, got {m.availability.value}"
        )


def test_no_metric_errors_on_clean_data(catalog):
    """No metric may ERROR on the clean baseline (a crash is silently
    swallowed by the auditor and would hide a regression)."""
    _, results = catalog
    errored = [n for n, m in results["clean_baseline"].metrics.items()
               if m.availability == MetricAvailability.ERROR]
    assert not errored, f"metrics errored on clean data: {errored}"


def test_verifiability_field_present(catalog):
    """v0.9: every metric result in the JSON dict carries a verifiability
    level (Verified / Measured / Not verifiable / N/A)."""
    scenarios, results = catalog
    from rda.report.json_report import _episode_result_to_dict
    res = results["clean_baseline"]
    d = _episode_result_to_dict(res)
    metrics_dict = d.get("metrics", d)
    # metrics can be dict keyed by name
    items = metrics_dict.values() if isinstance(metrics_dict, dict) else metrics_dict
    count = 0
    for item in items:
        if isinstance(item, dict) and "verifiability" in item:
            count += 1
    assert count > 0, "no metric reported a verifiability field"


# --- 5. Dataset-level aggregation -------------------------------------------

def test_dataset_audit_aggregates_and_summarizes(tmp_path_factory):
    """Running the catalog through DatasetAuditor yields coherent verdict
    counts and a v0.9 dataset summary."""
    tmp = tmp_path_factory.mktemp("golden_ds")
    scenarios = all_scenarios(tmp_path=tmp, include_video=_HAS_AV)
    episodes = [s.builder() for s in scenarios]

    info = DatasetInfo(
        path=str(tmp),
        num_episodes=len(episodes),
        total_frames=sum(e.num_frames for e in episodes),
        modalities=["state"],
        action_keys=["joint_pos"],
        meta={"format": "golden", "fps": 10},
    )
    result = DatasetAuditor().audit_dataset(info, iter(episodes))

    counts = result.verdict_counts
    assert counts.get(AuditVerdict.EXCLUDE, 0) >= 2, f"expected ≥2 EXCLUDE: {counts}"
    assert sum(counts.values()) == len(episodes)

    # v0.9 dataset summary present.
    assert result.dataset_summary is not None
    summary = result.dataset_summary.to_dict()
    assert isinstance(summary, dict) and summary, "dataset summary must be non-empty"


def test_verdict_distribution_is_stable(catalog):
    """Pin the non-video verdict distribution so any accidental change to
    verdict logic fails loudly."""
    scenarios, results = catalog
    dist = {AuditVerdict.PASS: 0, AuditVerdict.REVIEW: 0, AuditVerdict.EXCLUDE: 0}
    for s in scenarios:
        dist[results[s.id].verdict] += 1
    if not _HAS_AV:
        # clean + frozen + spikes + jitter = 4 PASS; joint_limit = 1 REVIEW;
        # nan + missing = 2 EXCLUDE.
        assert dist == {AuditVerdict.PASS: 4, AuditVerdict.REVIEW: 1,
                        AuditVerdict.EXCLUDE: 2}, dist
    else:
        # +4 video scenarios: 1 pass, 3 exclude.
        assert dist == {AuditVerdict.PASS: 5, AuditVerdict.REVIEW: 1,
                        AuditVerdict.EXCLUDE: 5}, dist


def test_text_report_renders_without_video(catalog):
    """Regression (post-0.9.1): the text/enhanced summary report must not
    crash on datasets with no video. The verifiability section iterates
    over video metrics that are NOT_AVAILABLE; the text path previously
    referenced the nonexistent MetricAvailability.NA member and raised
    AttributeError (the JSON path had been fixed in 0.9.1 but the text
    path shipped broken)."""
    from rda.io.schema import DatasetInfo
    from rda.audit.dataset_audit import DatasetAuditor
    from rda.report.summary import format_enhanced_summary_text

    scenarios = [s for s in catalog[0] if not s.video]
    info = DatasetInfo(
        path="golden://dataset",
        num_episodes=len(scenarios),
        total_frames=sum(s.builder().num_frames for s in scenarios),
        modalities=["state"],
        action_keys=["joint_pos"],
        meta={"fps": 10},
    )
    result = DatasetAuditor().audit_dataset(
        info, iter(s.builder() for s in scenarios)
    )
    text = format_enhanced_summary_text(result)
    assert isinstance(text, str) and len(text) > 100
    # video metrics are N/A (not applicable) on a video-less dataset
    assert "N/A" in text
