"""REQ-5 (v0.8.0): dataset-level relative baselines + acceptance summary.

Covers the parts that are easy to get silently wrong:

- Tukey IQR fence matches the [SLE] reference arithmetic
  (``score_lerobot_episodes`` uses ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``).
- Degenerate input: every episode the same duration gives IQR = 0, which
  must be *flagged*, not silently reported as "0 outliers = clean".
- Availability discipline: an unavailable metric is excluded from the
  baseline instead of being coerced to 0. Folding it in would drag the
  fences toward zero and manufacture false outliers.
- ``not_checked`` is always present, so "not measured" can never be read
  as "measured and fine".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.audit.dataset_audit import DatasetAuditResult  # noqa: E402
from rda.audit.episode_audit import EpisodeAuditResult  # noqa: E402
from rda.audit.rules import AuditVerdict  # noqa: E402
from rda.io.schema import DatasetInfo  # noqa: E402
from rda.metrics.base import MetricAvailability, MetricResult  # noqa: E402
from rda.report.acceptance import (  # noqa: E402
    build_acceptance_summary,
    compute_percentile_baselines,
    compute_runtime_outliers,
    is_tukey_outlier,
    tukey_fence,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _metric(
    name: str,
    measurement: dict | None = None,
    availability: MetricAvailability = MetricAvailability.AVAILABLE,
) -> MetricResult:
    return MetricResult(
        name=name,
        availability=availability,
        measurement=measurement or {},
    )


def _episode(
    idx: int,
    duration_sec: float | None = None,
    duration_available: bool = True,
) -> EpisodeAuditResult:
    """Episode carrying only the ``distribution`` metric.

    ``duration_sec=None`` + ``duration_available=False`` models a real
    case: the metric ran but had nothing to measure (e.g. no timestamps).
    """
    if not duration_available:
        m = _metric("distribution", {}, MetricAvailability.NOT_AVAILABLE)
    else:
        m = _metric("distribution", {"duration_sec": duration_sec})
    return EpisodeAuditResult(
        episode_index=idx,
        num_frames=0,
        metrics={"distribution": m},
        verdict=AuditVerdict.PASS,
    )


def _dataset(episodes: list[EpisodeAuditResult]) -> DatasetAuditResult:
    info = DatasetInfo(
        path="test://dataset",
        num_episodes=len(episodes),
        total_frames=sum(e.num_frames for e in episodes),
        meta={"fps": 50, "robot": "test"},
    )
    return DatasetAuditResult(
        dataset_info=info,
        episodes={e.episode_index: e for e in episodes},
    )


# ---------------------------------------------------------------------------
# Tukey fence arithmetic  [SLE]
# ---------------------------------------------------------------------------

def test_tukey_fence_matches_reference_arithmetic():
    """Hand-computed: [1..9, 100] → q1=3.25, q3=7.75, iqr=4.5."""
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=np.float64)
    f = tukey_fence(v)

    assert f["q1"] == pytest.approx(3.25)
    assert f["q3"] == pytest.approx(7.75)
    assert f["iqr"] == pytest.approx(4.5)
    assert f["lower_fence"] == pytest.approx(-3.5)   # 3.25 - 1.5*4.5
    assert f["upper_fence"] == pytest.approx(14.5)   # 7.75 + 1.5*4.5


def test_tukey_outlier_boundaries():
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=np.float64)
    f = tukey_fence(v)

    assert is_tukey_outlier(100.0, f) is True
    assert is_tukey_outlier(9.0, f) is False
    assert is_tukey_outlier(1.0, f) is False


def test_tukey_fence_empty_input_is_all_zero():
    f = tukey_fence(np.array([], dtype=np.float64))
    assert f == {"q1": 0.0, "q3": 0.0, "iqr": 0.0,
                 "lower_fence": 0.0, "upper_fence": 0.0}


# ---------------------------------------------------------------------------
# Runtime outliers
# ---------------------------------------------------------------------------

def test_runtime_outlier_detects_long_tail():
    """10..19 plus one 60s episode → only the 60s episode is flagged."""
    durations = [10.0, 11.0, 12.0, 13.0, 14.0,
                 15.0, 16.0, 17.0, 18.0, 19.0, 60.0]
    eps = [_episode(i, d) for i, d in enumerate(durations)]
    block = compute_runtime_outliers(_dataset(eps))

    # q1=12.5, q3=17.5, iqr=5.0 → upper fence 25.0
    assert block["available_episodes"] == 11
    assert block["iqr"] == pytest.approx(5.0)
    assert block["upper_fence"] == pytest.approx(25.0)
    assert block["count"] == 1
    assert block["episodes"][0]["episode_index"] == 10
    assert block["episodes"][0]["direction"] == "long"
    assert block["degenerate"] is False


def test_runtime_outlier_degenerate_when_all_durations_equal():
    """All-identical durations → IQR=0. Must be flagged, not read as clean."""
    eps = [_episode(i, 9.98) for i in range(50)]
    block = compute_runtime_outliers(_dataset(eps))

    assert block["iqr"] == 0.0
    assert block["degenerate"] is True
    assert "note" in block
    # The point of the flag: "0 outliers" here means "no spread", not "clean".
    assert "no spread" in block["note"]


def test_runtime_outlier_handles_no_available_metric():
    eps = [_episode(i, duration_available=False) for i in range(5)]
    block = compute_runtime_outliers(_dataset(eps))

    assert block["available_episodes"] == 0
    assert block["count"] == 0
    assert "note" in block   # absence must be explained, not silent


# ---------------------------------------------------------------------------
# Availability discipline — unavailable must never become 0
# ---------------------------------------------------------------------------

def test_baseline_excludes_unavailable_metrics():
    """3 available + 2 unavailable → baseline over 3, not 5.

    If unavailable episodes were coerced to 0.0, the P10 would collapse
    toward zero and every real episode would look like an outlier.
    """
    eps = [
        _episode(0, 10.0),
        _episode(1, 11.0),
        _episode(2, 12.0),
        _episode(3, duration_available=False),
        _episode(4, duration_available=False),
    ]
    baselines = compute_percentile_baselines(_dataset(eps))
    d = baselines["duration_sec"]

    assert d["available_episodes"] == 3
    assert d["min"] == pytest.approx(10.0)   # 0.0 never entered the sample
    assert d["max"] == pytest.approx(12.0)


def test_baseline_keeps_key_when_metric_nowhere_available():
    eps = [_episode(i, duration_available=False) for i in range(3)]
    baselines = compute_percentile_baselines(_dataset(eps))
    d = baselines["duration_sec"]

    # Key stays present so the gap is visible rather than silently absent.
    assert "duration_sec" in baselines
    assert d["available_episodes"] == 0
    assert "note" in d


def test_baseline_percentiles_p10_p50_p90():
    """[10..19] → p10=10.9, p50=14.5, p90=18.1 (numpy linear interp)."""
    eps = [_episode(i, float(10 + i)) for i in range(10)]
    baselines = compute_percentile_baselines(_dataset(eps))
    d = baselines["duration_sec"]

    assert d["p10"] == pytest.approx(10.9)
    assert d["p50"] == pytest.approx(14.5)
    assert d["p90"] == pytest.approx(18.1)
    assert d["unit"] == "s"


# ---------------------------------------------------------------------------
# Acceptance summary block
# ---------------------------------------------------------------------------

def test_acceptance_summary_always_carries_not_checked():
    eps = [_episode(i, 10.0) for i in range(3)]
    summary = build_acceptance_summary(_dataset(eps))

    # Even when nothing was skipped the key must exist — a missing key
    # reads as "everything was checked".
    assert "not_checked" in summary
    assert summary["not_checked"] == {}


def test_acceptance_summary_propagates_not_checked():
    eps = [_episode(i, 10.0) for i in range(3)]
    summary = build_acceptance_summary(
        _dataset(eps), not_checked={"av": 7}
    )
    assert summary["not_checked"] == {"av": 7}


def test_acceptance_summary_shape():
    eps = [_episode(i, float(10 + i)) for i in range(10)]
    summary = build_acceptance_summary(
        _dataset(eps), quality={"dhi": 72, "grade": "B"}
    )

    assert summary["schema_version"] == "1.0"
    assert "generated_at" in summary
    assert summary["dataset"]["num_episodes"] == 10
    assert summary["verdict_distribution"]["pass"] == 10
    assert summary["verdict_distribution"]["pass_rate"] == pytest.approx(1.0)
    assert summary["quality"] == {"dhi": 72, "grade": "B"}
    assert "baseline_percentiles" in summary
    assert "runtime_outliers" in summary
    assert summary["calibration"]["tier1_rho_vs_full_set"] == pytest.approx(0.960)


def test_acceptance_summary_does_not_adjudicate():
    """REQ-5 presents evidence; it must not emit an accept/reject verdict."""
    eps = [_episode(i, 10.0) for i in range(3)]
    summary = build_acceptance_summary(_dataset(eps))

    # No accept/reject style key anywhere at the top level.
    adjudicating = {"decision", "accept", "reject", "accepted", "rejected",
                    "pass_fail", "verdict"}
    assert adjudicating.isdisjoint(set(summary.keys()))
    assert "interpretation_note" in summary


# ---------------------------------------------------------------------------
# UI page smoke (AppTest) — skipped when streamlit.testing is unavailable
# ---------------------------------------------------------------------------

def _streamlit_testing_available() -> bool:
    try:
        from streamlit.testing.v1 import AppTest  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _streamlit_testing_available(), reason="streamlit.testing unavailable"
)
def test_apptest_acceptance_page_renders():
    """Page 8 renders end-to-end and shows baseline + outlier sections."""
    from streamlit.testing.v1 import AppTest

    ui_dir = Path(__file__).resolve().parents[1] / "rda" / "ui_app"

    # Pages import ``components.common`` by bare name. ``streamlit run``
    # gets that from app.py, which pushes ui_app onto sys.path;
    # AppTest.from_file runs the page directly and does not.
    if str(ui_dir) not in sys.path:
        sys.path.insert(0, str(ui_dir))

    # 10..19 plus a 60s episode → exactly one runtime outlier, so the
    # outlier table is exercised rather than the empty-state branch.
    durations = [10.0 + i for i in range(10)] + [60.0]
    result = _dataset([_episode(i, d) for i, d in enumerate(durations)])

    at = AppTest.from_file(
        str(ui_dir / "pages" / "8_Acceptance.py"), default_timeout=180
    )
    at.session_state["rda_lang"] = "zh"
    at.session_state["audit_result"] = result
    # Pre-seeded so the page does not call compute_dhi() on a synthetic
    # result that lacks the full metric set.
    at.session_state["dataset_report"] = {
        "dhi": 70,
        "grade": "B",
        "training_readiness": "conditional",
        "training_readiness_detail": "",
        "dimensions": {},
    }
    at.run()

    fatal = [
        e.value for e in at.exception
        if "Could not find page" not in e.value
    ]
    assert not fatal, fatal

    assert at.title[0].value.startswith("📋")
    assert "验收摘要" in at.title[0].value

    subheaders = [s.value for s in at.subheader]
    assert any("数据集基线" in s for s in subheaders), subheaders
    assert any("时长离群" in s for s in subheaders), subheaders
    assert any("未检查维度" in s for s in subheaders), subheaders

    text = "\n".join(
        [b.value for b in at.markdown]
        + [b.value for b in at.caption]
        + [b.value for b in at.info]
        + [b.value for b in at.warning]
        + [b.value for b in at.success]
    )
    # One outlier out of 11 episodes must be stated explicitly.
    assert "检出离群 1 条 / 共 11 条" in text, text[:800]
    # The degenerate warning must NOT appear — these durations do vary.
    assert "IQR = 0" not in text
