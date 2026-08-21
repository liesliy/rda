"""Evidence-boundary regression tests for user-facing RDA outputs."""

from __future__ import annotations

from types import SimpleNamespace

from rda.audit.rules import AuditVerdict
from rda.metrics.base import MetricAvailability, MetricResult
from rda.report.json_report import (
    _build_diagnosis,
    _determine_pattern_type,
    generate_episode_report,
    generate_dataset_report,
)
from rda.io.schema import DatasetInfo
from rda.audit.dataset_audit import DatasetAuditResult


def _obs(name: str, measurement: dict) -> MetricResult:
    return MetricResult.make_pass(
        name=name,
        measurement=measurement,
        message="observational",
    )


def _na(name: str) -> MetricResult:
    return MetricResult.make_na(name=name, reason="required input missing")


def _episode(metrics: dict, verdict: AuditVerdict = AuditVerdict.REVIEW):
    return SimpleNamespace(
        episode_index=0,
        num_frames=100,
        verdict=verdict,
        metrics=metrics,
    )


def _dataset(ep_result):
    info = DatasetInfo(
        path="fixture",
        num_episodes=1,
        total_frames=100,
        modalities=[],
        meta={},
    )
    return DatasetAuditResult(
        dataset_info=info,
        episodes={0: ep_result},
        verdict_counts={AuditVerdict.REVIEW: 1},
    )


def test_risk_pattern_and_diagnosis_are_not_failure_claims():
    ep = _episode({
        "distribution": _obs("distribution", {"duration_sec": 20.0, "path_length": 1.0}),
        "idle_ratio": _obs("idle_ratio", {"effective_motion_ratio": 0.05}),
        "action_discontinuity": _obs("action_discontinuity", {"spike_count": 1}),
    })
    reason_vector = {
        "duration_anomaly": 20.0,
        "motion_ratio_anomaly": 0.95,
        "spike_anomaly": 1.0,
    }
    pattern = _determine_pattern_type(ep, reason_vector)
    diagnosis = _build_diagnosis(ep, pattern, reason_vector)

    assert pattern["primary"] in {"FROZEN_LIKE", "STUCK_LIKE"}
    assert "not proof" in diagnosis["why"]
    assert any(token in diagnosis["next"].lower() for token in ("review", "confirm"))
    assert "exclude" not in diagnosis["next"].lower()
    assert "failed" not in diagnosis["next"].lower()


def test_episode_json_separates_risk_and_unverifiable():
    ep = _episode({
        "action_discontinuity": _obs("action_discontinuity", {"spike_count": 150}),
        "idle_ratio": _obs("idle_ratio", {"effective_motion_ratio": 0.9}),
        "sensor_synchronization": _na("sensor_synchronization"),
    })
    report = generate_episode_report(_dataset(ep), 0)

    assert report["evidence_level"] == "RISK_SIGNAL"
    assert "action_discontinuity" in report["risk_signals"]
    assert "sensor_synchronization" in report["unverifiable_metrics"]
    assert report["evidence_by_metric"]["sensor_synchronization"] == "UNVERIFIABLE"
    assert report["hard_fail_issues"] == []
    assert report["behavior_verdict"] == "NEEDS_REVIEW"


def test_hard_fail_remains_distinct_from_risk_signal():
    hard_fail = MetricResult.make_exclude(
        name="invalid_values",
        reason="NaN values detected",
        message="corrupt",
    )
    ep = _episode({"invalid_values": hard_fail}, verdict=AuditVerdict.EXCLUDE)
    report = generate_episode_report(_dataset(ep), 0)

    assert report["evidence_level"] == "HARD_FAIL"
    assert report["hard_fail_issues"] == ["INT-02"]
    assert report["risk_signals"] == []


def test_dataset_json_exposes_evidence_summary():
    ep = _episode({
        "action_discontinuity": _obs("action_discontinuity", {"spike_count": 150}),
        "sensor_synchronization": _na("sensor_synchronization"),
    })
    report = generate_dataset_report(_dataset(ep))

    evidence = report["evidence_summary"]
    assert evidence["hard_fail_episodes"] == 0
    assert evidence["risk_signal_episodes"] == 1
    assert evidence["unverifiable_metric_episodes"]["sensor_synchronization"] == 1
    assert "not" not in evidence["interpretation"].lower() or "deterministic" in evidence["interpretation"].lower()


def test_ui_issue_stats_counts_observational_signals():
    # UI extras are optional in the core test environment. Stub Plotly for
    # this pure aggregation test instead of installing the full UI stack.
    import sys
    import types

    if "plotly.graph_objects" not in sys.modules:
        plotly = types.ModuleType("plotly")
        graph_objects = types.ModuleType("plotly.graph_objects")
        graph_objects.Figure = object
        plotly.graph_objects = graph_objects
        sys.modules["plotly"] = plotly
        sys.modules["plotly.graph_objects"] = graph_objects

    from rda.ui_app.components.common import compute_issue_stats

    ep = _episode({
        "action_discontinuity": _obs("action_discontinuity", {"spike_count": 150}),
        "sensor_synchronization": _na("sensor_synchronization"),
    })
    stats = compute_issue_stats(_dataset(ep))

    assert stats["critical"] == 0
    assert stats["risk_signal"] == 1
    assert stats["unverifiable"] == 1
    action_issue = next(i for i in stats["top_issues"] if i["metric_name"] == "action_discontinuity")
    assert action_issue["evidence_level"] == "RISK_SIGNAL"
    assert any(token in action_issue["description"].lower() for token in ("risk signal", "风险信号"))
