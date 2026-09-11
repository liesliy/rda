"""Semantic invariant guard tests for RDA v0.9.

These tests verify that core architectural invariants (INV-003, INV-004,
INV-007) are preserved across code changes. Each test constructs
targeted scenarios to exercise the invariant boundary and asserts
the invariant holds.

Invariants tested:
  INV-003: Inferred/estimated values must never be reported as measured.
  INV-004: Recommendation layer must not mutate audit results.
  INV-007: Verdict can only become EXCLUDE due to L1 integrity gate.

Run:  pytest tests/test_invariant_guards.py -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rda.audit.episode_audit import EpisodeAuditor  # noqa: E402
from rda.audit.rules import AuditVerdict  # noqa: E402
from rda.io.schema import EpisodeData  # noqa: E402
from rda.metrics.base import MetricAvailability  # noqa: E402
from rda.metrics.visual_integrity import VideoStreamTemporalOffsetMetric  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants (mirror golden_set.py)
# ---------------------------------------------------------------------------
N_FRAMES = 120
FPS = 10
N_JOINTS = 6


def _clean_action(n: int = N_FRAMES, seed: int = 42) -> np.ndarray:
    """Smooth, slow, bounded joint trajectory."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    action = np.stack(
        [0.3 * np.sin(2 * np.pi * t / 3.0 + j * 0.4) for j in range(N_JOINTS)],
        axis=1,
    )
    action += rng.normal(0, 0.005, size=(n, N_JOINTS))
    return action.astype(np.float64)


def _timestamps(n: int = N_FRAMES, fps: int = FPS) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / fps


def _joint_limits() -> list:
    return [(-2.0, 2.0)] * N_JOINTS


# =========================================================================
# INV-003: No inference as measurement
# =========================================================================

class TestINV003NoInferenceAsMeasurement:
    """INV-003: Inferred/estimated values must never be reported as measured.

    When the video_stream_temporal_offset metric cannot extract per-frame
    PTS timestamps from video streams, it must NOT fabricate or infer a
    numeric offset. Instead it must return N/A (or measured=False) with
    a reason and suggestion, and must NOT report any numeric offset as
    an actual measurement.
    """

    def test_inv003_no_inference_as_measurement(self):
        """video_stream_temporal_offset must not fabricate offsets when
        frame-level timestamps are unavailable.

        Constructs an episode with video_features metadata referencing
        two camera streams, but no actual video files on disk so
        per-frame PTS extraction is impossible. The metric must return
        N/A and must not report measured=True or any numeric offset.
        """
        ep = EpisodeData(
            episode_index=0,
            num_frames=N_FRAMES,
            timestamps=_timestamps(),
            observation={"state": _clean_action()},
            action={"joint_pos": _clean_action()},
            meta={
                "dataset_root": "/nonexistent/path/for/test",
                "fps": FPS,
                "joint_limits": _joint_limits(),
                "video_features": {
                    "cam_a": {
                        "chunk_index": 0,
                        "file_index": 0,
                        "from_timestamp": 0.0,
                        "to_timestamp": N_FRAMES / FPS,
                    },
                    "cam_b": {
                        "chunk_index": 0,
                        "file_index": 0,
                        "from_timestamp": 0.0,
                        "to_timestamp": N_FRAMES / FPS,
                    },
                },
            },
        )

        metric = VideoStreamTemporalOffsetMetric()
        result = metric.compute(ep)

        # --- Core invariant: NOT measured ---
        measurement = result.measurement
        if "measured" in measurement:
            assert measurement["measured"] is False, (
                "INV-003 VIOLATION: measured must be False when frame-level "
                "timestamps are unavailable; got True -- the metric may be "
                "reporting an inferred offset as if it were a real measurement."
            )

        # --- Must not report any numeric offset as an actual value ---
        numeric_offset_keys = {
            "worst_p95_offset_ms", "worst_max_offset_ms",
            "offset_median_ms", "offset_p95_ms", "offset_p99_ms",
            "offset_max_ms",
        }
        reported_numeric = [
            k for k in numeric_offset_keys
            if k in measurement
            and isinstance(measurement[k], (int, float))
        ]
        assert not reported_numeric, (
            f"INV-003 VIOLATION: numeric offset values reported without real "
            f"measurement: {reported_numeric}. Inferred values must never be "
            f"presented as measured data."
        )

        # --- Must have an explanation (reason) ---
        assert result.availability == MetricAvailability.NOT_AVAILABLE, (
            f"INV-003: metric should be N/A when timestamps unavailable, "
            f"got {result.availability.value}"
        )
        reason = result.assessment.get("reason", "")
        assert reason, (
            "INV-003: assessment must include a reason field explaining "
            "why the measurement could not be performed."
        )

        # --- Must include a suggestion (in the message text) ---
        msg = result.message.lower()
        has_suggestion = any(
            kw in msg
            for kw in ("suggestion", "re-record", "enable", "re-encode",
                        " pts", "encoder", "hardware")
        )
        assert has_suggestion or "timestamp" in msg, (
            "INV-003: metric must provide a suggestion or guidance when "
            "measurement is not possible (e.g. how to obtain timestamps)."
        )


# =========================================================================
# INV-004: Recommendation does not mutate audit
# =========================================================================

class TestINV004RecommendDoesNotMutateAudit:
    """INV-004: Recommendation layer must not modify audit results.

    The recommendation pipeline (offline fallback or API) must treat the
    audit result as read-only. No field in the EpisodeAuditResult --
    verdict, measurement values, or findings -- may be altered by the
    recommendation layer.
    """

    def test_inv004_recommend_does_not_mutate_audit(self):
        """Run full audit -> deep copy -> invoke recommendation offline path
        -> verify audit result is byte-for-byte identical afterwards.
        """
        # 1. Build clean episode and run audit
        ep = EpisodeData(
            episode_index=0,
            num_frames=N_FRAMES,
            timestamps=_timestamps(),
            observation={"state": _clean_action()},
            action={"joint_pos": _clean_action()},
            meta={
                "fps": FPS,
                "source": "golden-clean",
                "joint_limits": _joint_limits(),
            },
        )
        auditor = EpisodeAuditor()
        result_before = auditor.audit(ep)

        # 2. Deep copy as baseline
        baseline_verdict = result_before.verdict
        baseline_metrics = copy.deepcopy(
            {n: {
                "measurement": dict(m.measurement),
                "assessment": dict(m.assessment),
                "has_finding": m.has_finding,
                "availability": m.availability,
            }
            for n, m in result_before.metrics.items()}
        )

        # 3. Run the recommendation offline fallback path.
        from rda.recommend.local_fallback import build_offline_result
        from rda.recommend.types import TargetPolicy
        from rda.recommend.temporal_metrics import (
            compute_temporal_sufficiency,
            aggregate_temporal_sufficiency,
        )

        ts = compute_temporal_sufficiency(ep)
        agg = aggregate_temporal_sufficiency(
            [ts], total_episodes=1, total_frames=ep.num_frames
        )
        rec_result = build_offline_result(
            agg, TargetPolicy.FRAME_WISE, lang="en"
        )

        # Verify recommendation was actually produced
        assert rec_result is not None
        assert rec_result.rules_version == "offline-fallback"

        # 4. Verify audit result is completely unchanged
        # 4a. Verdict unchanged
        assert result_before.verdict == baseline_verdict, (
            f"INV-004 VIOLATION: verdict changed from {baseline_verdict} "
            f"to {result_before.verdict} after recommendation pipeline."
        )

        # 4b. Every metric measurement and assessment unchanged
        for name, baseline in baseline_metrics.items():
            actual = result_before.metrics.get(name)
            assert actual is not None, (
                f"INV-004 VIOLATION: metric {name!r} was removed from "
                f"audit result by recommendation pipeline."
            )
            assert actual.measurement == baseline["measurement"], (
                f"INV-004 VIOLATION: metric {name!r} measurement mutated "
                f"by recommendation pipeline: "
                f"{baseline['measurement']} -> {actual.measurement}"
            )
            assert actual.assessment == baseline["assessment"], (
                f"INV-004 VIOLATION: metric {name!r} assessment mutated "
                f"by recommendation pipeline."
            )
            assert actual.has_finding == baseline["has_finding"], (
                f"INV-004 VIOLATION: metric {name!r} has_finding flag "
                f"changed by recommendation pipeline."
            )

        # 4c. No new metrics were injected
        assert set(result_before.metrics.keys()) == set(baseline_metrics.keys()), (
            "INV-004 VIOLATION: recommendation pipeline added or removed "
            "metrics from the audit result."
        )


# =========================================================================
# INV-007: EXCLUDE only from L1
# =========================================================================

class TestINV007ExcludeOnlyFromL1:
    """INV-007: Verdict can only become EXCLUDE due to L1 integrity gate.

    Diagnostic metrics (Layer 2) -- such as action_discontinuity,
    idle_ratio, sampling_jitter -- produce findings but must NEVER
    cause the episode verdict to escalate from PASS to REVIEW or
    EXCLUDE, even when multiple diagnostics fire simultaneously.
    """

    def test_inv007_exclude_only_from_l1(self):
        """Construct an episode where L1 integrity metrics all PASS but
        multiple L2/L3 diagnostic metrics show severe anomalies.

        The verdict MUST remain PASS. The diagnostics must still produce
        non-trivial measurements (proving the anomalies were detected),
        but they must not influence the verdict.
        """
        # --- Build action with frequent teleport spikes ---
        # Same pattern as golden_set _sc_action_spikes: spikes every 3 frames
        # ensure action_discontinuity detects them via MAD-based z-score.
        action = _clean_action(seed=200)
        for f in range(5, N_FRAMES - 1, 3):
            action[f, :] += 10.0

        # --- Frozen arm observation for idle_ratio ---
        obs_state = np.full((N_FRAMES, N_JOINTS), 0.123, dtype=np.float64)

        # --- Sampling jitter: alternating fast/slow timestamps ---
        # CV > 0.3, timestamps still monotonically increasing (L1 passes).
        factors = np.where(np.arange(N_FRAMES) % 2 == 0, 1.6, 0.4)
        dts = 0.1 * factors[: N_FRAMES - 1]
        ts = np.concatenate([[0.0], np.cumsum(dts)]).astype(np.float64)

        ep = EpisodeData(
            episode_index=0,
            num_frames=N_FRAMES,
            timestamps=ts,
            observation={"state": obs_state},
            action={"joint_pos": action},
            meta={
                "fps": FPS,
                "source": "inv007-multi-diagnostic",
                "joint_limits": _joint_limits(),
            },
        )

        # Run full audit
        auditor = EpisodeAuditor()
        result = auditor.audit(ep)

        # --- INV-007 core invariant: verdict must stay PASS ---
        assert result.verdict == AuditVerdict.PASS, (
            f"INV-007 VIOLATION: verdict is {result.verdict.value} but all "
            f"L1 integrity metrics should pass. Diagnostic anomalies "
            f"(action spikes + frozen arm + jitter) must NOT escalate "
            f"the verdict to REVIEW or EXCLUDE."
        )

        # --- Verify diagnostics actually detected the anomalies ---
        # At least one diagnostic must show non-trivial measurements
        disc = result.metrics.get("action_discontinuity")
        jitter = result.metrics.get("sampling_jitter")
        idle = result.metrics.get("idle_ratio")

        diagnostics_detected = False

        if disc is not None and disc.availability == MetricAvailability.AVAILABLE:
            spike_count = disc.measurement.get("spike_count", 0)
            if spike_count > 0:
                diagnostics_detected = True

        if jitter is not None and jitter.availability == MetricAvailability.AVAILABLE:
            cv = jitter.measurement.get("cv", jitter.measurement.get("jitter_ratio", 0))
            if cv > 0.1:
                diagnostics_detected = True

        if idle is not None and idle.availability == MetricAvailability.AVAILABLE:
            emr = idle.measurement.get("effective_motion_ratio", 1.0)
            if emr < 0.5:
                diagnostics_detected = True

        assert diagnostics_detected, (
            "INV-007 test setup error: none of the targeted diagnostic "
            "anomalies were detected. The test data may not be extreme "
            "enough to trigger the diagnostics."
        )

        # --- Verify no L1 metric was triggered by the diagnostic anomalies ---
        from rda.audit.rules import CRITICAL_METRICS
        for name in CRITICAL_METRICS:
            m = result.metrics.get(name)
            if m is None:
                continue
            if m.availability != MetricAvailability.AVAILABLE:
                continue
            status_val = m.assessment.get("status")
            assert status_val not in ("exclude", "review"), (
                f"INV-007: L1 critical metric {name!r} unexpectedly triggered "
                f"with status={status_val!r} "
                f"on a diagnostic-only anomaly episode."
            )


# =========================================================================
# INV-005: Thresholds are configurable
# =========================================================================

class TestINV005ThresholdsConfigurable:
    """INV-005: Heuristic thresholds must be configurable.

    All heuristic thresholds must be configurable via __init__ parameters,
    and the actual used values must be reflected in the result details.
    """

    def test_inv005_thresholds_are_configurable(self):
        """JointLimitMetric must accept and use non-default thresholds."""
        from rda.metrics.motion import JointLimitMetric

        # Non-default parameters
        custom_approach = 0.05
        custom_consecutive = 5
        custom_jump_mult = 3.0

        metric = JointLimitMetric(
            approach_threshold=custom_approach,
            consecutive_frames=custom_consecutive,
            jump_multiplier=custom_jump_mult,
        )

        # Build episode with joint values approaching limits
        action = _clean_action()
        obs_state = np.zeros((N_FRAMES, N_JOINTS), dtype=np.float64)
        # Set joint 0 to approach limit: 2.0 - 0.08 = 1.92 (margin = 0.04 < 0.05)
        obs_state[:, 0] = 1.92

        ep = EpisodeData(
            episode_index=0,
            num_frames=N_FRAMES,
            timestamps=_timestamps(),
            observation={"state": obs_state},
            action={"joint_pos": action},
            meta={"fps": FPS, "joint_limits": _joint_limits()},
        )

        result = metric.compute(ep)

        # Verify custom thresholds were used (stored in details dict)
        assert result.details.get("approach_threshold_used") == custom_approach, (
            f"INV-005: approach_threshold not configurable, "
            f"expected {custom_approach}, got {result.details.get('approach_threshold_used')}"
        )
        assert result.details.get("consecutive_frames_used") == custom_consecutive, (
            f"INV-005: consecutive_frames not configurable"
        )
        assert result.details.get("jump_multiplier_used") == custom_jump_mult, (
            f"INV-005: jump_multiplier not configurable"
        )


# =========================================================================
# INV-006: Report contains version trace
# =========================================================================

class TestINV006ReportVersionTrace:
    """INV-006: Reports must contain version and configuration trace.

    Every audit report must include RDA version, report schema version,
    and the configuration actually used, so results are reproducible.
    """

    def test_inv006_report_contains_version_trace(self):
        """JSON report must contain report_schema_version and tool_version."""
        import rda
        from rda.audit.dataset_audit import DatasetAuditor, DatasetInfo
        from rda.report.json_report import generate_json_report

        ep = EpisodeData(
            episode_index=0,
            num_frames=N_FRAMES,
            timestamps=_timestamps(),
            observation={"state": _clean_action()},
            action={"joint_pos": _clean_action()},
            meta={"fps": FPS, "joint_limits": _joint_limits()},
        )

        dataset_info = DatasetInfo(
            path="test/dataset",
            num_episodes=1,
            total_frames=N_FRAMES,
        )

        auditor = DatasetAuditor()
        result = auditor.audit_dataset(dataset_info, iter([ep]))

        # Generate JSON report
        report = generate_json_report(result)

        # Check required version fields
        assert "report_schema_version" in report, (
            "INV-006: report missing 'report_schema_version' field"
        )
        assert report["report_schema_version"] == "1.1", (
            f"INV-006: unexpected schema version {report['report_schema_version']}"
        )

        assert "tool_version" in report or "rda_version" in report, (
            "INV-006: report missing tool/rda version field"
        )

        version_field = report.get("tool_version") or report.get("rda_version")
        assert version_field == rda.__version__, (
            f"INV-006: version mismatch, expected {rda.__version__}, got {version_field}"
        )


# =========================================================================
# INV-008: Spike detection uses MAD
# =========================================================================

class TestINV008SpikeDetectionUsesMAD:
    """INV-008: Statistical anomaly detection must use MAD, not fixed σ.

    Diagnostic metrics (velocity_acceleration, action_discontinuity) must
    use reference + MAD z-score for spike detection, not standard deviation
    or fixed absolute thresholds.
    """

    def test_inv008_spike_detection_uses_mad(self):
        """Construct known distribution and verify spike_count matches
        MAD z-score expectation, not std-based expectation.
        """
        from rda.metrics.motion import ActionDiscontinuityMetric

        # Build action with known spike pattern:
        # - 100 frames of small uniform motion (delta = 0.1)
        # - 3 frames of huge spikes (delta = 100.0)
        n_normal = 100
        n_spike = 3
        n_total = n_normal + n_spike + 1  # +1 because diff reduces by 1

        action = np.zeros((n_total, N_JOINTS), dtype=np.float64)
        # Normal frames: small constant motion
        for i in range(1, n_normal + 1):
            action[i, :] = action[i-1, :] + 0.1
        # Spike frames: huge jumps
        for i in range(n_normal + 1, n_total):
            action[i, :] = action[i-1, :] + 100.0

        ep = EpisodeData(
            episode_index=0,
            num_frames=n_total,
            timestamps=np.arange(n_total, dtype=np.float64) / FPS,
            observation={"state": action},
            action={"joint_pos": action},
            meta={"fps": FPS, "joint_limits": _joint_limits()},
        )

        metric = ActionDiscontinuityMetric()
        result = metric.compute(ep)

        # Calculate expected spike_count using MAD
        deltas = np.diff(action, axis=0)
        delta_norms = np.linalg.norm(deltas, axis=1)
        median_val = np.median(delta_norms)
        mad_val = np.median(np.abs(delta_norms - median_val))
        if mad_val > 0:
            z_scores = 0.6745 * np.abs(delta_norms - median_val) / mad_val
            expected_mad_spikes = int(np.sum(z_scores > 5.0))
        else:
            expected_mad_spikes = 0

        # Calculate what std-based would give (should be different)
        std_val = np.std(delta_norms)
        if std_val > 0:
            z_std = np.abs(delta_norms - np.mean(delta_norms)) / std_val
            expected_std_spikes = int(np.sum(z_std > 5.0))
        else:
            expected_std_spikes = 0

        actual_spikes = result.measurement.get("spike_count", 0)

        # The actual spike count should match MAD expectation
        assert actual_spikes == expected_mad_spikes, (
            f"INV-008: spike_count={actual_spikes} doesn't match MAD expectation "
            f"({expected_mad_spikes}). Metric may not be using MAD z-score."
        )

        # If MAD and std give different results, we've proven it's MAD
        if expected_mad_spikes != expected_std_spikes:
            assert actual_spikes != expected_std_spikes, (
                f"INV-008: spike_count matches std expectation ({expected_std_spikes}) "
                f"not MAD ({expected_mad_spikes}). Metric may be using std instead of MAD."
            )


# =========================================================================
# INV-009: Verifiability levels not mixed
# =========================================================================

class TestINV009VerifiabilityNotMixed:
    """INV-009: Verifiability levels must not be mixed.

    The four levels (VERIFIED, MEASURED, NOT_AVAILABLE, N/A) must have
    distinct semantics and not be used interchangeably.
    """

    def test_inv009_verifiability_levels_not_mixed(self):
        """Verify all metrics report valid availability values and that
        video metrics on no-video episodes are NOT_AVAILABLE, not PASS.
        """
        # Build episode WITHOUT video
        ep = EpisodeData(
            episode_index=0,
            num_frames=N_FRAMES,
            timestamps=_timestamps(),
            observation={"state": _clean_action()},
            action={"joint_pos": _clean_action()},
            meta={"fps": FPS, "joint_limits": _joint_limits()},
        )

        auditor = EpisodeAuditor()
        result = auditor.audit(ep)

        # Valid availability values
        valid_availabilities = {
            MetricAvailability.AVAILABLE,
            MetricAvailability.NOT_AVAILABLE,
        }

        video_metric_names = [
            "video_freeze", "video_timestamp_alignment",
            "video_stream_presence", "video_stream_span_consistency",
            "video_stream_temporal_offset", "video_stream_temporal_drift",
            "video_frame_integrity",
        ]

        for name, metric_result in result.metrics.items():
            # Check availability is valid
            assert metric_result.availability in valid_availabilities, (
                f"INV-009: metric {name!r} has invalid availability "
                f"{metric_result.availability}"
            )

            # Video metrics on no-video episode must be NOT_AVAILABLE
            if name in video_metric_names:
                assert metric_result.availability == MetricAvailability.NOT_AVAILABLE, (
                    f"INV-009: video metric {name!r} on no-video episode should be "
                    f"NOT_AVAILABLE, got {metric_result.availability}. "
                    f"This may indicate verifiability levels are being mixed."
                )

        # Verify at least one non-video metric is AVAILABLE
        non_video_available = [
            name for name, m in result.metrics.items()
            if name not in video_metric_names
            and m.availability == MetricAvailability.AVAILABLE
        ]
        assert len(non_video_available) > 0, (
            "INV-009: no non-video metrics are AVAILABLE on a valid episode"
        )
