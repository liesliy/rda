"""REQ-15 / REQ-16 regression: "not judged" must never read as "clean".

The defect these tests pin was asymmetric, and that asymmetry is the point:

  * **nothing** downloaded  -> 50/50 PASS, exit 0  (a false pass)
  * **some** downloaded     -> 1200/1200 EXCLUDE, exit 2 (a false failure)

Both came from one root cause: ``MetricAvailability.NOT_AVAILABLE`` conflated
"this check does not apply" with "this check applies but its input is absent",
and ``classify_episode()`` exempted every non-``AVAILABLE`` metric from the
verdict. Fixing only one side is easy and wrong, so both are pinned here —
together with the boundary case (file present but corrupt must STILL be a hard
EXCLUDE) that keeps the fix from becoming an over-correction.

REQ-16 rides along: the report used to state "Visual quality analysis
included" based purely on the execution-tier constant, regardless of whether
a single frame had been decoded.
"""
from __future__ import annotations

import json

import pytest

from rda.audit.rules import (
    AuditVerdict,
    classify_episode,
    get_unverifiable_reasons,
)
from rda.metrics.base import (
    MetricAvailability,
    MetricResult,
    Verifiability,
)


# ---------------------------------------------------------------------------
# 1. Result-type contract
# ---------------------------------------------------------------------------

class TestMakeUnverifiable:
    def test_marks_verifiability_and_is_not_a_finding(self):
        """Nothing was *found* — the check simply could not run.

        ``has_finding`` must stay False, otherwise a missing download would
        start counting as evidence against the dataset.
        """
        r = MetricResult.make_unverifiable(
            "video_frame_integrity",
            reason="video_files_absent",
            message="declared videos are absent",
        )
        assert r.verifiability == Verifiability.UNVERIFIABLE
        assert r.has_finding is False
        assert r.assessment["status"] == "unverifiable"
        assert r.assessment["reason"] == "video_files_absent"

    def test_default_verifiability_is_backward_compatible(self):
        """Every pre-existing constructor keeps VERIFIABLE without changes.

        This is what makes the change additive: 59 ``make_na()`` call sites
        keep their exact current behaviour.
        """
        for r in (
            MetricResult.make_pass("m", {}),
            MetricResult.make_review("m", {}, reason="r"),
            MetricResult.make_exclude("m", reason="r"),
            MetricResult.make_na("m", reason="r"),
            MetricResult.make_error("m", reason="r"),
        ):
            assert r.verifiability == Verifiability.VERIFIABLE, r.name

    def test_with_reference_score_preserves_verifiability(self):
        """Calibration must not silently reset the flag to VERIFIABLE."""
        r = MetricResult.make_unverifiable("m", reason="absent")
        calibrated = r.with_reference_score(1.5, 0.9)
        assert calibrated.verifiability == Verifiability.UNVERIFIABLE


# ---------------------------------------------------------------------------
# 2. Verdict priority: EXCLUDE > UNVERIFIABLE > REVIEW > PASS
# ---------------------------------------------------------------------------

def _pass(name="m"):
    return MetricResult.make_pass(name, {"score_compat": 1.0})


def _unverifiable(name="m"):
    return MetricResult.make_unverifiable(name, reason="absent")


def _hard_exclude(name="m"):
    return MetricResult.make_exclude(name, reason="broken")


def _review(name="m"):
    return MetricResult.make_review(name, {}, reason="soft")


class TestVerdictPriority:
    def test_only_unverifiable_is_not_pass(self):
        """The original defect, pinned: this used to return PASS."""
        assert classify_episode([_pass(), _unverifiable("joint_limit")]) == (
            AuditVerdict.UNVERIFIABLE
        )

    def test_unverifiable_diagnostic_metric_does_not_affect_verdict(self):
        """Only *critical* checks losing their input can hold back a verdict."""
        assert classify_episode(
            [_pass(), _unverifiable("idle_ratio")],
            critical_metrics=["joint_limit"],
        ) == AuditVerdict.PASS

    def test_exclude_outranks_unverifiable(self):
        """Demonstrably corrupt data is a stronger finding than unknown data."""
        assert classify_episode(
            [
                _hard_exclude("joint_limit"),
                _unverifiable("video_stream_presence"),
            ]
        ) == AuditVerdict.EXCLUDE

    def test_unverifiable_outranks_review(self):
        """Soft findings plus an unchecked area reports "incomplete", not "review"."""
        assert classify_episode(
            [_review("joint_limit"), _unverifiable("video_stream_presence")]
        ) == AuditVerdict.UNVERIFIABLE

    def test_review_still_works_without_unverifiable(self):
        """No regression for the existing REVIEW path."""
        assert classify_episode([_review("joint_limit")]) == AuditVerdict.REVIEW

    def test_clean_episode_is_still_pass(self):
        assert classify_episode([_pass()]) == AuditVerdict.PASS

    def test_not_applicable_is_exempt(self):
        """A check that does not apply must not hold back the verdict."""
        m = MetricResult.make_na("video_stream_presence", reason="single_camera")
        m.verifiability = Verifiability.NOT_APPLICABLE
        assert classify_episode([_pass(), m]) == AuditVerdict.PASS


# ---------------------------------------------------------------------------
# 3. Reason reporting
# ---------------------------------------------------------------------------

class TestUnverifiableReasons:
    def test_lists_metric_and_reason(self):
        reasons = get_unverifiable_reasons(
            [_unverifiable("video_stream_presence"), _pass()]
        )
        assert reasons == ["video_stream_presence (L1): absent"]

    def test_ignores_non_critical_metrics(self):
        reasons = get_unverifiable_reasons(
            [_unverifiable("idle_ratio")], critical_metrics=["joint_limit"]
        )
        assert reasons == []


# ---------------------------------------------------------------------------
# 4. End-to-end: a v2.1 dataset that declares video but ships none
# ---------------------------------------------------------------------------

def _write_dataset_without_videos(root, *, cameras=("observation.images.cam",)):
    """Minimal v2.1 dataset whose schema declares cameras but stores no mp4.

    This is the ``aloha_insertion`` shape: ``meta/info.json`` promises video,
    ``videos/`` is empty. Before REQ-15 it audited as a clean PASS.
    """
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    meta = root / "meta"
    meta.mkdir(exist_ok=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["j0", "j1"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["j0", "j1"]},
    }
    for cam in cameras:
        features[cam] = {
            "dtype": "video",
            "shape": [32, 32, 3],
            "names": ["height", "width", "channels"],
        }

    # 100 frames, not 10: the video span is num_frames/fps while the parquet
    # span is (num_frames-1)/fps, so the relative gap is 1/num_frames. At 10
    # frames that is 11% and trips the timestamp-alignment check (hard tol
    # 0.10), which would mask the behaviour under test. Real datasets have
    # hundreds of frames — aloha_insertion measures 0.002.
    n_frames = 100
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "fps": 10,
        "chunks_size": 1000,
        "features": features,
    }))
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the block"}) + "\n"
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick up the block"],
                    "length": n_frames}) + "\n"
    )

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    pd.DataFrame({
        "episode_index": [0] * n_frames,
        "frame_index": list(range(n_frames)),
        "timestamp": [i / 10.0 for i in range(n_frames)],
        "index": list(range(n_frames)),
        "task_index": [0] * n_frames,
        "action": [list(np.full(2, 0.01 * i)) for i in range(n_frames)],
        "observation.state": [list(np.full(2, 0.01 * i)) for i in range(n_frames)],
    }).to_parquet(data_dir / "episode_000000.parquet")
    return root


class TestEndToEnd:
    def test_declared_video_with_no_files_is_unverifiable(self, tmp_path):
        from rda.audit.dataset_audit import DatasetAuditor
        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset

        ds = _write_dataset_without_videos(tmp_path / "ds")
        info = load_lerobot_dataset(str(ds))
        result = DatasetAuditor().audit_dataset(info, iter_episodes(str(ds)))

        ep = next(iter(result.episodes.values()))
        assert ep.verdict == AuditVerdict.UNVERIFIABLE, (
            f"expected UNVERIFIABLE, got {ep.verdict.value} — a dataset whose "
            f"videos were never downloaded must not audit as clean"
        )
        reasons = get_unverifiable_reasons(list(ep.metrics.values()))
        assert any("video_frame_integrity" in r for r in reasons), reasons

    def test_report_declares_schema_1_5_and_surfaces_unverifiable(self, tmp_path):
        from rda.audit.dataset_audit import DatasetAuditor
        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset
        from rda.report.json_report import generate_json_report

        ds = _write_dataset_without_videos(tmp_path / "ds2")
        info = load_lerobot_dataset(str(ds))
        result = DatasetAuditor().audit_dataset(info, iter_episodes(str(ds)))
        report = generate_json_report(result)

        assert report["report_schema_version"] == "1.5"
        assert report["summary"]["verdict_counts"].get("UNVERIFIABLE") == 1
        assert report["unverifiable_episodes"] == [0]
        assert report["episodes"][0]["unverifiable_reasons"]

    def test_video_quality_note_does_not_claim_analysis(self, tmp_path):
        """REQ-16: the note must reflect what happened, not what was requested."""
        from rda.audit.dataset_audit import DatasetAuditor
        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset
        from rda.report.json_report import generate_json_report

        ds = _write_dataset_without_videos(tmp_path / "ds3")
        info = load_lerobot_dataset(str(ds))
        auditor = DatasetAuditor()
        result = auditor.audit_dataset(info, iter_episodes(str(ds)))
        result.execution_tier = "full"
        report = generate_json_report(result)

        vq = report["video_quality"]
        assert vq["video_quality_executed"] is False
        assert vq["frames_sampled"] == 0
        assert vq["episodes_sampled"] == 0
        assert "included" not in vq["video_quality_note"].lower(), (
            f"report still claims analysis: {vq['video_quality_note']!r}"
        )


# ---------------------------------------------------------------------------
# 5. CLI exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:
    def test_exit_4_when_nothing_could_be_judged(self, tmp_path):
        from click.testing import CliRunner

        from rda.cli.main import cli

        ds = _write_dataset_without_videos(tmp_path / "ds4")
        out = tmp_path / "r.json"
        res = CliRunner().invoke(
            cli, ["audit", str(ds), "--full", "--format", "json", "-o", str(out)]
        )
        assert res.exit_code == 4, (
            f"expected exit 4 (unverifiable), got {res.exit_code}. "
            f"output={res.output[-400:]}"
        )

    def test_continue_on_unverifiable_downgrades_and_records_override(self, tmp_path):
        """The escape hatch must leave a trace, so a forced run is auditable."""
        from click.testing import CliRunner

        from rda.cli.main import cli

        ds = _write_dataset_without_videos(tmp_path / "ds5")
        out = tmp_path / "r5.json"
        res = CliRunner().invoke(
            cli,
            ["audit", str(ds), "--full", "--format", "json", "-o", str(out),
             "--continue-on-unverifiable"],
        )
        assert res.exit_code == 0, res.output[-400:]
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved.get("verifiability_override") is True
        # The verdicts themselves must NOT be rewritten to PASS.
        assert saved["summary"]["verdict_counts"].get("UNVERIFIABLE") == 1


# ---------------------------------------------------------------------------
# REQ-15 (0.12.1): the fourth state must also appear in the *text* report.
#
# 0.12.0 shipped the fourth state in JSON only. The text renderer still looped
# over a hard-coded ["PASS", "REVIEW", "EXCLUDE"], so a dataset where all 50
# episodes were UNVERIFIABLE printed:
#
#     PASS: 0 (0.0%)   REVIEW: 0 (0.0%)   EXCLUDE: 0 (0.0%)
#
# i.e. a table of zeros that reads as "nothing to report", while the exit code
# said 4 and the note at the bottom said the opposite. The verdict table is the
# most-read part of the report; leaving the fourth state out of it is the same
# class of defect REQ-15 exists to kill ("unknown != pass"), just one layer up.
# ---------------------------------------------------------------------------

def test_text_report_shows_unverifiable_row(tmp_path):
    from click.testing import CliRunner

    from rda.cli.main import cli

    ds = _write_dataset_without_videos(tmp_path / "ds_txt")
    res = CliRunner().invoke(cli, ["audit", str(ds), "--format", "text"])
    assert res.exit_code == 4, res.output[-600:]

    verdict_lines = [
        ln for ln in res.output.splitlines()
        if any(k in ln for k in ("PASS:", "REVIEW:", "EXCLUDE:", "UNVERIFIABLE:"))
    ]
    assert any("UNVERIFIABLE:" in ln for ln in verdict_lines), verdict_lines
    # And it must carry the real count, not a placeholder zero.
    unv = [ln for ln in verdict_lines if "UNVERIFIABLE:" in ln][0]
    assert " 1 " in unv or unv.rstrip().split()[-2].isdigit(), unv
    # The zero PASS row must not be mistaken for the whole story.
    assert any("PASS:" in ln and " 0 " in ln for ln in verdict_lines), verdict_lines


def test_text_summary_omits_unverifiable_row_when_zero():
    """No extra all-zero row on healthy datasets — noise erodes the signal."""
    from rda.report.summary import AuditSummary, format_summary_text

    clean = AuditSummary(
        total_episodes=10,
        verdict_counts={"PASS": 10, "REVIEW": 0, "EXCLUDE": 0, "UNVERIFIABLE": 0},
    )
    assert "UNVERIFIABLE" not in format_summary_text(clean)

    dirty = AuditSummary(
        total_episodes=10,
        verdict_counts={"PASS": 0, "REVIEW": 0, "EXCLUDE": 0, "UNVERIFIABLE": 10},
    )
    out = format_summary_text(dirty)
    assert "UNVERIFIABLE" in out
    assert "UNVERIFIABLE   : 10" in out
