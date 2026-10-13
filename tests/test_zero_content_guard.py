"""P0-1 / P0-2 guard tests: "nothing was audited" must never look like PASS.

Background (v0.9.21). Two independent mechanisms let an audit that never
happened report success:

1. ``classify_episode()`` skips non-available results by design. When the
   loader resolves no action / observation / video features, nearly every
   metric goes N/A, no finding is raised, and the verdict falls through to
   PASS — with exit code 0.
2. The CLI only exited non-zero on EXCLUDE, so a 0-episode dataset or a
   dataset where every episode was vacuous also exited 0.

Observed in the wild on the ``laa1991/lerobot-dataset-field-audit`` corpus:
``kitchen``, ``outdoor-depth`` and ``paris_street`` all returned exit 0 with
an empty ``episodes_usable`` and blank idle/occupancy values.

These tests pin the fixed behaviour. If any of them fails, a silent-pass
regression has been reintroduced.

Run:  pytest tests/test_zero_content_guard.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from click.testing import CliRunner  # noqa: E402

from rda.audit.dataset_audit import DatasetAuditor, DatasetInfo  # noqa: E402
from rda.audit.episode_audit import (  # noqa: E402
    EpisodeAuditor,
    has_auditable_content,
)
from rda.audit.execution_tier import ExecutionTier  # noqa: E402
from rda.audit.rules import AuditVerdict  # noqa: E402
from rda.cli.main import cli  # noqa: E402
from rda.io.schema import EpisodeData  # noqa: E402
from rda.report.json_report import _audit_effective, generate_json_report  # noqa: E402

N_FRAMES = 100
FPS = 30
N_JOINTS = 6


def _timestamps(n: int = N_FRAMES) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / float(FPS)


def _clean_action(n: int = N_FRAMES, joints: int = N_JOINTS) -> np.ndarray:
    tt = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack(
        [0.5 * np.sin(2 * np.pi * tt + j * 0.4) for j in range(joints)], axis=-1
    ).astype(np.float32)


def _episode(**kw) -> EpisodeData:
    """Build an EpisodeData with sensible defaults for the common case."""
    base = dict(
        episode_index=0,
        num_frames=N_FRAMES,
        timestamps=_timestamps(),
        observation={"state": _clean_action()},
        action={"action": _clean_action()},
        meta={"fps": FPS},
    )
    base.update(kw)
    return EpisodeData(**base)


# ---------------------------------------------------------------------------
# has_auditable_content — the predicate the whole guard rests on
# ---------------------------------------------------------------------------


def test_action_alone_counts_as_auditable():
    ep = _episode(observation={})
    assert has_auditable_content(ep) is True


def test_observation_alone_counts_as_auditable():
    ep = _episode(action={})
    assert has_auditable_content(ep) is True


def test_declared_video_features_count_as_auditable():
    """A video-only dataset is a legitimate audit target, not an empty one."""
    ep = _episode(
        observation={},
        action={},
        meta={
            "fps": FPS,
            "video_features": {"observation.images.top": {"chunk_index": 0}},
        },
    )
    assert has_auditable_content(ep) is True


def test_no_content_at_all_is_not_auditable():
    ep = _episode(observation={}, action={}, meta={"fps": FPS})
    assert has_auditable_content(ep) is False


def _video_only_episode() -> EpisodeData:
    return _episode(
        observation={},
        action={},
        meta={
            "fps": FPS,
            "video_features": {"observation.images.top": {"chunk_index": 0}},
        },
    )


def test_no_video_tier_makes_video_only_dataset_inauditable():
    """--no-video on a video-only dataset must not silently audit nothing.

    This is the exact trap created by recommending --no-video as the
    workaround for "videos were never downloaded".
    """
    ep = _video_only_episode()
    assert has_auditable_content(ep, ExecutionTier.NO_VIDEO) is False
    assert has_auditable_content(ep, ExecutionTier.FAST) is True


def test_video_only_tier_requires_video_features():
    ep = _video_only_episode()
    assert has_auditable_content(ep, ExecutionTier.VIDEO_ONLY) is True

    robot_only = _episode(meta={"fps": FPS})
    assert has_auditable_content(robot_only, ExecutionTier.VIDEO_ONLY) is False


def test_no_video_tier_still_accepts_robot_features():
    ep = _episode(meta={"fps": FPS})
    assert has_auditable_content(ep, ExecutionTier.NO_VIDEO) is True


def test_auditor_applies_tier_when_guarding():
    """The guard must use the auditor's own tier, not the default."""
    ep = _video_only_episode()
    blocked = EpisodeAuditor(execution_tier=ExecutionTier.NO_VIDEO).audit(ep)
    assert blocked.verdict == AuditVerdict.EXCLUDE
    assert "_zero_content_guard" in blocked.metrics

    allowed = EpisodeAuditor(execution_tier=ExecutionTier.FAST).audit(ep)
    assert "_zero_content_guard" not in allowed.metrics


def test_zero_frames_is_not_auditable():
    ep = _episode(num_frames=0, timestamps=np.zeros(0))
    assert has_auditable_content(ep) is False


# ---------------------------------------------------------------------------
# P0-1 — episode-level verdict guard
# ---------------------------------------------------------------------------


def test_zero_content_episode_is_excluded_not_passed():
    """The core regression: frames present, nothing resolved, must be EXCLUDE."""
    ep = _episode(observation={}, action={}, meta={"fps": FPS})
    result = EpisodeAuditor().audit(ep)

    assert result.verdict == AuditVerdict.EXCLUDE, (
        "zero-content episode fell through to a non-EXCLUDE verdict — "
        "the silent-PASS regression is back"
    )
    assert result.has_auditable_content is False
    assert any("zero_content_guard" in r for r in result.exclude_reasons), (
        f"expected a zero_content_guard reason, got {result.exclude_reasons}"
    )
    assert "_zero_content_guard" in result.metrics


def test_zero_content_guard_carries_a_high_severity_reason():
    ep = _episode(observation={}, action={}, meta={"fps": FPS})
    guard = EpisodeAuditor().audit(ep).metrics["_zero_content_guard"]
    assert guard.assessment["status"] == "fail"
    assert guard.assessment["severity"] == "high"
    assert "not audited" in guard.assessment["reason"]


def test_normal_episode_still_passes():
    """No over-blocking: a healthy episode must be untouched by the guard."""
    result = EpisodeAuditor().audit(_episode())
    assert result.verdict == AuditVerdict.PASS
    assert result.has_auditable_content is True
    assert "_zero_content_guard" not in result.metrics


def test_video_only_episode_is_not_blocked_by_the_guard():
    """Video integrity/temporal metrics are a real audit — do not exclude."""
    ep = _episode(
        observation={},
        action={},
        meta={
            "fps": FPS,
            "video_features": {"observation.images.top": {"chunk_index": 0}},
        },
    )
    result = EpisodeAuditor().audit(ep)
    assert "_zero_content_guard" not in result.metrics
    assert result.has_auditable_content is True


def test_existing_zero_frame_guard_still_wins_for_zero_frames():
    """0-frame episodes keep their own guard and are not double-flagged."""
    ep = _episode(num_frames=0, timestamps=np.zeros(0), observation={}, action={})
    result = EpisodeAuditor().audit(ep)

    assert result.verdict == AuditVerdict.EXCLUDE
    assert "_zero_frame_guard" in result.metrics
    assert "_zero_content_guard" not in result.metrics
    assert result.has_auditable_content is False


# ---------------------------------------------------------------------------
# P0-2 — dataset-level audit_effective and CLI exit code
# ---------------------------------------------------------------------------


def _dataset_result(episodes):
    info = DatasetInfo(path="test/dataset", num_episodes=len(episodes),
                       total_frames=sum(e.num_frames for e in episodes))
    return DatasetAuditor().audit_dataset(info, iter(episodes))


def test_audit_effective_false_when_every_episode_is_empty():
    result = _dataset_result([_episode(observation={}, action={}, meta={"fps": FPS})])
    assert _audit_effective(result) is False
    assert generate_json_report(result)["audit_effective"] is False


def test_audit_effective_false_for_empty_dataset():
    assert _audit_effective(_dataset_result([])) is False


def test_audit_effective_true_when_any_episode_has_content():
    episodes = [
        _episode(episode_index=0, observation={}, action={}, meta={"fps": FPS}),
        _episode(episode_index=1),
    ]
    assert _audit_effective(_dataset_result(episodes)) is True


def test_audit_effective_true_for_normal_dataset():
    result = _dataset_result([_episode(episode_index=0), _episode(episode_index=1)])
    assert _audit_effective(result) is True
    assert generate_json_report(result)["audit_effective"] is True


def test_audit_effective_is_exposed_in_json_report():
    result = _dataset_result([_episode()])
    report = generate_json_report(result)
    assert "audit_effective" in report, (
        "top-level audit_effective missing — consumers can no longer tell "
        "'audited and clean' from 'never audited'"
    )


def test_cli_exits_3_when_no_episode_is_auditable(tmp_path):
    """Exit 3 is the whole point: CI must not read a no-op audit as success."""
    ds = tmp_path / "empty_ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "data").mkdir(parents=True)
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["audit", str(ds), "--no-video", "--format", "json",
         "--output", str(tmp_path / "r.json")],
    )
    # A dataset with no readable episodes must not exit 0.
    assert res.exit_code != 0, (
        "an audit that produced no usable episode exited 0 — "
        "CI would treat this as a pass"
    )
