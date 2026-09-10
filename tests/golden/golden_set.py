"""Deterministic synthetic golden dataset for RDA v0.9 regression testing.

This module builds a fixed catalog of synthetic episodes with KNOWN ground
truth. Every episode is constructed so that exactly one (or a documented
set of) behaviour is triggered, and the expected verdict is pinned in
``tests/test_golden_regression.py``.

Design goals
------------
* **Deterministic**: fixed RNG seeds everywhere — a green run stays green.
* **End-to-end**: episodes are fed through the *real* metric pipeline
  (no mocked MetricResult objects), so this exercises the full path from
  raw arrays -> MetricResult -> verdict -> dataset summary.
* **Two tiers**:
    - non-video scenarios (always run, no heavy deps),
    - video scenarios (skipped cleanly when PyAV is unavailable).

The catalog is intentionally small and hand-tuned. It is NOT a substitute
for the per-metric unit tests; it is a *regression anchor* that proves the
v0.9 four-layer architecture behaves as specified across whole episodes.

Scenario contract
-----------------
Each scenario is a dict::

    {
        "id": str,                     # stable id, used in expected table
        "description": str,            # human-readable intent
        "episode": EpisodeData,
        "video": bool,                 # requires PyAV / real mp4 files
        "expected_verdict": str,       # "pass" | "review" | "exclude"
        "must_trigger": [metric_name], # metrics that MUST report a finding
        "must_not_trigger": [metric],  # metrics that MUST NOT hard-fail
        "measurements": {name: {key: (lo, hi)}},  # optional numeric bounds
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rda.io.schema import EpisodeData

# A fixed frame count for the non-video episodes.
N_FRAMES = 120
FPS = 10
N_JOINTS = 6


# ---------------------------------------------------------------------------
# Base episode construction
# ---------------------------------------------------------------------------

def _timestamps(n: int = N_FRAMES, fps: int = FPS) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / fps


def _clean_action(n: int = N_FRAMES, seed: int = 42) -> np.ndarray:
    """Smooth, slow, bounded joint trajectory — every metric should pass."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    # Smooth sinusoid sweep, small per-joint phase offset, tiny noise.
    action = np.stack(
        [0.3 * np.sin(2 * np.pi * t / 3.0 + j * 0.4) for j in range(N_JOINTS)],
        axis=1,
    )
    action += rng.normal(0, 0.005, size=(n, N_JOINTS))
    return action.astype(np.float64)


def _clean_episode(episode_index: int = 0) -> EpisodeData:
    action = _clean_action(seed=100 + episode_index)
    return EpisodeData(
        episode_index=episode_index,
        num_frames=N_FRAMES,
        timestamps=_timestamps(),
        observation={"state": action.copy()},
        action={"joint_pos": action.copy()},
        meta={"fps": FPS, "source": "golden-clean"},
    )


def _joint_limits_meta() -> Dict[str, Any]:
    """Permissive ±2 rad limits; clean trajectory is well inside.

    JointLimitMetric expects ``meta["joint_limits"]`` as an indexable
    collection of (low, high) tuples, one per joint.
    """
    return {"joint_limits": [(-2.0, 2.0)] * N_JOINTS}


# ---------------------------------------------------------------------------
# Scenario builders (non-video)
# ---------------------------------------------------------------------------

def _sc_clean() -> EpisodeData:
    ep = _clean_episode()
    ep.meta["joint_limits"] = _joint_limits_meta()["joint_limits"]
    return ep


def _sc_nan_values() -> EpisodeData:
    """Hard integrity failure: NaN injected into an action feature -> exclude."""
    ep = _clean_episode()
    a = ep.action["joint_pos"].copy()
    a[30:40, 2] = np.nan
    a[60, 0] = np.inf
    ep.action["joint_pos"] = a
    return ep


def _sc_joint_limit_violation() -> EpisodeData:
    """Critical metric REVIEW: joint command exceeds mechanical limit.

    JointLimitMetric reads ``observation.state`` (not action) and emits a
    REVIEW-level finding for a large run of violations (>5 frames). This is
    a critical metric but reports review status, so the verdict is REVIEW
    (not EXCLUDE).
    """
    ep = _clean_episode()
    violation = 5.0  # well beyond ±2
    for store in (ep.observation, ep.action):
        key = "state" if store is ep.observation else "joint_pos"
        a = store[key].copy()
        a[45:70, 3] = violation
        store[key] = a
    ep.meta["joint_limits"] = _joint_limits_meta()["joint_limits"]
    return ep


def _sc_missing_frames() -> EpisodeData:
    """Critical finding: one feature has fewer frames than the reference
    (sensor dropout / truncated stream). MissingFramesMetric compares
    per-feature array lengths against the reference feature, so we add a
    truncated secondary observation."""
    ep = _clean_episode()
    short = ep.observation["state"][:80].copy()  # 40 frames dropped
    ep.observation["secondary_arm"] = short
    return ep


def _sc_frozen_arm() -> EpisodeData:
    """Diagnostic only: arm held perfectly still -> idle finding, verdict stays PASS.

    Exact constant array (zero per-frame delta) so the data-driven idle
    threshold reliably classifies nearly all frames as idle ->
    effective_motion_ratio ~ 0 (< 0.1 severity band).
    """
    ep = _clean_episode()
    flat = np.full((N_FRAMES, N_JOINTS), 0.123, dtype=np.float64)
    ep.action["joint_pos"] = flat
    ep.observation["state"] = flat.copy()
    return ep


def _sc_action_spikes() -> EpisodeData:
    """Diagnostic only: many discontinuity spikes -> finding, verdict stays PASS.

    Repeated teleport jumps produce a large spike_count which trips the
    high-severity action_discontinuity band (>100).
    """
    ep = _clean_episode()
    a = ep.action["joint_pos"].copy()
    for f in range(5, N_FRAMES - 1, 3):
        a[f, :] += 10.0
    ep.action["joint_pos"] = a
    return ep


def _sc_sampling_jitter() -> EpisodeData:
    """Diagnostic only: highly irregular timestamps -> jitter finding, verdict pass."""
    ep = _clean_episode()
    # Alternate ~0.16s / 0.04s steps (mean ~0.1s) -> coefficient of variation
    # > 0.3, which trips the severe sampling-irregularity band.
    factors = np.where(np.arange(N_FRAMES) % 2 == 0, 1.6, 0.4)
    dts = 0.1 * factors[: N_FRAMES - 1]
    ts = np.concatenate([[0.0], np.cumsum(dts)]).astype(np.float64)
    ep.timestamps = ts
    return ep


# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldenScenario:
    id: str
    description: str
    builder: Callable[[], EpisodeData]
    video: bool = False
    expected_verdict: str = "pass"
    must_trigger: tuple = ()
    must_not_hard_fail: tuple = ()
    measurement_bounds: Optional[Dict[str, Dict[str, tuple]]] = None


# Non-video scenarios — always available.
NON_VIDEO_SCENARIOS: List[GoldenScenario] = [
    GoldenScenario(
        id="clean_baseline",
        description="Smooth bounded trajectory; every available metric passes.",
        builder=_sc_clean,
        expected_verdict="pass",
    ),
    GoldenScenario(
        id="nan_inf_values",
        description="NaN + Inf cells in an action feature (L1 integrity).",
        builder=_sc_nan_values,
        expected_verdict="exclude",
        must_trigger=("invalid_values",),
    ),
    GoldenScenario(
        id="joint_limit_violation",
        description="Joint command exceeds declared mechanical limit (critical metric, REVIEW-level).",
        builder=_sc_joint_limit_violation,
        expected_verdict="review",
        must_trigger=("joint_limit",),
    ),
    GoldenScenario(
        id="missing_frames_dropout",
        description="Secondary sensor stream truncated: feature-length mismatch (L1 integrity).",
        builder=_sc_missing_frames,
        # Length mismatch fires missing_dropout (review) AND schema_consistency
        # (hard exclude); verdict is exclude.
        expected_verdict="exclude",
        must_trigger=("missing_dropout",),
    ),
    GoldenScenario(
        id="frozen_arm_diagnostic",
        description="Arm nearly stationary: idle_ratio finding only, verdict must stay PASS.",
        builder=_sc_frozen_arm,
        expected_verdict="pass",
        must_trigger=("idle_ratio",),
    ),
    GoldenScenario(
        id="action_spikes_diagnostic",
        description="Teleport spikes: action_discontinuity finding only, verdict stays PASS.",
        builder=_sc_action_spikes,
        expected_verdict="pass",
        must_trigger=("action_discontinuity",),
    ),
    GoldenScenario(
        id="sampling_jitter_diagnostic",
        description="Irregular timestamps: sampling_jitter finding only, verdict stays PASS.",
        builder=_sc_sampling_jitter,
        expected_verdict="pass",
        must_trigger=("sampling_jitter",),
    ),
]


# ---------------------------------------------------------------------------
# Video scenarios — built lazily because they need PyAV + real mp4 files.
# ---------------------------------------------------------------------------

def _write_video(
    path, n_frames: int = 32, size: int = 64, fps: int = FPS,
    freeze: Optional[tuple] = None, moving: bool = True,
) -> None:
    """Write a tiny MP4 (mirrors the helper in test_req4_visual_integrity)."""
    import av  # local import; only called when PyAV is present.

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.width, stream.height = size, size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        stream.gop_size = n_frames
        cur = base.copy()
        for i in range(n_frames):
            if moving and (freeze is None or not (freeze[0] <= i <= freeze[1])):
                cur = np.roll(cur, 4, axis=1)
            frame = av.VideoFrame.from_ndarray(cur, format="gray")
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def _video_episode(
    root,
    index: int,
    cameras: Dict[str, Optional[tuple]],
    num_frames: int = 64,
    span_ratio: float = 1.0,
    delete_cam: Optional[str] = None,
) -> EpisodeData:
    from pathlib import Path

    root = Path(root)
    video_features: Dict[str, Any] = {}
    for cam, freeze in cameras.items():
        fpath = root / "videos" / cam / "chunk-000" / "file-000.mp4"
        _write_video(fpath, n_frames=num_frames, freeze=freeze, fps=FPS)
        if delete_cam == cam:
            fpath.unlink()
        video_features[cam] = {
            "chunk_index": 0,
            "file_index": 0,
            "from_timestamp": 0.0,
            # Frame-span convention: expected video frame count = span*fps,
            # which must equal num_frames -> span = num_frames / fps.
            "to_timestamp": num_frames / FPS * span_ratio,
        }
    t = np.arange(num_frames, dtype=np.float64) / FPS
    # Smooth large-amplitude sweep: continuous per-frame motion keeps
    # idle_ratio in the active band; ±1.2 rad stays inside the ±2 limits.
    tt = np.linspace(0, 1, num_frames)
    action = np.stack(
        [1.2 * np.sin(2 * np.pi * 3 * tt + j * 0.5) for j in range(4)],
        axis=1,
    ).astype(np.float64)
    return EpisodeData(
        episode_index=index,
        num_frames=num_frames,
        timestamps=t,
        observation={"state": action.copy()},
        action={"joint_pos": action.copy()},
        meta={
            "dataset_root": str(root),
            "fps": FPS,
            "joint_limits": [(-2.0, 2.0)] * 4,
            "video_features": video_features,
        },
    )


def build_video_scenarios(tmp_path) -> List[GoldenScenario]:
    """Video golden scenarios; only call when PyAV is importable."""
    from pathlib import Path

    base = Path(tmp_path)

    def _clean_multi():
        return _video_episode(base / "v_clean", 0,
                              {"cam_a": None, "cam_b": None})

    def _missing_cam():
        return _video_episode(base / "v_missing", 1,
                              {"cam_a": None, "cam_b": None},
                              delete_cam="cam_b")

    def _frozen_video():
        # 20-frame stall of 32 -> conclusive freeze (L1 exclude).
        return _video_episode(base / "v_freeze", 2,
                              {"cam": (5, 25)})

    def _timestamp_drift():
        # Declared video span 30% longer than parquet timeline -> exclude.
        return _video_episode(base / "v_drift", 3,
                              {"cam": None}, span_ratio=1.3)

    return [
        GoldenScenario(
            id="video_clean_multi_cam",
            description="Two present, in-sync moving cameras; all video checks pass.",
            builder=_clean_multi,
            video=True,
            expected_verdict="pass",
        ),
        GoldenScenario(
            id="video_missing_camera_stream",
            description="One camera file deleted: silent modality loss (L1 exclude).",
            builder=_missing_cam,
            video=True,
            expected_verdict="exclude",
            must_trigger=("video_stream_presence",),
        ),
        GoldenScenario(
            id="video_long_freeze",
            description="Long frozen span in video (L1 video_freeze exclude).",
            builder=_frozen_video,
            video=True,
            expected_verdict="exclude",
            must_trigger=("video_freeze",),
        ),
        GoldenScenario(
            id="video_timestamp_drift",
            description="Declared video span mismatches timeline (L1 exclude).",
            builder=_timestamp_drift,
            video=True,
            expected_verdict="exclude",
            must_trigger=("video_timestamp_alignment",),
        ),
    ]


def all_scenarios(tmp_path=None, include_video: bool = True) -> List[GoldenScenario]:
    """Return the full golden catalog (non-video always; video when available).

    Every scenario is wrapped so its built episode gets a unique
    ``episode_index``; without this, dataset-level aggregation overwrites
    episodes sharing index 0.
    """
    scenarios = list(NON_VIDEO_SCENARIOS)
    if include_video:
        try:
            import av  # noqa: F401
        except Exception:
            pass
        else:
            if tmp_path is not None:
                scenarios.extend(build_video_scenarios(tmp_path))

    wrapped: List[GoldenScenario] = []
    for idx, sc in enumerate(scenarios):
        original_builder = sc.builder

        def _make(b=original_builder, i=idx):
            ep = b()
            ep.episode_index = i
            return ep

        wrapped.append(GoldenScenario(
            id=sc.id,
            description=sc.description,
            builder=_make,
            video=sc.video,
            expected_verdict=sc.expected_verdict,
            must_trigger=tuple(sc.must_trigger),
            must_not_hard_fail=tuple(sc.must_not_hard_fail),
            measurement_bounds=dict(sc.measurement_bounds or {}),
        ))
    return wrapped
