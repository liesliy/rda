"""UI visual-audit panels (v0.7.x UI catch-up, REQ-4).

Three layers of protection against silent regressions:

1. ``compute_visual_stats`` dataset-level aggregation (pure function,
   no streamlit needed) — flagged counting, dep-missing counting.
2. Streamlit AppTest runs of the real pages (only when streamlit with
   ``testing`` support is installed; skipped otherwise so CI without
   the [ui] extra stays green).
3. Semantic guarantees: dep-missing never renders as pass; NA metrics
   hide the quality breakdown instead of faking empty tables.

Real-dataset paths below use libero_10 when available; otherwise the
metric results are synthesized (the UI contract is what is under test,
not the loader).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- streamlit stub for the pure-function layer (import-time only) ---
if "streamlit" not in sys.modules:
    _st = types.ModuleType("streamlit")
    _st.session_state = {}
    _st.radio = lambda *a, **k: None
    sys.modules["streamlit"] = _st

import pytest  # noqa: E402

from rda.audit.dataset_audit import DatasetAuditResult  # noqa: E402
from rda.io.schema import DatasetInfo  # noqa: E402
from rda.metrics.base import MetricAvailability, MetricResult  # noqa: E402
from rda.metrics.visual_integrity import VIDEO_DEPS_MISSING  # noqa: E402

DS_ROOT = Path(r"D:\workbuddy-data\datasets\libero_10")


def _make_result(n: int = 2) -> DatasetAuditResult:
    from rda.audit.dataset_audit import DatasetAuditResult as DAR
    from rda.audit.episode_audit import EpisodeAuditResult

    info = DatasetInfo(path=str(DS_ROOT), num_episodes=n, total_frames=n * 100)
    result = DAR(dataset_info=info)
    for i in range(n):
        result.episodes[i] = EpisodeAuditResult(episode_index=i, num_frames=100)
    result.compute_verdict_counts()
    return result


def _freeze_exclude() -> MetricResult:
    return MetricResult.make_exclude(
        name="video_freeze",
        reason="video_freeze_while_moving",
        message="froze 2 time(s)",
        details={
            "checked_features": ["observation.images.image"],
            "freeze_regions": [
                {"feature": "observation.images.image", "parquet_start": 10,
                 "parquet_end": 40, "duration_sec": 1.2, "moving_ratio_in_span": 0.8},
            ],
            "freeze_region_count": 1,
        },
    )


def _vq_review() -> MetricResult:
    return MetricResult.make_review(
        name="visual_quality",
        measurement={"penalty": 0.9, "worst_feature": "cam0"},
        message="penalty 0.90 (blur)",
        details={
            "per_feature": {"cam0": {"penalty": 0.9, "dominant_issue": "blur"}},
            "sample_count": 4,
            "thresholds": {},
        },
    )


def _na_all() -> dict:
    return {
        name: MetricResult.make_na(
            name=name, reason=VIDEO_DEPS_MISSING, message="PyAV missing.")
        for name in ("video_freeze", "video_timestamp_alignment",
                     "video_stream_sync", "visual_quality")
    }


# ---------------------------------------------------------------------------
# Layer 1: compute_visual_stats pure function
# ---------------------------------------------------------------------------

def test_stats_clean_data_no_flags():
    from rda.ui_app.components.common import compute_visual_stats

    result = _make_result()
    for ep in result.episodes.values():
        ep.metrics["video_freeze"] = MetricResult.make_pass(
            name="video_freeze", measurement={"freeze_region_count": 0})
    vis = compute_visual_stats(result)
    assert vis["has_video"] is True
    assert vis["checked_eps"] == result.num_episodes
    assert vis["flagged_eps"] == 0
    assert vis["dep_missing_eps"] == 0


def test_stats_counts_flagged_by_metric():
    from rda.ui_app.components.common import compute_visual_stats

    result = _make_result()
    ep0 = result.episodes[0]
    ep0.metrics["video_freeze"] = _freeze_exclude()
    ep0.metrics["visual_quality"] = _vq_review()
    vis = compute_visual_stats(result)
    assert vis["flagged_eps"] == 1
    assert vis["flagged_by_metric"] == {"video_freeze": 1, "visual_quality": 1}
    assert vis["checked_eps"] == 1


def test_stats_dep_missing_counts_and_has_video():
    from rda.ui_app.components.common import compute_visual_stats

    result = _make_result()
    ep0 = result.episodes[0]
    ep0.metrics.update(_na_all())
    vis = compute_visual_stats(result)
    # dep-missing still means "video modality exists but was not audited"
    assert vis["has_video"] is True
    assert vis["dep_missing_eps"] == 1
    assert vis["checked_eps"] == 0
    assert vis["flagged_eps"] == 0


# ---------------------------------------------------------------------------
# Layer 2: real-page AppTest runs (skipped without streamlit.testing)
# ---------------------------------------------------------------------------

def _streamlit_testing_available() -> bool:
    try:
        import streamlit  # noqa: F401
        from streamlit.testing.v1 import AppTest  # noqa: F401
        return True
    except Exception:
        return False


UI_PAGES_DIR = Path(__file__).resolve().parents[1] / "rda" / "ui_app"


def _load_real_dataset_or_none():
    if not DS_ROOT.exists():
        return None
    try:
        from rda.io.lerobot_loader import iter_episodes
        from rda.io.schema import DatasetInfo as DI

        info = DI(path=str(DS_ROOT), num_episodes=1, total_frames=0,
                  meta={"format": "lerobot"})
        eps = []
        for i, ep in enumerate(iter_episodes(str(DS_ROOT))):
            if i >= 1:
                break
            eps.append(ep)
            info.total_frames += ep.num_frames
        return DatasetAuditor().audit_dataset(info, iter(eps))
    except Exception:
        return None


@pytest.mark.skipif(not _streamlit_testing_available(), reason="streamlit not installed")
def test_apptest_explorer_visual_flagged(zeroshot=False):
    from streamlit.testing.v1 import AppTest

    result = _load_real_dataset_or_none()
    if result is None:
        pytest.skip("libero_10 dataset not available on this machine")
    ep0 = result.episodes[next(iter(result.episodes))]
    ep0.metrics["video_freeze"] = _freeze_exclude()
    ep0.metrics["visual_quality"] = _vq_review()

    at = AppTest.from_file(str(UI_PAGES_DIR / "pages" / "4_Episode_Explorer.py"),
                           default_timeout=180)
    at.session_state["rda_lang"] = "zh"
    at.session_state["audit_result"] = result
    at.session_state["episodes_df"] = None
    at.session_state["dataset_path"] = None
    at.session_state["user_verdicts"] = {}
    at.session_state["selected_episode"] = next(iter(result.episodes))
    at.run()

    fatal = [e.value for e in at.exception if "Could not find page" not in e.value]
    assert not fatal, fatal
    text = "\n".join([b.value for b in at.markdown] + [b.value for b in at.subheader]
                     + [b.value for b in at.caption] + [b.value for b in at.info]
                     + [b.value for b in at.warning] + [b.value for b in at.success])
    assert "视觉审计 (Layer 1C)" in text
    assert "视觉流完整性" in text
    assert "质量罚分" in text
    assert "未安装 PyAV" not in text


@pytest.mark.skipif(not _streamlit_testing_available(), reason="streamlit not installed")
def test_apptest_explorer_dep_missing_never_pass():
    from streamlit.testing.v1 import AppTest

    result = _load_real_dataset_or_none()
    if result is None:
        pytest.skip("libero_10 dataset not available on this machine")
    ep0 = result.episodes[next(iter(result.episodes))]
    ep0.metrics.update(_na_all())

    at = AppTest.from_file(str(UI_PAGES_DIR / "pages" / "4_Episode_Explorer.py"),
                           default_timeout=180)
    at.session_state["rda_lang"] = "en"
    at.session_state["audit_result"] = result
    at.session_state["episodes_df"] = None
    at.session_state["dataset_path"] = None
    at.session_state["user_verdicts"] = {}
    at.session_state["selected_episode"] = next(iter(result.episodes))
    at.run()

    fatal = [e.value for e in at.exception if "Could not find page" not in e.value]
    assert not fatal, fatal
    text = "\n".join([b.value for b in at.markdown] + [b.value for b in at.subheader]
                     + [b.value for b in at.caption] + [b.value for b in at.info]
                     + [b.value for b in at.warning] + [b.value for b in at.success])
    # not-audited must be loud and must not render quality tables
    assert "not installed" in text and "not audited" in text
    assert "per-camera penalty breakdown" not in text
