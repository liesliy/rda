"""LeRobot v2.0/v2.x format support regression.

These tests pin the three defects fixed in this change:

1. **P1 — version mislabeling**: ``_detect_format_version`` used to fold
   every ``v2.*`` declaration into ``"v2.1"``, so v2.0 datasets were
   mislabeled in audit reports. It must now return the declared version
   verbatim (normalized to a leading ``v``), and every consumer that
   branches on the format must treat the whole ``v2.*`` family as one
   layout (v2.0 and v2.1 share the same file layout).
2. **P1 — conform blind on v2.x**: ``DatasetView`` used to glob only the
   v3.0 layout (``file-*.parquet``, ``meta/episodes/**``, ``videos/<f>/**``),
   so on v2.x datasets it read zero data files and most items collapsed to
   UNDETERMINED. It must now read the v2.x layout:
   ``data/chunk-*/episode_*.parquet``, ``meta/episodes.jsonl`` and
   ``videos/chunk-*/<feature>/episode_*.mp4``.
3. **P3 — dropped task descriptions**: v2.x stores the task mapping in
   ``meta/tasks.jsonl``; it was never read, so episodes lost their
   human-readable task description.

v2.x layout reference (huggingface/lerobot ``convert_dataset_v21_to_v30.py``):

    data/chunk-000/episode_000000.parquet
    videos/chunk-000/CAMERA/episode_000000.mp4
    meta/episodes.jsonl  {"episode_index": 1, "tasks": ["..."], "length": 266}
    meta/tasks.jsonl     {"task_index": 1, "task": "..."}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

N_FRAMES = 20
FPS = 10
VIDEO_FEATURE = "observation.images.top"


def _write_v2x_dataset(
    root: Path,
    version: str = "v2.0",
    n_episodes: int = 2,
    with_videos: bool = True,
    with_tasks_jsonl: bool = True,
    with_tasks_in_rows: bool = True,
) -> Path:
    """Write a minimal but structurally faithful LeRobot v2.x dataset.

    ``with_videos`` writes placeholder mp4 files (existence only — no
    conform check in scope decodes them).
    """
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "meta"
    meta.mkdir()

    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        VIDEO_FEATURE: {"dtype": "video", "shape": [None, None, 3]},
    }
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": version,
        "robot_type": "panda",
        "total_episodes": n_episodes,
        "total_frames": n_episodes * N_FRAMES,
        "total_tasks": 2,
        "fps": FPS,
        "chunks_size": 1000,
        "features": features,
    }))

    if with_tasks_jsonl:
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "pick up the red block"}) + "\n"
            + json.dumps({"task_index": 1, "task": "pick up the blue block"}) + "\n"
        )

    rows = []
    for ep in range(n_episodes):
        row = {"episode_index": ep, "length": N_FRAMES}
        if with_tasks_in_rows:
            row["tasks"] = [f"pick up the {'red' if ep == 0 else 'blue'} block"]
        rows.append(row)
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    for ep in range(n_episodes):
        pd.DataFrame({
            "episode_index": [ep] * N_FRAMES,
            "frame_index": list(range(N_FRAMES)),
            "timestamp": [i / FPS for i in range(N_FRAMES)],
            "index": [ep * N_FRAMES + i for i in range(N_FRAMES)],
            "task_index": [ep] * N_FRAMES,
            "action": [list(np.full(2, 0.01 * i + 0.1 * ep)) for i in range(N_FRAMES)],
            "observation.state": [list(np.full(2, 0.01 * i + 0.1 * ep))
                                  for i in range(N_FRAMES)],
        }).to_parquet(data_dir / f"episode_{ep:06d}.parquet")

    if with_videos:
        for ep in range(n_episodes):
            vdir = root / "videos" / "chunk-000" / VIDEO_FEATURE
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / f"episode_{ep:06d}.mp4").write_bytes(b"\x00" * 16)

    return root


# ---------------------------------------------------------------------------
# 1. Format version detection (P1: no more folding v2.0 into v2.1)
# ---------------------------------------------------------------------------

class TestFormatVersionDetection:
    def test_v20_reported_verbatim(self, tmp_path: Path):
        from rda.io.lerobot_loader import _detect_format_version

        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        assert _detect_format_version(root) == "v2.0"

    def test_v21_reported_verbatim(self, tmp_path: Path):
        from rda.io.lerobot_loader import _detect_format_version

        root = _write_v2x_dataset(tmp_path / "v21", version="v2.1")
        assert _detect_format_version(root) == "v2.1"

    def test_bare_number_prefix_is_normalized(self, tmp_path: Path):
        """A declared ``2.1`` (no v) still normalizes to ``v2.1``."""
        from rda.io.lerobot_loader import _detect_format_version

        root = _write_v2x_dataset(tmp_path / "bare", version="2.1")
        assert _detect_format_version(root) == "v2.1"

    def test_structural_fallback_to_v21(self, tmp_path: Path):
        """No info.json, but episodes.jsonl present → v2.1 (v2.x layout)."""
        from rda.io.lerobot_loader import _detect_format_version

        root = tmp_path / "structural"
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "tasks": ["t"], "length": 5}) + "\n"
        )
        assert _detect_format_version(root) == "v2.1"

    def test_dataset_info_meta_carries_declared_version(self, tmp_path: Path):
        from rda.io.lerobot_loader import load_lerobot_dataset

        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        info = load_lerobot_dataset(str(root))
        assert info.meta["format_version"] == "v2.0", (
            "audit reports read the version from here — folding v2.0 into "
            "v2.1 mislabels the audited dataset"
        )

    def test_episode_meta_carries_declared_version(self, tmp_path: Path):
        from rda.io.lerobot_loader import iter_episodes

        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        ep = next(iter_episodes(str(root)))
        assert ep.meta["format_version"] == "v2.0"
        assert ep.meta["source"] == "lerobot"


# ---------------------------------------------------------------------------
# 2. v2.x task descriptions (P3: tasks.jsonl used to be dropped)
# ---------------------------------------------------------------------------

class TestTaskDescriptionV2x:
    def test_tasks_jsonl_surfaces_per_episode(self, tmp_path: Path):
        from rda.io.lerobot_loader import iter_episodes

        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        episodes = {ep.meta["task_index"]: ep
                    for ep in iter_episodes(str(root))}
        assert episodes[0].meta.get("task_description") == "pick up the red block"
        assert episodes[1].meta.get("task_description") == "pick up the blue block"

    def test_dataset_meta_tasks_mapping_from_jsonl(self, tmp_path: Path):
        from rda.io.lerobot_loader import load_lerobot_dataset

        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        info = load_lerobot_dataset(str(root))
        assert info.meta.get("tasks") == {
            0: "pick up the red block",
            1: "pick up the blue block",
        }

    def test_fallback_to_single_task_in_episode_row(self, tmp_path: Path):
        """No tasks.jsonl → episodes.jsonl's own single-element tasks list."""
        from rda.io.lerobot_loader import iter_episodes

        root = _write_v2x_dataset(tmp_path / "nojsonl", with_tasks_jsonl=False)
        ep = next(iter_episodes(str(root)))
        assert ep.meta.get("task_description") == "pick up the red block"

    def test_no_task_sources_leaves_description_absent(self, tmp_path: Path):
        from rda.io.lerobot_loader import iter_episodes

        root = _write_v2x_dataset(
            tmp_path / "notasks",
            with_tasks_jsonl=False,
            with_tasks_in_rows=False,
        )
        ep = next(iter_episodes(str(root)))
        assert "task_description" not in ep.meta
        # task_index itself still comes from the parquet column
        assert ep.meta.get("task_index") == 0


# ---------------------------------------------------------------------------
# 3. Full audit chain on a v2.0 dataset
# ---------------------------------------------------------------------------

class TestAuditV20:
    def test_full_audit_chain_runs_and_reports_v20(self, tmp_path: Path):
        import warnings

        from rda.audit.dataset_audit import DatasetAuditor
        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset
        from rda.report.json_report import generate_json_report

        root = _write_v2x_dataset(
            tmp_path / "v20", version="v2.0", with_videos=False
        )
        info = load_lerobot_dataset(str(root))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = DatasetAuditor().audit_dataset(info, iter_episodes(str(root)))
        assert len(result.episodes) == 2

        report = generate_json_report(result)
        assert report["dataset"]["format_version"] == "v2.0", (
            "the JSON report must carry the declared version, not the folded one"
        )


# ---------------------------------------------------------------------------
# 4. conform: DatasetView must see the v2.x layout (P1)
# ---------------------------------------------------------------------------

class TestConformDatasetViewV2x:
    def _view(self, root: Path):
        from rda.conform.engine import DatasetView

        return DatasetView(root)

    def test_data_files_read_v2x_layout(self, tmp_path: Path):
        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        files = self._view(root).data_files
        names = [p.name for p in files]
        assert names == ["episode_000000.parquet", "episode_000001.parquet"], names

    def test_episodes_meta_parsed_from_jsonl(self, tmp_path: Path):
        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        rows = self._view(root).episodes_meta
        assert len(rows) == 2
        assert [r["episode_index"] for r in rows] == [0, 1]
        assert all(r["length"] == N_FRAMES for r in rows)

    def test_video_payload_files_read_v2x_layout(self, tmp_path: Path):
        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        files = self._view(root).video_payload_files(VIDEO_FEATURE)
        names = [p.name for p in files]
        assert names == ["episode_000000.mp4", "episode_000001.mp4"], names

    def test_v30_layout_still_resolves(self, tmp_path: Path):
        """The v3 branch must not regress: videos/<feature>/** still wins."""
        root = _write_v2x_dataset(tmp_path / "v30", version="v3.0")
        # v3.0 declares file-* layout — rewrite data files to v3 naming so the
        # glob logic is what differs, and put videos in the v3 position.
        data_dir = root / "data" / "chunk-000"
        for ep in (0, 1):
            src = data_dir / f"episode_{ep:06d}.parquet"
            dst = data_dir / f"file-{ep:03d}.parquet"
            src.rename(dst)
        old = root / "videos" / "chunk-000" / VIDEO_FEATURE
        new = root / "videos" / VIDEO_FEATURE / "chunk-000"
        new.mkdir(parents=True)
        for f in old.iterdir():
            f.rename(new / f"file-{int(f.stem.split('_')[-1]):03d}.mp4")
        old.rmdir()
        old.parent.rmdir()

        view = self._view(root)
        assert [p.name for p in view.data_files] == [
            "file-000.parquet", "file-001.parquet"
        ]
        names = [p.name for p in view.video_payload_files(VIDEO_FEATURE)]
        assert names == ["file-000.mp4", "file-001.mp4"], names

    def test_timestamps_and_episode_ids_resolve_on_v2x(self, tmp_path: Path):
        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        view = self._view(root)
        ids = view.episode_ids()
        assert ids == [0, 1]
        ts = view.timestamps_by_episode()[0]
        assert ts.size == N_FRAMES
        assert abs(float(ts.max()) - (N_FRAMES - 1) / FPS) < 1e-9

    def test_run_conformance_coverage_above_broken_baseline(
        self, tmp_path: Path
    ):
        """Before the fix the v2.x path yielded 25% coverage (0 data files).

        With data-level checks able to run, coverage must clearly exceed
        that broken baseline.
        """
        from rda.conform.engine import run_conformance

        root = _write_v2x_dataset(tmp_path / "v20", version="v2.0")
        result = run_conformance(root)
        assert result.summary["coverage"] > 0.25, result.summary
        # data-level determinations must exist, not just metadata FAILs
        determined = [
            it for it in result.items if it.verdict.value not in
            ("UNDETERMINED", "NOT_APPLICABLE")
        ]
        assert len(determined) >= 4, (
            [f"{it.metric}:{it.verdict.value}" for it in result.items]
        )

    def test_v21_layout_equally_supported(self, tmp_path: Path):
        from rda.conform.engine import run_conformance

        root = _write_v2x_dataset(tmp_path / "v21", version="v2.1")
        result = run_conformance(root)
        assert result.summary["coverage"] > 0.25, result.summary
