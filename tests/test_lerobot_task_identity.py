"""Regression coverage for LeRobot v3.0 task identity (task_index / tasks.parquet).

Historically, RDA dropped task identity for multi-task v3.0 datasets: the
``task_index`` column in ``data/*.parquet`` was filtered out by the audit
projection and ``meta/tasks.parquet`` was never read, so reports could not
answer "which task is this episode from?". This is the same failure mode
reported for the community tool Calibra on ``lerobot/libero_10``.

These tests pin the fix: task count and mapping surface on ``DatasetInfo``,
and per-episode ``task_index``/``task_description`` surface on ``EpisodeData``,
while single-task / task-less datasets remain fully backward compatible.

Synthetic fixtures are used so the core assertions run without a real
dataset; a real ``lerobot/libero_10`` end-to-end check is included but
skipped when the dataset is not available.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

from rda.io.lerobot_loader import (  # noqa: E402
    _load_tasks_parquet_v30,
    _read_episode_parquet_v30,
    load_lerobot_dataset,
    iter_episodes,
)

REAL_LIBERO_10 = Path(
    os.environ.get("LIBERO10_DATASET_PATH", "D:/workbuddy-data/datasets/libero_10")
)


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

def _write_tasks_parquet(path: Path, tasks: dict):
    """Write meta/tasks.parquet in the libero_10 layout (index = description)."""
    import pandas as pd

    df = pd.DataFrame({"task_index": list(tasks.values())}, index=list(tasks.keys()))
    df.index.name = "task"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _build_synthetic_v30(root: Path, *, num_tasks: int = 2, with_task_index: bool = True):
    """Construct a minimal LeRobot v3.0 dataset directory for testing.

    ``num_tasks`` controls whether task identity is declared (tasks.parquet +
    ``total_tasks`` in info.json). ``num_tasks=0`` builds a genuinely
    task-less dataset (no tasks.parquet, no total_tasks, no task_index column).

    ``with_task_index`` controls only the per-frame ``task_index`` column in
    the data parquet (for modelling a declared-task-but-missing-column case).
    """
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    meta = root / "meta"
    meta.mkdir(exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "total_episodes": 2,
        "total_frames": 6,
        "fps": 10,
        "robot_type": "panda",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }

    if num_tasks > 0:
        tasks = {
            "pick up the red block": 0,
            "place the mug on the shelf": 1,
        }
        tasks = dict(list(tasks.items())[:num_tasks])
        _write_tasks_parquet(meta / "tasks.parquet", tasks)
        info["total_tasks"] = num_tasks

    (meta / "info.json").write_text(json.dumps(info))

    ep_meta = pd.DataFrame(
        [
            {"episode_index": 0, "length": 3, "data/chunk_index": 0, "data/file_index": 0},
            {"episode_index": 1, "length": 3, "data/chunk_index": 0, "data/file_index": 0},
        ]
    )
    ep_dir = meta / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_meta.to_parquet(ep_dir / "file-000.parquet")

    data_cols = {
        "episode_index": [0, 0, 0, 1, 1, 1],
        "timestamp": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
        "action": [[0.0, 0.0]] * 6,
        "observation.state": [[0.0, 0.0]] * 6,
    }
    if with_task_index and num_tasks > 0:
        data_cols["task_index"] = [0, 0, 0, 1, 1, 1]
    data = pd.DataFrame(data_cols)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(data_dir / "file-000.parquet")

    return root


# ---------------------------------------------------------------------------
# Synthetic (always-run) tests
# ---------------------------------------------------------------------------

def test_load_tasks_parquet_v30_returns_mapping(tmp_path):
    _write_tasks_parquet(tmp_path / "meta" / "tasks.parquet", {"task A": 0, "task B": 1})
    mapping = _load_tasks_parquet_v30(tmp_path)
    assert mapping == {0: "task A", 1: "task B"}


def test_load_tasks_parquet_v30_missing_file_returns_empty(tmp_path):
    assert _load_tasks_parquet_v30(tmp_path) == {}


def test_load_lerobot_dataset_surfaces_task_identity(tmp_path):
    root = _build_synthetic_v30(tmp_path / "ds")
    ds = load_lerobot_dataset(str(root))
    assert ds.meta["num_tasks"] == 2
    assert ds.meta["tasks"] == {0: "pick up the red block", 1: "place the mug on the shelf"}


def test_read_episode_v30_carries_task_index_and_description(tmp_path):
    import pandas as pd

    root = _build_synthetic_v30(tmp_path / "ds")
    ep_meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Episode 0 -> task 0
    ep0 = _read_episode_parquet_v30(root, ep_meta.iloc[0].to_dict(), fps=10)
    assert ep0.meta["task_index"] == 0
    assert ep0.meta["task_description"] == "pick up the red block"

    # Episode 1 -> task 1
    ep1 = _read_episode_parquet_v30(root, ep_meta.iloc[1].to_dict(), fps=10)
    assert ep1.meta["task_index"] == 1
    assert ep1.meta["task_description"] == "place the mug on the shelf"


def test_taskless_dataset_remains_backward_compatible(tmp_path):
    root = _build_synthetic_v30(tmp_path / "ds", num_tasks=0)
    ds = load_lerobot_dataset(str(root))
    # No task declaration at all -> no tasks mapping, no crash.
    assert "num_tasks" not in ds.meta
    assert "tasks" not in ds.meta
    ep = next(iter_episodes(str(root), max_episodes=1))
    assert "task_index" not in ep.meta
    assert "task_description" not in ep.meta


def test_declared_task_without_task_index_column_degrades_gracefully(tmp_path):
    """Declared tasks but missing per-frame task_index column must not crash."""
    root = _build_synthetic_v30(tmp_path / "ds", num_tasks=2, with_task_index=False)
    ds = load_lerobot_dataset(str(root))
    # Task identity is still declared at the dataset level (info.json + tasks.parquet).
    assert ds.meta["num_tasks"] == 2
    assert ds.meta["tasks"] == {0: "pick up the red block", 1: "place the mug on the shelf"}
    # But per-episode mapping is impossible without the column — degrade, don't crash.
    ep = next(iter_episodes(str(root), max_episodes=1))
    assert "task_index" not in ep.meta
    assert "task_description" not in ep.meta


# ---------------------------------------------------------------------------
# Real-dataset end-to-end (skipped when unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REAL_LIBERO_10 / "meta" / "tasks.parquet").exists(),
    reason="real lerobot/libero_10 dataset is not available",
)
def test_libero10_end_to_end_task_identity():
    ds = load_lerobot_dataset(str(REAL_LIBERO_10))
    assert ds.meta["num_tasks"] == 10
    assert len(ds.meta["tasks"]) == 10

    seen = set()
    for ep in iter_episodes(str(REAL_LIBERO_10)):
        assert ep.meta.get("task_index") is not None
        assert ep.meta.get("task_description")
        seen.add(ep.meta["task_index"])

    assert seen == set(range(10))
