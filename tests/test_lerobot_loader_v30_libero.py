"""Regression coverage for the LIBERO v3.0 parquet mapping mismatch."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from rda.io.lerobot_loader import (  # noqa: E402
    _build_episode_file_index_v30,
    _load_episode_metadata_v30,
    _read_episode_parquet_v30,
)


DATASET_PATH = Path(
    os.environ.get("LIBERO_DATASET_PATH", "D:/workbuddy-data/datasets/libero")
)


@pytest.mark.skipif(
    not (DATASET_PATH / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists(),
    reason="real LIBERO v3.0 dataset is not available",
)
def test_libero_meta_mismatch_falls_back_to_actual_data_file():
    """A stale meta file_index must not turn an existing episode into 0 frames."""
    metadata = _load_episode_metadata_v30(DATASET_PATH)
    file_index = _build_episode_file_index_v30(str(DATASET_PATH.resolve()))

    mismatches = []
    for _, row in metadata.iterrows():
        episode_index = int(row["episode_index"])
        claimed = (
            DATASET_PATH
            / "data"
            / f"chunk-{int(row['data/chunk_index']):03d}"
            / f"file-{int(row['data/file_index']):03d}.parquet"
        )
        actual = file_index.get(episode_index)
        if actual is not None and actual != claimed:
            mismatches.append((row, actual))

    if not mismatches:
        pytest.skip("the available LIBERO copy has no meta/data mismatch")

    row, actual_path = mismatches[0]
    episode = _read_episode_parquet_v30(DATASET_PATH, row, fps=50)
    assert episode.num_frames > 0
    assert actual_path.exists()


@pytest.mark.skipif(
    not (DATASET_PATH / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists(),
    reason="real LIBERO v3.0 dataset is not available",
)
def test_libero_missing_episode_stays_empty():
    """An episode absent from all downloaded data files remains genuinely empty."""
    metadata = _load_episode_metadata_v30(DATASET_PATH)
    file_index = _build_episode_file_index_v30(str(DATASET_PATH.resolve()))

    missing = metadata[~metadata["episode_index"].astype(int).isin(file_index)]
    if missing.empty:
        pytest.skip("the available LIBERO copy has no genuinely missing episode")

    row = missing.iloc[0]
    episode = _read_episode_parquet_v30(DATASET_PATH, row, fps=50)
    assert episode.num_frames == 0
