"""LeRobot dataset loader.

Loads datasets stored in the LeRobot format and converts them to the
canonical RDA :class:`EpisodeData` / :class:`DatasetInfo` representation.

Two entry points are provided:

- :func:`load_lerobot_dataset` — loads dataset-level metadata only
  (fast, for summary / overview).
- :func:`iter_episodes` — streams episodes one at a time as
  :class:`EpisodeData` objects (memory-efficient for large datasets).

Supports both lerobot v0.5.x (``from lerobot import LeRobotDataset``)
and v0.6.0+ (``from lerobot.datasets.lerobot_dataset import LeRobotDataset``).
Falls back to direct parquet file reading when the lerobot package is not
installed or when the API fails.

Direct parquet reading supports two format versions:

- **v3.0** — ``meta/episodes/chunk-XXX/file-XXX.parquet`` for episode
  metadata, and ``data/chunk-XXX/file-XXX.parquet`` for frame data
  (multiple episodes per file).
- **v2.1** — ``meta/episodes.jsonl`` (single JSONL file) for episode
  metadata, and ``data/chunk-XXX/episode_XXXXXX.parquet`` for frame data
  (one episode per file).

The format is auto-detected from ``meta/info.json`` (``codebase_version``
field) or from the presence of ``meta/episodes.jsonl``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from rda.io.schema import DatasetInfo, EpisodeData


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format_version(dataset_path: Path) -> str:
    """Auto-detect the LeRobot dataset format version.

    Detection order:

    1. If ``meta/info.json`` exists and contains ``codebase_version``,
       use it to determine major version.
    2. Otherwise, if ``meta/episodes.jsonl`` exists, infer v2.1.
    3. Otherwise, if ``meta/episodes/`` directory exists, infer v3.0.
    4. Otherwise, default to "v3.0" (let existing code raise its
       own descriptive error).

    Args:
        dataset_path: Root path of the LeRobot dataset.

    Returns:
        Version string: ``"v2.1"`` or ``"v3.0"``.
    """
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                info = json.load(f)
            version = info.get("codebase_version", "")
            if version:
                if version.startswith("v2") or version.startswith("2"):
                    return "v2.1"
                if version.startswith("v3") or version.startswith("3"):
                    return "v3.0"
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: structural detection
    if (dataset_path / "meta" / "episodes.jsonl").exists():
        return "v2.1"
    if (dataset_path / "meta" / "episodes").is_dir():
        return "v3.0"

    # Default — let downstream code raise its own error
    return "v3.0"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_info_json(dataset_path: Path) -> dict:
    """Load meta/info.json from a LeRobot dataset directory."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"meta/info.json not found in {dataset_path}. "
            "Is this a valid LeRobot dataset directory?"
        )
    with open(info_path) as f:
        return json.load(f)


def _get_feature_keys(info: dict) -> tuple:
    """Extract modality and action keys from info.json features dict."""
    features = info.get("features", {})
    modalities = []
    action_keys = []
    for key in features:
        if key.startswith("observation."):
            clean = key.replace("observation.", "", 1)
            modalities.append(clean)
        elif key == "action" or key.startswith("action."):
            clean = key.replace("action.", "", 1) if key.startswith("action.") else key
            action_keys.append(clean)
    return modalities, action_keys


def _extract_episode_from_dataframe(
    ep_df,
    ep_index: int,
    fps: int,
    format_version: str,
) -> EpisodeData:
    """Build an :class:`EpisodeData` from a per-episode DataFrame.

    Shared by both v2.1 and v3.0 direct-parquet loaders so that column
    extraction logic stays consistent.

    Args:
        ep_df: Pandas DataFrame containing rows for one episode.
        ep_index: Episode index.
        fps: Frames per second (used as fallback when ``timestamp``
            column is missing).
        format_version: Dataset format version string for metadata.

    Returns:
        EpisodeData populated from the DataFrame.
    """
    num_frames = len(ep_df)

    # Timestamps
    if "timestamp" in ep_df.columns:
        timestamps = ep_df["timestamp"].values.astype(np.float64)
        # Flatten if stored as arrays
        if timestamps.ndim > 1:
            timestamps = timestamps.squeeze(-1)
    else:
        timestamps = np.arange(num_frames, dtype=np.float64) / float(fps)

    # Observations
    observation: Dict[str, np.ndarray] = {}
    for key in ep_df.columns:
        if key.startswith("observation."):
            clean_key = key.replace("observation.", "", 1)
            vals = np.array(ep_df[key].tolist())
            observation[clean_key] = vals

    # Actions
    action: Dict[str, np.ndarray] = {}
    if "action" in ep_df.columns:
        action_vals = np.array(ep_df["action"].tolist())
        action["action"] = action_vals
    else:
        for key in ep_df.columns:
            if key.startswith("action."):
                clean_key = key.replace("action.", "", 1)
                action[clean_key] = np.array(ep_df[key].tolist())

    # Reward
    reward = None
    if "next.reward" in ep_df.columns:
        r = ep_df["next.reward"].values.astype(np.float32)
        if r.ndim > 1:
            r = r.squeeze(-1)
        reward = r

    # Done
    done = None
    if "next.done" in ep_df.columns:
        d = ep_df["next.done"].values.astype(np.bool_)
        if d.ndim > 1:
            d = d.squeeze(-1)
        done = d

    return EpisodeData(
        episode_index=ep_index,
        num_frames=num_frames,
        timestamps=timestamps,
        observation=observation,
        action=action,
        reward=reward,
        done=done,
        meta={"source": "lerobot", "format_version": format_version},
    )


def _try_lerobot_import():
    """Try importing LeRobotDataset with compatibility for old/new paths.

    Returns the LeRobotDataset class or None if not available.
    """
    # Try lerobot >= 0.6.0 path first
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        return LeRobotDataset
    except ImportError:
        pass

    # Fallback to lerobot < 0.6.0 path
    try:
        from lerobot import LeRobotDataset  # type: ignore
        return LeRobotDataset
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# v3.0 format helpers (original logic, preserved)
# ---------------------------------------------------------------------------


def _load_episode_metadata_v30(dataset_path: Path):
    """Load all episode metadata parquet files into a single DataFrame.

    LeRobot v3.0 format: ``meta/episodes/chunk-XXX/file-XXX.parquet``.

    Returns a pandas DataFrame with one row per episode, containing columns
    like ``episode_index``, ``length``, ``data/chunk_index``,
    ``data/file_index``, ``dataset_from_index``, ``dataset_to_index``, etc.
    """
    import pandas as pd

    ep_dir = dataset_path / "meta" / "episodes"
    if not ep_dir.exists():
        raise FileNotFoundError(
            f"meta/episodes/ directory not found in {dataset_path}. "
            "Is this a valid LeRobot v3.0 dataset directory?"
        )

    frames = []
    for chunk_dir in sorted(ep_dir.glob("chunk-*")):
        for fp in sorted(chunk_dir.glob("file-*.parquet")):
            frames.append(pd.read_parquet(fp))

    if not frames:
        raise FileNotFoundError(
            f"No episode parquet files found in {ep_dir}."
        )

    return pd.concat(frames, ignore_index=True)


# Backwards-compatible alias
_load_episode_metadata = _load_episode_metadata_v30


@lru_cache(maxsize=8)
def _build_episode_file_index_v30(dataset_path: str) -> Dict[int, Path]:
    """Build a cached ``episode_index`` -> data parquet path index.

    The v3.0 metadata normally points to the right file, but some local
    datasets have stale ``data/file_index`` values.  Only the small
    ``episode_index`` column is read while building this fallback index, so
    image/video columns are not loaded during the scan.
    """
    import pyarrow.parquet as pq

    data_dir = Path(dataset_path) / "data"
    if not data_dir.exists():
        return {}

    index: Dict[int, Path] = {}
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_path in sorted(chunk_dir.glob("*.parquet")):
            try:
                table = pq.read_table(parquet_path, columns=["episode_index"])
            except Exception:
                continue
            for value in set(table.column("episode_index").to_pylist()):
                if value is not None:
                    index.setdefault(int(value), parquet_path)
    return index


@lru_cache(maxsize=8)
def _read_v30_fallback_file(parquet_path: str):
    """Read one fallback file without image/video payload columns.

    Fallback reads are used after the metadata mapping has already failed,
    which is the audit path.  Keep timestamps, actions, numeric observations,
    reward, and done while avoiding the large camera/video columns.  The
    resulting per-file DataFrame is cached because a v3.0 file can contain
    many episodes.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    schema = pq.ParquetFile(path).schema_arrow
    columns = []
    for field in schema:
        name = field.name
        lower_name = name.lower()
        if name == "episode_index":
            columns.append(name)
        elif name in {"timestamp", "action", "next.reward", "next.done"}:
            columns.append(name)
        elif name.startswith("action."):
            columns.append(name)
        elif name.startswith("observation.") and not any(
            token in lower_name for token in ("image", "video")
        ):
            columns.append(name)

    return pd.read_parquet(path, columns=columns)


def _read_episode_parquet_v30(
    dataset_path: Path,
    ep_meta_row: dict,
    fps: int,
) -> EpisodeData:
    """Read a single episode from its parquet file (v3.0 format).

    v3.0 stores multiple episodes per ``data/chunk-XXX/file-XXX.parquet``
    file, so we filter by ``episode_index``.

    Args:
        dataset_path: Root path of the LeRobot dataset.
        ep_meta_row: A row from the episode metadata DataFrame (as dict).
        fps: Frames per second from info.json.

    Returns:
        EpisodeData for this episode.
    """
    import pandas as pd

    chunk_idx = int(ep_meta_row["data/chunk_index"])
    file_idx = int(ep_meta_row["data/file_index"])
    ep_index = int(ep_meta_row["episode_index"])

    parquet_path = (
        dataset_path
        / "data"
        / f"chunk-{chunk_idx:03d}"
        / f"file-{file_idx:03d}.parquet"
    )

    # Keep the metadata mapping as the fast path, but use an audit projection
    # so camera/video payloads are never loaded merely to locate an episode.
    # Some datasets have stale file_index values, so an empty match triggers
    # one cached scan of the actual data files.
    try:
        df = _read_v30_fallback_file(str(parquet_path))
        ep_df = df[df["episode_index"] == ep_index].reset_index(drop=True)
    except (FileNotFoundError, OSError, KeyError):
        ep_df = pd.DataFrame()

    if ep_df.empty:
        file_index = _build_episode_file_index_v30(str(dataset_path.resolve()))
        actual_path = file_index.get(ep_index)
        if actual_path is not None and actual_path != parquet_path:
            df = _read_v30_fallback_file(str(actual_path))
            ep_df = df[df["episode_index"] == ep_index].reset_index(drop=True)

    return _extract_episode_from_dataframe(ep_df, ep_index, fps, "v3.0")


# Backwards-compatible alias
_read_episode_parquet = _read_episode_parquet_v30


# ---------------------------------------------------------------------------
# v2.1 format helpers
# ---------------------------------------------------------------------------


def _load_episode_metadata_v21(dataset_path: Path):
    """Load episode metadata from ``meta/episodes.jsonl`` (v2.1 format).

    v2.1 stores episode metadata in a single JSONL file. Each line is a
    JSON object with keys like ``episode_index``, ``length``, ``tasks``,
    ``dataset_from_index``, ``dataset_to_index``.

    Args:
        dataset_path: Root path of the LeRobot dataset.

    Returns:
        pandas DataFrame with one row per episode, containing at least
        ``episode_index`` and ``length`` columns.
    """
    import pandas as pd

    ep_path = dataset_path / "meta" / "episodes.jsonl"
    if not ep_path.exists():
        raise FileNotFoundError(
            f"meta/episodes.jsonl not found in {dataset_path}. "
            "Is this a valid LeRobot v2.1 dataset directory?"
        )

    records = []
    with open(ep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise FileNotFoundError(
            f"No episode records found in {ep_path}."
        )

    return pd.DataFrame(records)


def _build_episode_file_index_v21(dataset_path: Path) -> Dict[int, Path]:
    """Build a mapping from ``episode_index`` to parquet file path (v2.1).

    Scans all ``data/chunk-XXX/episode_XXXXXX.parquet`` files and returns
    a dict keyed by integer episode index.

    Args:
        dataset_path: Root path of the LeRobot dataset.

    Returns:
        Dict mapping ``episode_index`` (int) to the parquet file Path.
    """
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"data/ directory not found in {dataset_path}."
        )

    index: Dict[int, Path] = {}
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for fp in sorted(chunk_dir.glob("episode_*.parquet")):
            # Filename format: episode_XXXXXX.parquet
            stem = fp.stem  # e.g. "episode_000000"
            try:
                ep_idx = int(stem.split("_")[-1])
            except (ValueError, IndexError):
                continue
            index[ep_idx] = fp

    return index


def _read_episode_parquet_v21(
    dataset_path: Path,
    ep_meta_row: dict,
    fps: int,
    file_index: Optional[Dict[int, Path]] = None,
) -> EpisodeData:
    """Read a single episode from its parquet file (v2.1 format).

    v2.1 stores one episode per file at
    ``data/chunk-XXX/episode_XXXXXX.parquet``.

    Args:
        dataset_path: Root path of the LeRobot dataset.
        ep_meta_row: A row from the episode metadata DataFrame (as dict).
        fps: Frames per second from info.json.
        file_index: Optional pre-built mapping from episode_index to
            parquet file path. If None, the index is built by scanning
            ``data/`` on first call (cached by caller for efficiency).

    Returns:
        EpisodeData for this episode.
    """
    import pandas as pd

    ep_index = int(ep_meta_row["episode_index"])

    if file_index is None:
        file_index = _build_episode_file_index_v21(dataset_path)

    if ep_index not in file_index:
        raise FileNotFoundError(
            f"No parquet file found for episode {ep_index} in {dataset_path / 'data'}."
        )

    parquet_path = file_index[ep_index]
    df = pd.read_parquet(parquet_path)

    # v2.1 files contain exactly one episode each, but filter defensively
    if "episode_index" in df.columns:
        ep_df = df[df["episode_index"] == ep_index].reset_index(drop=True)
    else:
        ep_df = df.reset_index(drop=True)

    return _extract_episode_from_dataframe(ep_df, ep_index, fps, "v2.1")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_lerobot_dataset(path: str) -> DatasetInfo:
    """Load a LeRobot dataset and return dataset-level metadata.

    This is a lightweight call that only reads metadata and the first
    episode to determine available modalities. It does not load all
    episode data into memory.

    Supports both direct parquet reading (LeRobot v2.1 and v3.0 formats)
    and the lerobot Python package (v0.5.x and v0.6.0+).

    The format version is auto-detected from ``meta/info.json``
    (``codebase_version`` field) or from structural cues.

    Args:
        path: Filesystem path to the LeRobot dataset directory, or
            a HuggingFace repository identifier (e.g.
            ``"lerobot/pusht"``).

    Returns:
        :class:`DatasetInfo` with summary metadata about the dataset,
        including ``modalities`` and ``action_keys`` derived from a
        sample episode.
    """
    dataset_path = Path(path)

    # --- Try direct parquet reading first (fastest, no lerobot needed) ---
    if dataset_path.is_dir() and (dataset_path / "meta" / "info.json").exists():
        info = _load_info_json(dataset_path)
        num_episodes = info.get("total_episodes", 0)
        total_frames = info.get("total_frames", 0)
        fps = info.get("fps", 50)
        modalities, action_keys = _get_feature_keys(info)

        format_version = _detect_format_version(dataset_path)

        meta = {
            "format": "lerobot",
            "format_version": format_version,
            "fps": fps,
        }
        if "robot_type" in info:
            meta["robot_type"] = info["robot_type"]

        # If info.json doesn't have total_episodes / total_frames,
        # compute them from episode metadata (common in v2.1).
        if not num_episodes or not total_frames:
            try:
                if format_version == "v2.1":
                    ep_meta = _load_episode_metadata_v21(dataset_path)
                else:
                    ep_meta = _load_episode_metadata_v30(dataset_path)
                num_episodes = len(ep_meta)
                if "length" in ep_meta.columns:
                    total_frames = int(ep_meta["length"].sum())
                else:
                    # Estimate from first episode
                    first = next(
                        iter_episodes(path, max_episodes=1), None
                    )
                    total_frames = first.num_frames * num_episodes if first else 0
            except Exception:
                pass

        return DatasetInfo(
            path=path,
            num_episodes=num_episodes,
            total_frames=total_frames,
            modalities=modalities,
            action_keys=action_keys,
            meta=meta,
        )

    # --- Fallback: try the lerobot Python package ---
    LRD = _try_lerobot_import()
    if LRD is None:
        raise ImportError(
            "Cannot load LeRobot dataset: direct parquet reading failed and "
            "the 'lerobot' package is not installed. "
            "Install it with: pip install lerobot"
        )

    # lerobot 0.6.0+ uses root= kwarg for local paths
    try:
        dataset = LRD(repo_id="", root=str(dataset_path))
    except Exception:
        # Older API: pass path as first positional arg
        dataset = LRD(str(dataset_path))

    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames

    # Determine features from the dataset
    features = dataset.features if hasattr(dataset, "features") else {}
    modalities = [
        k.replace("observation.", "", 1)
        for k in features
        if k.startswith("observation.")
    ]
    action_keys = [
        k.replace("action.", "", 1) if k.startswith("action.") else k
        for k in features
        if k == "action" or k.startswith("action.")
    ]

    return DatasetInfo(
        path=path,
        num_episodes=num_episodes,
        total_frames=total_frames,
        modalities=modalities,
        action_keys=action_keys,
        meta={"format": "lerobot"},
    )


def iter_episodes(
    path: str,
    max_episodes: Optional[int] = None,
) -> Iterator[EpisodeData]:
    """Iterate over episodes in a LeRobot dataset.

    Episodes are loaded one at a time and yielded as :class:`EpisodeData`
    objects, making this suitable for large datasets that don't fit in
    memory.

    Uses direct parquet file reading for LeRobot v2.1 and v3.0 datasets
    (fast, no lerobot package required). Falls back to the lerobot Python
    package for older formats.

    The format version is auto-detected from ``meta/info.json``
    (``codebase_version`` field) or from structural cues.

    Args:
        path: Filesystem path to the LeRobot dataset directory.
        max_episodes: Optional maximum number of episodes to yield.
            If None, yields all episodes.

    Yields:
        :class:`EpisodeData` for each episode in the dataset, in index
        order.
    """
    dataset_path = Path(path)

    # --- Try direct parquet reading first ---
    if dataset_path.is_dir() and (dataset_path / "meta" / "info.json").exists():
        info = _load_info_json(dataset_path)
        fps = info.get("fps", 50)

        format_version = _detect_format_version(dataset_path)

        if format_version == "v2.1":
            ep_metadata = _load_episode_metadata_v21(dataset_path)
            # Build file index once for efficiency
            file_index = _build_episode_file_index_v21(dataset_path)

            count = 0
            for _, ep_row in ep_metadata.iterrows():
                if max_episodes is not None and count >= max_episodes:
                    break
                try:
                    yield _read_episode_parquet_v21(
                        dataset_path, ep_row, fps, file_index=file_index
                    )
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Failed to read episode {int(ep_row['episode_index'])}: {e}"
                    )
                count += 1
            return
        else:
            # v3.0 path — original logic
            ep_metadata = _load_episode_metadata_v30(dataset_path)

            count = 0
            for _, ep_row in ep_metadata.iterrows():
                if max_episodes is not None and count >= max_episodes:
                    break
                try:
                    yield _read_episode_parquet_v30(dataset_path, ep_row, fps)
                except Exception as e:
                    # Skip unreadable episodes but continue
                    import warnings
                    warnings.warn(
                        f"Failed to read episode {int(ep_row['episode_index'])}: {e}"
                    )
                count += 1
            return

    # --- Fallback: try the lerobot Python package ---
    LRD = _try_lerobot_import()
    if LRD is None:
        raise ImportError(
            "Cannot load LeRobot dataset: direct parquet reading failed and "
            "the 'lerobot' package is not installed. "
            "Install it with: pip install lerobot"
        )

    try:
        dataset = LRD(repo_id="", root=str(dataset_path))
    except Exception:
        dataset = LRD(str(dataset_path))

    # For lerobot API, iterate using hf_dataset and episode_index grouping
    hf_ds = dataset.hf_dataset
    ep_indices = sorted(hf_ds.unique("episode_index"))

    count = 0
    for ep_idx in ep_indices:
        if max_episodes is not None and count >= max_episodes:
            break

        ep_frames = hf_ds.filter(
            lambda x: x["episode_index"] == ep_idx,
            load_from_cache_file=False,
        )
        num_frames = len(ep_frames)

        # Extract timestamps
        if "timestamp" in ep_frames.column_names:
            timestamps = np.array(ep_frames["timestamp"], dtype=np.float64)
            if timestamps.ndim > 1:
                timestamps = timestamps.squeeze(-1)
        else:
            fps_val = dataset.fps if hasattr(dataset, "fps") else 50
            timestamps = np.arange(num_frames, dtype=np.float64) / float(fps_val)

        # Observations
        observation = {}
        for key in ep_frames.column_names:
            if key.startswith("observation."):
                clean_key = key.replace("observation.", "", 1)
                observation[clean_key] = np.array(ep_frames[key])

        # Actions
        action = {}
        if "action" in ep_frames.column_names:
            action["action"] = np.array(ep_frames["action"])
        else:
            for key in ep_frames.column_names:
                if key.startswith("action."):
                    clean_key = key.replace("action.", "", 1)
                    action[clean_key] = np.array(ep_frames[key])

        # Reward / done
        reward = None
        if "next.reward" in ep_frames.column_names:
            r = np.array(ep_frames["next.reward"], dtype=np.float32)
            if r.ndim > 1:
                r = r.squeeze(-1)
            reward = r

        done = None
        if "next.done" in ep_frames.column_names:
            d = np.array(ep_frames["next.done"], dtype=np.bool_)
            if d.ndim > 1:
                d = d.squeeze(-1)
            done = d

        yield EpisodeData(
            episode_index=ep_idx,
            num_frames=num_frames,
            timestamps=timestamps,
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            meta={"source": "lerobot"},
        )
        count += 1
