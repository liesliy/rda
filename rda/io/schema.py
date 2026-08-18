"""Data schema definitions for RDA.

Defines the canonical data structures used across the audit pipeline,
decoupled from specific dataset storage formats (LeRobot, ROS bag, etc.).

All metrics and audit functions operate on these schema types, so adding
a new data format only requires writing a new loader that produces
:class:`EpisodeData` and :class:`DatasetInfo` objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeData:
    """Canonical representation of a single robot episode.

    This is the standard input type for all RDA metrics. Any dataset
    loader must produce objects of this type.

    Attributes:
        episode_index: Zero-based index of the episode within the dataset.
        num_frames: Total number of timesteps in the episode.
        timestamps: Array of shape ``(num_frames,)`` with timestamps
            in seconds. May be synthesized if not available in the
            source data.
        observation: Dict mapping modality keys (e.g. ``"state"``,
            ``"images.top"``) to observation arrays. Each array has
            shape ``(num_frames, ...)``.
        action: Dict mapping action keys (e.g. ``"joint_pos"``,
            ``"position"``) to action arrays. Each array has shape
            ``(num_frames, ...)``.
        reward: Optional array of shape ``(num_frames,)`` with
            per-step rewards.
        done: Optional array of shape ``(num_frames,)`` with done
            flags (boolean).
        meta: Arbitrary metadata dict for the episode. Common keys:
            ``"stream_timestamps"`` (per-modality timestamp dicts),
            ``"joint_limits"`` (robot joint limits for joint-limit check),
            ``"source"`` (data source identifier).
    """

    episode_index: int
    num_frames: int
    timestamps: np.ndarray
    observation: Dict[str, np.ndarray] = field(default_factory=dict)
    action: Dict[str, np.ndarray] = field(default_factory=dict)
    reward: Optional[np.ndarray] = None
    done: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetInfo:
    """Summary metadata for a full dataset.

    Attributes:
        path: Filesystem path or repository identifier of the dataset.
        num_episodes: Total number of episodes in the dataset.
        total_frames: Total number of frames across all episodes.
        modalities: List of observation modality keys (without the
            ``observation.`` prefix).
        action_keys: List of action keys (without the ``action.`` prefix).
        meta: Arbitrary dataset-level metadata dict. Common keys:
            ``"format"`` (e.g. ``"lerobot"``), ``"robot"``, ``"fps"``.
    """

    path: str
    num_episodes: int
    total_frames: int
    modalities: List[str] = field(default_factory=list)
    action_keys: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
