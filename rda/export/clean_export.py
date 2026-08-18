"""Clean Dataset Export — export a dataset excluding bad episodes.

Takes an audited LeRobot v3.0 dataset and a set of user verdicts
(KEEP / REMOVE / UNCERTAIN per episode), then produces a clean
output directory in the same LeRobot v3.0 structure containing
only the accepted episodes.

Decision logic
--------------
1. REMOVE  → always excluded
2. KEEP    → always included
3. UNCERTAIN → controlled by ``uncertainty_strategy``:
   - ``"keep"``   (default) → include
   - ``"remove"``            → exclude
4. Undecided episodes → fall back to AI verdict, then ``cleaning_strategy``:
   - PASS   → always include
   - REVIEW → controlled by cleaning_strategy
   - EXCLUDE → always exclude

Cleaning strategies
-------------------
- ``"conservative"`` (default): only remove AI-EXCLUDE undecided episodes.
  REVIEW episodes are kept.
- ``"aggressive"``: remove both AI-EXCLUDE and AI-REVIEW undecided episodes.
- ``custom`` (list): a list of AI verdict strings that should trigger removal,
  e.g. ``["EXCLUDE", "REVIEW"]``.

The exporter copies (never moves) parquet files from the source,
filters rows by ``episode_index`` column, and rewrites ``meta/info.json``
with updated counts.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExportReport:
    """Summary statistics returned after an export run.

    Attributes:
        source_path: Original dataset directory.
        output_path: Directory where the clean dataset was written.
        total_episodes: Episodes in the source dataset.
        kept_episodes: Episodes in the clean output.
        removed_episodes: Episodes excluded.
        removed_by_user: Excluded because user said REMOVE.
        removed_by_ai: Excluded because AI said EXCLUDE (undecided).
        removed_by_uncertainty: Excluded because UNCERTAIN + strategy=remove.
        removed_review: Excluded because AI said REVIEW + aggressive strategy.
        total_frames: Frames in the clean output.
        kept_episode_ids: Sorted list of episode indices kept.
        removed_episode_ids: Sorted list of episode indices removed.
        uncertainty_strategy: The strategy used for UNCERTAIN episodes.
        cleaning_strategy: The strategy used for undecided episodes.
        review_count: Number of undecided episodes with AI verdict REVIEW.
        undecided_count: Number of episodes with no user verdict.
    """

    source_path: str
    output_path: str
    total_episodes: int
    kept_episodes: int
    removed_episodes: int
    removed_by_user: int = 0
    removed_by_ai: int = 0
    removed_by_uncertainty: int = 0
    removed_review: int = 0
    total_frames: int = 0
    kept_episode_ids: List[int] = field(default_factory=list)
    removed_episode_ids: List[int] = field(default_factory=list)
    uncertainty_strategy: str = "keep"
    cleaning_strategy: str = "conservative"
    review_count: int = 0
    undecided_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "total_episodes": self.total_episodes,
            "kept_episodes": self.kept_episodes,
            "removed_episodes": self.removed_episodes,
            "removed_by_user": self.removed_by_user,
            "removed_by_ai": self.removed_by_ai,
            "removed_by_uncertainty": self.removed_by_uncertainty,
            "removed_review": self.removed_review,
            "total_frames": self.total_frames,
            "kept_episode_ids": self.kept_episode_ids,
            "removed_episode_ids": self.removed_episode_ids,
            "uncertainty_strategy": self.uncertainty_strategy,
            "cleaning_strategy": self.cleaning_strategy,
            "review_count": self.review_count,
            "undecided_count": self.undecided_count,
        }


# ---------------------------------------------------------------------------
# Main exporter
# ---------------------------------------------------------------------------

class CleanDatasetExporter:
    """Export a clean LeRobot v3.0 dataset after excluding bad episodes.

    Parameters
    ----------
    source_path : str | Path
        Root directory of the source LeRobot v3.0 dataset.
    output_path : str | Path
        Root directory where the clean dataset will be written.
    user_verdicts : dict
        Mapping ``episode_id (int) → {"decision": "KEEP"|"REMOVE"|"UNCERTAIN", ...}``.
    uncertainty_strategy : str
        How to treat UNCERTAIN episodes: ``"keep"`` (default) or ``"remove"``.
    ai_verdicts : dict | None
        Optional mapping ``episode_id (int) → "PASS"|"REVIEW"|"EXCLUDE"`` for
        episodes that have no user verdict.  When ``None``, the exporter
        reads them from the audit report if available, otherwise assumes PASS.
    cleaning_strategy : str | list
        How to treat undecided episodes (those without user verdict) based on
        their AI verdict.  Built-in options:

        - ``"conservative"`` (default) – only AI-EXCLUDE → remove;
          AI-REVIEW → keep.
        - ``"aggressive"`` – both AI-EXCLUDE and AI-REVIEW → remove.
        - Custom: pass a list of AI verdict strings that should trigger
          removal, e.g. ``["EXCLUDE", "REVIEW"]``.

    Examples
    --------
    >>> exporter = CleanDatasetExporter(
    ...     source_path="./data",
    ...     output_path="./clean_data",
    ...     user_verdicts={3: {"decision": "REMOVE"}},
    ... )
    >>> report = exporter.export()
    >>> print(report.kept_episodes, report.removed_episodes)
    """

    def __init__(
        self,
        source_path: str | Path,
        output_path: str | Path,
        user_verdicts: Dict[int, Dict[str, Any]],
        uncertainty_strategy: str = "keep",
        ai_verdicts: Optional[Dict[int, str]] = None,
        cleaning_strategy: str | list = "conservative",
    ) -> None:
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.user_verdicts = user_verdicts or {}
        self.uncertainty_strategy = uncertainty_strategy.lower()
        self.ai_verdicts = ai_verdicts or {}

        if self.uncertainty_strategy not in ("keep", "remove"):
            raise ValueError(
                f"uncertainty_strategy must be 'keep' or 'remove', "
                f"got '{self.uncertainty_strategy}'"
            )

        # Parse cleaning_strategy
        if isinstance(cleaning_strategy, list):
            self.cleaning_strategy = [v.upper() for v in cleaning_strategy]
            self._ai_remove_set = set(self.cleaning_strategy)
        else:
            cs = cleaning_strategy.lower()
            if cs == "conservative":
                self._ai_remove_set = {"EXCLUDE"}
            elif cs == "aggressive":
                self._ai_remove_set = {"EXCLUDE", "REVIEW"}
            else:
                raise ValueError(
                    f"cleaning_strategy must be 'conservative', 'aggressive', "
                    f"or a list of AI verdict strings, got '{cleaning_strategy}'"
                )
            self.cleaning_strategy = cs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, progress_callback: Optional[Any] = None) -> ExportReport:
        """Run the clean export and return an :class:`ExportReport`.

        Parameters
        ----------
        progress_callback : callable, optional
            Called as ``callback(current_step, total_steps, message)``
            so that UI code can display progress.

        Returns
        -------
        ExportReport
            Statistics about the export (kept / removed counts, frame
            totals, etc.).
        """
        self._validate_source()

        # Determine episode sets
        all_episode_ids = self._get_all_episode_ids()
        kept_ids, removed_ids, stats = self._compute_episode_sets(all_episode_ids)

        total_steps = self._count_data_chunks() + 2  # +meta + info.json
        step = 0

        # Copy meta files (excluding episodes/ parquet which we regenerate)
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, "Copying meta files...")
        self._copy_meta()

        # Copy and filter data parquet files
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, "Filtering data files...")
        kept_frame_count = self._copy_and_filter_data(kept_ids)

        # Copy tasks parquet and update info.json
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, "Updating metadata...")
        self._update_info_json(kept_ids, kept_frame_count)

        # Build report
        report = ExportReport(
            source_path=str(self.source_path),
            output_path=str(self.output_path),
            total_episodes=len(all_episode_ids),
            kept_episodes=len(kept_ids),
            removed_episodes=len(removed_ids),
            removed_by_user=stats["by_user"],
            removed_by_ai=stats["by_ai"],
            removed_by_uncertainty=stats["by_uncertainty"],
            removed_review=stats["by_review"],
            total_frames=kept_frame_count,
            kept_episode_ids=sorted(kept_ids),
            removed_episode_ids=sorted(removed_ids),
            uncertainty_strategy=self.uncertainty_strategy,
            cleaning_strategy=(
                self.cleaning_strategy
                if isinstance(self.cleaning_strategy, str)
                else list(self.cleaning_strategy)
            ),
            review_count=stats["review_count"],
            undecided_count=stats["undecided_count"],
        )
        return report

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_source(self) -> None:
        """Ensure the source dataset path exists and has the right structure."""
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Source dataset not found: {self.source_path}"
            )
        data_dir = self.source_path / "data"
        meta_dir = self.source_path / "meta"
        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Source dataset missing 'data/' directory: {self.source_path}"
            )
        if not meta_dir.is_dir():
            raise FileNotFoundError(
                f"Source dataset missing 'meta/' directory: {self.source_path}"
            )

    # ------------------------------------------------------------------
    # Episode set computation
    # ------------------------------------------------------------------

    def _get_all_episode_ids(self) -> Set[int]:
        """Scan all data parquet files and return the set of episode_index values."""
        data_dir = self.source_path / "data"
        episode_ids: Set[int] = set()
        for chunk_dir in sorted(data_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for pf in sorted(chunk_dir.iterdir()):
                if pf.suffix == ".parquet":
                    df = pd.read_parquet(pf, columns=["episode_index"])
                    episode_ids.update(df["episode_index"].unique().tolist())
        return episode_ids

    def _compute_episode_sets(
        self, all_ids: Set[int]
    ) -> tuple[Set[int], Set[int], Dict[str, int]]:
        """Decide which episodes to keep and which to remove.

        Uses ``cleaning_strategy`` to determine which AI verdicts for
        undecided episodes trigger removal.

        Returns
        -------
        kept_ids, removed_ids, stats
            stats has keys: by_user, by_ai, by_uncertainty, by_review,
            review_count, undecided_count
        """
        kept: Set[int] = set()
        removed: Set[int] = set()
        stats = {
            "by_user": 0, "by_ai": 0, "by_uncertainty": 0, "by_review": 0,
            "review_count": 0, "undecided_count": 0,
        }

        for ep_id in all_ids:
            user_v = self.user_verdicts.get(ep_id, {})
            decision = (user_v.get("decision") or "").upper()

            if decision == "REMOVE":
                removed.add(ep_id)
                stats["by_user"] += 1
            elif decision == "KEEP":
                kept.add(ep_id)
            elif decision == "UNCERTAIN":
                if self.uncertainty_strategy == "remove":
                    removed.add(ep_id)
                    stats["by_uncertainty"] += 1
                else:
                    kept.add(ep_id)
            else:
                # Undecided → use AI verdict + cleaning strategy
                stats["undecided_count"] += 1
                ai_verdict = self._get_ai_verdict(ep_id)
                if ai_verdict in self._ai_remove_set:
                    removed.add(ep_id)
                    if ai_verdict == "EXCLUDE":
                        stats["by_ai"] += 1
                    elif ai_verdict == "REVIEW":
                        stats["by_review"] += 1
                        stats["review_count"] += 1
                    else:
                        stats["by_ai"] += 1
                else:
                    kept.add(ep_id)
                    if ai_verdict == "REVIEW":
                        stats["review_count"] += 1

        return kept, removed, stats

    def _get_ai_verdict(self, ep_id: int) -> str:
        """Get the AI verdict for an episode, defaulting to PASS."""
        return self.ai_verdicts.get(ep_id, "PASS").upper()

    # ------------------------------------------------------------------
    # Data copying
    # ------------------------------------------------------------------

    def _count_data_chunks(self) -> int:
        """Count the number of chunk directories."""
        data_dir = self.source_path / "data"
        if not data_dir.exists():
            return 0
        return sum(1 for d in data_dir.iterdir() if d.is_dir())

    def _copy_meta(self) -> None:
        """Copy meta directory, excluding episode-level parquets that
        need to be regenerated."""
        src_meta = self.source_path / "meta"
        dst_meta = self.output_path / "meta"
        dst_meta.mkdir(parents=True, exist_ok=True)

        for item in src_meta.iterdir():
            dst_item = dst_meta / item.name
            if item.name == "episodes":
                # We'll handle episodes/ separately
                continue
            if item.is_file():
                shutil.copy2(str(item), str(dst_item))
            elif item.is_dir():
                if dst_item.exists():
                    shutil.rmtree(str(dst_item))
                shutil.copytree(str(item), str(dst_item))

        # Copy episodes/ directory (episode metadata parquet files)
        src_episodes = src_meta / "episodes"
        if src_episodes.exists():
            dst_episodes = dst_meta / "episodes"
            dst_episodes.mkdir(parents=True, exist_ok=True)
            for chunk_dir in src_episodes.iterdir():
                if not chunk_dir.is_dir():
                    continue
                dst_chunk = dst_episodes / chunk_dir.name
                dst_chunk.mkdir(parents=True, exist_ok=True)
                for pf in chunk_dir.iterdir():
                    if pf.suffix == ".parquet":
                        shutil.copy2(str(pf), str(dst_chunk / pf.name))

    def _copy_and_filter_data(self, kept_ids: Set[int]) -> int:
        """Copy data/ parquet files, filtering rows to keep only kept episodes.

        Returns the total number of frames (rows) in the clean output.
        """
        src_data = self.source_path / "data"
        dst_data = self.output_path / "data"
        dst_data.mkdir(parents=True, exist_ok=True)

        total_frames = 0

        for chunk_dir in sorted(src_data.iterdir()):
            if not chunk_dir.is_dir():
                continue
            dst_chunk = dst_data / chunk_dir.name
            dst_chunk.mkdir(parents=True, exist_ok=True)

            for pf in sorted(chunk_dir.iterdir()):
                if pf.suffix != ".parquet":
                    continue
                # Read, filter, write
                df = pd.read_parquet(pf)
                filtered = df[df["episode_index"].isin(kept_ids)]
                total_frames += len(filtered)

                if len(filtered) > 0:
                    # Reset index to avoid gaps
                    filtered = filtered.reset_index(drop=True)
                    out_path = dst_chunk / pf.name
                    filtered.to_parquet(out_path, index=False)
                else:
                    # Write an empty file with the same schema
                    empty = df.iloc[:0].reset_index(drop=True)
                    out_path = dst_chunk / pf.name
                    empty.to_parquet(out_path, index=False)

        return total_frames

    def _update_info_json(
        self, kept_ids: Set[int], total_frames: int
    ) -> None:
        """Update meta/info.json with correct episode and frame counts."""
        info_path = self.output_path / "meta" / "info.json"
        if not info_path.exists():
            # If no info.json existed, create a minimal one
            info = {}
        else:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)

        info["total_episodes"] = len(kept_ids)
        info["total_frames"] = total_frames

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helper: load user verdicts from JSON file
# ---------------------------------------------------------------------------

def load_verdicts_from_json(path: str | Path) -> Dict[int, Dict[str, Any]]:
    """Load user verdicts from a JSON file.

    Expected format:
    ``{"0": {"decision": "KEEP", "notes": "..."}, "3": {"decision": "REMOVE"}}``

    Keys are episode IDs as strings (JSON object keys are always strings);
    they are converted to int.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_ai_verdicts_from_report(report_path: str | Path) -> Dict[int, str]:
    """Extract AI verdicts from a JSON audit report.

    Reads ``episodes[].verdict`` and ``episodes[].episode_index``.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    verdicts: Dict[int, str] = {}
    for ep in report.get("episodes", []):
        ep_idx = ep.get("episode_index")
        verdict = ep.get("verdict")
        if ep_idx is not None and verdict is not None:
            verdicts[int(ep_idx)] = verdict
    return verdicts
