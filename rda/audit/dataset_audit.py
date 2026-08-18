"""Dataset-level audit logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rda.audit.episode_audit import EpisodeAuditResult, EpisodeAuditor
from rda.audit.rules import AuditVerdict
from rda.calibration.reference import ReferenceProfile
from rda.io.schema import DatasetInfo


@dataclass
class DatasetAuditResult:
    """Result of auditing an entire dataset.

    Attributes:
        dataset_info: Metadata about the dataset.
        episodes: Mapping from episode index to EpisodeAuditResult.
        verdict_counts: Summary count of each verdict across all episodes.
    """

    dataset_info: DatasetInfo
    episodes: Dict[int, EpisodeAuditResult] = field(default_factory=dict)
    verdict_counts: Dict[AuditVerdict, int] = field(default_factory=dict)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def compute_verdict_counts(self) -> Dict[AuditVerdict, int]:
        """Recount verdict tallies from the episode results."""
        counts = {v: 0 for v in AuditVerdict}
        for ep_result in self.episodes.values():
            counts[ep_result.verdict] += 1
        self.verdict_counts = counts
        return counts


class DatasetAuditor:
    """Orchestrates the full audit of a dataset, episode by episode.

    Args:
        episode_auditor: EpisodeAuditor instance to use for each episode.
            If None, a default EpisodeAuditor is created.
        reference: Optional :class:`ReferenceProfile` for behavioral scoring.
            When provided and no ``episode_auditor`` is given, a
            :class:`~rda.calibration.BehavioralScorer` is automatically
            created and passed to the EpisodeAuditor.
    """

    def __init__(
        self,
        episode_auditor: Optional[EpisodeAuditor] = None,
        reference: Optional[ReferenceProfile] = None,
    ) -> None:
        """Initialize the dataset auditor.

        Args:
            episode_auditor: EpisodeAuditor instance to use for each episode.
                If None, a default EpisodeAuditor with all metrics is created.
            reference: Optional ReferenceProfile for behavioral scoring.
                When provided and episode_auditor is None, a BehavioralScorer
                is automatically created and injected into the EpisodeAuditor.
        """
        if episode_auditor is not None:
            self.episode_auditor = episode_auditor
        elif reference is not None:
            from rda.calibration.scorer import BehavioralScorer
            scorer = BehavioralScorer(reference)
            self.episode_auditor = EpisodeAuditor(scorer=scorer)
        else:
            self.episode_auditor = EpisodeAuditor()

    def audit_dataset(
        self,
        dataset_info: DatasetInfo,
        episode_iter,
    ) -> DatasetAuditResult:
        """Audit all episodes in the dataset.

        Args:
            dataset_info: Dataset metadata.
            episode_iter: Iterator yielding EpisodeData objects.

        Returns:
            DatasetAuditResult with per-episode results and summary counts.
        """
        result = DatasetAuditResult(dataset_info=dataset_info)

        for episode in episode_iter:
            ep_result = self.episode_auditor.audit(episode)
            result.episodes[episode.episode_index] = ep_result

        result.compute_verdict_counts()
        return result
