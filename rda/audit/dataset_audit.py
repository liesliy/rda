"""Dataset-level audit logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rda.audit.episode_audit import EpisodeAuditResult, EpisodeAuditor
from rda.audit.rules import AuditVerdict
from rda.calibration.reference import ReferenceProfile
from rda.io.schema import DatasetInfo
from rda.report.dataset_summary import DatasetSummaryResult, compute_dataset_summary


@dataclass
class DatasetAuditResult:
    """Result of auditing an entire dataset.

    Attributes:
        dataset_info: Metadata about the dataset.
        episodes: Mapping from episode index to EpisodeAuditResult.
        verdict_counts: Summary count of each verdict across all episodes.
        dataset_summary: Aggregated dataset-level summary statistics (v0.9).
    """

    dataset_info: DatasetInfo
    episodes: Dict[int, EpisodeAuditResult] = field(default_factory=dict)
    verdict_counts: Dict[AuditVerdict, int] = field(default_factory=dict)
    dataset_summary: Optional[DatasetSummaryResult] = None

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "dataset_info": {
                "name": self.dataset_info.name if hasattr(self.dataset_info, "name") else str(self.dataset_info),
                "num_episodes": self.num_episodes,
            },
            "verdict_counts": {v.name: c for v, c in self.verdict_counts.items()},
            "episodes": {
                str(idx): ep.to_dict() if hasattr(ep, "to_dict") else str(ep)
                for idx, ep in self.episodes.items()
            },
        }
        if self.dataset_summary is not None:
            d["dataset_summary"] = self.dataset_summary.to_dict().get("dataset_summary", {})
        return d


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

        After all episodes are audited, computes dataset-level summary
        statistics (v0.9) from L2 diagnostic measurements.

        Args:
            dataset_info: Dataset metadata.
            episode_iter: Iterator yielding EpisodeData objects.

        Returns:
            DatasetAuditResult with per-episode results, summary counts,
            and dataset-level summary statistics.
        """
        result = DatasetAuditResult(dataset_info=dataset_info)

        for episode in episode_iter:
            ep_result = self.episode_auditor.audit(episode)
            result.episodes[episode.episode_index] = ep_result

        result.compute_verdict_counts()

        # Compute dataset-level summary (v0.9)
        episode_results_for_summary = []
        for idx, ep_result in result.episodes.items():
            ep_dict = {
                "episode_index": idx,
                "verdict": ep_result.verdict.value if hasattr(ep_result.verdict, "value") else str(ep_result.verdict),
                "measurements": {},
                "findings": [],
            }
            # Extract measurements from episode result
            if hasattr(ep_result, "results"):
                for metric_result in ep_result.results:
                    if hasattr(metric_result, "name") and hasattr(metric_result, "measurement"):
                        if metric_result.measurement:
                            ep_dict["measurements"][metric_result.name] = metric_result.measurement
            if hasattr(ep_result, "findings"):
                ep_dict["findings"] = ep_result.findings
            episode_results_for_summary.append(ep_dict)

        result.dataset_summary = compute_dataset_summary(episode_results_for_summary)

        return result
