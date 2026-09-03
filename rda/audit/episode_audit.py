"""Episode-level audit logic.

The :class:`EpisodeAuditor` runs a set of metrics against a single episode
and produces an :class:`EpisodeAuditResult` with per-metric results and an
overall classification verdict.

P0-1: Reference Calibration Engine integration
-----------------------------------------------
When a :class:`~rda.calibration.BehavioralScorer` is provided (i.e. a
reference profile has been calibrated), each episode is also scored against
the reference distribution. Two integration points:

1. **Per-metric z-score / percentile**: Portable metrics get their
   ``z_score`` and ``percentile`` fields populated from the reference.
2. **Overall deviation score**: The result carries an aggregate
   ``deviation_score`` computed via PCA (or RMS fallback).

When no scorer is provided, the auditor falls back to the original
rule-based verdict — full backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rda.audit.rules import AuditVerdict, classify_episode, upgrade_verdict_by_behavior, compute_behavior_severity
from rda.io.schema import EpisodeData
from rda.metrics.base import MetricAvailability, MetricBase, MetricResult


# Mapping from portable metric name to the corresponding audit metric name
# (the one that computes it). Used to inject reference scores.
_PORTABLE_TO_METRIC: Dict[str, str] = {
    "duration_sec": "distribution",
    "spike_count": "action_discontinuity",
    "effective_motion_ratio": "idle_ratio",
}


@dataclass
class EpisodeAuditResult:
    """Result of auditing a single episode.

    Attributes:
        episode_index: Index of the audited episode.
        num_frames: Number of frames in the episode.
        metrics: Mapping from metric name to MetricResult.
        verdict: Final classification verdict (PASS / REVIEW / EXCLUDE).
        deviation_score: Primary deviation score relative to a reference
            profile (mirrors ``combined_score`` by default).
            ``None`` when no reference is available.
        deviation_method: How the deviation score was computed
            (``"pca"``, ``"rms"``, or ``None``).
        portable_score: Portable-only (Tier-1) deviation score.
            Cross-platform comparable.
        platform_score: Platform-specific deviation score. Only
            meaningful when the reference was calibrated on the same
            platform.
        combined_score: Combined deviation score (portable + platform).
            For same-platform deep dive.
        has_platform_metrics: Whether the reference profile contains
            platform-specific metrics (and therefore ``platform_score``
            is meaningful).
    """

    episode_index: int
    num_frames: int
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    verdict: AuditVerdict = AuditVerdict.PASS
    task_index: Optional[int] = None
    task_description: Optional[str] = None
    deviation_score: Optional[float] = None
    deviation_method: Optional[str] = None
    portable_score: Optional[float] = None
    platform_score: Optional[float] = None
    combined_score: Optional[float] = None
    has_platform_metrics: bool = False
    behavior_severity: float = 0.0


class EpisodeAuditor:
    """Runs a set of metrics against a single episode and classifies it.

    Args:
        metrics: List of metric instances to run. If ``None``, defaults
            to all registered metrics.
        scorer: Optional :class:`~rda.calibration.BehavioralScorer` for
            reference-based behavioral scoring. When provided, per-metric
            z-scores / percentiles and an overall deviation score are
            computed in addition to the rule-based verdict.
    """

    def __init__(
        self,
        metrics: Sequence[MetricBase] | None = None,
        scorer: object | None = None,
    ) -> None:
        """Initialize the episode auditor.

        Args:
            metrics: List of metric instances to run.
                If None, defaults to all registered metrics.
            scorer: Optional BehavioralScorer instance for reference-based
                deviation scoring. The type is ``object`` to avoid a
                hard import dependency on the calibration module.
        """
        if metrics is None:
            from rda.metrics import ALL_METRICS

            metrics = [cls() for cls in ALL_METRICS]
        self.metrics: List[MetricBase] = list(metrics)
        self.scorer = scorer

    def audit(self, episode: EpisodeData) -> EpisodeAuditResult:
        """Run all metrics on an episode and return the audit result.

        Args:
            episode: The episode data to audit.

        Returns:
            EpisodeAuditResult with per-metric results, overall verdict,
            and (when a scorer is configured) reference-based deviation
            scores.
        """
        metric_results: Dict[str, MetricResult] = {}
        for metric in self.metrics:
            try:
                result = metric.compute(episode)
            except Exception as exc:
                # Single metric failure must not abort the entire audit.
                # Record as ERROR so it's visible in the report but doesn't
                # affect the verdict (skipped metrics are ignored by classify_episode).
                result = MetricResult.make_error(
                    name=metric.name,
                    reason=f"metric_computation_error: {type(exc).__name__}: {exc}",
                    message=f"Metric {metric.name} failed with {type(exc).__name__}; skipped.",
                )
            metric_results[result.name] = result

        # --- Reference-calibrated scoring (P0-1) ---
        deviation_score: Optional[float] = None
        deviation_method: Optional[str] = None
        portable_score: Optional[float] = None
        platform_score: Optional[float] = None
        combined_score: Optional[float] = None
        has_platform_metrics: bool = False

        if self.scorer is not None:
            behavior = self.scorer.score_episode(episode)
            deviation_score = behavior.deviation_score
            deviation_method = behavior.method
            portable_score = behavior.portable_score
            platform_score = behavior.platform_score
            combined_score = behavior.combined_score
            has_platform_metrics = behavior.has_platform_metrics

            # Inject z-score and percentile into corresponding metric results
            for portable_name, metric_name in _PORTABLE_TO_METRIC.items():
                if metric_name not in metric_results:
                    continue
                z = behavior.metric_scores.get(portable_name)
                if z is None:
                    continue
                # Approximate per-metric percentile from the profile
                pct = self._approx_percentile_for_metric(
                    portable_name, z
                )
                old = metric_results[metric_name]
                metric_results[metric_name] = old.with_reference_score(z, pct)

        # Rule-based verdict (unchanged — rule-based still determines
        # PASS/REVIEW/EXCLUDE; deviation score is additional signal)
        verdict = classify_episode(list(metric_results.values()))

        # Behavior-aware verdict upgrade: if rule-based is PASS but
        # behavioral metrics show anomalies, upgrade to REVIEW
        verdict = upgrade_verdict_by_behavior(verdict, list(metric_results.values()))

        # Compute behavior severity and generate findings for explainability
        behavior_severity, behavior_findings = compute_behavior_severity(list(metric_results.values()))

        # Attach behavioral findings to the corresponding metric results
        # so they appear in the audit report with explanations
        for finding in behavior_findings:
            metric_name = finding["metric"]
            if metric_name in metric_results:
                old_result = metric_results[metric_name]
                # Create a new MetricResult with has_finding=True and updated assessment
                new_assessment = {
                    "status": "review",
                    "severity": "medium" if finding["severity"] >= 20 else "low",
                    "reason": finding["reason"],
                }
                metric_results[metric_name] = MetricResult(
                    name=old_result.name,
                    availability=old_result.availability,
                    measurement=dict(old_result.measurement),
                    assessment=new_assessment,
                    details=dict(old_result.details),
                    message=old_result.message,
                    z_score=old_result.z_score,
                    percentile=old_result.percentile,
                    has_finding=True,
                )

        # --- P0 fix (v0.4.12): 0-frame episodes must not PASS ---
        # If an episode has 0 frames (e.g. meta/data mapping mismatch),
        # the verdict is silently PASS because no metrics fire. Force EXCLUDE
        # so this failure mode is visible.
        if episode.num_frames == 0:
            verdict = AuditVerdict.EXCLUDE
            # Add a synthetic finding so the report explains why
            metric_results["_zero_frame_guard"] = MetricResult(
                name="_zero_frame_guard",
                availability=MetricAvailability.AVAILABLE,
                measurement={"num_frames": 0},
                assessment={
                    "status": "fail",
                    "severity": "high",
                    "reason": "Episode has 0 frames — meta/data mapping mismatch or missing data file. Cannot evaluate quality.",
                },
                details={"guard": "zero_frame_exclusion"},
                message="EXCLUDE: 0 frames read — data may be missing or misaligned.",
                has_finding=True,
            )

        return EpisodeAuditResult(
            episode_index=episode.episode_index,
            num_frames=episode.num_frames,
            metrics=metric_results,
            verdict=verdict,
            task_index=(episode.meta or {}).get("task_index"),
            task_description=(episode.meta or {}).get("task_description"),
            deviation_score=deviation_score,
            deviation_method=deviation_method,
            portable_score=portable_score,
            platform_score=platform_score,
            combined_score=combined_score,
            has_platform_metrics=has_platform_metrics,
            behavior_severity=behavior_severity,
        )

    def __call__(self, episode: EpisodeData) -> EpisodeAuditResult:
        """Convenience callable wrapper around :meth:`audit`."""
        return self.audit(episode)

    # --- Internal helpers ---

    def _approx_percentile_for_metric(
        self, portable_name: str, z_score: float
    ) -> float:
        """Approximate percentile rank for a single metric from its z-score.

        Uses the reference profile's percentiles via the scorer.
        Falls back to 0.5 if the scorer doesn't have a profile.
        """
        try:
            profile = self.scorer.reference  # type: ignore[union-attr]
        except AttributeError:
            return 0.5

        stats = profile.metrics.get(portable_name)
        if stats is None:
            return 0.5

        # Reverse the robust z formula to get raw value, then look up percentile
        if stats.mad == 0.0:
            return 0.5
        raw_value = stats.median + (z_score * stats.mad) / 0.6745
        return stats.percentile_rank(raw_value)
