"""Base class and result type for all RDA metrics.

This module defines the abstract interface that every metric must implement,
along with the standard result type (MetricResult) used throughout the audit
pipeline.

Design principles:
  - **Unified interface**: every metric subclasses :class:`MetricBase` and
    implements :meth:`MetricBase.compute`, which takes an :class:`EpisodeData`
    and returns a :class:`MetricResult`.
  - **Three-schema result**: each result carries ``availability``,
    ``measurement``, and ``assessment`` — so callers can distinguish "can't
    compute" from "computed and found problems" from "computed and clean".
  - **Backward compatibility**: ``passed`` and ``score`` properties are derived
    read-only views for legacy consumers.
  - **Reference-calibrated fields**: ``z_score`` and ``percentile`` are
    populated by the :class:`~rda.calibration.BehavioralScorer` when a
    reference profile is available; they default to ``None`` otherwise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from rda.io.schema import EpisodeData


class MetricAvailability(str, Enum):
    """Whether the metric could be computed for this episode.

    Attributes:
        AVAILABLE: Required data was present and computation succeeded.
        NOT_AVAILABLE: Required data is missing (e.g. no ``stream_timestamps``,
            no ``observation.state``) — metric is not applicable.
        ERROR: Required data was present but computation failed
            (e.g. all-NaN array, shape mismatch).
    """

    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"
    ERROR = "error"


class AssessmentStatus(str, Enum):
    """Outcome of the metric assessment.

    Attributes:
        PASS: Data meets quality expectations (Layer 1), or the metric is
            purely observational (Layer 2 / 3) and reports measurements only.
        REVIEW: Statistical anomaly detected; warrants human review.
        EXCLUDE: Deterministic error — data is structurally unusable.
        SKIPPED: Computation was skipped due to an upstream failure.
        NA: Not applicable — required data not available.
    """

    PASS = "pass"
    REVIEW = "review"
    EXCLUDE = "exclude"
    SKIPPED = "skipped"
    NA = "na"


@dataclass
class MetricResult:
    """Standard result returned by every metric.

    Three-layer schema:
      availability:  whether the metric could be computed.
      measurement:   raw numerical output (no abstraction into score).
      assessment:    structured verdict {status, severity, reason}.

    Reference-calibrated fields (P0-1):
      z_score:    Robust z-score relative to a reference profile (MAD-based).
                  ``None`` when no reference profile is available.
      percentile: Percentile rank relative to the reference population [0, 1].
                  ``None`` when no reference profile is available.

    Backward-compatible properties ``passed`` and ``score`` are provided
    as read-only derivations for legacy code.
    """

    name: str
    availability: MetricAvailability = MetricAvailability.AVAILABLE
    measurement: Dict[str, Any] = field(default_factory=dict)
    assessment: Dict[str, Any] = field(default_factory=lambda: {"status": "pass", "severity": None, "reason": None})
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    z_score: Optional[float] = None
    percentile: Optional[float] = None
    has_finding: bool = False

    # --- Backward-compatible read-only properties ---

    @property
    def passed(self) -> bool:
        """Derive pass/fail from assessment status (backward compat)."""
        status = self.assessment.get("status", "pass")
        return status in ("pass",)

    @property
    def score(self) -> float:
        """Derive a float score from assessment (backward compat).

        pass → 1.0, review → 0.5, exclude/skipped/na → 0.0
        Falls back to measurement["score_compat"] if available.
        """
        status = self.assessment.get("status", "pass")
        if status == "pass":
            return self.measurement.get("score_compat", 1.0)
        elif status == "review":
            return self.measurement.get("score_compat", 0.5)
        else:
            return 0.0

    # --- Reference-calibration mutators ---

    def with_reference_score(self, z_score: float, percentile: float) -> "MetricResult":
        """Return a copy with z_score and percentile populated.

        Used by :class:`~rda.audit.episode_audit.EpisodeAuditor` when a
        :class:`~rda.calibration.ReferenceProfile` is available.

        Args:
            z_score: Robust z-score relative to reference.
            percentile: Percentile rank [0, 1] relative to reference.

        Returns:
            A new :class:`MetricResult` with the same fields except
            ``z_score`` and ``percentile`` are set.
        """
        return MetricResult(
            name=self.name,
            availability=self.availability,
            measurement=dict(self.measurement),
            assessment=dict(self.assessment),
            details=dict(self.details),
            message=self.message,
            z_score=float(z_score),
            percentile=float(percentile),
            has_finding=self.has_finding,
        )

    # --- Convenience constructors ---

    @classmethod
    def make_pass(cls, name: str, measurement: Dict[str, Any],
                  message: str = "", details: Dict[str, Any] | None = None,
                  severity: Optional[str] = None,
                  baseline: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Construct a PASS result.

        Args:
            baseline: Optional baseline metadata (method, scope, threshold)
                for observational metrics that need transparency about
                how the assessment was derived.
        """
        assessment: Dict[str, Any] = {"status": "pass", "severity": severity, "reason": None}
        if baseline:
            assessment["baseline"] = baseline
        return cls(
            name=name,
            availability=MetricAvailability.AVAILABLE,
            measurement=measurement,
            assessment=assessment,
            details=details or {},
            message=message,
        )

    @classmethod
    def make_review(cls, name: str, measurement: Dict[str, Any],
                    reason: str = "", message: str = "",
                    details: Dict[str, Any] | None = None,
                    severity: str = "medium",
                    baseline: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Construct a REVIEW-level finding result.

        has_finding is set to True so the rule engine can classify this
        metric's finding. The assessment.status is informational.

        Args:
            baseline: Optional baseline metadata (method, scope, threshold)
                for observational metrics that need transparency about
                how the assessment was derived.
        """
        assessment: Dict[str, Any] = {"status": "review", "severity": severity, "reason": reason}
        if baseline:
            assessment["baseline"] = baseline
        return cls(
            name=name,
            availability=MetricAvailability.AVAILABLE,
            measurement=measurement,
            assessment=assessment,
            details=details or {},
            message=message,
            has_finding=True,
        )

    @classmethod
    def make_exclude(cls, name: str, reason: str = "",
                     message: str = "", details: Dict[str, Any] | None = None) -> MetricResult:
        """Construct an EXCLUDE result (deterministic error).

        has_finding is set to True so the rule engine classifies this
        as a critical finding that triggers the EXCLUDE verdict.
        """
        return cls(
            name=name,
            availability=MetricAvailability.AVAILABLE,
            measurement={},
            assessment={"status": "exclude", "severity": "high", "reason": reason},
            details=details or {},
            message=message,
            has_finding=True,
        )

    @classmethod
    def make_na(cls, name: str, reason: str = "",
                message: str = "", details: Dict[str, Any] | None = None) -> MetricResult:
        """Construct a N/A result (data not available)."""
        return cls(
            name=name,
            availability=MetricAvailability.NOT_AVAILABLE,
            measurement={},
            assessment={"status": "na", "severity": None, "reason": reason},
            details=details or {},
            message=message,
        )

    @classmethod
    def make_error(cls, name: str, reason: str = "",
                   message: str = "", details: Dict[str, Any] | None = None) -> MetricResult:
        """Construct an ERROR result."""
        return cls(
            name=name,
            availability=MetricAvailability.ERROR,
            measurement={},
            assessment={"status": "skipped", "severity": "high", "reason": reason},
            details=details or {},
            message=message,
        )


class MetricBase(ABC):
    """Abstract base class for all RDA audit metrics.

    Subclasses must implement :meth:`compute` which accepts an
    :class:`EpisodeData` and returns a :class:`MetricResult`.
    """

    name: str = "base"
    description: str = "Base metric"

    @abstractmethod
    def compute(self, episode: EpisodeData) -> MetricResult:
        """Compute the metric for a single episode.

        Args:
            episode: The canonical episode data to evaluate.

        Returns:
            MetricResult containing the metric score, pass/fail status,
            and structured details.
        """
        ...

    def __call__(self, episode: EpisodeData) -> MetricResult:
        """Convenience callable wrapper around :meth:`compute`."""
        return self.compute(episode)
