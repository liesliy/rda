"""Metric implementations for RDA.

All metrics inherit from MetricBase and return a MetricResult when
computed over an episode.

Two orthogonal classification systems are provided:

**Architectural Layers** (how the metric is computed / what it checks):
  Layer 1 — Data Integrity: deterministic pass/fail hard checks
  Layer 2 — Temporal & Motion Anomaly: observational measurements
  Layer 3 — Dataset Utility: training data efficiency analysis

**Portability Tiers** (cross-platform comparability, per MVP Spec v0.2.0 §1.5):
  Tier 1 — Universal: directly comparable across platforms
      duration_sec, spike_count, effective_motion_ratio
  Tier 2 — Normalizable: comparable after platform normalization
      velocity, acceleration, jerk, path_length
  Tier 3 — Platform-specific: only meaningful on specific platforms
      joint_limit, workspace, torque, force, tactile
"""

from typing import List, Type

from rda.metrics.base import MetricBase, MetricResult, MetricAvailability, AssessmentStatus
from rda.metrics.integrity import (
    MissingFramesMetric,
    NaNInfMetric,
    SchemaShapeMetric,
)
from rda.metrics.temporal import (
    TimestampValidityMetric,
    SensorSyncMetric,
    JitterMetric,
    TemporalSufficiencyMetric,
)
from rda.metrics.motion import (
    JointLimitMetric,
    VelocityMetric,
    ActionDiscontinuityMetric,
    IdleRatioMetric,
)
from rda.metrics.distribution import DistributionMetric, CoverageMetric

# ---------------------------------------------------------------------------
# Architectural Layer classification (how the metric works)
# ---------------------------------------------------------------------------

LAYER1_INTEGRITY: List[Type[MetricBase]] = [
    MissingFramesMetric,
    NaNInfMetric,
    SchemaShapeMetric,
    TimestampValidityMetric,
    JointLimitMetric,
]
"""Layer 1 — Data Integrity: deterministic hard checks (pass/exclude)."""

LAYER2_TEMPORAL_MOTION: List[Type[MetricBase]] = [
    SensorSyncMetric,
    JitterMetric,
    VelocityMetric,
    ActionDiscontinuityMetric,
    TemporalSufficiencyMetric,
]
"""Layer 2 — Temporal & Motion Anomaly: observational measurements."""

LAYER3_DATASET_UTILITY: List[Type[MetricBase]] = [
    IdleRatioMetric,
    DistributionMetric,
    CoverageMetric,
]
"""Layer 3 — Dataset Utility: training data efficiency and coverage."""

ALL_METRICS: List[Type[MetricBase]] = LAYER1_INTEGRITY + LAYER2_TEMPORAL_MOTION + LAYER3_DATASET_UTILITY
"""All 12 metric classes, in architectural layer order."""

# ---------------------------------------------------------------------------
# Portability Tier classification (per MVP Spec v0.2.0 §1.5)
# ---------------------------------------------------------------------------
# Tier 1 — Universal metrics: directly comparable across platforms.
#   These form the core ranking signal (3 metrics, proven ρ=0.96 vs full set).
TIER1_UNIVERSAL: List[Type[MetricBase]] = [
    ActionDiscontinuityMetric,  # → spike_count
    IdleRatioMetric,            # → effective_motion_ratio (and implicitly duration)
    DistributionMetric,         # → duration_sec, path_length
]
"""Tier 1 — Universal metrics comparable across any robot platform.

Maps to MVP Spec v0.2.0 §1.5:
  - duration_sec             ← DistributionMetric.measurement.duration_sec
  - spike_count              ← ActionDiscontinuityMetric.measurement.spike_count
  - effective_motion_ratio   ← IdleRatioMetric.measurement.effective_motion_ratio
"""

# Tier 2 — Normalizable metrics: comparable after platform-specific scaling.
#   Useful as diagnostic signals but not for raw cross-platform ranking.
TIER2_NORMALIZABLE: List[Type[MetricBase]] = [
    VelocityMetric,      # velocity / acceleration
    JitterMetric,        # sampling jitter (related to temporal quality)
    SensorSyncMetric,    # sensor sync offset (needs platform baseline)
]
"""Tier 2 — Normalizable metrics requiring platform calibration.

Per MVP Spec v0.2.0 §1.5: velocity, acceleration, jerk, path_length.
These metrics are valid diagnostic signals but require per-platform
normalization before cross-platform comparison.
"""

# Tier 3 — Platform-specific metrics: only meaningful on certain platforms.
TIER3_PLATFORM_SPECIFIC: List[Type[MetricBase]] = [
    JointLimitMetric,    # requires robot joint limits config
    CoverageMetric,      # workspace / state space is task-specific
    MissingFramesMetric,
    NaNInfMetric,
    SchemaShapeMetric,
    TimestampValidityMetric,
]
"""Tier 3 — Platform-specific or integrity metrics.

Joint-limit and workspace metrics require robot-specific configuration.
Integrity metrics are pass/fail checks rather than ranking signals.
"""

__all__ = [
    "MetricBase",
    "MetricResult",
    "MetricAvailability",
    "AssessmentStatus",
    # Metric classes — Layer 1 (Integrity)
    "MissingFramesMetric",
    "NaNInfMetric",
    "SchemaShapeMetric",
    "TimestampValidityMetric",
    "JointLimitMetric",
    # Metric classes — Layer 2 (Temporal & Motion)
    "SensorSyncMetric",
    "JitterMetric",
    "VelocityMetric",
    "ActionDiscontinuityMetric",
    "TemporalSufficiencyMetric",
    # Metric classes — Layer 3 (Dataset Utility)
    "IdleRatioMetric",
    "DistributionMetric",
    "CoverageMetric",
    # Architectural Layer classification
    "LAYER1_INTEGRITY",
    "LAYER2_TEMPORAL_MOTION",
    "LAYER3_DATASET_UTILITY",
    "ALL_METRICS",
    # Portability Tier classification (MVP Spec v0.2.0 §1.5)
    "TIER1_UNIVERSAL",
    "TIER2_NORMALIZABLE",
    "TIER3_PLATFORM_SPECIFIC",
]
