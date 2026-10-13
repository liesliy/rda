"""Metric implementations for RDA.

All metrics inherit from MetricBase and return a MetricResult when
computed over an episode.

Two orthogonal classification systems are provided:

**Architectural Layers** (how the metric is computed / what it checks):
  Layer 1 — Data Integrity: deterministic pass/fail hard checks
  Layer 2 — Temporal & Motion Anomaly: observational measurements
  Layer 3 — Dataset Utility: training data efficiency analysis
  Layer 4 — Security & Privacy: PII / sensitive exposure (v0.9.22)

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
    NumericSanityMetric,
)
from rda.metrics.temporal import (
    TimestampValidityMetric,
    SensorSyncMetric,
    JitterMetric,
    TemporalStructureMetric,
)
from rda.metrics.motion import (
    JointLimitMetric,
    VelocityMetric,
    ActionDiscontinuityMetric,
    IdleRatioMetric,
)
from rda.metrics.distribution import DistributionMetric, CoverageMetric
from rda.metrics.pii_scan import PIIScanMetric
from rda.metrics.video_integrity import VideoFrameIntegrityMetric
from rda.metrics.visual_integrity import (
    VideoFreezeMetric,
    VideoStreamPresenceMetric,
    VideoStreamSpanConsistencyMetric,
    VideoStreamTemporalOffsetMetric,
    VideoStreamTemporalDriftMetric,
    VideoTimestampAlignmentMetric,
)
from rda.metrics.visual_quality import VisualQualityMetric

# ---------------------------------------------------------------------------
# Architectural Layer classification (how the metric works)
# ---------------------------------------------------------------------------

LAYER1_INTEGRITY: List[Type[MetricBase]] = [
    MissingFramesMetric,
    NaNInfMetric,
    SchemaShapeMetric,
    TimestampValidityMetric,
    JointLimitMetric,
    VideoFrameIntegrityMetric,
    VideoFreezeMetric,
    VideoTimestampAlignmentMetric,
    VideoStreamPresenceMetric,
    NumericSanityMetric,
]
"""Layer 1 — Data Integrity: deterministic hard checks (pass/exclude).

v0.9: VideoStreamSyncMetric split into VideoStreamPresenceMetric (L1)
plus three diagnostic metrics in L2.
v0.9.18: Added NumericSanityMetric (WARNING-level, does not affect verdict).
"""

LAYER2_TEMPORAL_MOTION: List[Type[MetricBase]] = [
    SensorSyncMetric,
    JitterMetric,
    VelocityMetric,
    ActionDiscontinuityMetric,
    VisualQualityMetric,
    VideoStreamSpanConsistencyMetric,
    VideoStreamTemporalOffsetMetric,
    VideoStreamTemporalDriftMetric,
]
"""Layer 2 — Temporal & Motion Anomaly: observational measurements.

v0.9: Added three video stream diagnostics from the split of
VideoStreamSyncMetric — span_consistency, temporal_offset, temporal_drift.
"""

LAYER3_DATASET_UTILITY: List[Type[MetricBase]] = [
    IdleRatioMetric,
    DistributionMetric,
    CoverageMetric,
    TemporalStructureMetric,
]
"""Layer 3 — Dataset Utility: training data efficiency and coverage.

v0.9: Added TemporalStructureMetric (renamed from TemporalSufficiencyMetric).
Moved from L2 to L3 since it describes dataset-level temporal structure
rather than per-episode motion anomalies."""

LAYER4_SECURITY_PRIVACY: List[Type[MetricBase]] = [
    PIIScanMetric,
]
"""Layer 4 — Security & Privacy: personally identifiable exposure.

v0.9.22: added for YD/T 6771-2026 安全性 (the only P0 admission gate among
the eight quality dimensions). Deliberately a separate layer rather than
folding into L1: ``rda/report/json_report.py`` groups results by
``LAYER1_INTEGRITY`` for the integrity section, and a privacy finding is
neither an integrity failure nor a quality measurement.

WARNING-level by default (findings visible, verdict untouched); promoted to
the critical gate only under ``--pii-strict``."""

ALL_METRICS: List[Type[MetricBase]] = (
    LAYER1_INTEGRITY + LAYER2_TEMPORAL_MOTION
    + LAYER3_DATASET_UTILITY + LAYER4_SECURITY_PRIVACY
)
"""All metric classes, in architectural layer order.

v0.9: grew from 18 to 20 with the video_stream_sync split into
4 independent metrics (1 L1 + 3 L2).
v0.9.22: grew to 23 with the addition of pii_scan (L4)."""

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
    "VideoFrameIntegrityMetric",
    "VideoFreezeMetric",
    "VideoTimestampAlignmentMetric",
    "VideoStreamPresenceMetric",
    "NumericSanityMetric",
    # Metric classes — Layer 2 (Temporal & Motion)
    "SensorSyncMetric",
    "JitterMetric",
    "VelocityMetric",
    "ActionDiscontinuityMetric",
    "VisualQualityMetric",
    "VideoStreamSpanConsistencyMetric",
    "VideoStreamTemporalOffsetMetric",
    "VideoStreamTemporalDriftMetric",
    # Metric classes — Layer 3 (Dataset Utility)
    "IdleRatioMetric",
    "DistributionMetric",
    "CoverageMetric",
    "TemporalStructureMetric",
    # Metric classes — Layer 4 (Security & Privacy)
    "PIIScanMetric",
    # Backward compatibility alias (deprecated, use TemporalStructureMetric)
    "TemporalSufficiencyMetric",
    # Architectural Layer classification
    "LAYER1_INTEGRITY",
    "LAYER2_TEMPORAL_MOTION",
    "LAYER3_DATASET_UTILITY",
    "LAYER4_SECURITY_PRIVACY",
    "ALL_METRICS",
    # Portability Tier classification (MVP Spec v0.2.0 §1.5)
    "TIER1_UNIVERSAL",
    "TIER2_NORMALIZABLE",
    "TIER3_PLATFORM_SPECIFIC",
]
