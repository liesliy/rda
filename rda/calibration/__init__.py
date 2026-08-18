"""Reference Calibration Engine.

Implements reference-based behavioral scoring that replaces the simple
rule-based metric checker with a statistically-grounded deviation score.

Key components:

- :class:`ReferenceProfile` — robust distribution stats for portable metrics
- :func:`calibrate` — build a reference profile from calibration episodes
- :class:`BehavioralScorer` — score new episodes against a reference profile
- :class:`BehavioralScore` — result dataclass with deviation + breakdown

Example::

    from rda.calibration import calibrate, BehavioralScorer

    profile = calibrate(calibration_episodes, platform="unitree_h1")
    scorer  = BehavioralScorer(profile)
    score   = scorer.score_episode(new_episode)

    print(f"Deviation score: {score.deviation_score:.2f} ({score.method})")
"""
from __future__ import annotations

from rda.calibration.portable import (
    ALL_SCORE_METRICS,
    METRIC_DIRECTIONS,
    PLATFORM_METRICS,
    PORTABLE_METRICS,
    extract_all_score_metrics,
    extract_platform_metrics,
    extract_portable_metrics,
)
from rda.calibration.reference import (
    MetricStats,
    ReferenceProfile,
    load_reference,
    save_reference,
)
from rda.calibration.scorer import BehavioralScore, BehavioralScorer, calibrate

__all__ = [
    "MetricStats",
    "ReferenceProfile",
    "BehavioralScore",
    "BehavioralScorer",
    "calibrate",
    "PORTABLE_METRICS",
    "PLATFORM_METRICS",
    "ALL_SCORE_METRICS",
    "METRIC_DIRECTIONS",
    "extract_portable_metrics",
    "extract_platform_metrics",
    "extract_all_score_metrics",
    "save_reference",
    "load_reference",
]
