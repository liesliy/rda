"""Recommendation module for RDA data optimization suggestions.

Provides:
  - Temporal sufficiency metrics (idle structure, active runs, valid windows)
  - Type definitions for recommendations (actions, confidence levels)
  - Text formatting for CLI output
  - API client for remote rule evaluation (with local caching)

The recommendation engine (rule logic, thresholds, evidence strings) runs
on the RDA API server. The CLI sends only aggregated metrics (<1KB) to
the API — no raw episode data is uploaded.

For private deployment, set the RDA_API_URL environment variable.
"""
from __future__ import annotations

from rda.recommend.temporal_metrics import (
    DatasetTemporalSufficiency,
    TemporalSufficiency,
    aggregate_temporal_sufficiency,
    compute_idle_mask,
    compute_temporal_sufficiency,
)
from rda.recommend.types import (
    ConfidenceLevel,
    Recommendation,
    RecommendationAction,
    RecommendationResult,
    TargetPolicy,
)
from rda.recommend.formatter import format_recommendation_text
from rda.recommend.api_client import run_recommendation

__all__ = [
    # Temporal metrics
    "TemporalSufficiency",
    "DatasetTemporalSufficiency",
    "compute_temporal_sufficiency",
    "compute_idle_mask",
    "aggregate_temporal_sufficiency",
    # Types
    "Recommendation",
    "RecommendationResult",
    "RecommendationAction",
    "ConfidenceLevel",
    "TargetPolicy",
    # Formatting
    "format_recommendation_text",
    # API client
    "run_recommendation",
]
