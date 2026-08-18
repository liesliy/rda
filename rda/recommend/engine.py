"""DEPRECATED: The recommendation engine has been moved to the server.

This module is kept as a stub for backwards compatibility. The rule logic,
thresholds, and recommendation builders now live in the RDA API server's
private codebase (server/engine_core.py).

If you need to run recommendations:
  - CLI: use `rda recommend <path> --policy <frame-wise|temporal>`
  - API:  POST to https://rda.niusu2026.cn/api/v1/recommend
  - Private deployment: set RDA_API_URL and run your own server

For local-only metrics (no recommendations), use:
  - `rda audit <path>`
  - `from rda.recommend.temporal_metrics import compute_temporal_sufficiency`
"""

# Re-export types for backwards compatibility
from rda.recommend.types import (
    ConfidenceLevel,
    Recommendation,
    RecommendationAction,
    RecommendationResult,
    TargetPolicy,
    GENERAL_CAVEATS,
    HELD_OUT_VALIDATION,
)
from rda.recommend.formatter import format_recommendation_text
from rda.recommend.api_client import run_recommendation


def RecommendationEngine(*args, **kwargs):
    """Raise informative error — engine moved to server."""
    raise NotImplementedError(
        "RecommendationEngine has been moved to the RDA API server. "
        "Use `rda recommend` CLI command or the API endpoint instead. "
        "For private deployment, set RDA_API_URL and run your own server."
    )


__all__ = [
    "ConfidenceLevel",
    "Recommendation",
    "RecommendationAction",
    "RecommendationResult",
    "TargetPolicy",
    "format_recommendation_text",
    "run_recommendation",
]
