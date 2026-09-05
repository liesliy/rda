"""Type definitions for the recommendation system (OPEN SOURCE).

This module contains all type definitions, enums, and data classes
that are shared between the open-source CLI and the closed-source
recommendation engine. It is intentionally free of any rule logic,
thresholds, or business logic.

Copyright (c) 2026 Niu Su Tech. MIT License.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from rda.recommend.temporal_metrics import DatasetTemporalSufficiency


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RecommendationAction(str, Enum):
    """What action the recommendation suggests."""
    TRIM_INITIAL = "TRIM_INITIAL"
    TRIM_IDLE_MILD = "TRIM_IDLE_MILD"
    DO_NOT_PRUNE = "DO_NOT_PRUNE"
    DO_NOT_PRUNE_AGGRESSIVELY = "DO_NOT_PRUNE_AGGRESSIVELY"
    NO_RECOMMENDATION = "NO_RECOMMENDATION"
    # REQ-1 (v0.5.9): emitted when the preflight verdict evidence shows
    # structurally broken episodes. Repair must precede optimization;
    # all TRIM_* suggestions are suppressed. Older clients that receive
    # this value degrade to NO_RECOMMENDATION via from_dict (safe).
    REPAIR_FIRST = "REPAIR_FIRST"
    # REQ-3 (v0.6.0): an episode (or dataset) is essentially all-idle
    # with near-zero chunk-usable content. Mirrors openpi/DROID's
    # practice of dropping near-static demonstrations. Also degrades
    # to NO_RECOMMENDATION on older clients via from_dict (safe).
    DISCARD_STATIC = "DISCARD_STATIC"
    # REQ-2 (v0.6.0): review suggestion for action spikes / jerk
    # outliers (possible smoothing post-processing or teleop glitches).
    # Human review, never auto-deletion.
    SMOOTHING_REVIEW = "SMOOTHING_REVIEW"
    # REQ-2 (v0.6.0): sensor timestamp sync p95 exceeds the dataset's
    # own baseline (×2) — check calibration / timestamp alignment.
    CALIBRATION_CHECK = "CALIBRATION_CHECK"
    # REQ-2 (v0.6.0): state-space occupancy below the dataset's own
    # distribution quantiles — consider richer coverage in collection.
    COVERAGE_SUGGESTION = "COVERAGE_SUGGESTION"
    # REQ-4 (v0.7.0): dataset contains episodes with PROVEN visual-stream
    # corruption (camera freeze while moving, missing camera stream, or
    # video/parquet timeline mismatch). Repair-first framing; older
    # clients degrade to NO_RECOMMENDATION via from_dict (safe).
    VISUAL_REPAIR_FIRST = "VISUAL_REPAIR_FIRST"
    # REQ-4 (v0.7.0): episodes with a VA-B visual-quality penalty
    # (blur/exposure/contrast) — measurement-level review hint, never a
    # deletion suggestion. Degrades safely on older clients.
    VISUAL_QUALITY_REVIEW = "VISUAL_QUALITY_REVIEW"


class ConfidenceLevel(str, Enum):
    """Confidence level for a recommendation."""
    HIGH = "HIGH"
    EXPERIMENTAL = "EXPERIMENTAL"
    NOT_RECOMMENDED = "NOT_RECOMMENDED"


class TargetPolicy(str, Enum):
    """Type of model the user intends to train."""
    FRAME_WISE = "frame_wise"      # MLP, BC, per-frame policies
    TEMPORAL = "temporal"          # ACT, Diffusion Policy, Transformer

    @classmethod
    def from_cli_name(cls, name: str) -> "TargetPolicy":
        """Map CLI-friendly names to enum values."""
        mapping = {
            "frame-wise": cls.FRAME_WISE,
            "frame_wise": cls.FRAME_WISE,
            "mlp": cls.FRAME_WISE,
            "bc": cls.FRAME_WISE,
            "temporal": cls.TEMPORAL,
            "act": cls.TEMPORAL,
            "dp": cls.TEMPORAL,
            "diffusion": cls.TEMPORAL,
            "diffusion_policy": cls.TEMPORAL,
            "transformer": cls.TEMPORAL,
        }
        normalized = name.strip().lower().replace("-", "_")
        if normalized not in mapping:
            raise ValueError(
                f"Unknown policy type: '{name}'. "
                f"Valid options: frame-wise, temporal"
            )
        return mapping[normalized]


# ---------------------------------------------------------------------------
# Recommendation result
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """A single optimization recommendation."""
    action: RecommendationAction
    confidence: ConfidenceLevel
    title: str
    summary: str
    details: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    expected_impact: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence.value,
            "title": self.title,
            "summary": self.summary,
            "details": list(self.details),
            "caveats": list(self.caveats),
            "expected_impact": self.expected_impact,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recommendation":
        """Reconstruct a Recommendation from a dict (API response)."""
        try:
            action = RecommendationAction(data.get("action", "NO_RECOMMENDATION"))
        except ValueError:
            action = RecommendationAction.NO_RECOMMENDATION
        try:
            confidence = ConfidenceLevel(data.get("confidence", "HIGH"))
        except ValueError:
            confidence = ConfidenceLevel.HIGH
        return cls(
            action=action,
            confidence=confidence,
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            details=data.get("details", []),
            caveats=data.get("caveats", []),
            expected_impact=data.get("expected_impact", ""),
        )


@dataclass
class RecommendationResult:
    """Full result of running the recommendation engine."""
    target_policy: TargetPolicy
    temporal_sufficiency: DatasetTemporalSufficiency
    recommendations: List[Recommendation] = field(default_factory=list)
    rules_version: str = ""
    engine_version: str = ""
    # REQ-1: verdict evidence attached by the client-side gate when the
    # dataset has excluded episodes. Absent (None) for healthy datasets.
    verdict_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "target_policy": self.target_policy.value,
            "temporal_sufficiency": self.temporal_sufficiency.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "rules_version": self.rules_version,
            "engine_version": self.engine_version,
        }
        if self.verdict_summary is not None:
            d["verdict_summary"] = self.verdict_summary
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationResult":
        """Reconstruct a RecommendationResult from a dict (API response).

        The temporal_sufficiency field is typically provided by the CLI
        (computed locally), while recommendations/rules_version come from
        the API response.
        """
        try:
            policy = TargetPolicy(data.get("target_policy", "frame_wise"))
        except ValueError:
            policy = TargetPolicy.FRAME_WISE

        # Reconstruct temporal_sufficiency from dict if present
        ts_data = data.get("temporal_sufficiency", {})
        ts = DatasetTemporalSufficiency(
            total_episodes=ts_data.get("total_episodes", 0),
            total_frames=ts_data.get("total_frames", 0),
            computed_episodes=ts_data.get("computed_episodes", 0),
            idle_total_ratio=ts_data.get("idle_total_ratio", {}),
            idle_prefix_ratio=ts_data.get("idle_prefix_ratio", {}),
            active_run_p50=ts_data.get("active_run_p50", {}),
            active_run_p90=ts_data.get("active_run_p90", {}),
            active_run_max=ts_data.get("active_run_max", {}),
            transition_count=ts_data.get("transition_count", {}),
            valid_window_ratio_5=ts_data.get("valid_window_ratio_5", {}),
            valid_window_ratio_10=ts_data.get("valid_window_ratio_10", {}),
            valid_window_ratio_20=ts_data.get("valid_window_ratio_20", {}),
        )

        recs = [
            Recommendation.from_dict(r)
            for r in data.get("recommendations", [])
        ]

        return cls(
            target_policy=policy,
            temporal_sufficiency=ts,
            recommendations=recs,
            rules_version=data.get("rules_version", ""),
            engine_version=data.get("engine_version", ""),
            verdict_summary=data.get("verdict_summary"),
        )


# ---------------------------------------------------------------------------
# Standard caveats (shared constants)
# ---------------------------------------------------------------------------

GENERAL_CAVEATS = [
    "效果随任务域和模型架构变化。"
    "Effect varies by task domain and model architecture.",
    "本建议基于实验观察，不保证成功率提升。"
    "This recommendation is based on experimental observations and "
    "does not guarantee success rate improvement.",
]

HELD_OUT_VALIDATION = (
    "请先在 held-out 验证集上验证效果后再应用到完整数据集。 "
    "Always validate on a held-out dataset before applying to the full dataset."
)
