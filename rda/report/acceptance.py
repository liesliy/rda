"""REQ-5 — dataset-level relative baselines and the acceptance summary.

Three deliverables:

1. **Tukey IQR fence outlier detection** on episode runtime, ported from
   the [SLE] reference implementation (``score_lerobot_episodes``).
2. **Dataset percentile baselines** (P10 / P50 / P90) — "what is typical
   for *this* dataset", so recommendations are relative to the audited
   population instead of an absolute threshold borrowed from another
   robot platform.
3. **Acceptance summary** — folds the two above plus the Tier-1
   calibration layer into one deliverable the data-acceptance party can
   receive as a single piece of evidence.

Red line honoured: RDA measures and presents, it does not adjudicate
semantics. This module reports counts, baselines and outliers. It does
**not** emit its own pass/fail quality verdict — the acceptance decision
belongs to the human reviewer; RDA only makes the evidence explicit.

Authority references:
- [SLE] ``score_lerobot_episodes`` (Apache-2.0): ``is_time_outlier`` uses
  the Tukey fence ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``.
- Tier-1 portable metrics and their rho=0.960 correlation with the full
  metric set: see :mod:`rda.calibration.portable`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rda.audit.dataset_audit import DatasetAuditResult
from rda.metrics.base import MetricAvailability

__all__ = [
    "ACCEPTANCE_SCHEMA_VERSION",
    "tukey_fence",
    "is_tukey_outlier",
    "compute_percentile_baselines",
    "compute_runtime_outliers",
    "build_acceptance_summary",
]


# Bumped when the layout of the ``acceptance_summary`` block changes.
ACCEPTANCE_SCHEMA_VERSION = "1.0"

# How many individual outlier episodes to enumerate before truncating.
# Keeps the JSON report small on large datasets; the count is always exact.
OUTLIER_DETAIL_LIMIT = 50

# Tukey fence multiplier. 1.5 is the [SLE] / Tukey default; exposed as a
# module constant so a future policy can widen it without touching callers.
TUKEY_K = 1.5


# ---------------------------------------------------------------------------
# Baseline specs — mirror the Tier-1 portable metric set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _BaselineSpec:
    """Which measurement backs a baseline entry."""

    key: str            # output key
    metric: str         # metric name on EpisodeAuditResult.metrics
    field: str          # measurement field
    unit: str           # display unit
    higher_is_worse: bool


# These three are exactly PORTABLE_METRICS in rda.calibration.portable —
# the Tier-1 set that correlates at rho=0.960 with the full metric set and
# stays comparable across robot platforms. Reusing the same set means the
# baseline block and the calibration layer speak about the same numbers.
_BASELINE_SPECS: Tuple[_BaselineSpec, ...] = (
    _BaselineSpec(
        key="duration_sec",
        metric="distribution",
        field="duration_sec",
        unit="s",
        higher_is_worse=True,
    ),
    _BaselineSpec(
        key="effective_motion_ratio",
        metric="idle_ratio",
        field="effective_motion_ratio",
        unit="ratio",
        higher_is_worse=False,
    ),
    _BaselineSpec(
        key="spike_count",
        metric="action_discontinuity",
        field="spike_count",
        unit="count",
        higher_is_worse=True,
    ),
)


# ---------------------------------------------------------------------------
# Tukey IQR fence  [SLE]
# ---------------------------------------------------------------------------

def tukey_fence(values: np.ndarray) -> Dict[str, float]:
    """Compute the Tukey IQR fences for a sample.

    Mirrors ``score_lerobot_episodes.scores.build_time_stats`` /
    ``is_time_outlier`` (mode="iqr")::

        q1, q3 = np.percentile(values, (25, 75))
        iqr    = q3 - q1
        lower  = q1 - 1.5 * iqr
        upper  = q3 + 1.5 * iqr

    Args:
        values: 1-D array of measurements. Empty is tolerated.

    Returns:
        Dict with ``q1``, ``q3``, ``iqr``, ``lower_fence``,
        ``upper_fence``. All zero when ``values`` is empty — callers must
        check ``available_episodes`` before reading the fences.
    """
    if values.size == 0:
        return {
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower_fence": 0.0,
            "upper_fence": 0.0,
        }

    q1, q3 = np.percentile(values, (25.0, 75.0))
    q1 = float(q1)
    q3 = float(q3)
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": q1 - TUKEY_K * iqr,
        "upper_fence": q3 + TUKEY_K * iqr,
    }


def is_tukey_outlier(value: float, fences: Dict[str, float]) -> bool:
    """True when ``value`` falls outside ``[lower_fence, upper_fence]``."""
    return bool(value < fences["lower_fence"] or value > fences["upper_fence"])


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def _collect_indexed(
    result: DatasetAuditResult,
    metric_name: str,
    field: str,
) -> List[Tuple[int, float]]:
    """Collect ``(episode_index, value)`` for episodes where the metric ran.

    Episodes whose metric is absent or not AVAILABLE are skipped rather
    than coerced to 0.0 — an unavailable measurement is not a zero
    measurement, and folding it in would corrupt the fences.
    """
    out: List[Tuple[int, float]] = []
    for idx, ep in result.episodes.items():
        m = ep.metrics.get(metric_name)
        if m is None:
            continue
        if m.availability != MetricAvailability.AVAILABLE:
            continue
        val = m.measurement.get(field)
        if val is None:
            continue
        out.append((int(idx), float(val)))
    return out


# ---------------------------------------------------------------------------
# Deliverable 2: dataset percentile baselines (P10 / P50 / P90)
# ---------------------------------------------------------------------------

def compute_percentile_baselines(
    result: DatasetAuditResult,
) -> Dict[str, Any]:
    """P10 / P50 / P90 per Tier-1 portable metric, over available episodes.

    These are *relative* baselines: they describe the audited population,
    so a downstream consumer can say "this episode sits above P90 of its
    own dataset" without importing an absolute threshold from a different
    robot platform.

    Returns a dict keyed by baseline key, each with ``p10``/``p50``/``p90``,
    ``min``/``max``, ``unit`` and ``available_episodes``. Metrics with no
    available episode keep the key with ``available_episodes: 0`` so the
    absence stays visible instead of silently disappearing.
    """
    out: Dict[str, Any] = {}
    for spec in _BASELINE_SPECS:
        pairs = _collect_indexed(result, spec.metric, spec.field)
        if not pairs:
            out[spec.key] = {
                "unit": spec.unit,
                "available_episodes": 0,
                "note": (
                    f"metric '{spec.metric}' unavailable on every episode; "
                    "no baseline computed"
                ),
            }
            continue

        values = np.array([v for _, v in pairs], dtype=np.float64)
        p10, p50, p90 = np.percentile(values, (10.0, 50.0, 90.0))
        out[spec.key] = {
            "unit": spec.unit,
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "available_episodes": int(values.size),
        }
    return out


# ---------------------------------------------------------------------------
# Deliverable 1: runtime outliers via Tukey IQR fence  [SLE]
# ---------------------------------------------------------------------------

def compute_runtime_outliers(
    result: DatasetAuditResult,
) -> Dict[str, Any]:
    """Flag duration outliers with the Tukey IQR fence.

    Why this matters: a dataset whose episodes are "all fine individually"
    can still contain runtime outliers that skew training batch
    composition. The fence is computed *within* the dataset, so no
    external duration assumption is smuggled in.

    Returns fences, exact outlier count, and up to
    ``OUTLIER_DETAIL_LIMIT`` individual episodes with their direction
    (``"long"`` / ``"short"``).
    """
    pairs = _collect_indexed(result, "distribution", "duration_sec")

    if not pairs:
        return {
            "method": "tukey_iqr",
            "reference": "[SLE] score_lerobot_episodes is_time_outlier(mode='iqr')",
            "available_episodes": 0,
            "count": 0,
            "episodes": [],
            "note": (
                "duration_sec unavailable on every episode; "
                "no outlier test performed"
            ),
        }

    values = np.array([v for _, v in pairs], dtype=np.float64)
    fences = tukey_fence(values)

    # Degenerate case: every episode has the same duration, so IQR is 0 and
    # the fence collapses to a single point. Tukey then flags even a
    # negligible difference. We still report what the rule says (it is the
    # [SLE] behaviour) but mark it, because "0 outliers" here means "no
    # spread to detect against" — NOT "checked and clean".
    degenerate = fences["iqr"] == 0.0

    outliers: List[Dict[str, Any]] = []
    for idx, val in pairs:
        if not is_tukey_outlier(val, fences):
            continue
        outliers.append({
            "episode_index": idx,
            "duration_sec": round(val, 4),
            "direction": "long" if val > fences["upper_fence"] else "short",
        })

    # Longest first — the long tail is what usually matters for batch
    # composition and storage cost.
    outliers.sort(key=lambda d: d["duration_sec"], reverse=True)

    block: Dict[str, Any] = {
        "method": "tukey_iqr",
        "reference": "[SLE] score_lerobot_episodes is_time_outlier(mode='iqr')",
        "tukey_k": TUKEY_K,
        "available_episodes": int(values.size),
        "q1": fences["q1"],
        "q3": fences["q3"],
        "iqr": fences["iqr"],
        "lower_fence": fences["lower_fence"],
        "upper_fence": fences["upper_fence"],
        "degenerate": degenerate,
        "count": len(outliers),
        "truncated": len(outliers) > OUTLIER_DETAIL_LIMIT,
        "episodes": outliers[:OUTLIER_DETAIL_LIMIT],
    }

    if degenerate:
        block["note"] = (
            "Every episode has the same duration (IQR = 0), so the Tukey "
            "fence collapses to a single point and even a negligible "
            "difference counts as an outlier. Read this as 'no spread to "
            "detect against', not as 'checked and clean'."
        )

    return block


# ---------------------------------------------------------------------------
# Deliverable 3: acceptance summary
# ---------------------------------------------------------------------------

def build_acceptance_summary(
    result: DatasetAuditResult,
    *,
    quality: Optional[Dict[str, Any]] = None,
    not_checked: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Assemble the one-page acceptance evidence block.

    Args:
        result: The dataset audit result.
        quality: Optional pre-computed quality block (DHI / grade /
            training readiness). Kept as an argument so this module never
            imports :mod:`rda.report.json_report` — that would be a
            circular import.
        not_checked: Optional ``{dependency: skipped_count}`` map, i.e.
            the ``skipped_by_missing_dep`` block. Shown verbatim so the
            acceptance party can see which dimensions were *not*
            measured — an absent key reading as "checked and fine" is
            exactly the failure mode this field prevents.

    Returns:
        The ``acceptance_summary`` dict for the JSON report.
    """
    # Count verdicts from the episodes themselves rather than reading
    # ``result.verdict_counts``: that map is only populated once
    # ``compute_verdict_counts()`` has been called, so trusting it makes
    # this block depend on call order. Tallying directly is both correct
    # and order-independent.
    pass_n = review_n = exclude_n = 0
    for ep in result.episodes.values():
        verdict = getattr(ep.verdict, "value", ep.verdict)
        if verdict == "PASS":
            pass_n += 1
        elif verdict == "REVIEW":
            review_n += 1
        elif verdict == "EXCLUDE":
            exclude_n += 1
    total = result.num_episodes

    info = result.dataset_info
    baselines = compute_percentile_baselines(result)
    outliers = compute_runtime_outliers(result)

    summary: Dict[str, Any] = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "path": info.path,
            "num_episodes": info.num_episodes,
            "total_frames": info.total_frames,
            "fps": info.meta.get("fps"),
            "robot": info.meta.get("robot", "unknown"),
            "modalities": list(info.modalities) if info.modalities else [],
        },
        "verdict_distribution": {
            "pass": pass_n,
            "review": review_n,
            "exclude": exclude_n,
            "total": total,
            "pass_rate": (pass_n / total) if total else None,
        },
        "baseline_percentiles": baselines,
        "runtime_outliers": outliers,
        "calibration": {
            "tier1_metrics": [s.key for s in _BASELINE_SPECS],
            "tier1_source": "rda.calibration.portable.PORTABLE_METRICS",
            "tier1_rho_vs_full_set": 0.960,
            "note": (
                "Tier-1 portable metrics are cross-platform comparable; "
                "platform-specific metrics (velocity_p95, path_length) are "
                "excluded from the baseline on purpose."
            ),
        },
    }

    if quality:
        summary["quality"] = quality

    # Always emit, even when empty: ""not checked" must be visible".
    summary["not_checked"] = not_checked or {}

    summary["interpretation_note"] = (
        "This block presents measurements and relative baselines only. "
        "It does not issue an accept/reject decision — that judgement "
        "belongs to the data acceptance party. RDA measures and presents; "
        "it does not adjudicate semantics."
    )

    return summary
