"""API client for RDA Recommend (OPEN SOURCE).

This module replaces the local recommendation engine with a remote API
call. It handles:
  1. Local computation of temporal sufficiency metrics (stays offline)
  2. Local caching of API responses (offline fallback)
  3. Remote API call to the closed-source engine
  4. Graceful degradation when offline or rate-limited

Privacy: Only aggregated metrics (<1KB) are sent to the API server.
No raw episode data, images, or action arrays are ever uploaded.

Copyright (c) 2026 Niu Su Tech. MIT License.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rda.recommend.types import (
    RecommendationResult,
    TargetPolicy,
)
from rda.recommend.local_fallback import build_offline_result
from rda.recommend.preflight import (
    PreflightAuditor,
    PreflightSummary,
    aggregate_preflight,
    gate_result_by_verdict,
)
from rda.recommend.temporal_metrics import (
    DatasetTemporalSufficiency,
    TemporalSufficiency,
    aggregate_temporal_sufficiency,
    compute_temporal_sufficiency,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://rda.niusu2026.cn"
CACHE_DIR = Path.home() / ".rda" / "cache"
CACHE_TTL_DAYS = 30
REQUEST_TIMEOUT = 30  # seconds
# REQ-1 (v0.5.9): payload contract version. v2 adds verdict_summary;
# the server reads it only when >= 2, so v1 servers ignore it safely.
# v4 (v0.9.0): adds audit_signals (smoothness / calibration / coverage).
CONTRACT_VERSION = 4


def get_api_url() -> str:
    """Get the API URL from env var or default."""
    return os.environ.get("RDA_API_URL", DEFAULT_API_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _cache_key(
    temporal_sufficiency: DatasetTemporalSufficiency,
    policy: str,
    lang: str = "zh",
    verdict_summary: Optional[Dict[str, Any]] = None,
    policy_chunk_size: Optional[int] = None,
    audit_signals: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a stable cache key from metrics + policy + language.

    v3 (REQ-3): policy_chunk_size changes rule outcomes (dynamic valid
    window alignment + tail-trim frame counts), so it participates in
    the key. The "v3-" prefix also invalidates v2 entries that lacked
    the new DROID-aligned retention metrics. NOTE: no colon in the
    prefix — colons are reserved characters on Windows and a
    "v2:<hash>" filename silently fails to write there.

    v4 (v0.9.0): audit_signals (smoothness / calibration / coverage)
    participate in the key so that different diagnostic profiles don't
    collide on the same temporal sufficiency hash.
    """
    ts_dict = temporal_sufficiency.to_dict()
    # Sort keys for stable hash
    payload = json.dumps(
        {
            "policy": policy,
            "lang": lang,
            "ts": ts_dict,
            "vs": verdict_summary or {},
            "pcs": policy_chunk_size,
            "as": audit_signals or {},
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"v4-{digest}"


def _cache_path(key: str) -> Path:
    """Get the cache file path for a given key."""
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> Optional[Dict[str, Any]]:
    """Read a cached result. Returns None if not found or expired."""
    path = _cache_path(key)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Check TTL
    import time
    cached_at = data.get("cached_at", 0)
    age_seconds = time.time() - cached_at
    if age_seconds > CACHE_TTL_DAYS * 86400:
        return None

    return data


def _write_cache(key: str, data: Dict[str, Any]) -> None:
    """Write a result to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import time
        data["cached_at"] = time.time()
        _cache_path(key).write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError:
        # Cache write failure is non-fatal
        pass


# ---------------------------------------------------------------------------
# Local metrics computation (stays fully offline)
# ---------------------------------------------------------------------------

def compute_local_metrics(
    episodes_iter,
    total_episodes: int,
    total_frames: int,
    progress_callback=None,
    preflight: Optional[PreflightAuditor] = None,
) -> Tuple[DatasetTemporalSufficiency, List[TemporalSufficiency], Optional[PreflightSummary]]:
    """Compute temporal sufficiency metrics locally.

    This is the heavy lifting that stays on the client side.
    The result is a small aggregated dict that gets sent to the API.

    REQ-1: when ``preflight`` is provided, the deterministic CRITICAL
    metrics are re-evaluated in the SAME pass (episodes_iter is a
    single-use generator — a second traversal would double load time).

    Args:
        episodes_iter: Iterator yielding EpisodeData objects.
        total_episodes: Total number of episodes.
        total_frames: Total number of frames.
        progress_callback: Optional callback(step, total, message).
        preflight: Optional PreflightAuditor; when given, per-episode
            verdict evidence is collected and aggregated.

    Returns:
        Tuple of (aggregated_dataset_metrics, per_episode_list,
        preflight_summary_or_None).
    """
    per_episode: List[TemporalSufficiency] = []
    verdicts = [] if preflight is not None else None

    for i, episode in enumerate(episodes_iter):
        ts = compute_temporal_sufficiency(episode)
        per_episode.append(ts)
        if preflight is not None:
            verdicts.append(preflight.evaluate(episode))
        if progress_callback is not None:
            progress_callback(i + 1, total_episodes, f"episode {episode.episode_index}")

    agg = aggregate_temporal_sufficiency(
        per_episode,
        total_episodes=total_episodes,
        total_frames=total_frames,
    )

    summary = aggregate_preflight(verdicts) if verdicts is not None else None
    return agg, per_episode, summary


# ---------------------------------------------------------------------------
# Audit signals extraction (v0.9)
# ---------------------------------------------------------------------------

def extract_audit_signals(audit_result) -> Dict[str, Any]:
    """Extract aggregated diagnostic signals from a DatasetAuditResult.

    Computes compact summaries for three v0.9 diagnostic dimensions:
      - smoothness_summary: from action_discontinuity measurements
      - calibration_summary: from sensor_synchronization measurements
      - coverage_summary: from coverage (state-space occupancy) measurements

    These are sent to the recommendation API so the server-side engine
    can factor them into recommendation rules (e.g. SMOOTHING_REVIEW,
    CALIBRATION_CHECK, COVERAGE_SUGGESTION).

    Args:
        audit_result: A DatasetAuditResult from the audit pipeline.

    Returns:
        Dict with three optional sub-dicts. Missing data yields empty dicts.
    """
    signals: Dict[str, Any] = {}

    if audit_result is None or not hasattr(audit_result, "episodes"):
        return signals

    # Collect per-episode measurements
    spike_counts: List[int] = []
    jerk_p95_values: List[float] = []
    sync_p95_values: List[float] = []
    occupancy_values: List[float] = []

    for ep_result in audit_result.episodes.values():
        # EpisodeAuditResult.metrics is Dict[str, MetricResult]
        metrics = getattr(ep_result, "metrics", {})

        # action_discontinuity
        disc = metrics.get("action_discontinuity")
        if disc is not None:
            measurement = getattr(disc, "measurement", {}) or {}
            sc = measurement.get("spike_count", 0)
            spike_counts.append(sc)
            jerk = measurement.get("jerk_p95")
            if jerk is not None:
                jerk_p95_values.append(float(jerk))

        # sensor_synchronization
        sync = metrics.get("sensor_synchronization")
        if sync is not None:
            measurement = getattr(sync, "measurement", {}) or {}
            offset = measurement.get("worst_p95_offset_ms")
            if offset is not None:
                sync_p95_values.append(float(offset))

        # coverage (state-space occupancy)
        cov = metrics.get("coverage")
        if cov is not None:
            measurement = getattr(cov, "measurement", {}) or {}
            occ = measurement.get("occupancy_rate")
            if occ is not None:
                occupancy_values.append(float(occ))

    # Build smoothness_summary
    if spike_counts:
        signals["smoothness_summary"] = {
            "total_spikes": sum(spike_counts),
            "episodes_with_spikes": sum(1 for s in spike_counts if s > 0),
            "total_episodes": len(spike_counts),
            "jerk_p95_median": (
                sorted(jerk_p95_values)[len(jerk_p95_values) // 2]
                if jerk_p95_values else None
            ),
        }

    # Build calibration_summary
    if sync_p95_values:
        sorted_sync = sorted(sync_p95_values)
        n = len(sorted_sync)
        signals["calibration_summary"] = {
            "median_worst_p95_offset_ms": sorted_sync[n // 2],
            "max_worst_p95_offset_ms": sorted_sync[-1],
            "total_episodes": n,
        }

    # Build coverage_summary
    if occupancy_values:
        sorted_occ = sorted(occupancy_values)
        n = len(sorted_occ)
        signals["coverage_summary"] = {
            "median_occupancy": sorted_occ[n // 2],
            "min_occupancy": sorted_occ[0],
            "max_occupancy": sorted_occ[-1],
            "total_episodes": n,
        }

    return signals


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_api(
    policy: str,
    temporal_sufficiency: DatasetTemporalSufficiency,
    episode_count: int,
    total_frames: int,
    lang: str = "zh",
    verdict_summary: Optional[Dict[str, Any]] = None,
    policy_chunk_size: Optional[int] = None,
    audit_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the remote recommendation API.

    Returns the raw JSON response dict.

    Raises:
        ImportError: if `requests` is not installed.
        ConnectionError: if the API is unreachable.
        RuntimeError: if the API returns an error.
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' package is required for rda recommend. "
            "Install it with: pip install requests"
        )

    api_url = get_api_url()
    ts_dict = temporal_sufficiency.to_dict()

    body: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "policy": policy,
        "temporal_sufficiency": ts_dict,
        "episode_count": episode_count,
        "total_frames": total_frames,
        "lang": lang,
        # REQ-1: verdict evidence from the client-side preflight
        # pass. v1 servers ignore unknown keys; v2 servers gate
        # TRIM suggestions on this.
        "verdict_summary": verdict_summary or {},
    }
    # REQ-3: optional policy chunk size for DROID-aligned dynamic
    # window rules. v1/v2 servers ignore unknown keys (lenient
    # parsing); None means "not provided".
    if policy_chunk_size is not None:
        body["policy_chunk_size"] = int(policy_chunk_size)

    # v0.9: audit_signals for SMOOTHING_REVIEW / CALIBRATION_CHECK /
    # COVERAGE_SUGGESTION rules. v1/v2/v3 servers ignore unknown keys.
    if audit_signals:
        body["audit_signals"] = audit_signals

    resp = requests.post(
        f"{api_url}/api/v1/recommend",
        json=body,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": f"rda-cli/0.9.0"},
    )

    if resp.status_code == 429:
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        raise RuntimeError(
            f"Rate limit exceeded: {body.get('message', 'Daily limit reached (20/day).')}"
        )

    if resp.status_code != 200:
        try:
            body = resp.json()
            raise RuntimeError(
                f"API error ({resp.status_code}): {body.get('message', 'Unknown error')}"
            )
        except ValueError:
            raise RuntimeError(f"API error ({resp.status_code}): {resp.text[:200]}")

    return resp.json()


def _get_remote_rules_version() -> Optional[str]:
    """Check the API's current rules_version via health endpoint.

    Returns None if the API is unreachable.
    """
    try:
        import requests
        resp = requests.get(
            f"{get_api_url()}/api/v1/health",
            timeout=10,
            headers={"User-Agent": "rda-cli/0.9.0"},
        )
        if resp.status_code == 200:
            return resp.json().get("rules_version")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API: run_recommendation (drop-in replacement for old function)
# ---------------------------------------------------------------------------

def run_recommendation(
    episodes_iter,
    target_policy: TargetPolicy,
    total_episodes: int,
    total_frames: int,
    progress_callback=None,
    lang: str = "zh",
    offline: bool = False,
    policy_chunk_size: Optional[int] = None,
    audit_signals: Optional[Dict[str, Any]] = None,
) -> RecommendationResult:
    """Run the full recommendation pipeline.

    This is a drop-in replacement for the old rda.recommend.engine.run_recommendation().
    The key difference: rule evaluation happens on a remote server, not locally.

    Flow:
      1. Compute temporal sufficiency metrics locally (heavy lifting)
      2. Check local cache for a previous result
      3. Try calling the API
      4. If API fails -> use cache if available
      5. If no cache -> conservative local fallback (never a hard error)
      6. Save successful API response to cache

    Args:
        episodes_iter: Iterator yielding EpisodeData objects.
        target_policy: The user's intended model architecture.
        total_episodes: Total number of episodes.
        total_frames: Total number of frames.
        progress_callback: Optional callback(step, total, message).
        lang: Language of the returned copy ("zh" or "en"). Cached
            per-language so switching languages does not mix copy.
        offline: Skip the API entirely and use the conservative local
            fallback. Use this on machines that must never make network
            calls (air-gapped clusters, private deployment QA, etc.).
        policy_chunk_size: Optional action chunk size of the target
            policy (REQ-3, DROID-aligned). When provided, the server
            aligns valid-window and tail-trim rules to this chunk
            length; None keeps the legacy fixed window tiers.
        audit_signals: Optional dict of aggregated diagnostic signals
            from the v0.9 audit pipeline (smoothness_summary,
            calibration_summary, coverage_summary). Sent to the server
            so it can trigger SMOOTHING_REVIEW / CALIBRATION_CHECK /
            COVERAGE_SUGGESTION rules. Older servers ignore this field.

    Returns:
        RecommendationResult with recommendations and rules_version.
    """
    policy_name = "frame-wise" if target_policy == TargetPolicy.FRAME_WISE else "temporal"
    lang = lang if lang in ("zh", "en") else "zh"

    # Step 1: Compute metrics locally (temporal sufficiency + REQ-1
    # preflight verdicts in a single pass over the episode stream).
    if progress_callback:
        progress_callback(0, total_episodes, "Computing temporal sufficiency metrics...")

    preflight = PreflightAuditor()
    agg, _, verdict_summary = compute_local_metrics(
        episodes_iter,
        total_episodes=total_episodes,
        total_frames=total_frames,
        progress_callback=progress_callback,
        preflight=preflight,
    )
    verdict_payload = verdict_summary.to_dict() if verdict_summary else {}

    # Explicit offline mode: never touch the network, straight to fallback
    if offline:
        import click
        click.echo(
            "  [Offline mode] Using conservative local fallback "
            "(rules not evaluated server-side).",
            err=True,
        )
        result = build_offline_result(
            agg, target_policy, lang=lang, policy_chunk_size=policy_chunk_size
        )
        return gate_result_by_verdict(result, verdict_summary, lang=lang)

    # Step 2: Check cache (v4 keys include audit signals)
    key = _cache_key(agg, policy_name, lang, verdict_payload, policy_chunk_size, audit_signals)
    cached = _read_cache(key)

    # Step 3: Try API
    api_error: Optional[str] = None
    api_result: Optional[Dict[str, Any]] = None

    try:
        api_result = _call_api(
            policy=policy_name,
            temporal_sufficiency=agg,
            episode_count=total_episodes,
            total_frames=total_frames,
            lang=lang,
            verdict_summary=verdict_payload,
            policy_chunk_size=policy_chunk_size,
            audit_signals=audit_signals,
        )
    except ImportError as e:
        api_error = str(e)
    except Exception as e:
        api_error = str(e)

    # Step 4: Handle API result or fallback
    if api_result is not None:
        # Success — save to cache and return
        _write_cache(key, api_result)

        # Build RecommendationResult from API response + local metrics
        full_data = {
            **api_result,
            "temporal_sufficiency": agg.to_dict(),
        }
        result = RecommendationResult.from_dict(full_data)
        return gate_result_by_verdict(result, verdict_summary, lang=lang)

    # API failed — try cache
    if cached is not None:
        # Warn user about offline/cache mode
        import click
        click.echo(
            f"  [Warning] API unavailable ({api_error[:80] if api_error else 'unknown'}). "
            f"Using cached result (rules v{cached.get('rules_version', '?')}).",
            err=True,
        )
        full_data = {
            **cached,
            "temporal_sufficiency": agg.to_dict(),
        }
        result = RecommendationResult.from_dict(full_data)
        return gate_result_by_verdict(result, verdict_summary, lang=lang)

    # No cache, no API — conservative local fallback instead of a hard error.
    # The metrics (heavy lifting) are already computed client-side; grading
    # them with built-in conservative rules is strictly more useful than
    # failing. Every paid / private-deployment scenario must survive an
    # unreachable recommendation server.
    import click
    click.echo(
        f"  [Warning] API unavailable ({api_error[:80] if api_error else 'unknown'}) "
        "and no cached result. Falling back to conservative local evaluation "
        "(rules_version=offline-fallback). Re-run online for the full "
        "server-side evaluation.",
        err=True,
    )
    result = build_offline_result(
        agg, target_policy, lang=lang, policy_chunk_size=policy_chunk_size
    )
    return gate_result_by_verdict(result, verdict_summary, lang=lang)
