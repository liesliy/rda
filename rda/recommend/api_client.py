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
) -> str:
    """Generate a stable cache key from metrics + policy + language."""
    ts_dict = temporal_sufficiency.to_dict()
    # Sort keys for stable hash
    payload = json.dumps(
        {"policy": policy, "lang": lang, "ts": ts_dict},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


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
) -> Tuple[DatasetTemporalSufficiency, List[TemporalSufficiency]]:
    """Compute temporal sufficiency metrics locally.

    This is the heavy lifting that stays on the client side.
    The result is a small aggregated dict that gets sent to the API.

    Args:
        episodes_iter: Iterator yielding EpisodeData objects.
        total_episodes: Total number of episodes.
        total_frames: Total number of frames.
        progress_callback: Optional callback(step, total, message).

    Returns:
        Tuple of (aggregated_dataset_metrics, per_episode_list).
    """
    per_episode: List[TemporalSufficiency] = []

    for i, episode in enumerate(episodes_iter):
        ts = compute_temporal_sufficiency(episode)
        per_episode.append(ts)
        if progress_callback is not None:
            progress_callback(i + 1, total_episodes, f"episode {episode.episode_index}")

    agg = aggregate_temporal_sufficiency(
        per_episode,
        total_episodes=total_episodes,
        total_frames=total_frames,
    )

    return agg, per_episode


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_api(
    policy: str,
    temporal_sufficiency: DatasetTemporalSufficiency,
    episode_count: int,
    total_frames: int,
    lang: str = "zh",
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

    resp = requests.post(
        f"{api_url}/api/v1/recommend",
        json={
            "policy": policy,
            "temporal_sufficiency": ts_dict,
            "episode_count": episode_count,
            "total_frames": total_frames,
            "lang": lang,
        },
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": f"rda-cli/0.5.0"},
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
            headers={"User-Agent": "rda-cli/0.5.0"},
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

    Returns:
        RecommendationResult with recommendations and rules_version.
    """
    policy_name = "frame-wise" if target_policy == TargetPolicy.FRAME_WISE else "temporal"
    lang = lang if lang in ("zh", "en") else "zh"

    # Step 1: Compute metrics locally
    if progress_callback:
        progress_callback(0, total_episodes, "Computing temporal sufficiency metrics...")

    agg, _ = compute_local_metrics(
        episodes_iter,
        total_episodes=total_episodes,
        total_frames=total_frames,
        progress_callback=progress_callback,
    )

    # Explicit offline mode: never touch the network, straight to fallback
    if offline:
        import click
        click.echo(
            "  [Offline mode] Using conservative local fallback "
            "(rules not evaluated server-side).",
            err=True,
        )
        return build_offline_result(agg, target_policy, lang=lang)

    # Step 2: Check cache
    key = _cache_key(agg, policy_name, lang)
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
        return RecommendationResult.from_dict(full_data)

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
        return RecommendationResult.from_dict(full_data)

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
    return build_offline_result(agg, target_policy, lang=lang)
