# Server deploy verification (engine_version=2)

**When**: 2026-08-18, after the v0.5.2 server-side deploy.
**Endpoint**: `https://rda.niusu2026.cn` (self-hosted on TencentCloud Lighthouse,
Ubuntu 24.04, systemd unit `rda-api`, uvicorn `server.main:app`).

## What was verified

The v0.5.2 release moved the recommendation engine to a bilingual
`RecommendEngine(lang=)` with `ENGINE_VERSION = "2"`. The deploy check
confirms the live server is actually running that engine and that the
`lang` field round-trips through the API correctly.

## Method (reproducible)

```bash
# health
curl -s https://rda.niusu2026.cn/api/v1/health

# zh recommend (payload abbreviated — full temporal_sufficiency dict omitted)
curl -s -X POST https://rda.niusu2026.cn/api/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{"policy":"DO_NOT_PRUNE","episode_count":100,"total_frames":10000,"lang":"zh", ...}'

# en recommend
curl -s -X POST https://rda.niusu2026.cn/api/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{"policy":"DO_NOT_PRUNE","episode_count":100,"total_frames":10000,"lang":"en", ...}'

# invalid lang → must 422
curl -s -o /dev/null -w '%{http_code}' -X POST .../api/v1/recommend \
  -d '{"lang":"fr", ...}'
```

## Results (v0.5.2)

| Check | Expected | Actual |
|---|---|---|
| `health` returns `engine_version` | `"2"` | ✓ `"2"` |
| `health` returns `rules_version` | `"0.5.0"` | ✓ `"0.5.0"` |
| zh recommend body language | all Chinese | ✓ 1183 chars, zero English fragments |
| en recommend body language | all English | ✓ 2225 chars, zero Chinese fragments |
| zh/en recommend content semantic parity | same findings, different language | ✓ |
| `lang` field echoed in response | matches request | ✓ |
| Invalid `lang` (e.g. `"fr"`) | HTTP 422 + error message | ✓ |
| Client cache key includes `lang` | zh/en don't collide | ✓ (local cache, 30-day TTL) |

## Deploy gotcha (recorded for future-you)

First deploy attempt copied `server/` to `/opt/rda-api/` (repo root). The
systemd unit actually loads `/opt/rda-api/server/` (uvicorn
`server.main:app`), so the root copy did nothing — `engine_version` stayed
at the old value. Fix: copy into the `server/` subdirectory, restart unit.
Now part of the deploy checklist.

## What this does NOT prove

- It proves the deployed engine is bilingual and reachable. It does **not**
  prove the recommendation *logic* is correct — only that the plumbing
  works. A labelled-dataset benchmark of recommendation quality is roadmap
  L2.
- The `rda.niusu2026.cn` endpoint is the author's own deployment. Users who
  self-host (set `RDA_API_URL`) get the same engine code but this write-up
  makes no claim about their uptime or parity — that's their responsibility.

## Backing artifact

- Engine code: [`server/engine_core.py`](../server/engine_core.py) (not
  shipped to PyPI — see [wheel_leak_guard.md](./wheel_leak_guard.md)).
- Client: [`rda/recommend/api_client.py`](../rda/recommend/api_client.py).
- This is a live-server check, not a committed test. Re-run the curl
  commands above to reproduce.
