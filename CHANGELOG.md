# Changelog

## 0.5.9 - 2026-09-05

### Added

- **REQ-1: verdict-gated recommendations.** Audit verdicts now enter the recommendation chain: `rda recommend` re-evaluates the five deterministic CRITICAL metrics (missing/dropout, NaN/Inf, schema, timestamp validity, joint limits) in the same single pass as the temporal metrics (zero extra dataset traversal, near-zero cost), and any episode failure suppresses all pruning advice. The dataset receives a single `REPAIR_FIRST` recommendation instead, citing per-episode evidence (episode index, failed metrics) and a damage class: `INVALID` (NaN/Inf, broken time axis — irrecoverable at audit time) vs `REPAIRABLE` (frame loss, schema mismatch, joint excursion — fixable by trimming/re-export). This holds on ALL result paths: server response, cache, and offline fallback (client-side final gate), so the guarantee is independent of server upgrade order. New module `rda/recommend/preflight.py`; API payload carries `contract_version: 2` + `verdict_summary` (aggregates only, <1KB — privacy posture unchanged); server rules v0.6.0/engine v3 grade the same gate; v1 clients and servers are fully unaffected.
- `RecommendationResult.verdict_summary` field: JSON reports now embed the verdict evidence (`exclude_count`, `dominant_reason_code`, `excluded_episodes[]` with `reason_code`/`failed_metrics`) next to the REPAIR_FIRST recommendation.
- Regression coverage: `tests/test_preflight.py` (17 tests) — per-episode verdict classification (NaN → INVALID, dropout → REPAIRABLE, 0-frame guard), aggregation with INVALID-dominates-REPAIRABLE, payload size budget (<1KB for 200+ episodes), the gate itself (TRIM blocked on excluded datasets, passthrough on healthy, zh/en copy), single-pass integration, and cache-key namespacing.

### Fixed

- Windows cache-write failure: the v2 cache-key namespace originally used a colon (`v2:<hash>`), which is a reserved character on Windows — cache writes silently produced 0-byte files. The namespace is now `v2-<hash>`; old `v2:`-less (v1) cache entries simply age out via TTL.

## 0.5.8 - 2026-09-03

### Fixed

- `video_frame_integrity` false-flagged every episode of a chunked v3.0 dataset: it compared a shared MP4's *total* frame count against a single episode's parquet count (e.g. 101469 vs 214), producing a 100% hard-mismatch on `lerobot/libero_10`. The metric now derives each episode's own frame span inside the chunk via `(to_timestamp - from_timestamp) * fps` (timestamps are preserved as floats and `fps` is injected by the loader), and separately detects genuine file truncation (when the span runs past the end of the MP4).
- Episode verdicts carried no task identity: `EpisodeAuditResult` now surfaces `task_index` and `task_description` (propagated from `EpisodeData.meta`), so per-episode results in reports answer "which task is this episode from?" — completing the report layer of the v3.0 task-identity fix started in 0.5.7.

### Added

- Regression coverage for chunked-video frame-count cross-check and file-truncation detection (`tests/test_video_frame_integrity.py`).
- Regression coverage for episode-level task-identity propagation to the audit result (`tests/test_lerobot_task_identity.py`).

## 0.5.7 - 2026-09-03

### Fixed

- LeRobot v3.0 task identity was silently dropped for multi-task datasets: the per-frame `task_index` column was filtered out by the audit projection and `meta/tasks.parquet` was never read, so reports could not answer "which task is this episode from?" (the same failure mode reported for the community tool Calibra on `lerobot/libero_10`). `DatasetInfo.meta` now surfaces `num_tasks` and the `task_index -> description` mapping, and `EpisodeData.meta` now carries `task_index` and `task_description`. Single-task and task-less datasets remain fully backward compatible.
- `video_frame_integrity` no longer decodes every frame of videos lacking the `nb_frames` metadata key: it now reads the exact count from PyAV's `stream.frames` (FFmpeg's MP4/MOV `stts` box) and memoizes the result per file, so chunked videos shared across many episodes are parsed once. This cut the metric's cost from >1h to milliseconds on `lerobot/libero_10`.

### Added

- Regression coverage for v3.0 task identity: `tests/test_lerobot_task_identity.py` (synthetic fixtures + a real `lerobot/libero_10` end-to-end check).
- Regression coverage for the video frame-count fast paths: `tests/test_video_frame_integrity.py` (asserts the decode fallback is only reached when neither `nb_frames` metadata nor `stream.frames` is available).

## 0.5.5 - 2026-08-31

### Added

- `rda recommend --offline`: conservative local fallback when the recommendation API is unreachable and no cache exists. Previously this scenario exited with an error even though all required measurements were already computed locally.
- Offline mode only emits the "clearly safe" rule set (`DO_NOT_PRUNE`, and `TRIM_INITIAL` for long initial idle prefixes under frame-wise temporal policy); it never invents risky pruning advice. Output is clearly labeled as offline conservative mode with rule version `offline-fallback`.
- Offline recommendation results are cached like online ones, so a later successful API call can reuse them.
- Test suite for offline fallback behavior and for `tools/rda_render.py` (badge color semantics, HTML report rendering).

### Fixed

- `recommend` no longer fails hard when network is unavailable: it now degrades to local fallback with a stderr warning instead of `exit 1`.

## 0.5.4 - 2026-08-21

### Added

- Evidence-aware report fields for `HARD_FAIL`, `RISK_SIGNAL`, and `UNVERIFIABLE`.
- Episode-level evidence summaries in JSON reports.
- Blind-validation helpers for annotator-safe sample manifests and external QC comparison.
- Regression coverage for evidence boundaries and blind QC comparison.

### Changed

- CLI, Markdown, and UI copy now distinguishes confirmed structural failures from statistical risk signals.
- Risk signals such as action discontinuity, low effective motion, and unusual distributions are described as observations requiring human review, not confirmed corruption.
- Unverifiable metrics are not presented as PASS.
- Recommendation UI copy clarifies that recommendation evidence and audit-risk evidence are separate evidence chains.

### Validation boundary

The release does not claim independent precision, recall, threshold optimality, or training/rollout improvement without human labels, customer QC, or held-out outcome data.

## 0.5.3 - 2026-08-20

- Fixed LeRobot v3.0 metadata/data-file episode mapping fallback.
- Re-ran 12 real datasets with corrected LIBERO handling.
