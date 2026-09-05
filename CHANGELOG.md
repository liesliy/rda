# Changelog

## 0.7.0 - 2026-09-05

### Added

- **REQ-4: visual-stream audit — VA-A integrity trio + VA-B quality measurement.** RDA now audits the visual modality with the same hard-evidence discipline as the kinematics:
  - `video_freeze` (EXCLUDE-grade): detects camera drop-out — consecutive codec-identical video frames while the arm is moving. Decodes 64×64 grayscale spans via PyAV; frozen detection uses an adaptive epsilon (`max(0.10, 0.25 × p10` of frame diffs`)`) calibrated on libero_10 so slow motion is not misclassified as a stall. A single brief freeze is REVIEW; conclusive corruption requires repeated spans, a ≥3 s stall, or ≥30% frozen time.
  - `video_timestamp_alignment` (EXCLUDE beyond 10% drift, REVIEW 2–10%): compares the chunk-video time span against the parquet timeline, catching systematic video/state misalignment that frame-count checks miss.
  - `video_stream_sync` (EXCLUDE): all referenced camera streams must resolve (a missing wrist camera is silent modality loss) and multi-camera spans must agree.
  - `visual_quality` (REVIEW-grade, never a veto — VA-B): blur (Laplacian variance), exposure (dark/blown-out bands), and contrast (P5–P95) on 10 uniformly sampled frames; aggregates as `max(blur, exposure)` per SLE, reports the worst-frame timestamp for direct human review. Thresholds are SLE-derived priors, calibrated on healthy data (penalty 0.00 on libero_10) — calibrate per camera before treating them as gates.
- **Server engine v5 (rules 0.8.0)**: new `VISUAL_REPAIR_FIRST` (HIGH) and `VISUAL_QUALITY_REVIEW` (EXPERIMENTAL) recommendations driven by an extended `audit_signals.visual_summary` (whitelisted, bounded). v1–v3 clients and payloads behave exactly as before; new actions degrade to `NO_RECOMMENDATION` on old clients via `from_dict` (safe).
- Metric provenance extended: all four new metrics ship the four-file record (`docs/provenance/<metric>/`), 18 metrics × 4 files.

### Fixed

- `VisualQualityMetric` and alignment metric now read `episode.timestamps` (the canonical field name) — verified against the schema dataclass.

## 0.6.0 - 2026-09-05

### Added

- **REQ-3: DROID-aligned idle rules.** Per-episode temporal metrics now include `usable_retention_ratio` (fraction of frames inside consecutive non-idle runs of ≥16 frames — openpi's DROID `min_non_idle_len=16` ≈ 1s at typical 15–30Hz) and `max_idle_run_frames` (longest static stretch). New `rda recommend --policy-chunk-size N` flag (e.g. 100 for ACT, 16 for Diffusion Policy) sends `policy_chunk_size` under `contract_version: 3`; the server then evaluates window integrity AT the chunk length instead of the legacy fixed 5/10/20 tiers, and may emit a chunk-window-collapse `DO_NOT_PRUNE` warning plus an EXPERIMENTAL tail-trim suggestion (DROID `filter_last_n_in_ranges=10` reference). Near-static datasets (median idle >95% AND usable retention <5%) now receive a single `DISCARD_STATIC` recommendation, mirroring openpi/DROID dropping near-static demonstrations — human review required, never automated. Offline fallback implements the same DISCARD_STATIC branch conservatively (dataset-level signal only). New enum actions degrade to `NO_RECOMMENDATION` on older clients via `from_dict` (safe).
- **REQ-2: advisory recommendation types.** New `SMOOTHING_REVIEW` (action spikes beyond 5×MAD of the episode's own motion — teleop glitches, sensor noise, or post-hoc smoothing; human review, never auto-deletion), `CALIBRATION_CHECK` (sensor sync p95 > 2× the dataset's own baseline — calibration drift detection without absolute hardware thresholds), and `COVERAGE_SUGGESTION` (episodes below the dataset's P5 state-space occupancy — collection guidance for data producers, not a deletion hint). All three are driven by optional `audit_signals` in the v3 payload (whitelisted keys, bounded size), advisory-only, and rendered with an explicit "human-review signal" marker in the CLI output.
- **REQ-8: metric provenance.** `docs/provenance/<metric>/` ships a four-file record (algorithm.md, source.md, implementation_origin.md, license.md) for all 14 metrics, with a README index — making the IP boundary explicit: public precedents inform ideas only, all implementation is original.
- Server engine v4 (rules 0.7.0): chunk-aligned window evaluation, DISCARD_STATIC, tail-trim, and the three REQ-2 advisory rules. v1/v2 requests behave exactly as before (all new parameters optional, unknown payload fields tolerated).

### Fixed

- Audit JSON report top-level `version: "0.2.0"` was a hardcoded report-schema version that read like a tool version next to `tool_version`. Renamed to `report_schema_version: "1.0"`; no consumer ever read the old key (verified repo-wide).
- `rda audit` on a LeRobot v3.0 dataset without pandas/pyarrow installed suggested `pip install lerobot` — wrong for the direct-parquet path, which needs only pandas/pyarrow. The loader now raises a precise message naming the actual missing packages; the CLI only suggests lerobot when the issue is really about lerobot.
- CLI output now reports the new DROID-aligned metrics (`usable_retention`, `max_idle_run`) in the temporal-sufficiency overview.

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
