# Changelog

## 0.9.2 - 2026-09-11

Governance-driven metric refinements based on semantic invariant
review and independent re-implementation audit.

### Changed
- `rda.metrics.integrity.MissingFramesMetric` (D-02): Sensor dropout
  / frame-length mismatch now returns **EXCLUDE** instead of REVIEW,
  matching spec §4.1 which mandates hard exclusion for data with
  missing frames (downstream frame-index training would misalign).
- `rda.metrics.motion.JointLimitMetric` (D-14): Replaced binary
  pass/review with **three-level classification**:
  - **PASS** – all frames within limits; inter-frame jumps normal.
  - **REVIEW** – any frame within `approach_threshold` (±2%) of a
    limit boundary, or ≥ `consecutive_frames` (3) consecutive frames
    glued to a boundary.
  - **EXCLUDE** – any frame actually exceeds a limit, or inter-frame
    jump exceeds `jump_multiplier` (2×) the theoretical max increment.
  - New measurement outputs: `min_margin_ratio`, `max_jump_ratio`.
  - Constructor accepts `approach_threshold`, `consecutive_frames`,
    `jump_multiplier` (all with sensible defaults).

### Fixed
- `rda.metrics.integrity.MissingFramesMetric`: `make_exclude()` call
  passed unsupported `measurement` and `severity` keyword arguments,
  causing a `TypeError` whenever dropout was detected.

### Added — tests
- New golden scenario `joint_limit_approaching`: smooth Gaussian bump
  approaching the limit boundary without exceeding it, verifying the
  REVIEW path of the three-level classifier.
- Updated `test_joint_limit_three_level_classification` to assert
  both EXCLUDE (actual violation) and REVIEW (approaching boundary).
- Pinned verdict distribution updated: 4 PASS / 1 REVIEW / 3 EXCLUDE
  (non-video); 5 PASS / 1 REVIEW / 6 EXCLUDE (with video).

## 0.9.1 - 2026-09-10

Patch release fixing two latent bugs surfaced by the new golden
regression set, plus the golden set itself.

### Fixed
- `rda.report.json_report`: `_verifiability_for_metric` referenced
  `MetricAvailability.NA`, an enum member that does not exist
  (the enum only defines `AVAILABLE` / `NOT_AVAILABLE` /
  `ERROR`). Any available metric crashed report generation with
  `AttributeError`. N/A is now derived from the not-available
  reason (`single_camera`, `no_video_features`, `dep_*`); other
  unavailable metrics report "Not verifiable".
- `rda.metrics.temporal`: `SamplingJitterMetric` pass path omitted
  `jitter_ratio` from its measurement, so
  `compute_behavior_severity` never saw it and the sampling-jitter
  behavior diagnostic could never fire. `jitter_ratio` (= CV) is
  now emitted on every path.

### Added — tests
- `tests/golden/`: deterministic golden dataset — 7 non-video and 4
  PyAV-synthesized video episodes with pinned verdicts and
  must-trigger metrics, covering all four audit layers (Integrity
  Gate / Trajectory Diagnostics / Dataset Profile / Dataset
  Summary).
- `tests/test_golden_regression.py`: end-to-end regression guards —
  verdict per scenario, must-trigger metrics, diagnostics never
  escalate to hard excludes, N/A behaviour without video,
  verifiability fields, dataset aggregation, and a pinned verdict
  distribution.

## 0.9.0 - 2026-09-10

Four-layer audit architecture: Integrity Gate → Trajectory Diagnostics →
Dataset Profile → Dataset Summary. The audit verdict is now driven
exclusively by hard integrity checks; diagnostic metrics report
measurements and findings without flipping episode verdicts. See
`RDA_v0.9_核心参数规范.md` / `RDA_v0.9_迁移计划.md` for the full design.

### Changed — rules layer (Phase 1)
- `REVIEW_METRICS` renamed to `DIAGNOSTIC_METRICS` (backward-compatible
  alias `REVIEW_METRICS` kept). Diagnostic findings no longer trigger a
  REVIEW verdict — they produce measurements + findings only.
- `upgrade_verdict_by_behavior` is now opt-in (`enabled=False` by
  default).
- `compute_behavior_severity`: added `sampling_jitter` branch
  (jitter_ratio > 0.3 → +30 severity, > 0.1 → +10); fixed the
  `distribution` → `coverage` field mapping (`occupancy_rate`).
- `video_frame_integrity` wired into `CRITICAL_METRICS` (fixes the v0.8
  wiring gap; a failed frame-integrity check now hard-excludes).

### Changed — video stream metrics split (Phase 2)
- `video_stream_sync` split into four independent metrics:
  - `video_stream_presence` (L1, hard check — missing camera stream →
    EXCLUDE),
  - `video_stream_span_consistency` (L2 diagnostic),
  - `video_stream_temporal_offset` (L2 measurement, pairwise frame-level
    offset; "Not verifiable" without frame timestamps),
  - `video_stream_temporal_drift` (L2 measurement, clock drift rate).
  Old `VideoStreamSyncMetric` names are kept as compatibility aliases.

### Changed — naming and layer placement (Phase 3)
- `temporal_sufficiency` renamed to `temporal_structure` (compatibility
  alias kept). It now belongs to the Dataset Profile (L3) layer.
- L3 metrics (`distribution`, `coverage`, `temporal_structure`) no
  longer participate in episode verdicts; their measurements feed the
  Dataset Profile aggregation only.

### Added — Dataset Summary layer (Phase 4)
- `rda/report/dataset_summary.py` (new): cross-episode aggregation of
  idle_ratio (median/P10/P90), sensor sync (median/p90/max of worst p95
  offset) and visual quality (median blur variance + exposure anomaly
  ratio). Attached to `DatasetAuditResult.dataset_summary` and emitted
  in both text and JSON reports.

### Added — verifiability levels + report format (Phase 5)
- Every metric result now carries a `verifiability` field:
  **Verified** (hard-checked), **Measured** (observational
  measurement), **Not verifiable** (missing reference/timestamps),
  **N/A** (modality absent).
- JSON report: `report_schema_version` bumped to 1.1; new top-level
  `dataset_summary` section.
- Text report restructured into Integrity Gate / Trajectory
  Diagnostics / Dataset Profile / Dataset Summary sections with a
  dedicated Video Temporal Verification block.

### Added — recommend audit_signals (Phase 6)
- Recommend contract bumped to v4: clients may send `audit_signals`
  (smoothness_summary / calibration_summary / coverage_summary)
  extracted from the audit pipeline, enabling SMOOTHING_REVIEW /
  CALIBRATION_CHECK / COVERAGE_SUGGESTION server-side rules. Older
  servers ignore the unknown key; older clients are unaffected.
- CLI `rda recommend --audit-report <rda_report.json>`: extracts
  diagnostic signals from a pre-computed audit report and sends them
  with the recommendation request.
- Cache key namespaced `v4-` (includes audit signals).

### Versioning
- Package version bumped 0.8.0 → 0.9.0 (`__init__.py`, `pyproject.toml`,
  User-Agent all aligned; enforced by `test_version_consistency.py`).

## 0.8.0 - 2026-09-06

### Added (REQ-5 — dataset-level relative baselines + acceptance summary)
- `rda/report/acceptance.py` (new): dataset-level relative baselines.
  - `tukey_fence` / `is_tukey_outlier` — episode-runtime outlier detection
    using the Tukey fence `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`, ported from the
    [SLE] reference implementation (`score_lerobot_episodes`
    `is_time_outlier(mode="iqr")`). No threshold is invented here.
  - `compute_percentile_baselines` — P10 / P50 / P90 (+ min / max) for the
    Tier-1 portable metric set (`duration_sec`, `spike_count`,
    `effective_motion_ratio`; rho = 0.960 vs the full metric set, see
    `rda/calibration/portable.py`). Baselines are relative to the audited
    population, not an absolute threshold borrowed from another platform.
  - `build_acceptance_summary` — folds verdict distribution, percentile
    baselines, runtime outliers, the calibration layer and the
    not-checked inventory into one block.
- `acceptance_summary` is emitted by **both** report formats: the engine
  (CLI `--format json`) and the product/dataset report. `schema_version`
  `1.0`.
- UI: new page **8 · Acceptance** (`rda/ui_app/pages/8_Acceptance.py`) —
  dataset identity, verdict distribution, quality layer, baseline table,
  outlier table, not-checked inventory, calibration layer, and an export +
  disclaimer section. Registered in `app.py` navigation.
- i18n: 48 new bilingual (zh/en) `acc_*` keys plus `nav_acceptance`.
- `skipped_by_missing_dep` is now a public function in `json_report`
  (the private `_skipped_by_missing_dep` name is kept as an alias so
  existing callers and tests keep working).

### Red line honoured
REQ-5 measures and presents; it does **not** adjudicate. The acceptance
summary contains no accept/reject decision — that call belongs to the
human reviewer. `test_acceptance_summary_does_not_adjudicate` pins this.

### Degenerate-fence guard
When every episode has the same duration, `IQR = 0` and the Tukey fence
collapses to a single point. Reporting "0 outliers" there would read as
"checked and clean" when the truth is "no spread to detect against" — the
same class of trap as REQ-11's "not checked must not read as pass". The
block now sets `degenerate: true` with an explicit note, and the UI shows
a warning. Verified on aloha_insertion (50 episodes, all 9.98 s).

### Fixed
- **Test infrastructure silently faking green (pre-existing).**
  `tests/test_i18n.py` and `tests/test_ui_visual_audit.py` injected a stub
  `streamlit` module into `sys.modules` unconditionally, which broke
  `streamlit.testing.v1` imports for the rest of the session — every
  `AppTest` case was skipped instead of run. The stub now applies only
  when the real streamlit is genuinely unavailable.
- `tests/test_ui_visual_audit.py` never imported `DatasetAuditor`
  (a `NameError` swallowed by a bare `except:`). Combined with the above,
  **the REQ-4 UI page tests had never actually executed**. Now imported;
  both cases run and pass.
- `AppTest.from_file` does not put `rda/ui_app` on `sys.path` (unlike
  `streamlit run`, where `app.py` handles it), so pages importing
  `components.common` raised `ModuleNotFoundError`. Tests now insert the
  directory explicitly. Diagnostics were also added to the previously
  silent `except:` so skipped-for-missing-data is visible rather than
  silent.
- Version strings had drifted apart for two releases: `rda.__version__`
  and the `rda-cli/<ver>` User-Agent both said 0.7.2 while
  `pyproject.toml` said 0.7.3. All three are now 0.8.0, and
  `tests/test_version_consistency.py` pins them together.

### Tests
- `tests/test_req5_acceptance_summary.py` (new, 14 cases): Tukey fence
  arithmetic against hand-computed values, degenerate-fence branch,
  unavailable metrics never counted as 0, outlier ordering/truncation,
  no-adjudication guarantee, i18n key parity, and a Streamlit `AppTest`
  end-to-end render of the new page.
- `tests/test_version_consistency.py` (new, 2 cases).
- Suite: 154 passed, 2 skipped (the 2 skips need a real LIBERO v3.0
  dataset that is not present on this machine).

### Measured on real datasets
- libero_10 (379 episodes): 36 runtime outliers, fence `[16.5 s, 35.7 s]`,
  p50 = 25.8 s, max = 50.4 s — all on the long side.
- aloha_insertion (50 episodes): all durations 9.98 s → degenerate branch,
  correctly flagged rather than reported as "0 outliers".

## 0.7.3 - 2026-09-05

### Added (v0.7.x UI catch-up + REQ-3②)
- UI: Visual Audit panel in Episode Explorer detail — VA-A integrity
  trio status lines with freeze-region table (per-stream span,
  duration, moving ratio) and VA-B quality breakdown (per-camera
  penalty table + per-sample quality curves with clipped-frac on a
  secondary axis). `video_deps_missing` renders a loud "not audited ≠
  pass" warning and hides the quality tables instead of faking them.
- UI: Visual Integrity Overview section in Health Overview —
  dataset-level checked / flagged / not-checked episode counts with a
  per-metric findings list and a page link into the review queue.
- i18n: 45 new bilingual (zh/en) keys covering the visual panels,
  visual recommendation texts (`rec_video_freeze`,
  `rec_video_timestamp_alignment`, `rec_video_stream_sync`,
  `rec_visual_quality`) and the health visual overview.
- REQ-3②: fps-aware usable-run floor — `usable_retention_ratio` now
  uses `max(16, 1s-in-frames)` as the minimum run length when the
  episode's fps is known (from dataset info.json via the loader), and
  exactly 16 otherwise (byte-identical to 0.6.x at ≤16 Hz). Rationale:
  DROID's `min_non_idle_len=16` encodes "≈1 s at 15-30 Hz"; at 50 Hz
  16 frames is 0.32 s, letting sub-second twitches count as usable
  (measured on aloha_insertion @50 Hz: usable retention 24.6% → 0.0%
  on a genuine static episode). New field `usable_run_floor_frames`
  (per episode and dataset median) reports which floor produced the
  ratio; server payload gains the field additively (v0.6.1+ parsers
  ignore unknown keys).
- Tests: `tests/test_ui_visual_audit.py` (stats layer + Streamlit
  AppTest end-to-end runs of both pages, incl. the dep-missing
  never-pass guarantee) and `tests/test_req3b_fps_floor.py` (floor
  semantics + fallbacks). Suite: 136 passed, 4 skipped.

## 0.7.2 - 2026-09-05

### Fixed
- `skipped_by_missing_dep` crashed on real `DatasetAuditResult` shapes
  (`EpisodeAuditResult` exposes `metrics`, not `metric_results`) — the
  0.7.1 wheel shipped before this fix was caught in the post-upload
  smoke test. 0.7.2 is the first wheel containing the corrected
  attribute access. All 0.7.1 features otherwise unchanged.

## 0.7.1 - 2026-09-05

### Added (REQ-10 — visual detection calibration closure)
- `benchmarks/visual_inject.py`: corruption-injection benchmark — five
  visual corruption families (blur gradient, dark, blown-out, freeze
  spans, static camera) re-encoded onto real episodes; produces
  per-scenario detection tables. Calibration run (libero_10 × 5
  episodes × 11 scenarios): all acceptance checks passed, including
  both boundary cases (majority-frozen episode and fully static camera
  — the adaptive epsilon does not desensitize).
- `visual_quality`: highlight-clipping detection (`clipped_frac`,
  fraction of pixels ≥ 250; ≥ 20% → blown-out). Calibration finding:
  mean luminance alone misses saturation on dark scenes (×3 gain clips
  22–33% of pixels while mean stays ≈ 180, under the 230 band).
- Multi-dataset threshold check (libero_10 / jaco_play / xarm_lift):
  penalty p50 = 0.00 everywhere; jaco_play shows expected REVIEW
  signals on genuinely blurred episodes (blur_var p10 = 61.5 vs xarm
  464.6). Thresholds hold across resolutions.
- Decode performance numbers (README): freeze ≈ 0.11 s/episode,
  quality ≈ 0.17 s/episode (worst ≈ 0.17 s) on 10k-frame 224×224
  chunk files.

### Added (REQ-11 — optional-dependency UX)
- Visual metrics grade NA with reason `video_deps_missing` when PyAV is
  absent ("not audited", never "pass").
- JSON report: top-level `skipped_by_missing_dep` counts (e.g.
  `{"av": 12}`); CLI prints a warning with the install hint.
- `[video]` extra (`pip install robot-data-audit[video]`); README
  dependency-tier table.

### Added (UI catch-up)
- `7_Recommend.py`: policy_chunk_size input (CLI had it since 0.6.0)
  and icons for the visual / advice actions.

### Tests
- 9 new tests (REQ-10 detection anchors + REQ-11 semantics);
  122 passed, 2 skipped, zero regressions.

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
