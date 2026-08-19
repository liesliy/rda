# Experiments

Lightweight write-ups of the experiments we actually ran.
Each entry links to the concrete artifact (test file, CI job, or doc) that
backs its claims — if there's no link, the claim isn't made.

## Index

| Write-up | What it covers | Backed by |
|---|---|---|
| [apptest_i18n_smoke.md](./apptest_i18n_smoke.md) | 40/40 headless UI smoke test across zh/en × 8 pages (v0.5.2) | `tests/test_i18n.py` |
| [server_deploy_verify.md](./server_deploy_verify.md) | Online API verification: engine_version=2, zh/en recommend parity | `https://rda.niusu2026.cn` (live) |
| [negative_control_gate.md](./negative_control_gate.md) | The spike-computed / verdict-ignored bug: story + regression pin | `tests/test_negative_control.py` |
| [wheel_leak_guard.md](./wheel_leak_guard.md) | Build-time guarantee: `server/` and `engine_core` never ship in the PyPI wheel | `.github/workflows/ci.yml` |

For the 11-dataset / 4,909-episode audit results, see
[`docs/benchmark.md`](../docs/benchmark.md) — that's a results table, not an
experiment write-up, so it lives in `docs/`.

## What this directory is NOT

- **No AUC / precision / recall numbers.** We have not run a labelled-ground-truth
  benchmark. Numbers like "AUC 0.916" you may have seen elsewhere are not
  reproducible from this repo and are deliberately omitted.
- **No cross-tool comparison.** A head-to-head against Calibra or other
  data-health tools is planned (see roadmap L2) but not done.
- **No framework.** Each write-up is a standalone markdown file. We will not
  build a structured JSON schema until there are 3+ real-team case studies to
  generalize from.

If you want to add a case study (your dataset, your finding), open an issue —
we'll write it up here in the same flat-markdown style.
