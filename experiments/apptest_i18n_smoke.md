# AppTest: zh/en headless UI smoke (40/40)

**When**: 2026-08-18, during the v0.5.2 bilingual release.
**Scope**: `rda ui` (Streamlit app + 7 pages) rendered headlessly in both
`zh` and `en` modes — 8 surfaces × 2 languages = 16 checks, plus 24
component-level checks (charts, export, language switcher cache, CJK leak
scan) = **40 assertions, all passed**.

## Why this matters

The i18n refactor touched every page, the formatter, the API client cache
key, and the server-side engine. A static "keys exist" check (now
[`tests/test_i18n.py`](../tests/test_i18n.py)) catches missing translations,
but it can't catch:

- a page that calls `t("wrong_key")` and silently renders the key literal,
- a chart axis label hardcoded in one language,
- a session-state cache that doesn't invalidate when the user switches
  language (shows stale zh text in en mode).

Only rendering the page and asserting on the output catches those. So the
smoke test runs each page through Streamlit's `AppTest` API and checks the
rendered text.

## Method (reproducible)

```bash
# requires: streamlit, playwright (chromium), the rda dev install
python -m streamlit.testing.app_test_runner rda/ui_app/app.py
```

Per-page assertions (abbreviated — see the v0.5.2 release regression report
for the full matrix):

1. Page renders without exception.
2. Language switcher reflects current `lang`.
3. In `en` mode, no CJK characters appear in rendered body text
   (the switcher's own "Language / 语言" dual label is whitelisted —
   it's an industry-convention affordance, see `i18n.py`).
4. In `zh` mode, body text is Chinese (not English fallback).
5. Exported report text matches the active language.

## Results (v0.5.2)

| Surface | zh | en |
|---|---|---|
| app (home) | ✓ | ✓ |
| Page 1 — Overview | ✓ | ✓ |
| Page 2 — Per-episode | ✓ | ✓ |
| Page 3 — Metrics | ✓ | ✓ |
| Page 4 — Recommend | ✓ | ✓ |
| Page 5 — Export | ✓ | ✓ |
| Page 6 — Settings | ✓ | ✓ |
| Page 7 — About | ✓ | ✓ |

Plus: en-mode CJK leak scan = 0 hits (whitelist applied); zh/en recommend
formatter output = 1183 / 2225 chars, zero mixed-language fragments.

## What this does NOT prove

- It doesn't prove the recommendations are *correct* — only that the right
  language string is wired through. Recommendation quality is a separate
  question (see [server_deploy_verify.md](./server_deploy_verify.md) for the
  online parity check, and the L2 roadmap for a labelled benchmark).
- It doesn't cover every interaction path through the UI, only the
  first-render smoke. Deeper interaction tests (drag, upload, re-run) are
  not automated yet.

## Backing artifact

- Static key parity: [`tests/test_i18n.py`](../tests/test_i18n.py) (5/5 pass,
  runs in CI without Streamlit installed, via a stub).
- The 40-assertion AppTest matrix itself was run ad-hoc during the release
  and documented in the v0.5.2 regression report. It is **not** committed as
  a runnable test file (it requires a browser runtime the CI runner doesn't
  have). If you want to re-run it, the command above is the entry point.
