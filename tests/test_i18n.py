"""i18n catalog integrity tests.

The bilingual UI (zh/en) is only trustworthy if the two string tables
stay perfectly aligned: same keys, valid values, matching format
placeholders. A missing key falls back silently (by design, to keep
pages usable), which means a drifted table produces *mixed-language
pages* that nobody notices until a user screenshots them. These tests
make that drift loud.

Runs without streamlit installed: the catalog module only touches
``st`` at render time, so a stub is injected at import time. This also
keeps the test green in CI, where the [ui] extra is not installed.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Inject a streamlit stub BEFORE importing the catalog (import-time only).
if "streamlit" not in sys.modules:
    _st = types.ModuleType("streamlit")
    _st.session_state = {}
    _st.radio = lambda *a, **k: None
    sys.modules["streamlit"] = _st

from rda.ui_app.i18n import DEFAULT_LANG, LANGS, STRINGS  # noqa: E402

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _tables() -> dict:
    return {lang: STRINGS[lang] for lang in LANGS}


def test_both_languages_present():
    assert set(LANGS) == {"zh", "en"}
    assert DEFAULT_LANG in LANGS
    for lang in LANGS:
        assert isinstance(STRINGS[lang], dict), f"table for {lang} missing"
        assert len(STRINGS[lang]) > 0, f"table for {lang} is empty"


def test_key_sets_identical():
    tables = _tables()
    zh_keys, en_keys = set(tables["zh"]), set(tables["en"])
    only_zh = zh_keys - en_keys
    only_en = en_keys - zh_keys
    assert not only_zh, f"keys only in zh (en pages show raw keys): {sorted(only_zh)[:10]}"
    assert not only_en, f"keys only in en (zh pages show raw keys): {sorted(only_en)[:10]}"


# Keys where an empty string is by design: zh appends a suffix ("级" in
# "A级") while en renders the bare grade ("A"). Anything new landing here
# must justify itself.
EMPTY_BY_DESIGN = {"health_grade_suffix"}


def test_values_are_nonempty_strings():
    for lang, table in _tables().items():
        for key, value in table.items():
            assert isinstance(value, str), f"{lang}:{key} is {type(value).__name__}, not str"
            if key in EMPTY_BY_DESIGN:
                continue
            assert value.strip(), f"{lang}:{key} is empty or whitespace"


def test_format_placeholders_match():
    """``{name}`` placeholders must be identical across languages.

    A mismatch means ``s.format(**kwargs)`` raises KeyError for one
    language but not the other — a crash that only reproduces in the
    language you didn't test.
    """
    for key in _tables()["zh"]:
        zh_ph = set(_PLACEHOLDER_RE.findall(_tables()["zh"][key]))
        en_ph = set(_PLACEHOLDER_RE.findall(_tables()["en"][key]))
        assert zh_ph == en_ph, (
            f"key '{key}': zh placeholders {sorted(zh_ph)} != "
            f"en placeholders {sorted(en_ph)}"
        )


def test_no_double_braces():
    """Escaped/typo'd braces (``{{``) would survive t() verbatim."""
    for lang, table in _tables().items():
        for key, value in table.items():
            assert "{{" not in value and "}}" not in value, (
                f"{lang}:{key} contains doubled braces"
            )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
