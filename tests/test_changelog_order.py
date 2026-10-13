"""CHANGELOG hygiene guard.

v0.9.21 (P0-4). The 0.9.18 entry carried the date ``2026-10-14`` — later
than 0.9.19 (10-02) and 0.9.20 (10-08), and in the future at the time it
was noticed. A version history whose dates do not descend is, to anyone
auditing the project, evidence that releases are not under control.

This test pins two cheap invariants so the next occurrence fails CI
instead of shipping:

1. version headings must be in strictly descending semver order;
2. their dates must be non-increasing down the file (a newer release can
   never carry an older date than the release above it).

The ``Unreleased`` heading is skipped — it has no date by definition.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"

_HEADING = re.compile(
    r"^##\s+(?:Unreleased\b.*|(\d+\.\d+\.\d+)\s*-\s*(\d{4}-\d{2}-\d{2}))",
    re.MULTILINE,
)


def _key(version: str) -> tuple:
    return tuple(int(p) for p in version.split("."))


def _releases() -> list[tuple[str, date]]:
    out = []
    for m in _HEADING.finditer(CHANGELOG.read_text(encoding="utf-8")):
        if not m.group(1):
            continue  # Unreleased
        y, mo, d = (int(x) for x in m.group(2).split("-"))
        out.append((m.group(1), date(y, mo, d)))
    return out


def test_changelog_has_releases():
    assert _releases(), "no version headings found in CHANGELOG.md"


def test_versions_descend():
    versions = [v for v, _ in _releases()]
    for earlier, later in zip(versions, versions[1:]):
        assert _key(earlier) > _key(later), (
            f"CHANGELOG version order broken: {earlier} appears above {later}"
        )


def test_dates_are_non_increasing_down_the_file():
    releases = _releases()
    for (v_hi, d_hi), (v_lo, d_lo) in zip(releases, releases[1:]):
        assert d_hi >= d_lo, (
            f"CHANGELOG date order broken: {v_hi} ({d_hi}) is dated earlier "
            f"than {v_lo} ({d_lo}) — a newer release cannot ship before an "
            f"older one"
        )


def test_no_future_dates():
    today = date.today()
    for version, d in _releases():
        assert d <= today, (
            f"CHANGELOG entry {version} is dated {d}, which is in the future"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
