"""Version consistency guard.

RDA's version string is duplicated in three places that drifted apart in
0.7.x: ``rda.__version__`` still said 0.7.2 while ``pyproject.toml`` said
0.7.3, and the recommend API client hard-codes its own ``rda-cli/<ver>``
User-Agent. Two releases shipped with that mismatch unnoticed, so this
test pins all three to the same value.
"""
from __future__ import annotations

import re
from pathlib import Path

from rda import __version__

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no version = ... found in pyproject.toml"
    return match.group(1)


def test_dunder_version_matches_pyproject():
    assert __version__ == _pyproject_version(), (
        f"rda.__version__={__version__!r} but pyproject declares "
        f"{_pyproject_version()!r}"
    )


def test_api_client_user_agent_version_matches():
    """The recommend client advertises its version; keep it in sync."""
    source = (ROOT / "rda" / "recommend" / "api_client.py").read_text(
        encoding="utf-8"
    )
    agents = set(re.findall(r'"rda-cli/([^"]+)"', source))
    assert agents, "no rda-cli/<version> User-Agent found"
    assert agents == {__version__}, (
        f"User-Agent versions {sorted(agents)} != __version__ {__version__!r}"
    )
