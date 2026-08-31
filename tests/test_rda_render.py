"""Tests for ``tools/rda_render.py`` (HTML report + SVG badge renderer).

The renderer turns an RDA JSON audit report into:
- a self-contained HTML report (verdict cards, three-layer table,
  per-episode SVG strip, top observations), and
- a shields-style SVG badge whose color carries meaning:
  EXCLUDE>0 -> red, else REVIEW>0 -> amber, else green.

Reports below are hand-constructed minimal fixtures — every expected
string and count is exact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_PATH = REPO_ROOT / "tools" / "rda_render.py"

spec = importlib.util.spec_from_file_location("rda_render", RENDER_PATH)
rda_render = importlib.util.module_from_spec(spec)
sys.modules["rda_render"] = rda_render
spec.loader.exec_module(rda_render)


# --- fixtures ----------------------------------------------------------------


def _episode(idx: int, verdict: str) -> dict:
    return {
        "episode_index": idx,
        "verdict": verdict,
        "metrics": {},
        "verdict_reasons": [],
    }


@pytest.fixture
def report_mixed() -> dict:
    """206-episode-like summary: 43 PASS / 163 REVIEW / 0 EXCLUDE."""
    episodes = ([_episode(i, "PASS") for i in range(43)]
                + [_episode(i, "REVIEW") for i in range(43, 206)])
    return {
        "tool_version": "0.5.4",
        "dataset": {"path": "lerobot/pusht", "total_frames": 25650,
                    "modalities": ["vision", "state"]},
        "summary": {
            "total_episodes": 206,
            "verdict_counts": {"PASS": 43, "REVIEW": 163, "EXCLUDE": 0},
            "pass_rate": 43 / 206,
            "review_episodes": list(range(43, 206)),
            "exclude_episodes": [],
        },
        "three_layer_aggregates": {
            "layer1_integrity": {
                "missing_dropout": {"available": 206, "pass_rate": 1.0,
                                    "failed": 0, "na": 0},
            },
            "layer2_temporal_motion": {
                "timestamp_validity": {"available": 206, "pass_rate": 0.79,
                                       "failed": 43, "na": 0},
            },
            "layer3_dataset_utility": {
                "coverage": {"available": 0, "pass_rate": None,
                             "failed": 0, "na": 206},
            },
        },
        "top_observations": [
            {"metric": "timestamp_validity", "description": "43 episodes with reversed timestamps",
             "evidence_level": "HARD_FAIL"},
        ],
        "episodes": episodes,
    }


# --- badge --------------------------------------------------------------------


def test_badge_all_pass_is_green():
    svg = rda_render.make_badge({"PASS": 10, "REVIEW": 0, "EXCLUDE": 0}, 10, "ds", "0.5.4")
    assert "#2da44e" in svg  # PASS green
    assert "#cf222e" not in svg and "#bf8700" not in svg
    assert "10 pass" in svg


def test_badge_review_is_amber():
    svg = rda_render.make_badge({"PASS": 43, "REVIEW": 163, "EXCLUDE": 0}, 206, "ds", "0.5.4")
    assert "#bf8700" in svg  # REVIEW amber
    assert "#cf222e" not in svg
    assert "43 pass" in svg and "163 review" in svg


def test_badge_exclude_is_red():
    svg = rda_render.make_badge({"PASS": 3, "REVIEW": 2, "EXCLUDE": 1}, 6, "ds", "0.5.4")
    assert "#cf222e" in svg  # EXCLUDE red wins over amber
    assert "1 excl" in svg


def test_badge_zero_episodes():
    svg = rda_render.make_badge({"PASS": 0, "REVIEW": 0, "EXCLUDE": 0}, 0, "ds", "0.5.4")
    assert "no episodes" in svg


def test_badge_is_valid_xml():
    import xml.etree.ElementTree as ET
    svg = rda_render.make_badge({"PASS": 1, "REVIEW": 1, "EXCLUDE": 1}, 3,
                                'ds"&<>', "0.5.4")
    ET.fromstring(svg)  # raises on malformed XML (escapes checked)


# --- HTML ----------------------------------------------------------------------


def test_html_contains_verdict_cards(report_mixed):
    html = rda_render.build_html(report_mixed, "lerobot/pusht", "report.json")
    assert "43" in html and "163" in html
    assert "PASS — clean" in html
    assert "REVIEW — worth a human look" in html
    assert "EXCLUDE — integrity failure" in html


def test_html_three_layer_table(report_mixed):
    html = rda_render.build_html(report_mixed, "lerobot/pusht", "report.json")
    assert "Layer 1 · Integrity" in html
    assert "Layer 2 · Temporal &amp; Motion" in html or "Layer 2 · Temporal & Motion" in html
    assert "79%" in html          # failed metric pass rate
    assert "n/a" in html          # unavailable metric rendered as n/a


def test_html_top_observations_and_escape(report_mixed):
    html = rda_render.build_html(report_mixed, 'ds"<script>', "report.json")
    assert "<script>" not in html.replace("</script>", "")  # title escaped
    assert "&lt;script&gt;" in html
    assert "timestamp_validity" in html
    assert "HARD_FAIL" in html


def test_html_self_contained(report_mixed):
    """No external asset references: single-file requirement."""
    html = rda_render.build_html(report_mixed, "ds", "report.json")
    assert "http://" not in html
    assert "https://" not in html
    assert "<style>" in html


def test_html_episode_strip_count(report_mixed):
    html = rda_render.build_html(report_mixed, "ds", "report.json")
    # one strip cell per episode: 43 PASS (green) + 163 REVIEW (amber)
    assert html.count('height="14" rx="1" fill="#2da44e"') == 43
    assert html.count('height="14" rx="1" fill="#bf8700"') == 163


def test_cli_end_to_end(tmp_path, report_mixed):
    """main() writes both outputs from a JSON file on disk."""
    src = tmp_path / "report.json"
    src.write_text(json.dumps(report_mixed), encoding="utf-8")
    html_out = tmp_path / "out.html"
    badge_out = tmp_path / "out.svg"

    sys.argv = ["rda_render.py", str(src), "--html", str(html_out),
                "--badge", str(badge_out), "--title", "t"]
    rda_render.main()

    assert html_out.exists() and "<!DOCTYPE html>" in html_out.read_text(encoding="utf-8")
    svg = badge_out.read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "163 review" in svg
