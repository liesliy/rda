#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rda-render: RDA JSON audit report -> shareable self-contained HTML + README badge.

Usage:
    python rda_render.py <rda_report.json> [--html out.html] [--badge out.svg] [--title "My dataset"]

Outputs:
- A single-file HTML report (no external assets, no JS dependencies) with:
  verdict summary, three-layer metric pass rates, per-episode strip, top observations.
- An SVG badge (shields.io style, flat) showing overall verdict composition,
  ready to paste into a README as: ![RDA audit](path/to/badge.svg)

Design goals: the audited dataset carries its own proof — anyone who receives
the dataset folder (or the HTML) can see the audit result without running RDA.
"""

import argparse
import base64
import hashlib
import html
import json
import sys
from pathlib import Path

VERDICT_COLORS = {"PASS": "#2da44e", "REVIEW": "#bf8700", "EXCLUDE": "#cf222e"}
VERDICT_ORDER = ["PASS", "REVIEW", "EXCLUDE"]

LAYER_LABELS = {
    "layer1_integrity": "Layer 1 · Integrity",
    "layer2_temporal_motion": "Layer 2 · Temporal & Motion",
    "layer3_dataset_utility": "Layer 3 · Dataset Utility",
}

METRIC_LABELS = {
    "missing_dropout": "missing/dropout",
    "invalid_values": "NaN/Inf",
    "schema_consistency": "schema",
    "timestamp_validity": "timestamps",
    "joint_limit": "joint limits",
    "sensor_synchronization": "sensor sync",
    "velocity_acceleration": "velocity/acc",
    "action_discontinuity": "action spikes",
    "temporal_sufficiency": "sufficiency",
    "idle_ratio": "idle ratio",
    "coverage": "coverage",
    "distribution": "distribution",
}


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:64] or "dataset"


def make_badge(counts: dict, total: int, dataset_name: str, rda_version: str) -> str:
    """shields-style flat SVG badge: `RDA | 43 pass · 163 review · 0 excl`."""
    parts = []
    for v in VERDICT_ORDER:
        c = counts.get(v, 0)
        if c > 0 or v != "EXCLUDE":
            parts.append(f"{c} {v.lower()}")
    right = " · ".join(parts)
    if total == 0:
        right = "no episodes"

    left = "RDA audit"
    font = "11px Verdana,Geneva,DejaVu Sans,sans-serif"
    # width estimation: ~6.2px per char at 11px Verdana, plus padding
    w1 = int(len(left) * 6.6) + 12
    w2 = int(len(right) * 6.6) + 12
    h = 20

    def esc(s):
        return html.escape(s, quote=True)

    # overall color: EXCLUDE>0 red, else REVIEW>0 amber, else green
    if counts.get("EXCLUDE", 0) > 0:
        accent = VERDICT_COLORS["EXCLUDE"]
    elif counts.get("REVIEW", 0) > 0:
        accent = VERDICT_COLORS["REVIEW"]
    else:
        accent = VERDICT_COLORS["PASS"]

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w1 + w2}" height="{h}" role="img" aria-label="{esc(left)}: {esc(right)}">
  <title>RDA audit: {esc(right)}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r"><rect width="{w1 + w2}" height="{h}" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{w1}" height="{h}" fill="#555"/>
    <rect x="{w1}" width="{w2}" height="{h}" fill="{accent}"/>
    <rect width="{w1 + w2}" height="{h}" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="{font}" text-rendering="geometricPrecision">
    <text x="{w1 / 2 + 1}" y="{h / 2 + 3.8}" fill="#010101" fill-opacity=".3">{esc(left)}</text>
    <text x="{w1 / 2 + 1}" y="{h / 2 + 2.8}">{esc(left)}</text>
    <text x="{w1 + w2 / 2 + 1}" y="{h / 2 + 3.8}" fill="#010101" fill-opacity=".3">{esc(right)}</text>
    <text x="{w1 + w2 / 2 + 1}" y="{h / 2 + 2.8}">{esc(right)}</text>
  </g>
</svg>'''
    return svg


def fmt_pct(x):
    return f"{x * 100:.0f}%"


def build_html(report: dict, title: str, source_file: str) -> str:
    summary = report.get("summary", {})
    counts = summary.get("verdict_counts", {})
    total = summary.get("total_episodes", 0)
    ds = report.get("dataset", {})
    version = report.get("tool_version", "?")
    generated = report.get("version", "")

    counts = {v: counts.get(v, 0) for v in VERDICT_ORDER}
    pass_rate = summary.get("pass_rate")

    # ---- three layer aggregates ----
    agg = report.get("three_layer_aggregates", {})
    layer_rows = []
    for layer, label in LAYER_LABELS.items():
        metrics = agg.get(layer, {})
        for m, st in metrics.items():
            avail = st.get("available", 0)
            pr = st.get("pass_rate")
            failed = st.get("failed", 0)
            na = st.get("na", 0)
            if pr is None:
                status, color = "n/a", "#8b949e"
            elif failed == 0 and na == 0:
                status, color = "100%", VERDICT_COLORS["PASS"]
            elif failed == 0:
                status, color = f"{fmt_pct(pr)} (+{na} n/a)", VERDICT_COLORS["REVIEW"]
            else:
                status, color = fmt_pct(pr), VERDICT_COLORS["EXCLUDE"]
            layer_rows.append(
                (label, METRIC_LABELS.get(m, m), avail, status, color, failed))

    # ---- per-episode strip (SVG) ----
    eps = report.get("episodes", [])
    strip_cells = []
    cell_w = 3.0
    gap = 0.6
    for e in eps:
        v = e.get("verdict", "PASS")
        x = len(strip_cells) * (cell_w + gap)
        strip_cells.append(
            f'<rect x="{x:.1f}" y="0" width="{cell_w}" height="14" rx="1" fill="{VERDICT_COLORS.get(v, "#999")}">'
            f"<title>ep {e.get('episode_index', '?')} · {v}</title></rect>")
    strip_w = len(strip_cells) * (cell_w + gap)
    strip_svg = ""
    if strip_cells:
        strip_svg = (
            f'<svg width="100%" height="18" viewBox="0 0 {strip_w:.0f} 14" preserveAspectRatio="none" '
            f'style="max-width:{strip_w:.0f}px">{"".join(strip_cells)}</svg>'
            '<div class="legend">'
            + "".join(
                f'<span class="dot" style="background:{c}"></span>{v} {counts.get(v, 0)}'
                for v, c in VERDICT_COLORS.items())
            + "</div>")

    # ---- top observations ----
    obs_rows = []
    for o in report.get("top_observations", [])[:8]:
        obs_rows.append(
            f"<tr><td><code>{html.escape(str(o.get('metric', '')))}</code></td>"
            f"<td>{html.escape(str(o.get('description', '')))}</td>"
            f"<td>{html.escape(str(o.get('evidence_level', '')))}</td></tr>")

    # ---- review episode list (first N) ----
    review_ids = summary.get("review_episodes", []) or []
    exclude_ids = summary.get("exclude_episodes", []) or []

    def idlist(ids, n=60):
        s = ", ".join(str(i) for i in ids[:n])
        return s + (f" … (+{len(ids) - n} more)" if len(ids) > n else "")

    rows_html = "".join(
        f"<tr><td>{lbl}</td><td>{met}</td><td>{avail}</td>"
        f'<td style="color:{color};font-weight:600">{status}</td></tr>'
        for lbl, met, avail, status, color, failed in layer_rows)

    # hero metrics (compact)
    hero = report.get("hero_metrics", {})
    hero_items = []
    if hero.get("action_discontinuity"):
        ad = hero["action_discontinuity"]
        hero_items.append(("Action spikes", f"{ad.get('total_spikes', 0)} across {ad.get('affected_episodes', 0)} eps"))
    if hero.get("state_space_occupancy"):
        so = hero["state_space_occupancy"]
        hero_items.append(("State occupancy", f"median {so.get('median_occupancy', '-')} ({so.get('interpretation', '-')})"))
    if hero.get("sensor_synchronization"):
        ss = hero["sensor_synchronization"]
        avail_eps = ss.get("available_episodes", 0)
        if avail_eps:
            hero_items.append(("Sensor sync p95", f"{ss.get('median_p95_offset_ms', '-')} ms median"))
        else:
            hero_items.append(("Sensor sync", "n/a on this dataset"))

    title_esc = html.escape(title)
    src_esc = html.escape(source_file)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RDA audit · {title_esc}</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ font: 15px/1.55 -apple-system, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
       color: #1f2328; margin: 0; background: #f6f8fa; }}
.wrap {{ max-width: 960px; margin: 24px auto 48px; padding: 0 16px; }}
header {{ background: #1f2328; color: #fff; border-radius: 10px; padding: 22px 26px; margin-bottom: 18px; }}
header h1 {{ margin: 0 0 4px; font-size: 21px; }}
header .sub {{ color: #9ea7b3; font-size: 13px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 10px; padding: 14px 16px; }}
.card .num {{ font-size: 26px; font-weight: 700; }}
.card .lbl {{ font-size: 12px; color: #656d76; }}
section {{ background: #fff; border: 1px solid #d0d7de; border-radius: 10px; padding: 18px 22px; margin-bottom: 16px; }}
h2 {{ font-size: 15px; margin: 0 0 12px; color: #57606a; text-transform: uppercase; letter-spacing: .4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eaeef2; }}
th {{ color: #656d76; font-weight: 600; font-size: 12.5px; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 5px 0 12px; }}
.legend {{ font-size: 12.5px; color: #57606a; margin-top: 6px; }}
.legend .dot {{ margin-left: 0; }}
code {{ background: #f6f8fa; border-radius: 4px; padding: 1px 5px; font-size: 12.5px; }}
.foot {{ color: #8b949e; font-size: 12px; margin-top: 14px; }}
ul {{ margin: 6px 0; padding-left: 20px; }}
.big {{ display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }}
.big .v {{ font-size: 30px; font-weight: 700; }}
</style>
</head>
<body><div class="wrap">

<header>
  <h1>{title_esc}</h1>
  <div class="sub">Independent quality audit · RDA v{html.escape(str(version))} · LeRobot format · {total} episodes · {ds.get('total_frames', '?')} frames · modalities: {html.escape(', '.join(ds.get('modalities', []) or ['n/a']))}</div>
</header>

<div class="cards">
  <div class="card"><div class="num" style="color:{VERDICT_COLORS['PASS']}">{counts['PASS']}</div><div class="lbl">PASS — clean</div></div>
  <div class="card"><div class="num" style="color:{VERDICT_COLORS['REVIEW']}">{counts['REVIEW']}</div><div class="lbl">REVIEW — worth a human look</div></div>
  <div class="card"><div class="num" style="color:{VERDICT_COLORS['EXCLUDE']}">{counts['EXCLUDE']}</div><div class="lbl">EXCLUDE — integrity failure</div></div>
  <div class="card"><div class="num">{fmt_pct(pass_rate) if pass_rate is not None else '—'}</div><div class="lbl">PASS rate</div></div>
</div>

<section>
  <h2>Per-episode verdict strip</h2>
  {strip_svg}
</section>

<section>
  <h2>Metric pass rates (three layers, default thresholds, zero per-dataset tuning)</h2>
  <table>
    <tr><th>Layer</th><th>Metric</th><th>Episodes measured</th><th>Pass rate</th></tr>
    {rows_html}
  </table>
</section>

{('<section><h2>Top observations</h2><table><tr><th>Metric</th><th>Finding</th><th>Evidence</th></tr>' + "".join(obs_rows) + "</table></section>") if obs_rows else ""}

{('<section><h2>Hero metrics</h2><ul>' + "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in hero_items) + "</ul></section>") if hero_items else ""}

<section>
  <h2>Episode lists</h2>
  <p><b>REVIEW ({counts['REVIEW']})</b>: {html.escape(idlist(review_ids)) or "—"}</p>
  <p><b>EXCLUDE ({counts['EXCLUDE']})</b>: {html.escape(idlist(exclude_ids)) or "—"}</p>
  <p style="color:#656d76;font-size:13px">REVIEW is a statistical signal worth human inspection before training — not a verdict of "bad data". EXCLUDE means an integrity-layer hard failure or unreadable episode.</p>
</section>

<div class="foot">Generated from <code>{src_esc}</code> · RDA audit runs fully offline on LeRobot v2.1/v3.0 datasets · <code>pip install robot-data-audit</code> to reproduce · GitHub: github.com/liesliy/rda</div>

</div></body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="Render RDA JSON report to shareable HTML + badge")
    ap.add_argument("report", help="path to rda audit JSON report")
    ap.add_argument("--html", dest="html_out", default=None, help="output HTML path")
    ap.add_argument("--badge", dest="badge_out", default=None, help="output SVG badge path")
    ap.add_argument("--title", default=None, help="dataset title shown in HTML/badge context")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    ds_name = safe_name(Path(report.get("dataset", {}).get("path", args.report)).stem
                        or Path(args.report).stem)
    title = args.title or report.get("dataset", {}).get("path", ds_name)

    html_out = args.html_out or f"rda_report_{ds_name}.html"
    badge_out = args.badge_out or f"rda_badge_{ds_name}.svg"

    counts = {v: report.get("summary", {}).get("verdict_counts", {}).get(v, 0)
              for v in VERDICT_ORDER}
    total = report.get("summary", {}).get("total_episodes", 0)
    version = report.get("tool_version", "?")

    Path(html_out).write_text(build_html(report, title, args.report), encoding="utf-8")
    Path(badge_out).write_text(make_badge(counts, total, ds_name, version), encoding="utf-8")

    print(f"HTML  -> {html_out}")
    print(f"badge -> {badge_out}")
    print(f"README snippet:\n![RDA audit]({Path(badge_out).name})")


if __name__ == "__main__":
    main()
