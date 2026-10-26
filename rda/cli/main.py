"""RDA CLI entry point.

Provides the `rda` command with subcommands for auditing robot datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from rda import __version__


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------

@click.group(
    help=(
        "Robot Data Audit (RDA) — Quality auditing tool for robot datasets.\n\n"
        "RDA automatically audits robot datasets, identifies suspicious "
        "demonstrations, explains why they are problematic, and helps teams "
        "curate training-ready data.\n\n"
        "Quick start:\n"
        "  rda audit /path/to/lerobot/dataset\n"
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(__version__, "-V", "--version", prog_name="rda")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """RDA command-line interface."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# audit subcommand
# ---------------------------------------------------------------------------

@cli.command(
    "audit",
    short_help="Audit a LeRobot dataset at the given PATH.",
)
@click.argument(
    "path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to save the JSON audit report. Defaults to <path>/rda_report.json.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for the audit report.",
)
@click.option(
    "--platform",
    "platform",
    type=str,
    default=None,
    show_default=False,
    help=(
        "Robot platform name (e.g. 'so101', 'droid'). Used for Tier 3 "
        "platform-specific metrics. If not provided, only Tier 1 universal "
        "metrics are used for ranking."
    ),
)
@click.option(
    "--ui",
    "launch_ui",
    is_flag=True,
    default=False,
    help=(
        "Launch the Streamlit web UI after the audit completes. "
        "Requires the optional UI extras: pip install robot-data-audit[ui]"
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "--blind",
    "blind",
    is_flag=True,
    default=False,
    help=(
        "Anonymize the output report for blind sharing. "
        "Replaces filesystem paths with hashes so the report can be "
        "shared externally without revealing your data location. "
        "No raw trajectory data is included — only aggregated metrics."
    ),
)
@click.option(
    "--video-quality",
    "video_quality",
    is_flag=True,
    default=False,
    help=(
        "Enable visual quality analysis (blur, exposure, contrast). "
        "By default, visual quality checks are disabled (Fast Audit mode). "
        "Use this flag to opt-in for video quality assessment."
    ),
)
@click.option(
    "--no-video",
    "no_video",
    is_flag=True,
    default=False,
    help=(
        "Skip all video-related metrics entirely. "
        "Useful for datasets without video modalities or when video "
        "analysis is not needed."
    ),
)
@click.option(
    "--video-only",
    "video_only",
    is_flag=True,
    default=False,
    help=(
        "Run only video-related metrics (video integrity, visual quality, "
        "video temporal). Skips all non-video metrics."
    ),
)
@click.option(
    "--full",
    "full_audit",
    is_flag=True,
    default=False,
    help=(
        "Run the full audit with all metrics including visual quality. "
        "Equivalent to --video-quality but more explicit."
    ),
)
@click.option(
    "--pii-video",
    "pii_video",
    is_flag=True,
    default=False,
    help=(
        "Extend the privacy scan into video: sample frames per camera and "
        "run face detection (requires opencv-python). Off by default "
        "because it is the only expensive part of pii_scan."
    ),
)
@click.option(
    "--pii-strict",
    "pii_strict",
    is_flag=True,
    default=False,
    help=(
        "Treat PII findings as an acceptance gate: promote pii_scan into "
        "the critical metric list for this run, so episodes with privacy "
        "findings are graded EXCLUDE. Default is WARNING-level "
        "(findings reported, verdict untouched)."
    ),
)
@click.option(
    "--freeze-motion-source",
    "freeze_motion_source",
    type=click.Choice(["action", "state", "auto"], case_sensitive=False),
    default="action",
    show_default=True,
    help=(
        "Motion signal source for video_freeze detection. "
        "action=pure visual (default), "
        "state=cross-validate with state data, "
        "auto=choose automatically."
    ),
)
def audit(
    path: Path,
    output: Optional[Path],
    output_format: str,
    platform: Optional[str],
    launch_ui: bool,
    verbose: bool,
    blind: bool,
    video_quality: bool,
    no_video: bool,
    video_only: bool,
    full_audit: bool,
    freeze_motion_source: str,
    pii_video: bool,
    pii_strict: bool,
) -> None:
    """Audit a LeRobot dataset at the given PATH.

    Runs all RDA metrics against every episode in the dataset and produces
    a verdict (PASS / REVIEW / EXCLUDE) for each episode.

    \b
    Execution Tiers (D-17):
      (default)       Fast Audit — all metrics except visual quality
      --video-quality Include visual quality analysis (blur, exposure)
      --no-video      Skip all video-related metrics
      --video-only    Run only video-related metrics
      --full          Full audit with all metrics including visual quality

    \b
    Examples:
      rda audit /path/to/lerobot/dataset
      rda audit ./my_dataset --format json --output report.json
      rda audit ./my_dataset --platform so101 -v
      rda audit ./my_dataset --blind  # anonymized report for external sharing
      rda audit ./my_dataset --video-quality  # include visual quality checks
      rda audit ./my_dataset --no-video  # skip all video metrics

    \b
    Privacy / PII scan (v0.9.22):
      (default)     pii_scan runs on text surfaces and numeric features;
                    findings are reported as WARNING and do NOT change
                    the episode verdict.
      --pii-video   additionally sample video frames and detect faces
                    (requires opencv-python).
      --pii-strict  promote pii_scan into the critical list for this run,
                    so privacy findings grade EXCLUDE. Use for acceptance
                    gating, not for routine development runs.
      rda audit ./my_dataset --full  # full audit with all metrics

    \b
    Exit codes:
      0  Audited, no EXCLUDE verdicts
      1  Usage / structural error (bad path, unreadable dataset)
      2  At least one EXCLUDE verdict
      3  No effective audit — no episode yielded auditable content
         (no action / observation / video features resolved, or 0 episodes).
         A report with exit 3 must not be read as "data is clean".
    """
    from rda.audit.dataset_audit import DatasetAuditor
    from rda.audit.execution_tier import ExecutionTier, get_tier_config
    from rda.report.json_report import save_json_report
    from rda.report.summary import build_summary, format_enhanced_summary_text

    # --- Validate mutually exclusive flags ---------------------------------
    tier_flags = {
        "--video-quality": video_quality,
        "--no-video": no_video,
        "--video-only": video_only,
        "--full": full_audit,
    }
    active_flags = [name for name, active in tier_flags.items() if active]
    if len(active_flags) > 1:
        click.echo(
            f"Error: Options {', '.join(active_flags)} are mutually exclusive. "
            f"Please specify only one execution tier.",
            err=True,
        )
        sys.exit(1)

    # --- Determine execution tier ------------------------------------------
    if full_audit:
        execution_tier = ExecutionTier.FULL
    elif video_only:
        execution_tier = ExecutionTier.VIDEO_ONLY
    elif no_video:
        execution_tier = ExecutionTier.NO_VIDEO
    elif video_quality:
        execution_tier = ExecutionTier.VIDEO_QUALITY
    else:
        execution_tier = ExecutionTier.FAST

    tier_config = get_tier_config(execution_tier)

    path_str = str(path)

    # --- Validate input path ------------------------------------------------
    if not path.exists():
        click.echo(f"Error: Path does not exist: {path_str}", err=True)
        click.echo(
            "  Tip: Provide the path to a LeRobot dataset directory.",
            err=True,
        )
        click.echo(
            "  Example: rda audit /path/to/lerobot/dataset",
            err=True,
        )
        sys.exit(1)

    if not path.is_dir():
        click.echo(f"Error: Path is not a directory: {path_str}", err=True)
        sys.exit(1)

    if verbose:
        click.echo(f"Loading dataset from: {path_str}")
        click.echo(f"Execution tier: {tier_config['description']}")
        if platform:
            click.echo(f"Platform: {platform}")

    # --- Load dataset -------------------------------------------------------
    try:
        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset
        dataset_info = load_lerobot_dataset(path_str)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        # The loader raises a specific message when pandas/pyarrow are
        # missing (direct-parquet path); only suggest lerobot otherwise.
        if "pandas and pyarrow" not in str(e):
            click.echo("  Install it with: pip install lerobot", err=True)
        sys.exit(1)
    except Exception as e:
        # Catch-all for malformed / unreadable datasets
        click.echo(
            f"Error: Failed to load dataset from '{path_str}': {e}",
            err=True,
        )
        click.echo(
            "  Make sure the path points to a valid LeRobot dataset directory.",
            err=True,
        )
        sys.exit(1)

    if dataset_info.num_episodes == 0:
        click.echo(
            "Error: Dataset contains no episodes. Nothing to audit.",
            err=True,
        )
        sys.exit(1)

    if verbose:
        click.echo(
            f"Dataset: {dataset_info.num_episodes} episodes, "
            f"{dataset_info.total_frames} total frames"
        )
        click.echo(
            f"Modalities: {', '.join(dataset_info.modalities) or '(none)'}"
        )
        click.echo(
            f"Actions: {', '.join(dataset_info.action_keys) or '(none)'}"
        )
        click.echo("")

    # --- Run the audit ------------------------------------------------------
    # Video face scanning is meaningless (and dishonest) under --no-video:
    # the tier declares that no video was read, so the report must not imply
    # frames were inspected.
    scan_video = bool(pii_video) and execution_tier != ExecutionTier.NO_VIDEO
    metric_kwargs = {
        "video_freeze": {"motion_source": freeze_motion_source},
        "pii_scan": {"scan_video": scan_video},
    }
    # v0.9.22: pii_scan is WARNING-level unless explicitly promoted to the
    # acceptance gate. The promotion is per-run so it never silently changes
    # how a report produced without the flag should be read.
    critical_metrics = None
    if pii_strict:
        from rda.audit.rules import critical_with

        critical_metrics = critical_with(["pii_scan"])

    auditor = DatasetAuditor(
        execution_tier=execution_tier,
        metric_kwargs=metric_kwargs,
        critical_metrics=critical_metrics,
    )
    try:
        episode_iter = iter_episodes(path_str)
        result = auditor.audit_dataset(dataset_info, episode_iter)
    except Exception as e:
        click.echo(f"Error: Audit failed: {e}", err=True)
        sys.exit(1)

    # Store execution tier info in result for report generation
    result.execution_tier = execution_tier.value
    result.video_quality_executed = tier_config["video_quality_executed"]

    # --- Build and display summary -----------------------------------------
    summary = build_summary(result)

    # Generate the JSON report dict once (used for both stdout and file)
    import json as _json
    from rda.report.json_report import generate_json_report, anonymize_report

    report_dict = generate_json_report(result)
    if blind:
        report_dict = anonymize_report(report_dict)

    if output_format.lower() == "text":
        text = format_enhanced_summary_text(result)
        if blind:
            # Redact the dataset path in text output
            text = text.replace(
                f"Dataset: {path_str}", "Dataset: [redacted]"
            )
        click.echo(text)
    else:
        click.echo(
            _json.dumps(report_dict, indent=2, default=str)
        )

    # --- REQ-11 (v0.7.1): surface skipped-by-missing-dependency metrics ------
    skipped = report_dict.get("skipped_by_missing_dep") or {}
    if skipped:
        dep_desc = ", ".join(
            f"{n} metric-result(s) skipped (missing '{dep}')"
            for dep, n in sorted(skipped.items())
        )
        click.echo(
            f"Warning: {dep_desc}. Those checks were NOT performed — "
            f"install with: pip install {' '.join(sorted(skipped.keys()))}",
            err=True,
        )

    # --- Save JSON report ---------------------------------------------------
    if output is None:
        if blind:
            output = Path.cwd() / "rda_blind_report.json"
        else:
            output = path / "rda_report.json"

    try:
        # Save the (possibly anonymized) report dict directly
        Path(output).write_text(
            _json.dumps(report_dict, indent=2, default=str),
            encoding="utf-8",
        )
        if verbose or blind:
            click.echo(f"\nReport saved to: {output}")
            if blind:
                click.echo(
                    "  [Blind mode] Path and identifying info redacted. "
                    "This file is safe to share externally."
                )
    except OSError as e:
        click.echo(f"Warning: Could not save report: {e}", err=True)

    # --- UI flag ------------------------------------------------------------
    if launch_ui:
        _launch_streamlit_ui()

    # --- P0-2 (v0.9.21): "no effective audit" must not look like success ---
    # Checked *before* the EXCLUDE branch: when every episode resolved no
    # auditable content they are now EXCLUDEd by the zero-content guard, and
    # exit 2 would mask the far more actionable truth — the audit never ran.
    if not report_dict.get("audit_effective", True):
        click.echo(
            "ERROR: no effective audit — no episode yielded auditable content.\n"
            "  The loader resolved no action / observation / video features.\n"
            "  Likely causes: unsupported schema dialect, field naming the\n"
            "  loader does not recognise, an empty dataset, or videos that\n"
            "  were never downloaded.\n"
            "  This report must NOT be read as 'data is clean'.",
            err=True,
        )
        sys.exit(3)

    # --- Exit with non-zero if any EXCLUDE verdicts -------------------------
    if summary.verdict_counts.get("EXCLUDE", 0) > 0:
        sys.exit(2)


# ---------------------------------------------------------------------------
# ui subcommand
# ---------------------------------------------------------------------------

def _launch_streamlit_ui(
    port: int = 8501,
    headless: bool = False,
) -> None:
    """Launch the bundled Streamlit web UI in a subprocess."""
    import subprocess
    import rda.ui_app as _ui_pkg  # noqa: F401  (existence check)

    try:
        import streamlit  # noqa: F401
    except ImportError:
        click.echo(
            "Error: Streamlit is required for the web UI but not installed.",
            err=True,
        )
        click.echo(
            "  Install it with: pip install robot-data-audit[ui]",
            err=True,
        )
        sys.exit(1)

    app_path = Path(_ui_pkg.__file__).resolve().parent / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--global.developmentMode",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    if headless:
        cmd.extend(["--server.headless", "true"])

    click.echo(f"Launching RDA web UI at http://localhost:{port} ...")
    click.echo("  (Press Ctrl+C to stop)")
    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        click.echo("\nUI stopped.")


@cli.command(
    "ui",
    short_help="Launch the Streamlit web UI.",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=8501,
    show_default=True,
    help="Port for the Streamlit server.",
)
@click.option(
    "--headless",
    is_flag=True,
    default=False,
    help="Run Streamlit in headless mode (don't auto-open a browser).",
)
def ui(port: int, headless: bool) -> None:
    """Launch the RDA Streamlit web UI.

    The web UI supports uploading a LeRobot dataset, running the full
    audit with progress feedback, exploring per-episode results,
    generating optimization recommendations, and exporting reports.

    \b
    Requires the optional UI extras:
      pip install robot-data-audit[ui]

    \b
    Examples:
      rda ui
      rda ui --port 8502 --headless
    """
    _launch_streamlit_ui(port=port, headless=headless)


# ---------------------------------------------------------------------------
# example subcommand
# ---------------------------------------------------------------------------

@cli.command(
    "example",
    short_help="Show example usage and sample dataset paths.",
)
def example() -> None:
    """Show example usage and sample dataset paths.

    Prints the path to bundled example datasets (if available) and
    common usage examples to help you get started quickly.
    """
    import importlib.resources

    click.echo("RDA — Example Usage")
    click.echo("=" * 40)
    click.echo("")

    # Check for bundled example datasets
    example_dir = Path.cwd() / "examples"
    found_examples: list[Path] = []

    if example_dir.is_dir():
        for item in sorted(example_dir.iterdir()):
            if item.is_dir():
                found_examples.append(item)

    if found_examples:
        click.echo("Bundled example datasets:")
        for ex in found_examples:
            click.echo(f"  {ex}")
        click.echo("")
        click.echo("To audit an example dataset:")
        click.echo(f"  rda audit {found_examples[0]}")
    else:
        click.echo("No bundled example datasets found in ./examples/")
        click.echo("")
        click.echo("Example commands:")
        click.echo("  rda audit /path/to/lerobot/dataset")
        click.echo("  rda audit ./my_dataset -v")
        click.echo("  rda audit ./my_dataset --format json --output report.json")
        click.echo("  rda audit ./my_dataset --platform so101")
        click.echo("  rda audit ./my_dataset --video-quality  # include visual quality")
        click.echo("  rda audit ./my_dataset --no-video       # skip video metrics")
        click.echo("  rda audit ./my_dataset --full           # full audit")

    click.echo("")
    click.echo("For more help: rda audit --help")


# ---------------------------------------------------------------------------
# conform subcommand (v0.10.0)
# ---------------------------------------------------------------------------

def _parse_workspace_bounds(text: Optional[str]):
    """Parse ``--workspace-bounds`` into (bounds6, max_distance).

    Accepts either:
      * ``"xmin,xmax,ymin,ymax,zmin,zmax"`` — Cartesian workspace bounds; the
        feasible-space max distance is then the box diagonal;
      * ``"d=<float>"`` — the max distance declared directly.

    Anything else is a usage error: the value feeds a standard threshold, so
    guessing would put a fabricated number into the report.
    """
    import re

    if text is None or not str(text).strip():
        return None, None
    s = str(text).strip()
    m = re.match(r"^(?:d|dist|max_distance)\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)$", s)
    if m:
        return None, float(m.group(1))
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(
            f"--workspace-bounds 需要 6 个数 'xmin,xmax,ymin,ymax,zmin,zmax'，"
            f"或 'd=<最大距离>'；无法解析：{text!r}"
        ) from None
    if len(vals) != 6:
        raise ValueError(
            f"--workspace-bounds 需要 6 个数 'xmin,xmax,ymin,ymax,zmin,zmax'，"
            f"实际给了 {len(vals)} 个：{text!r}"
        )
    return vals, None


@cli.command(
    "conform",
    short_help="Conformance assessment against GB/Z 218.1-2026.",
)
@click.argument(
    "dataset",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--standard",
    "standard",
    type=click.Choice(["gbz218.1-2026"], case_sensitive=False),
    default="gbz218.1-2026",
    show_default=True,
    help="Standard to assess against. Only GB/Z 218.1-2026 ships in v0.10.0.",
)
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "JSON profile overriding individual thresholds. Keys are the same "
        "names as in rda/conform/standards/gbz218_1.py; omitted entries "
        "keep the standard's own value."
    ),
)
@click.option(
    "--workspace-bounds",
    "workspace_bounds",
    type=str,
    default=None,
    help=(
        "Feasible-space declaration required by 6.2.3.4 a. Either "
        "'xmin,xmax,ymin,ymax,zmin,zmax' or 'd=<max distance>'. "
        "Without it the Hausdorff check stays UNDETERMINED."
    ),
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["md", "json"], case_sensitive=False),
    default="md",
    show_default=True,
    help="Output format of the conformance report.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Directory to write the report into. Defaults to the current directory.",
)
@click.option(
    "--strict-coverage",
    "strict_coverage",
    is_flag=True,
    default=False,
    help=(
        "Request the coverage gate. NOTE: the gate is a stub in v0.10.0 — "
        "it records the request and the coverage number but does not gate "
        "anything yet, so it never triggers exit code 3."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
def conform(
    dataset: Path,
    standard: str,
    profile_path: Optional[Path],
    workspace_bounds: Optional[str],
    output_format: str,
    out_dir: Optional[Path],
    strict_coverage: bool,
    verbose: bool,
) -> None:
    """Assess a dataset against GB/Z 218.1-2026 and issue four-state verdicts.

    Unlike 'rda audit' (which asks "is this data any good?"), 'rda conform'
    asks "does this data meet the standard?" — every judgement is tied to a
    numbered clause and a binding ("应" mandatory / "宜" advisory), and the
    result is one of four states:

    \b
      PASS            meets the threshold
      FAIL            violates the threshold (both 应 and 宜)
      NOT_APPLICABLE  clause does not apply / standard asks for output only
      UNDETERMINED    inputs missing — never silently reported as PASS

    \b
    Examples:
      rda conform /path/to/lerobot/dataset
      rda conform ./ds --format json --out ./reports
      rda conform ./ds --workspace-bounds "-0.5,0.5,-0.5,0.5,0.0,0.6"
      rda conform ./ds --profile my_thresholds.json

    \b
    Exit codes:
      0  No FAIL at all (PASS / NOT_APPLICABLE / UNDETERMINED only)
      1  At least one FAIL. 应-class violations are counted as
         mandatory_fail, 宜-class violations as advisory_fail; both exit 1,
         so a FAIL can never be mistaken for a clean run.
      2  Execution error (bad path, unreadable dataset, bad profile)
      3  Coverage gate tripped — STUB in v0.10.0, never returned yet
    """
    from rda.conform import (
        ConformOptions,
        build_conformance_node,
        load_profile,
        render_conformance_markdown,
        run_conformance,
    )

    path_str = str(dataset)

    if not dataset.exists():
        click.echo(f"Error: Path does not exist: {path_str}", err=True)
        sys.exit(2)
    if not dataset.is_dir():
        click.echo(f"Error: Path is not a directory: {path_str}", err=True)
        sys.exit(2)

    # --- profile -----------------------------------------------------------
    try:
        profile = load_profile(profile_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except ValueError as e:
        click.echo(f"Error: invalid profile: {e}", err=True)
        sys.exit(2)

    # --- workspace bounds --------------------------------------------------
    try:
        bounds, max_distance = _parse_workspace_bounds(workspace_bounds)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    options = ConformOptions(
        workspace_bounds=bounds,
        workspace_max_distance=max_distance,
        strict_coverage=strict_coverage,
    )

    if verbose:
        click.echo(f"Dataset : {path_str}")
        click.echo(f"Standard: {standard}")
        click.echo(f"Profile : {profile.source or 'built-in defaults'}")
        if max_distance is not None:
            click.echo(f"Feasible-space max distance: {max_distance}")
        click.echo("")

    # --- run ---------------------------------------------------------------
    try:
        result = run_conformance(path_str, profile=profile, options=options)
    except Exception as e:
        click.echo(f"Error: conformance assessment failed: {e}", err=True)
        sys.exit(2)

    # --- render ------------------------------------------------------------
    if output_format.lower() == "json":
        import json as _json

        payload = build_conformance_node(result)
        text = _json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        filename = f"conformance_report_{standard}.json"
    else:
        text = render_conformance_markdown(result)
        filename = f"conformance_report_{standard}.md"

    click.echo(text)

    target_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / filename
        out_file.write_text(text, encoding="utf-8")
        click.echo(f"\nReport saved to: {out_file}")
    except OSError as e:
        click.echo(f"Warning: could not write report: {e}", err=True)

    # --- gates -------------------------------------------------------------
    summary = result.summary
    if summary["undetermined_metrics"]:
        tail = ""
        if summary["fail"] == 0:
            # 最容易踩的坑：没有 FAIL → 退出码 0，但报告里躺着若干"没查"的项。
            # 退出码 0 被 CI 读成"干净通过"，而报告其实不足以宣称符合，
            # 必须把这句话说在明面上。
            tail = (
                " —— 本次无 FAIL，退出码为 0；但存在未裁定项，本份报告"
                "不足以宣称该数据集「符合」标准（UNDETERMINED 不是 PASS）。"
            )
        click.echo(
            "Note: UNDETERMINED (not judged, NOT a pass): "
            + ", ".join(summary["undetermined_metrics"]) + tail,
            err=True,
        )
    if strict_coverage:
        click.echo(
            "Note: --strict-coverage requested, but the coverage gate is a "
            "stub in v0.10.0 (TODO: define the threshold and gate semantics). "
            f"coverage={summary['coverage']} recorded, no gate applied.",
            err=True,
        )
    if summary["mandatory_fail"] > 0:
        click.echo(
            f"FAIL: {summary['mandatory_fail']} mandatory (应) clause(s) violated.",
            err=True,
        )
        sys.exit(1)
    if summary["fail"] > 0:
        click.echo(
            f"FAIL: {summary['advisory_fail']} advisory (宜) clause(s) violated.",
            err=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# recommend subcommand
# ---------------------------------------------------------------------------

def _extract_audit_signals_from_report(report_dict: dict) -> dict:
    """Extract audit signals from a pre-computed audit report JSON.

    Reads per-episode metric measurements from the report and computes
    compact summaries for smoothness, calibration and coverage signals.
    These are sent to the recommendation API so the server-side engine
    can factor them into recommendation rules.
    """
    signals: dict = {}

    episodes = report_dict.get("episodes", {})
    if not episodes:
        return signals

    spike_counts: list[int] = []
    jerk_p95_values: list[float] = []
    sync_p95_values: list[float] = []
    occupancy_values: list[float] = []

    for _ep_key, ep_data in episodes.items():
        measurements = ep_data.get("measurements", {})

        # action_discontinuity → smoothness
        disc = measurements.get("action_discontinuity", {})
        if disc:
            sc = disc.get("spike_count", 0)
            spike_counts.append(sc)
            jerk = disc.get("jerk_p95")
            if jerk is not None:
                jerk_p95_values.append(float(jerk))

        # sensor_synchronization → calibration
        sync = measurements.get("sensor_synchronization", {})
        if sync:
            offset = sync.get("worst_p95_offset_ms")
            if offset is not None:
                sync_p95_values.append(float(offset))

        # coverage → coverage_summary
        cov = measurements.get("coverage", {})
        if cov:
            occ = cov.get("occupancy_rate")
            if occ is not None:
                occupancy_values.append(float(occ))

    if spike_counts:
        signals["smoothness_summary"] = {
            "total_spikes": sum(spike_counts),
            "episodes_with_spikes": sum(1 for s in spike_counts if s > 0),
            "total_episodes": len(spike_counts),
            "jerk_p95_median": (
                sorted(jerk_p95_values)[len(jerk_p95_values) // 2]
                if jerk_p95_values else None
            ),
        }

    if sync_p95_values:
        sorted_sync = sorted(sync_p95_values)
        n = len(sorted_sync)
        signals["calibration_summary"] = {
            "median_worst_p95_offset_ms": sorted_sync[n // 2],
            "max_worst_p95_offset_ms": sorted_sync[-1],
            "total_episodes": n,
        }

    if occupancy_values:
        sorted_occ = sorted(occupancy_values)
        n = len(sorted_occ)
        signals["coverage_summary"] = {
            "median_occupancy": sorted_occ[n // 2],
            "min_occupancy": sorted_occ[0],
            "max_occupancy": sorted_occ[-1],
            "total_episodes": n,
        }

    return signals


@cli.command(
    "recommend",
    short_help="Generate data optimization recommendations.",
)
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--policy",
    type=click.Choice(["frame-wise", "temporal"], case_sensitive=False),
    default="frame-wise",
    show_default=True,
    help=(
        "Target model architecture type. "
        "frame-wise = MLP / BC (per-frame policies). "
        "temporal = ACT / Diffusion Policy / Transformer policies."
    ),
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for the recommendation.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to save the recommendation JSON report.",
)
@click.option(
    "--lang",
    type=click.Choice(["zh", "en"], case_sensitive=False),
    default="zh",
    show_default=True,
    help="Language of the recommendation text (zh / en).",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help=(
        "Never contact the recommendation API; grade with the built-in "
        "conservative local rules instead (rules_version=offline-fallback). "
        "For air-gapped machines and strict private deployments."
    ),
)
@click.option(
    "--policy-chunk-size",
    type=click.IntRange(2, 200),
    default=None,
    help=(
        "Action chunk size of the target policy (REQ-3, DROID-aligned). "
        "Valid-window and tail-trim evaluation is aligned to this length "
        "instead of the legacy fixed tiers. E.g. 100 for ACT, 16 for "
        "Diffusion Policy."
    ),
)
@click.option(
    "--audit-report",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to a pre-computed RDA audit report JSON (from 'rda audit'). "
        "When provided, diagnostic audit signals (smoothness, calibration, "
        "coverage) are extracted and sent to the recommendation API for "
        "more informed suggestions (v0.9)."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
def recommend(
    path: Path,
    policy: str,
    output_format: str,
    output: Optional[Path],
    lang: str,
    offline: bool,
    policy_chunk_size: Optional[int],
    audit_report: Optional[Path],
    verbose: bool,
) -> None:
    """Generate data optimization recommendations for the dataset at PATH.

    \b
    Analyzes the dataset's temporal structure (idle ratios, active run
    distribution, valid window ratios) and produces conservative, evidence-
    graded optimization suggestions based on your intended model architecture.

    \b
    IMPORTANT: RDA is a data quality diagnosis + low-risk optimization
    suggestion tool. It is NOT an automatic optimizer that guarantees
    success rate improvement. Always validate on a held-out set.

    \b
    The recommend command computes temporal metrics locally and sends
    only aggregated statistics (<1KB) to the RDA API for rule evaluation.
    No raw episode data is uploaded. Results are cached locally for
    offline reuse.

    \b
    Policy types:
      frame-wise   MLP, BC, and other per-frame policies
      temporal     ACT, Diffusion Policy, Transformer policies

    \b
    Examples:
      rda recommend /path/to/dataset --policy frame-wise
      rda recommend /path/to/dataset --policy temporal
      rda recommend /path/to/dataset --lang en --format json -o rec.json
      rda recommend /path/to/dataset --offline   (never contacts the API)
      rda recommend /path/to/dataset --audit-report rda_report.json  (v0.9: send audit signals)

    \b
    Environment:
      RDA_API_URL  Override the API endpoint (for private deployment)
    """
    from rda.recommend.types import TargetPolicy
    from rda.recommend.api_client import run_recommendation
    from rda.recommend.formatter import format_recommendation_text
    from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset

    path_str = str(path)
    target_policy = TargetPolicy.from_cli_name(policy)
    lang = lang.lower()

    if verbose:
        click.echo(f"Loading dataset from: {path_str}")
        click.echo(f"Target policy: {target_policy.value}")
        click.echo(f"Language: {lang}")

    # --- Load dataset ---
    try:
        dataset_info = load_lerobot_dataset(path_str)
    except Exception as e:
        click.echo(f"Error: Failed to load dataset: {e}", err=True)
        sys.exit(1)

    if dataset_info.num_episodes == 0:
        click.echo("Error: Dataset contains no episodes.", err=True)
        sys.exit(1)

    if verbose:
        click.echo(
            f"Dataset: {dataset_info.num_episodes} episodes, "
            f"{dataset_info.total_frames} total frames"
        )
        click.echo("")

    # --- Load audit signals from pre-computed report (v0.9) ---
    audit_signals = None
    if audit_report is not None:
        import json as _json
        try:
            report_data = _json.loads(Path(audit_report).read_text(encoding="utf-8"))
            audit_signals = _extract_audit_signals_from_report(report_data)
            if verbose and audit_signals:
                signal_names = ", ".join(audit_signals.keys())
                click.echo(f"Audit signals extracted: {signal_names}")
            elif verbose:
                click.echo("Audit report loaded but no diagnostic signals found.")
        except Exception as e:
            click.echo(
                f"Warning: Failed to load audit report '{audit_report}': {e}. "
                "Proceeding without audit signals.",
                err=True,
            )

    # --- Run recommendation engine ---
    try:
        episode_iter = iter_episodes(path_str)
        result = run_recommendation(
            episode_iter,
            target_policy=target_policy,
            total_episodes=dataset_info.num_episodes,
            total_frames=dataset_info.total_frames,
            progress_callback=(
                lambda step, total, msg: click.echo(
                    f"  Analyzing {step}/{total}: {msg}"
                ) if verbose else None
            ),
            lang=lang,
            offline=offline,
            policy_chunk_size=policy_chunk_size,
            audit_signals=audit_signals,
        )
    except Exception as e:
        click.echo(f"Error: Recommendation failed: {e}", err=True)
        sys.exit(1)

    # --- Output ---
    if output_format.lower() == "json":
        import json
        report = result.to_dict()
        if output:
            output.write_text(json.dumps(report, indent=2, default=str))
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(json.dumps(report, indent=2, default=str))
    else:
        text = format_recommendation_text(result, lang=lang)
        click.echo(text)
        if output:
            output.write_text(text)
            click.echo(f"Report saved to: {output}")


