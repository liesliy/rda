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
        "(Not yet available — coming in a future release)"
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
def audit(
    path: Path,
    output: Optional[Path],
    output_format: str,
    platform: Optional[str],
    launch_ui: bool,
    verbose: bool,
) -> None:
    """Audit a LeRobot dataset at the given PATH.

    Runs all RDA metrics against every episode in the dataset and produces
    a verdict (PASS / REVIEW / EXCLUDE) for each episode.

    \b
    Examples:
      rda audit /path/to/lerobot/dataset
      rda audit ./my_dataset --format json --output report.json
      rda audit ./my_dataset --platform so101 -v
    """
    from rda.audit.dataset_audit import DatasetAuditor
    from rda.report.json_report import save_json_report
    from rda.report.summary import build_summary, format_enhanced_summary_text

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
        if platform:
            click.echo(f"Platform: {platform}")

    # --- Load dataset -------------------------------------------------------
    try:
        from rda.io.lerobot_loader import iter_episodes, load_lerobot_dataset
        dataset_info = load_lerobot_dataset(path_str)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo(
            "  Install it with: pip install lerobot",
            err=True,
        )
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
    auditor = DatasetAuditor()
    try:
        episode_iter = iter_episodes(path_str)
        result = auditor.audit_dataset(dataset_info, episode_iter)
    except Exception as e:
        click.echo(f"Error: Audit failed: {e}", err=True)
        sys.exit(1)

    # --- Build and display summary -----------------------------------------
    summary = build_summary(result)

    if output_format.lower() == "text":
        click.echo(format_enhanced_summary_text(result))
    else:
        # JSON summary on stdout
        import json
        from rda.report.json_report import generate_json_report
        click.echo(
            json.dumps(generate_json_report(result), indent=2, default=str)
        )

    # --- Save JSON report ---------------------------------------------------
    if output is None:
        output = path / "rda_report.json"

    try:
        save_json_report(result, output)
        if verbose:
            click.echo(f"\nReport saved to: {output}")
    except OSError as e:
        click.echo(f"Warning: Could not save report: {e}", err=True)

    # --- UI flag ------------------------------------------------------------
    if launch_ui:
        click.echo("")
        click.echo(
            "Streamlit UI is not yet available. "
            "Stay tuned for the web-based dashboard in a future release."
        )

    # --- Exit with non-zero if any EXCLUDE verdicts -------------------------
    if summary.verdict_counts.get("EXCLUDE", 0) > 0:
        sys.exit(2)


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

    click.echo("")
    click.echo("For more help: rda audit --help")


# ---------------------------------------------------------------------------
# recommend subcommand
# ---------------------------------------------------------------------------

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
      rda recommend /path/to/dataset --format json -o rec.json

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

    if verbose:
        click.echo(f"Loading dataset from: {path_str}")
        click.echo(f"Target policy: {target_policy.value}")

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
        text = format_recommendation_text(result)
        click.echo(text)
        if output:
            output.write_text(text)
            click.echo(f"Report saved to: {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the rda console script."""
    cli()


if __name__ == "__main__":
    main()
