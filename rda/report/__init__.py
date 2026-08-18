"""Report generation for RDA audit results.

Two JSON output formats are provided:

**Engine format** (``generate_json_report``):
  Detailed three-layer metric aggregates with full per-metric detail.
  Used for programmatic consumption.

**MVP Spec v0.2.0 product format**:
  User-facing schema defined in MVP Product Spec §7.3 / §7.4.
  - Episode-level: ``generate_episode_report``
  - Dataset-level: ``generate_dataset_report``
"""

from rda.report.aggregation import aggregate_dataset_metrics
from rda.report.json_report import (
    format_json_report,
    generate_dataset_report,
    generate_episode_report,
    generate_json_report,
    save_json_report,
)
from rda.report.summary import (
    AuditSummary,
    build_summary,
    format_enhanced_summary_text,
    format_summary_text,
)
from rda.report.top_issues import compute_hero_metrics, compute_top_issues, compute_top_observations

__all__ = [
    # Engine-format JSON report
    "generate_json_report",
    "format_json_report",
    "save_json_report",
    # MVP Spec v0.2.0 product-format reports
    "generate_episode_report",
    "generate_dataset_report",
    # Aggregation
    "aggregate_dataset_metrics",
    # Summary / text
    "build_summary",
    "format_summary_text",
    "format_enhanced_summary_text",
    "AuditSummary",
    # Top observations / hero metrics
    "compute_top_issues",
    "compute_top_observations",
    "compute_hero_metrics",
]

# Audit history tracking
from rda.report.audit_history import (
    save_audit_snapshot,
    load_audit_history,
    compute_trend,
    discover_datasets_with_history,
)

__all__ += [
    # Audit history
    "save_audit_snapshot",
    "load_audit_history",
    "compute_trend",
    "discover_datasets_with_history",
]