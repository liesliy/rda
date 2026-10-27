"""Standard threshold tables bundled with RDA conform.

Currently ships GB/Z 218.1-2026 only. A new standard is added by dropping
a module here that exposes ``STANDARD_ID`` and ``all_thresholds()``.
"""
from __future__ import annotations

from rda.conform.standards.gbz218_1 import (  # noqa: F401
    APPENDIX_A_RECOMMENDED,
    BINDING_ADVISORY,
    BINDING_INFORMATIONAL,
    BINDING_MANDATORY,
    STANDARD_ID,
    STANDARD_TITLE,
    CORE_FIELDS,
    FRAME_LOSS_FACTOR,
    Threshold,
    all_thresholds,
)

__all__ = [
    "APPENDIX_A_RECOMMENDED",
    "BINDING_ADVISORY",
    "BINDING_INFORMATIONAL",
    "BINDING_MANDATORY",
    "STANDARD_ID",
    "STANDARD_TITLE",
    "CORE_FIELDS",
    "FRAME_LOSS_FACTOR",
    "Threshold",
    "all_thresholds",
]
