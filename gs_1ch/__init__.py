# ============================================================
# gs_1ch/__init__.py
# Public API — everything a user needs is importable from here
# ============================================================

from gs_1ch.core.transform import gs_transform
from gs_1ch.core.diagnostic import run_diagnostics, get_gpu_info
from gs_1ch.executor.policy import DiagnosticPolicy, DEFAULT_POLICY
from gs_1ch.reporting.error_report import (
    ClientResult,
    make_ok,
    make_error,
    make_skipped,
    consolidate,
)
from gs_1ch.core.preprocessing import (
    validate_and_normalize_images,
    describe_array,
)


__version__ = "0.1.0"
__all__ = [
    "gs_transform",
    "run_diagnostics",
    "get_gpu_info",
    "DiagnosticPolicy",
    "DEFAULT_POLICY",
    "ClientResult",
    "make_ok",
    "make_error",
    "make_skipped",
    "consolidate",
    "validate_and_normalize_images",
    "describe_array",
]