# ============================================================
# gs_1ch/executor/policy.py
# DiagnosticPolicy — replaces all input() calls in gs_diagnose()
# Configured once at executor startup, consulted at every
# decision point during the diagnostic lifecycle.
# ============================================================

from dataclasses import dataclass

# ── Valid policy values ───────────────────────────────────────────────────────
_VALID_ON_FIRST_RUN   = {"auto"}          # only one valid option for now
_VALID_ON_GPU_CHANGE  = {"auto"}          # only one valid option for now
_VALID_ON_OOM         = {"abort_job"}     # only one valid option for now


# ── Policy dataclass ──────────────────────────────────────────────────────────

@dataclass
class DiagnosticPolicy:
    """
    Controls diagnostic behavior in non-interactive NVFlare environment.

    Parameters
    ----------
    on_first_run  : str
        What to do when no diagnostic file exists.
        "auto" — run the GPU probe immediately, save results, continue.

    on_gpu_change : str
        What to do when the saved diagnostic was recorded on a different GPU.
        "auto" — re-run the probe silently, update the file, continue.

    on_oom        : str
        What to do when a single image is too large for available VRAM.
        "abort_job" — raise MemoryError with a detailed message so the
                      executor catches it, packages it as a ClientResult
                      error envelope, and the server aborts cleanly.

    verbose       : bool
        Whether to emit info-level log messages during diagnostic steps.
        Errors are always logged regardless of this setting.
    """
    on_first_run  : str  = "auto"
    on_gpu_change : str  = "auto"
    on_oom        : str  = "abort_job"
    verbose       : bool = True

    def __post_init__(self):
        """Validate policy values at construction time, not at runtime."""
        if self.on_first_run not in _VALID_ON_FIRST_RUN:
            raise ValueError(
                f"[gs_1ch] DiagnosticPolicy: invalid on_first_run="
                f"'{self.on_first_run}'. "
                f"Valid options: {_VALID_ON_FIRST_RUN}"
            )
        if self.on_gpu_change not in _VALID_ON_GPU_CHANGE:
            raise ValueError(
                f"[gs_1ch] DiagnosticPolicy: invalid on_gpu_change="
                f"'{self.on_gpu_change}'. "
                f"Valid options: {_VALID_ON_GPU_CHANGE}"
            )
        if self.on_oom not in _VALID_ON_OOM:
            raise ValueError(
                f"[gs_1ch] DiagnosticPolicy: invalid on_oom="
                f"'{self.on_oom}'. "
                f"Valid options: {_VALID_ON_OOM}"
            )

    def to_dict(self) -> dict:
        """Serialize to plain dict for passing into run_diagnostics()."""
        return {
            "on_first_run" : self.on_first_run,
            "on_gpu_change": self.on_gpu_change,
            "on_oom"       : self.on_oom,
            "verbose"      : self.verbose,
        }
    
    
# ── Default policy ────────────────────────────────────────────────────────────
# Used by the executor unless overridden in config_fed_client.json.
# Fully automatic — no human input required at any stage.

DEFAULT_POLICY = DiagnosticPolicy(
    on_first_run  = "auto",
    on_gpu_change = "auto",
    on_oom        = "abort_job",
    verbose       = True,
)