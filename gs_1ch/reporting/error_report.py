# ============================================================
# gs_1ch/reporting/error_report.py
# Structured result envelopes + multi-client error consolidation
# ============================================================

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_OK      = "ok"
STATUS_ERROR   = "error"
STATUS_SKIPPED = "skipped"   # diagnostic ran but transform was not attempted



# ── Result envelope ───────────────────────────────────────────────────────────

@dataclass
class ClientResult:
    """
    Structured result from one client's gs_1ch execution.
    Returned as a dict inside NVFlare Shareable.
    """
    client_id      : str
    status         : str                        # STATUS_OK / ERROR / SKIPPED
    timestamp      : str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # ── Populated on success ──────────────────────────────────────────────────
    output_path    : Optional[str]  = None      # path where transformed data was saved
    input_shape    : Optional[tuple] = None     # (B, H, W) of input
    output_shape   : Optional[tuple] = None     # (B, H, W) of output
    elapsed_sec    : Optional[float] = None     # wall time for transform
    gpu_name       : Optional[str]  = None      # GPU used
    n_chunks       : Optional[int]  = None      # how many VRAM chunks were used

    # ── Populated on error ────────────────────────────────────────────────────
    error_type     : Optional[str]  = None      # exception class name
    error_message  : Optional[str]  = None      # full error message
    error_stage    : Optional[str]  = None      # "diagnostic" | "transform" | "io"

    def to_dict(self):
        return {
            "client_id"    : self.client_id,
            "status"       : self.status,
            "timestamp"    : self.timestamp,
            "output_path"  : self.output_path,
            "input_shape"  : self.input_shape,
            "output_shape" : self.output_shape,
            "elapsed_sec"  : self.elapsed_sec,
            "gpu_name"     : self.gpu_name,
            "n_chunks"     : self.n_chunks,
            "error_type"   : self.error_type,
            "error_message": self.error_message,
            "error_stage"  : self.error_stage,
        }

    @staticmethod
    def from_dict(d: dict):
        return ClientResult(**d)
    
    
    
# ── Convenience constructors ──────────────────────────────────────────────────

def make_ok(
    client_id   : str,
    output_path : str,
    input_shape : tuple,
    output_shape: tuple,
    elapsed_sec : float,
    gpu_name    : str,
    n_chunks    : int,
) -> ClientResult:
    return ClientResult(
        client_id    = client_id,
        status       = STATUS_OK,
        output_path  = output_path,
        input_shape  = input_shape,
        output_shape = output_shape,
        elapsed_sec  = elapsed_sec,
        gpu_name     = gpu_name,
        n_chunks     = n_chunks,
    )


def make_error(
    client_id   : str,
    error_stage : str,
    exception   : Exception,
) -> ClientResult:
    return ClientResult(
        client_id     = client_id,
        status        = STATUS_ERROR,
        error_type    = type(exception).__name__,
        error_message = str(exception),
        error_stage   = error_stage,
    )


def make_skipped(client_id: str, reason: str) -> ClientResult:
    return ClientResult(
        client_id     = client_id,
        status        = STATUS_SKIPPED,
        error_message = reason,
    )

# ── Server-side consolidation ─────────────────────────────────────────────────

def consolidate(results: list[ClientResult], logger=None) -> dict:
    """
    Consolidate results from all clients into a summary report.

    Parameters
    ----------
    results : list of ClientResult — one per client
    logger  : NVFlare logger or None

    Returns
    -------
    summary : dict with keys:
                abort       : bool — True if any client errored
                ok_clients  : list of client_id strings
                failed      : list of client_id strings
                report      : str — full human-readable report
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    ok_clients  = [r for r in results if r.status == STATUS_OK]
    failed      = [r for r in results if r.status == STATUS_ERROR]
    skipped     = [r for r in results if r.status == STATUS_SKIPPED]
    abort       = len(failed) > 0

    lines = [
        "",
        "  ╔══════════════════════════════════════════════════╗",
        "  ║           GS — Client Transform Report           ║",
        "  ╚══════════════════════════════════════════════════╝",
        f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Clients    : {len(results)} total  |  "
        f"{len(ok_clients)} ok  |  "
        f"{len(failed)} failed  |  "
        f"{len(skipped)} skipped",
        "  " + "─" * 50,
    ]

    # ── OK clients ────────────────────────────────────────────────────────────
    if ok_clients:
        lines.append("  ✅  Successful clients:")
        for r in ok_clients:
            mins, secs = divmod(int(r.elapsed_sec or 0), 60)
            time_str   = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            lines.append(
                f"      • {r.client_id}"
                f"  |  {r.gpu_name}"
                f"  |  {r.input_shape} → {r.output_shape}"
                f"  |  {time_str}"
                f"  |  {r.n_chunks} chunk(s)"
                f"  |  saved → {r.output_path}"
            )

    # ── Skipped clients ───────────────────────────────────────────────────────
    if skipped:
        lines.append("  ⏭   Skipped clients:")
        for r in skipped:
            lines.append(f"      • {r.client_id}  |  {r.error_message}")

    # ── Failed clients ────────────────────────────────────────────────────────
    if failed:
        lines.append("  ❌  Failed clients:")
        for r in failed:
            lines.append(
                f"      • {r.client_id}"
                f"  |  stage: {r.error_stage}"
                f"  |  {r.error_type}: {r.error_message}"
            )
        lines += [
            "  " + "─" * 50,
            "  ⛔  Job aborted — one or more clients failed.",
            "      Fix the errors above and re-submit the job.",
        ]
    else:
        lines += [
            "  " + "─" * 50,
            "  ✓  All clients completed successfully.",
        ]

    lines.append("")
    report = "\n".join(lines)
    _log(report)

    return {
        "abort"      : abort,
        "ok_clients" : [r.client_id for r in ok_clients],
        "failed"     : [r.client_id for r in failed],
        "report"     : report,
    }



