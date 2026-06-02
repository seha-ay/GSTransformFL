# ============================================================
# gs_1ch/executor/gs_executor.py
# NVFlare Executor — wires diagnostic + transform + reporting
# into the federated client lifecycle.
# ============================================================

import os
import time
import numpy as np
from pathlib import Path

# ── NVFlare ───────────────────────────────────────────────────────────────────
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import ReturnCode

# ── gs_1ch ────────────────────────────────────────────────────────────────────
from gs_1ch.core.diagnostic import run_diagnostics, get_gpu_info
from gs_1ch.core.transform import gs_transform
from gs_1ch.executor.policy import DiagnosticPolicy, DEFAULT_POLICY
from gs_1ch.reporting.error_report import (
    make_ok, make_error, make_skipped,
    STATUS_OK, STATUS_ERROR
)

# ── Task name constant ────────────────────────────────────────────────────────
# Must match the task name in config_fed_server.json
TASK_GS_TRANSFORM = "gs_transform"



# ── Executor ──────────────────────────────────────────────────────────────────

class GS1chExecutor(Executor):
    """
    NVFlare Executor for gs_1ch single-channel GS transform.

    Runs once per client before training starts.
    Reads input data from input_path, writes transformed
    data to output_path, returns a structured result envelope
    to the server.

    Parameters (set via config_fed_client.json)
    ----------
    input_path       : str  — path to input .npy file (B, H, W) float32
    output_path      : str  — path where transformed .npy will be saved
    iter_count       : int  — GS iterations (default 50)
    maskP            : float — frequency-domain mask probability (default 0.0)
    auto_chunk       : bool — enable automatic VRAM chunking (default True)
    verbose          : bool — emit progress logs (default True)
    time_budget_warn : int  — seconds above which → warning (default 300)
    time_budget_slow : int  — seconds above which → slow warning (default 1800)
    """

    def __init__(
        self,
        input_path       : str,
        output_path      : str,
        iter_count       : int   = 50,
        maskP            : float = 0.0,
        auto_chunk       : bool  = True,
        verbose          : bool  = True,
        time_budget_warn : int   = 300,
        time_budget_slow : int   = 1800,
    ):
        super().__init__()

        # ── Validate paths at construction time ───────────────────────────────
        self.input_path  = Path(input_path)
        self.output_path = Path(output_path)

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"[gs_1ch] input_path does not exist: {self.input_path}\n"
                f"  This path must be accessible on the client machine "
                f"before the job starts."
            )

        # ── Store transform settings ──────────────────────────────────────────
        self.iter_count       = iter_count
        self.maskP            = maskP
        self.auto_chunk       = auto_chunk
        self.verbose          = verbose
        self.time_budget_warn = time_budget_warn
        self.time_budget_slow = time_budget_slow

        # ── Policy: fully automatic, no human input ───────────────────────────
        self.policy = DEFAULT_POLICY

        # ── Diagnostic cache: populated on first execute() call ───────────────
        self._diag = None
        
        

# ── Helpers ───────────────────────────────────────────────────────────────

    def _get_diag_path(self, fl_ctx: FLContext) -> Path:
        """
        Resolve diagnostic file path from NVFlare workspace.
        Each client gets its own path — never shared across machines.
        """
        workspace = fl_ctx.get_engine().get_workspace()
        site_dir  = Path(workspace.get_site_config_dir())
        return site_dir / "gs_1ch_diagnostic.txt"

    def _get_logger(self, fl_ctx: FLContext):
        """Return NVFlare's logger for this component."""
        return self.logger
    
    
# ── Main task handler ─────────────────────────────────────────────────────

    def execute(
        self,
        task_name : str,
        shareable : Shareable,
        fl_ctx    : FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        Called by NVFlare when server dispatches a task to this client.
        Returns a Shareable containing a ClientResult dict.
        """
        logger    = self._get_logger(fl_ctx)
        client_id = fl_ctx.get_identity_name()

        # ── Unknown task guard ────────────────────────────────────────────────
        if task_name != TASK_GS_TRANSFORM:
            logger.warning(
                f"[gs_1ch] Unknown task received: '{task_name}'. "
                f"Expected '{TASK_GS_TRANSFORM}'. Skipping."
            )
            result   = make_skipped(
                client_id = client_id,
                reason    = f"Unknown task: {task_name}"
            )
            reply    = make_reply(ReturnCode.OK)
            reply["gs_1ch_result"] = result.to_dict()
            return reply

        # ── Abort signal check ────────────────────────────────────────────────
        if abort_signal.triggered:
            result = make_skipped(
                client_id = client_id,
                reason    = "Abort signal received before task started."
            )
            reply  = make_reply(ReturnCode.OK)
            reply["gs_1ch_result"] = result.to_dict()
            return reply
        
        # ── Stage 1: Diagnostics ──────────────────────────────────────────────
        if self._diag is None:
            diag_path = self._get_diag_path(fl_ctx)
            try:
                self._diag = run_diagnostics(
                    diag_path = diag_path,
                    policy    = self.policy.to_dict(),
                    logger    = logger,
                )
            except Exception as e:
                result = make_error(
                    client_id   = client_id,
                    error_stage = "diagnostic",
                    exception   = e,
                )
                reply  = make_reply(ReturnCode.EXECUTION_EXCEPTION)
                reply["gs_1ch_result"] = result.to_dict()
                return reply

        # ── Stage 2: Load input data ──────────────────────────────────────────
        try:
            data = np.load(self.input_path)
            if data.ndim != 3:
                raise ValueError(
                    f"Input data must be 3D (B, H, W), "
                    f"got shape {data.shape}."
                )
            data = data.astype(np.float32)
            input_shape = data.shape
        except Exception as e:
            result = make_error(
                client_id   = client_id,
                error_stage = "io",
                exception   = e,
            )
            reply  = make_reply(ReturnCode.EXECUTION_EXCEPTION)
            reply["gs_1ch_result"] = result.to_dict()
            return reply

        # ── Stage 3: Transform ────────────────────────────────────────────────
        # Check for server-side overrides first.
        # If the server injected iter_count or maskP via the task Shareable,
        # those take precedence over local config_fed_client.json values.
        
        
        iter_count = shareable.get("iter_count", None)
        maskP      = shareable.get("maskP",      None)

        if iter_count is not None:
            logger.info(
                f"[gs_1ch] iter_count overridden by server: {iter_count}"
            )
        else:
            iter_count = self.iter_count
            logger.info(
                f"[gs_1ch] iter_count from local config: {iter_count}"
            )

        if maskP is not None:
            logger.info(
                f"[gs_1ch] maskP overridden by server: {maskP}"
            )
        else:
            maskP = self.maskP
            logger.info(
                f"[gs_1ch] maskP from local config: {maskP}"
            )

        t_start = time.time()
        try:
            out = gs_transform(
                data             = data,
                diag             = self._diag,
                iter_count       = iter_count,
                maskP            = maskP,
                auto_chunk       = self.auto_chunk,
                verbose          = self.verbose,
                time_budget_warn = self.time_budget_warn,
                time_budget_slow = self.time_budget_slow,
                logger           = logger,
            )
        except Exception as e:
            result = make_error(
                client_id   = client_id,
                error_stage = "transform",
                exception   = e,
            )
            reply  = make_reply(ReturnCode.EXECUTION_EXCEPTION)
            reply["gs_1ch_result"] = result.to_dict()
            return reply

        elapsed = time.time() - t_start

        # ── Stage 4: Save output ──────────────────────────────────────────────
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.output_path, out)
        except Exception as e:
            result = make_error(
                client_id   = client_id,
                error_stage = "io",
                exception   = e,
            )
            reply  = make_reply(ReturnCode.EXECUTION_EXCEPTION)
            reply["gs_1ch_result"] = result.to_dict()
            return reply

        # ── Stage 5: Report success ───────────────────────────────────────────
        gpu_info = get_gpu_info()
        B, H, W  = input_shape
        B_chunk  = max(1, int(
            self._diag['safe_vram_bytes'] // (H * W * 64)
        ))
        n_chunks = int(np.ceil(B / B_chunk))

        result = make_ok(
            client_id    = client_id,
            output_path  = str(self.output_path),
            input_shape  = tuple(input_shape),
            output_shape = tuple(out.shape),
            elapsed_sec  = elapsed,
            gpu_name     = gpu_info['gpu_name'],
            n_chunks     = n_chunks,
        )

        if self.verbose:
            logger.info(
                f"[gs_1ch] ✓ Client '{client_id}' complete — "
                f"output saved to {self.output_path}"
            )

        reply = make_reply(ReturnCode.OK)
        reply["gs_1ch_result"] = result.to_dict()
        return reply