# # ============================================================
# # gs_1ch/controller/gs_controller.py
# # NVFlare Controller — server-side task dispatch,
# # result collection, and abort-on-failure logic.
# # ============================================================

# from typing import List

# # ── NVFlare ───────────────────────────────────────────────────────────────────
# from nvflare.apis.controller import Controller
# from nvflare.apis.fl_context import FLContext
# from nvflare.apis.shareable import Shareable, make_reply
# from nvflare.apis.signal import Signal
# from nvflare.apis.fl_constant import ReturnCode
# from nvflare.apis.client import Client
# from nvflare.apis.impl.task import Task, TaskCompletionStatus

# # ── gs_1ch ────────────────────────────────────────────────────────────────────
# from gs_1ch.executor.gs_executor import TASK_GS_TRANSFORM
# from gs_1ch.reporting.error_report import (
#     ClientResult, consolidate, STATUS_ERROR
# )


# # ── Controller ────────────────────────────────────────────────────────────────

# class GS1chController(Controller):
#     """
#     NVFlare Controller for gs_1ch single-channel GS transform.

#     Dispatches the gs_transform task to all participating clients,
#     collects their ClientResult envelopes, and runs consolidation.
#     Aborts the job if any client reports an error.

#     Parameters (set via config_fed_server.json)
#     ----------
#     task_timeout : int — seconds to wait per client before timeout (default 3600)
#     """

#     def __init__(
#         self,
#         task_timeout         : int   = 3600,
#         override_params_path : str   = None,
#     ):
#         super().__init__()
#         self.task_timeout         = task_timeout
#         self.override_params_path = override_params_path
#         self._results: List[ClientResult] = []
        
        
        
#     def start_controller(self, fl_ctx: FLContext):
#             """Called once when the server starts the job."""
#             logger = self.get_logger()
#             logger.info(
#                 "[gs_1ch] Controller started — "
#                 "waiting for clients to connect before dispatching."
#             )
#             self._results = []

#     def stop_controller(self, fl_ctx: FLContext):
#         """Called once when the job ends (success or abort)."""
#         logger = self.get_logger()
#         logger.info("[gs_1ch] Controller stopped.")
        
        
#     def handle_event(self, event_type: str, fl_ctx: FLContext):
#         """React to NVFlare lifecycle events."""
#         from nvflare.apis.event_type import EventType
#         if event_type == EventType.ROUND_STARTED:
#             logger = self.get_logger()
#             logger.info(
#                 "[gs_1ch] All clients connected — dispatching transform task."
#             )
            
#     def run_controller(self, abort_signal: Signal, fl_ctx: FLContext):
#         """
#         Main job body. Called once by NVFlare after start_controller.
#         Dispatches task, collects results, consolidates, aborts if needed.
#         """
#         logger = self.get_logger()

#         # ── Build task ────────────────────────────────────────────────────────
#         # Base config lives in each client's config_fed_client.json.
#         # If override_params_path is set and the file exists, those values
#         # are injected here and take precedence on all clients.
#         task_data = Shareable()

#         if self.override_params_path is not None:
#             import json
#             from pathlib import Path
#             override_path = Path(self.override_params_path)
#             if override_path.exists():
#                 try:
#                     with open(override_path, "r") as f:
#                         overrides = json.load(f)
#                     for key, val in overrides.items():
#                         task_data[key] = val
#                     logger.info(
#                         f"[gs_1ch] Override params loaded from "
#                         f"{override_path}: {overrides}"
#                     )
#                 except Exception as e:
#                     logger.error(
#                         f"[gs_1ch] Failed to load override params from "
#                         f"{override_path}: {e}. "
#                         f"Clients will use their local config values."
#                     )
#             else:
#                 logger.warning(
#                     f"[gs_1ch] override_params_path is set but file not found: "
#                     f"{override_path}. "
#                     f"Clients will use their local config values."
#                 )

#         task = Task(
#             name      = TASK_GS_TRANSFORM,
#             data      = task_data,
#             timeout   = self.task_timeout,
#         )

#         # ── Dispatch to all clients and wait ──────────────────────────────────
#         logger.info(
#             f"[gs_1ch] Dispatching '{TASK_GS_TRANSFORM}' to all clients "
#             f"(timeout: {self.task_timeout}s)."
#         )
#         self.broadcast_and_wait(
#             task          = task,
#             fl_ctx        = fl_ctx,
#             abort_signal  = abort_signal,
#             min_responses = 1,
#         )

#         # ── Abort signal check ────────────────────────────────────────────────
#         if abort_signal.triggered:
#             logger.warning("[gs_1ch] Job aborted by external signal.")
#             return

#         # ── Collect results from all clients ──────────────────────────────────
#         results = []
#         clients = self._engine.get_clients()

#         for client in clients:
#             client_id = client.name
#             response  = task.get_client_reply(client)

#             # ── Timeout or no response ────────────────────────────────────────
#             if (
#                 response is None
#                 or task.completion_status == TaskCompletionStatus.TIMEOUT
#             ):
#                 from gs_1ch.reporting.error_report import make_error
#                 results.append(make_error(
#                     client_id   = client_id,
#                     error_stage = "timeout",
#                     exception   = TimeoutError(
#                         f"Client '{client_id}' did not respond within "
#                         f"{self.task_timeout}s."
#                     ),
#                 ))
#                 continue

#             # ── Unpack ClientResult envelope ──────────────────────────────────
#             try:
#                 result_dict = response["gs_1ch_result"]
#                 result      = ClientResult.from_dict(result_dict)
#             except Exception as e:
#                 from gs_1ch.reporting.error_report import make_error
#                 results.append(make_error(
#                     client_id   = client_id,
#                     error_stage = "reporting",
#                     exception   = ValueError(
#                         f"Could not unpack result envelope "
#                         f"from '{client_id}': {e}"
#                     ),
#                 ))
#                 continue

#             results.append(result)

#         # ── Consolidate and abort if needed ───────────────────────────────────
#         summary = consolidate(results, logger=logger)

#         if summary["abort"]:
#             logger.error(
#                 "[gs_1ch] Job aborted — see report above for details."
#             )
#             fl_ctx.set_prop(
#                 key        = "gs_1ch_abort",
#                 value      = True,
#                 private    = False,
#                 sticky     = True,
#             )
#             self.system_panic(
#                 reason = (
#                     f"gs_1ch transform failed on "
#                     f"{len(summary['failed'])} client(s): "
#                     f"{summary['failed']}. See server log for details."
#                 ),
#                 fl_ctx = fl_ctx,
#             )
#             return

#         logger.info(
#             f"[gs_1ch] All {len(summary['ok_clients'])} client(s) "
#             f"completed successfully. Job continuing."
#         )







# ============================================================
# gs_1ch/controller/gs_controller.py
# NVFlare 2.7.2 compatible — uses BroadcastAndProcess +
# ResponseProcessor pattern
# ============================================================

import json
from pathlib import Path
from typing import List

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.response_processor import ResponseProcessor
from nvflare.app_common.workflows.broadcast_and_process import BroadcastAndProcess

from gs_1ch.executor.gs_executor import TASK_GS_TRANSFORM
from gs_1ch.reporting.error_report import ClientResult, consolidate


# ── Response Processor ────────────────────────────────────────────────────────

class GS1chResponseProcessor(ResponseProcessor):
    """
    Collects ClientResult envelopes from all clients.
    Called by BroadcastAndProcess after each client responds.
    """

    def __init__(self, override_params_path: str = None):
        super().__init__()
        self.override_params_path = override_params_path
        self._results: List[ClientResult] = []
        self._failed = False

    def create_task_data(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        """
        Build the task Shareable sent to every client.
        Injects override params if override_params_path is set.
        """
        self._results = []
        self._failed  = False

        task_data = Shareable()

        if self.override_params_path is not None:
            override_path = Path(self.override_params_path)
            if override_path.exists():
                try:
                    with open(override_path, "r") as f:
                        overrides = json.load(f)
                    for key, val in overrides.items():
                        task_data[key] = val
                    self.logger.info(
                        f"[gs_1ch] Override params loaded from "
                        f"{override_path}: {overrides}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"[gs_1ch] Failed to load override params: {e}. "
                        f"Clients will use local config values."
                    )
            else:
                self.logger.warning(
                    f"[gs_1ch] override_params_path set but not found: "
                    f"{override_path}. Clients will use local config values."
                )

        return task_data

    def process_client_response(
        self,
        client: Client,
        task_name: str,
        response: Shareable,
        fl_ctx: FLContext,
    ) -> bool:
        """
        Called once per client response. Unpacks ClientResult envelope.
        Returns False if client failed — triggers system_panic in controller.
        """
        client_id = client.name

        try:
            result_dict = response.get("gs_1ch_result")
            if result_dict is None:
                raise ValueError("No gs_1ch_result key in response.")
            result = ClientResult.from_dict(result_dict)
        except Exception as e:
            from gs_1ch.reporting.error_report import make_error
            result = make_error(
                client_id   = client_id,
                error_stage = "reporting",
                exception   = ValueError(
                    f"Could not unpack result from '{client_id}': {e}"
                ),
            )

        self._results.append(result)

        if result.status != "ok":
            self._failed = True
            self.logger.error(
                f"[gs_1ch] Client '{client_id}' failed at stage "
                f"'{result.error_stage}': {result.error_message}"
            )
            return False

        self.logger.info(
            f"[gs_1ch] Client '{client_id}' completed successfully."
        )
        return True

    def final_process(self, fl_ctx: FLContext) -> bool:
        """
        Called after all clients respond. Runs consolidation report.
        Returns False if any client failed — triggers system_panic.
        """
        summary = consolidate(self._results, logger=self.logger)
        return not summary["abort"]


# ── Controller config ─────────────────────────────────────────────────────────

def make_gs1ch_controller(
    task_timeout         : int  = 3600,
    override_params_path : str  = None,
) -> BroadcastAndProcess:
    """
    Factory function — returns a configured BroadcastAndProcess controller
    for the gs_1ch transform job.

    Use this path in config_fed_server.json:
        gs_1ch.controller.gs_controller.make_gs1ch_controller
    """
    return BroadcastAndProcess(
        processor                    = GS1chResponseProcessor(
            override_params_path = override_params_path
        ),
        task_name                    = TASK_GS_TRANSFORM,
        min_responses_required       = 0,
        wait_time_after_min_received = 10,
        timeout                      = task_timeout,
    )