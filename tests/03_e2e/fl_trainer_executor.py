# ============================================================
# tests/03_e2e/fl_trainer_executor.py
# NVFlare Executor for federated CNN training.
#
# Each client:
#   1. Receives global model weights from server
#   2. Trains locally on GS-transformed data for N epochs
#   3. Returns updated weights to server
#
# Works in conjunction with fl_trainer_controller.py
# which handles FedAvg aggregation server-side.
# ============================================================

import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import ReturnCode

# ── Add repo root to path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from simple_cnn import SimpleCNN, NumpyImageDataset, train_one_epoch, get_model_weights, set_model_weights

# ── Task name constant ────────────────────────────────────────────────────────
TASK_TRAIN  = "train"
TASK_ASSESS = "assess"


# ── Executor ──────────────────────────────────────────────────────────────────

class FLTrainerExecutor(Executor):
    """
    NVFlare Executor for local CNN training.

    Receives global model weights from the server each round,
    trains locally on GS-transformed data, returns updated weights.

    Parameters (set via config_fed_client.json)
    ----------
    images_path  : str   — path to GS-transformed .npy file (B, H, W)
    labels_path  : str   — path to labels .npy file
    batch_size   : int   — training batch size (default 32)
    local_epochs : int   — local training epochs per round (default 1)
    lr           : float — learning rate (default 0.001)
    num_classes  : int   — number of output classes (default 2)
    """

    def __init__(
        self,
        images_path  : str,
        labels_path  : str,
        batch_size   : int   = 32,
        local_epochs : int   = 1,
        lr           : float = 0.001,
        num_classes  : int   = 2,
    ):
        super().__init__()

        self.images_path  = Path(images_path)
        self.labels_path  = Path(labels_path)
        self.batch_size   = batch_size
        self.local_epochs = local_epochs
        self.lr           = lr
        self.num_classes  = num_classes

        # ── Validate paths at startup ─────────────────────────────────────────
        if not self.images_path.exists():
            raise FileNotFoundError(
                f"[FLTrainer] images_path not found: {self.images_path}\n"
                f"  Run GSTransformFL first to generate transformed data."
            )
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"[FLTrainer] labels_path not found: {self.labels_path}"
            )

        # ── Device ────────────────────────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Build dataset and loader ──────────────────────────────────────────
        dataset          = NumpyImageDataset(
            str(self.images_path),
            str(self.labels_path)
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle    = True,
            num_workers= 0,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = SimpleCNN(num_classes=self.num_classes).to(self.device)
        
        
    def execute(
        self,
        task_name    : str,
        shareable    : Shareable,
        fl_ctx       : FLContext,
        abort_signal : Signal,
    ) -> Shareable:

        client_id = fl_ctx.get_identity_name()
        logger    = self.logger

        # ── Unknown task guard ────────────────────────────────────────────────
        if task_name != TASK_TRAIN:
            logger.warning(
                f"[FLTrainer] Unknown task: '{task_name}'. "
                f"Expected '{TASK_TRAIN}'."
            )
            return make_reply(ReturnCode.TASK_UNKNOWN)

        # ── Abort signal check ────────────────────────────────────────────────
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # ── Round number ──────────────────────────────────────────────────────
        current_round = shareable.get("current_round", 0)
        logger.info(
            f"[FLTrainer] Client '{client_id}' — "
            f"Round {current_round + 1} starting."
        )

        # ── Load global weights from server ───────────────────────────────────
        global_weights = shareable.get("global_weights", None)
        if global_weights is not None:
            try:
                set_model_weights(self.model, global_weights)
                logger.info(
                    f"[FLTrainer] Global weights loaded from server."
                )
            except Exception as e:
                logger.error(
                    f"[FLTrainer] Failed to load global weights: {e}"
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # ── Local training ────────────────────────────────────────────────────
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr
        )

        t_start = time.time()
        losses  = []

        for epoch in range(self.local_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            loss = train_one_epoch(
                self.model, self.train_loader,
                optimizer, self.device
            )
            losses.append(loss)
            logger.info(
                f"[FLTrainer] Client '{client_id}' — "
                f"Round {current_round+1} Epoch {epoch+1}/{self.local_epochs} "
                f"loss={loss:.4f}"
            )

        elapsed = time.time() - t_start

        # ── Return updated weights ────────────────────────────────────────────
        updated_weights = get_model_weights(self.model)
        n_samples       = len(self.train_loader.dataset)

        reply = make_reply(ReturnCode.OK)
        reply["updated_weights"] = updated_weights
        reply["n_samples"]       = n_samples
        reply["avg_loss"]        = float(np.mean(losses))
        reply["client_id"]       = client_id
        reply["current_round"]   = current_round
        reply["elapsed_sec"]     = elapsed

        logger.info(
            f"[FLTrainer] Client '{client_id}' — "
            f"Round {current_round+1} complete. "
            f"avg_loss={np.mean(losses):.4f} | "
            f"samples={n_samples} | "
            f"time={elapsed:.1f}s"
        )

        return reply
        
        
    