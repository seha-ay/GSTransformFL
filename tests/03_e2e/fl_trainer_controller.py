# ============================================================
# tests/03_e2e/fl_trainer_controller.py
# NVFlare Controller for federated CNN training.
#
# Runs N rounds of FedAvg:
#   1. Broadcasts global model weights to all clients
#   2. Collects updated local weights
#   3. Aggregates via weighted FedAvg
#   4. Evaluates global model on shared test set
#   5. Saves final model weights to disk
# ============================================================

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import ReturnCode

# ── Add repo root to path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from simple_cnn import (
    SimpleCNN, NumpyImageDataset,
    evaluate, get_model_weights, set_model_weights, fedavg
)

TASK_TRAIN = "train"



# ── Controller ────────────────────────────────────────────────────────────────

class FLTrainerController(Controller):
    """
    NVFlare Controller for federated CNN training with FedAvg.

    Parameters (set via config_fed_server.json)
    ----------
    num_rounds      : int  — number of FL rounds (default 3)
    test_images_path: str  — path to shared test .npy file
    test_labels_path: str  — path to shared test labels .npy file
    output_dir      : str  — where to save final model and results
    num_classes     : int  — number of output classes (default 2)
    task_timeout    : int  — seconds per round before timeout (default 300)
    """

    def __init__(
        self,
        num_rounds       : int = 3,
        test_images_path : str = "",
        test_labels_path : str = "",
        output_dir       : str = "",
        num_classes      : int = 2,
        task_timeout     : int = 300,
    ):
        super().__init__()
        self.num_rounds        = num_rounds
        self.test_images_path  = Path(test_images_path)
        self.test_labels_path  = Path(test_labels_path)
        self.output_dir        = Path(output_dir)
        self.num_classes       = num_classes
        self.task_timeout      = task_timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Global model ──────────────────────────────────────────────────────
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.global_model = SimpleCNN(
            num_classes=self.num_classes
        ).to(self.device)
        self.round_results = []
        
    def start_controller(self, fl_ctx: FLContext):
        self.logger.info(
            f"[FLTrainer] Controller started — "
            f"{self.num_rounds} rounds, device={self.device}"
        )
        self.round_results = []

    def stop_controller(self, fl_ctx: FLContext):
        self.logger.info("[FLTrainer] Controller stopped.")
        
        
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        """
        Main federated training loop.
        Runs num_rounds of FedAvg then evaluates on test set.
        """
        logger = self.logger

        for round_idx in range(self.num_rounds):
            if abort_signal.triggered:
                logger.warning("[FLTrainer] Abort signal received.")
                return

            logger.info(
                f"[FLTrainer] ── Round {round_idx+1}/{self.num_rounds} ──"
            )

            # ── Build task with current global weights ─────────────────────
            global_weights = get_model_weights(self.global_model)
            task_data      = Shareable()
            task_data["global_weights"] = global_weights
            task_data["current_round"]  = round_idx

            task = Task(
                name    = TASK_TRAIN,
                data    = task_data,
                timeout = self.task_timeout,
                result_received_cb = self._process_client_result,
            )

            # ── Reset round state ─────────────────────────────────────────
            self._round_client_results = []

            # ── Broadcast and wait ────────────────────────────────────────
            self.broadcast_and_wait(
                task         = task,
                fl_ctx       = fl_ctx,
                abort_signal = abort_signal,
                min_responses= 1,
            )

            if abort_signal.triggered:
                return

            # ── FedAvg aggregation ────────────────────────────────────────
            if not self._round_client_results:
                logger.error(
                    f"[FLTrainer] Round {round_idx+1}: "
                    f"no client results received. Aborting."
                )
                self.system_panic(
                    "No client results received.", fl_ctx
                )
                return

            weight_list = [r["updated_weights"]
                           for r in self._round_client_results]
            sample_list = [r["n_samples"]
                           for r in self._round_client_results]
            total_samples = sum(sample_list)

            # ── Weighted FedAvg ───────────────────────────────────────────
            avg_weights = {}
            for key in weight_list[0].keys():
                avg_weights[key] = np.sum(
                    [w[key] * (n / total_samples)
                     for w, n in zip(weight_list, sample_list)],
                    axis=0
                )

            set_model_weights(self.global_model, avg_weights)

            # ── Evaluate on test set ──────────────────────────────────────
            test_acc, test_loss = self._evaluate_global_model()

            round_result = {
                "round"       : round_idx + 1,
                "n_clients"   : len(self._round_client_results),
                "total_samples": total_samples,
                "avg_train_loss": float(np.mean(
                    [r["avg_loss"] for r in self._round_client_results]
                )),
                "test_accuracy": test_acc,
                "test_loss"   : test_loss,
            }
            self.round_results.append(round_result)

            logger.info(
                f"[FLTrainer] Round {round_idx+1} complete — "
                f"test_acc={test_acc:.4f} | "
                f"test_loss={test_loss:.4f} | "
                f"clients={len(self._round_client_results)}"
            )

        # ── Save final model and results ──────────────────────────────────────
        self._save_results(fl_ctx)

    def _process_client_result(
        self, client_task, fl_ctx: FLContext
    ):
        """Called by NVFlare when each client returns results."""
        response  = client_task.result
        client    = client_task.client

        if response is None:
            self.logger.error(
                f"[FLTrainer] No response from {client.name}"
            )
            return

        try:
            result = {
                "client_id"     : response.get("client_id", client.name),
                "updated_weights": response["updated_weights"],
                "n_samples"     : response["n_samples"],
                "avg_loss"      : response["avg_loss"],
            }
            self._round_client_results.append(result)
            self.logger.info(
                f"[FLTrainer] Received results from "
                f"'{result['client_id']}' — "
                f"samples={result['n_samples']} | "
                f"loss={result['avg_loss']:.4f}"
            )
        except Exception as e:
            self.logger.error(
                f"[FLTrainer] Failed to unpack result "
                f"from {client.name}: {e}"
            )
            
            
    def _evaluate_global_model(self):
        """Evaluate global model on shared test set."""
        if not self.test_images_path.exists():
            self.logger.warning(
                "[FLTrainer] Test set not found — skipping evaluation."
            )
            return 0.0, 0.0

        try:
            dataset = NumpyImageDataset(
                str(self.test_images_path),
                str(self.test_labels_path)
            )
            loader = DataLoader(
                dataset, batch_size=64,
                shuffle=False, num_workers=0
            )
            acc, loss = evaluate(
                self.global_model, loader, self.device
            )
            return acc, loss
        except Exception as e:
            self.logger.error(
                f"[FLTrainer] Evaluation failed: {e}"
            )
            return 0.0, 0.0

    def _save_results(self, fl_ctx: FLContext):
        """Save final model weights and round results to disk."""
        logger = self.logger

        # ── Save model weights ────────────────────────────────────────────────
        model_path = self.output_dir / "global_model.pt"
        torch.save(
            self.global_model.state_dict(),
            str(model_path)
        )
        logger.info(f"[FLTrainer] Model saved to {model_path}")

        # ── Save round results ────────────────────────────────────────────────
        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    "num_rounds"   : self.num_rounds,
                    "num_classes"  : self.num_classes,
                    "round_results": self.round_results,
                    "final_test_accuracy": (
                        self.round_results[-1]["test_accuracy"]
                        if self.round_results else None
                    ),
                },
                f, indent=2
            )
        logger.info(
            f"[FLTrainer] Results saved to {results_path}"
        )

        # ── Print summary ─────────────────────────────────────────────────────
        logger.info("")
        logger.info(
            "  ╔══════════════════════════════════════════════════╗"
        )
        logger.info(
            "  ║       GSTransformFL — Training Summary           ║"
        )
        logger.info(
            "  ╠══════════════════════════════════════════════════╣"
        )
        for r in self.round_results:
            logger.info(
                f"  ║  Round {r['round']}  "
                f"test_acc={r['test_accuracy']:.4f}  "
                f"train_loss={r['avg_train_loss']:.4f}  ║"
            )
        logger.info(
            "  ╚══════════════════════════════════════════════════╝"
        )

    def process_result_of_unknown_task(
        self, client, task_name, client_task_id, result, fl_ctx
    ):
        pass