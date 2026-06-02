# ============================================================
# tests/03_e2e/run_e2e_test.py
# End-to-end test orchestrator for GSTransformFL.
#
# Pipeline:
#   Step 1 — Prepare PneumoniaMNIST data (if not already done)
#   Step 2 — Run GSTransformFL NVFlare simulation
#   Step 3 — Run FL training NVFlare simulation (3 rounds)
#   Step 4 — Print final results summary
#
# Usage:
#   python tests/03_e2e/run_e2e_test.py
#
# Requirements:
#   source tests/environment/gs_test_env/bin/activate
# ============================================================

import sys
import os
import json
import time
import shutil
import subprocess
from pathlib import Path

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
DATA_DIR    = SCRIPT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
OUTPUT_DIR  = DATA_DIR / "output"
RESULTS_DIR = SCRIPT_DIR / "results"
WS_GS       = SCRIPT_DIR / "workspace_gs"
WS_TRAIN    = SCRIPT_DIR / "workspace_train"
JOB_GS      = SCRIPT_DIR / "job_gs"
JOB_TRAIN   = SCRIPT_DIR / "job_train"

NVFLARE_BIN = sys.executable.replace("python", "nvflare")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Result tracking ───────────────────────────────────────────────────────────
_steps_passed = []
_steps_failed = []

def step_passed(name):
    _steps_passed.append(name)
    print(f"  ✅  {name}")

def step_failed(name, reason):
    _steps_failed.append(name)
    print(f"  ❌  {name}")
    print(f"      {reason}")

print()
print("  GSTransformFL — End-to-End Test")
print("  " + "═" * 50)
print(f"  Repo root   : {REPO_ROOT}")
print(f"  Data dir    : {DATA_DIR}")
print(f"  Results dir : {RESULTS_DIR}")
print()



# ── Step 1: Data preparation ──────────────────────────────────────────────────
print("  Step 1 — Data preparation")
print("  " + "─" * 50)

site1_raw = RAW_DIR / "site1_train.npy"
site2_raw = RAW_DIR / "site2_train.npy"
test_raw  = RAW_DIR / "test.npy"

if site1_raw.exists() and site2_raw.exists() and test_raw.exists():
    print("  ℹ️   Data already prepared — skipping download.")
    step_passed("Data preparation (cached)")
else:
    print("  Downloading and splitting PneumoniaMNIST...")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "prepare_data.py")],
        capture_output = False,
        text           = True,
        env            = {**os.environ,
                          "PYTHONPATH": str(REPO_ROOT)}
    )
    if result.returncode == 0:
        step_passed("Data preparation")
    else:
        step_failed("Data preparation",
                    "prepare_data.py failed — check output above")
        sys.exit(1)

# ── Load split info ───────────────────────────────────────────────────────────
with open(RAW_DIR / "split_info.json") as f:
    split_info = json.load(f)

print()
print(f"  Site 1 : {split_info['site1']['n_images']:,} images")
print(f"  Site 2 : {split_info['site2']['n_images']:,} images")
print(f"  Test   : {split_info['test']['n_images']:,} images")
print()



# ── Step 2: Build and run GS transform job ────────────────────────────────────
print("  Step 2 — GS Transform (GSTransformFL)")
print("  " + "─" * 50)

def build_gs_job():
    """Build NVFlare job folder for GSTransformFL."""
    if JOB_GS.exists():
        shutil.rmtree(JOB_GS)
    JOB_GS.mkdir(parents=True)

    # ── Server config ─────────────────────────────────────────────────────────
    server_config_dir = JOB_GS / "app_server" / "config"
    server_config_dir.mkdir(parents=True)

    server_config = {
        "format_version": 2,
        "task_data_filters"  : [],
        "task_result_filters": [],
        "components": [
            {
                "id"  : "gs_1ch_processor",
                "path": "gs_1ch.controller.gs_controller.GS1chResponseProcessor",
                "args": {"override_params_path": None}
            }
        ],
        "workflows": [
            {
                "id"  : "gs_1ch_workflow",
                "path": "nvflare.app_common.workflows.broadcast_and_process.BroadcastAndProcess",
                "args": {
                    "processor"                   : "gs_1ch_processor",
                    "task_name"                   : "gs_transform",
                    "min_responses_required"      : 0,
                    "wait_time_after_min_received": 10,
                    "timeout"                     : 3600,
                }
            }
        ]
    }

    import json as _json
    with open(server_config_dir / "config_fed_server.json", "w") as f:
        _json.dump(server_config, f, indent=2)

    # ── Per-client configs ────────────────────────────────────────────────────
    client_configs = {
        "site-1": {
            "input_path" : str(RAW_DIR / "site1_train.npy"),
            "output_path": str(OUTPUT_DIR / "site-1" / "images_transformed.npy"),
        },
        "site-2": {
            "input_path" : str(RAW_DIR / "site2_train.npy"),
            "output_path": str(OUTPUT_DIR / "site-2" / "images_transformed.npy"),
        },
    }

    for site_name, paths in client_configs.items():
        client_config_dir = JOB_GS / f"app_{site_name}" / "config"
        client_config_dir.mkdir(parents=True)
        custom_dir = JOB_GS / f"app_{site_name}" / "custom"
        custom_dir.mkdir(parents=True)
        gs_dst = custom_dir / "gs_1ch"
        if not gs_dst.exists():
            shutil.copytree(REPO_ROOT / "gs_1ch", gs_dst)

        client_config = {
            "format_version"     : 2,
            "task_data_filters"  : [],
            "task_result_filters": [],
            "executors": [
                {
                    "tasks": ["gs_transform"],
                    "executor": {
                        "path": "gs_1ch.executor.gs_executor.GS1chExecutor",
                        "args": {
                            "input_path"       : paths["input_path"],
                            "output_path"      : paths["output_path"],
                            "iter_count"       : 50,
                            "maskP"            : 0.0,
                            "auto_chunk"       : True,
                            "verbose"          : True,
                            "time_budget_warn" : 300,
                            "time_budget_slow" : 1800,
                        }
                    }
                }
            ]
        }
        with open(client_config_dir / "config_fed_client.json", "w") as f:
            _json.dump(client_config, f, indent=2)

    # ── Server custom ─────────────────────────────────────────────────────────
    server_custom = JOB_GS / "app_server" / "custom"
    server_custom.mkdir(parents=True)
    gs_dst = server_custom / "gs_1ch"
    if not gs_dst.exists():
        shutil.copytree(REPO_ROOT / "gs_1ch", gs_dst)

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta = {
        "name"        : "gs_transform",
        "resource_spec": {},
        "deploy_map"  : {
            "app_server": ["server"],
            "app_site-1": ["site-1"],
            "app_site-2": ["site-2"],
        },
        "min_clients" : 2,
    }
    with open(JOB_GS / "meta.json", "w") as f:
        _json.dump(meta, f, indent=2)


# ── Run GS transform ──────────────────────────────────────────────────────────
site1_out = OUTPUT_DIR / "site-1" / "images_transformed.npy"
site2_out = OUTPUT_DIR / "site-2" / "images_transformed.npy"

if site1_out.exists() and site2_out.exists():
    print("  ℹ️   Transformed data already exists — skipping GS transform.")
    step_passed("GS transform (cached)")
else:
    print("  Building GS transform job...")
    build_gs_job()

    if WS_GS.exists():
        shutil.rmtree(WS_GS)

    # ── Pre-run GPU diagnostic before simulator starts ────────────────────────
    # NVFlare simulator heartbeat timeout (~350ms) is shorter than the
    # GPU probe (~2s). We pre-run the diagnostic here so the executor
    # finds an existing diagnostic file and skips the probe entirely.
    print("  Running GPU diagnostic pre-flight...")
    preflight_script = f"""
import sys
sys.path.insert(0, '{REPO_ROOT}')
from pathlib import Path
from gs_1ch.core.diagnostic import run_diagnostics

policy = {{
    'on_first_run' : 'auto',
    'on_gpu_change': 'auto',
    'on_oom'       : 'abort_job',
    'verbose'      : True,
}}

# Run diagnostic for each simulated client workspace
for site in ['site-1', 'site-2']:
    diag_path = Path('{WS_GS}') / site / 'local' / 'gs_1ch_diagnostic.txt'
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag = run_diagnostics(diag_path=diag_path, policy=policy)
    print(f'  Diagnostic ready for {{site}}: {{diag[\"gpu_name\"]}}')
"""
    preflight_result = subprocess.run(
        [sys.executable, "-c", preflight_script],
        capture_output = False,
        text           = True,
        env            = {**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if preflight_result.returncode != 0:
        step_failed("GPU diagnostic pre-flight", "Probe failed")
        sys.exit(1)
    print()

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    cmd = [
        NVFLARE_BIN, "simulator", str(JOB_GS),
        "--workspace", str(WS_GS),
        "--n_clients", "2",
        "--clients",   "site-1,site-2",
        "--threads",   "2",
    ]

    print(f"  Running: {' '.join(cmd)}")
    t_start = time.time()
    result  = subprocess.run(
        cmd, capture_output=True, text=True, env=env
    )
    elapsed = time.time() - t_start

    if result.returncode == 0 and site1_out.exists() and site2_out.exists():
        step_passed(
            f"GS transform — site-1 and site-2 "
            f"({elapsed:.1f}s)"
        )
    else:
        print(result.stdout[-3000:])
        print(result.stderr[-2000:])
        step_failed(
            "GS transform",
            "Simulator failed or output files missing"
        )
        sys.exit(1)

print()



# ── Step 3: FL training ───────────────────────────────────────────────────────
print("  Step 3 — Federated Training (3 rounds FedAvg)")
print("  " + "─" * 50)

def build_train_job():
    """Build NVFlare job folder for FL training."""
    import json as _json

    if JOB_TRAIN.exists():
        shutil.rmtree(JOB_TRAIN)
    JOB_TRAIN.mkdir(parents=True)

    # ── Server config ─────────────────────────────────────────────────────────
    server_config_dir = JOB_TRAIN / "app_server" / "config"
    server_config_dir.mkdir(parents=True)

    server_config = {
        "format_version"     : 2,
        "task_data_filters"  : [],
        "task_result_filters": [],
        "workflows": [
            {
                "id"  : "fl_trainer",
                "path": "fl_trainer_controller.FLTrainerController",
                "args": {
                    "num_rounds"       : 3,
                    "test_images_path" : str(RAW_DIR / "test.npy"),
                    "test_labels_path" : str(RAW_DIR / "test_labels.npy"),
                    "output_dir"       : str(RESULTS_DIR),
                    "num_classes"      : 2,
                    "task_timeout"     : 300,
                }
            }
        ]
    }
    with open(server_config_dir / "config_fed_server.json", "w") as f:
        _json.dump(server_config, f, indent=2)

    # ── Per-client configs ────────────────────────────────────────────────────
    client_configs = {
        "site-1": {
            "images_path": str(OUTPUT_DIR / "site-1" / "images_transformed.npy"),
            "labels_path": str(RAW_DIR / "site1_labels.npy"),
        },
        "site-2": {
            "images_path": str(OUTPUT_DIR / "site-2" / "images_transformed.npy"),
            "labels_path": str(RAW_DIR / "site2_labels.npy"),
        },
    }

    for site_name, paths in client_configs.items():
        client_config_dir = JOB_TRAIN / f"app_{site_name}" / "config"
        client_config_dir.mkdir(parents=True)
        custom_dir = JOB_TRAIN / f"app_{site_name}" / "custom"
        custom_dir.mkdir(parents=True)

        # Copy training scripts to custom dir
        for script in ["simple_cnn.py", "fl_trainer_executor.py"]:
            shutil.copy(SCRIPT_DIR / script, custom_dir / script)

        client_config = {
            "format_version"     : 2,
            "task_data_filters"  : [],
            "task_result_filters": [],
            "executors": [
                {
                    "tasks": ["train"],
                    "executor": {
                        "path": "fl_trainer_executor.FLTrainerExecutor",
                        "args": {
                            "images_path" : paths["images_path"],
                            "labels_path" : paths["labels_path"],
                            "batch_size"  : 32,
                            "local_epochs": 1,
                            "lr"          : 0.001,
                            "num_classes" : 2,
                        }
                    }
                }
            ]
        }
        with open(client_config_dir / "config_fed_client.json", "w") as f:
            _json.dump(client_config, f, indent=2)

    # ── Server custom ─────────────────────────────────────────────────────────
    server_custom = JOB_TRAIN / "app_server" / "custom"
    server_custom.mkdir(parents=True)
    for script in ["simple_cnn.py", "fl_trainer_controller.py"]:
        shutil.copy(SCRIPT_DIR / script, server_custom / script)

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta = {
        "name"        : "fl_training",
        "resource_spec": {},
        "deploy_map"  : {
            "app_server": ["server"],
            "app_site-1": ["site-1"],
            "app_site-2": ["site-2"],
        },
        "min_clients" : 2,
    }
    with open(JOB_TRAIN / "meta.json", "w") as f:
        _json.dump(meta, f, indent=2)


print("  Building FL training job...")
build_train_job()

if WS_TRAIN.exists():
    shutil.rmtree(WS_TRAIN)

env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
cmd = [
    NVFLARE_BIN, "simulator", str(JOB_TRAIN),
    "--workspace", str(WS_TRAIN),
    "--n_clients", "2",
    "--clients",   "site-1,site-2",
    "--threads",   "2",
]

print(f"  Running: {' '.join(cmd)}")
t_start = time.time()
result  = subprocess.run(
    cmd, capture_output=True, text=True, env=env
)
elapsed = time.time() - t_start

if result.returncode == 0:
    step_passed(f"FL training — 3 rounds ({elapsed:.1f}s)")
else:
    print(result.stdout[-3000:])
    print(result.stderr[-2000:])
    step_failed("FL training", "Simulator failed — check output above")
    sys.exit(1)

print()


# ── Step 4: Results summary ───────────────────────────────────────────────────
print("  Step 4 — Results")
print("  " + "─" * 50)

results_path = RESULTS_DIR / "training_results.json"

if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║     GSTransformFL — E2E Test Results             ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Dataset   : PneumoniaMNIST                      ║")
    print(f"  ║  Transform : GS (iter=50, maskP=0.0)             ║")
    print(f"  ║  FL rounds : {results['num_rounds']}                                   ║")
    print(f"  ║  Clients   : 2 simulated sites                   ║")
    print("  ╠══════════════════════════════════════════════════╣")

    for r in results["round_results"]:
        print(
            f"  ║  Round {r['round']}  "
            f"test_acc={r['test_accuracy']:.4f}  "
            f"train_loss={r['avg_train_loss']:.4f}          ║"
        )

    final_acc = results.get("final_test_accuracy", 0.0)
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Final test accuracy : {final_acc:.4f}                     ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    step_passed(
        f"Results saved — final accuracy: {final_acc:.4f}"
    )
else:
    step_failed(
        "Results",
        f"training_results.json not found at {results_path}"
    )

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("  " + "═" * 50)
print(f"  Steps passed : {len(_steps_passed)}")
print(f"  Steps failed : {len(_steps_failed)}")
print()

if _steps_failed:
    print("  ❌  E2E test FAILED.")
    print("      Failed steps:")
    for s in _steps_failed:
        print(f"        • {s}")
    sys.exit(1)
else:
    print("  ✅  E2E test PASSED.")
    print()
    print("  Output files:")
    print(f"    Transformed site-1 : {site1_out}")
    print(f"    Transformed site-2 : {site2_out}")
    print(f"    Global model       : {RESULTS_DIR / 'global_model.pt'}")
    print(f"    Training results   : {RESULTS_DIR / 'training_results.json'}")
    print()
    sys.exit(0)