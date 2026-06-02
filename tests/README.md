# GSTransformFL — Test Suite

This directory contains three levels of tests for the GSTransformFL package.
Run them in order — each level builds on the previous.

---

## Structure

```
tests/
├── environment/
│   └── setup_test_env.sh        # creates a clean venv from scratch
├── 01_unit/
│   └── test_core_transform.py   # no GPU required — tests package logic only
├── 02_integration/
│   └── NVFlare_test.ipynb       # interactive NVFlare POC (2 simulated clients)
└── 03_e2e/
    ├── prepare_data.py          # downloads PneumoniaMNIST, splits by site
    ├── simple_cnn.py            # minimal CNN definition
    ├── fl_trainer_executor.py   # NVFlare training executor
    ├── fl_trainer_controller.py # NVFlare FedAvg controller
    └── run_e2e_test.py          # orchestrates full pipeline
```

---

## Requirements

- Linux (Ubuntu 20.04+)
- Python 3.10+
- NVIDIA GPU with CUDA 11.x or 12.x
- `nvidia-smi` accessible on PATH

No other dependencies need to be pre-installed — `setup_test_env.sh`
handles everything from scratch.

---

## Step 1 — Create the test environment

Run from the repo root (`GS-NVFlare/`):

```bash
bash tests/environment/setup_test_env.sh
```

This creates a clean virtual environment at
`tests/environment/gs_test_env/` and installs all dependencies
including the correct CuPy variant for your CUDA toolkit version.

Expected output ends with:

```
  ✅  Setup complete.
```

---

## Step 2 — Run unit tests (no GPU required)

```bash
source tests/environment/gs_test_env/bin/activate
python tests/01_unit/test_core_transform.py
```

These tests verify package logic — imports, diagnostic utilities,
policy validation, error reporting — without touching the GPU or NVFlare.
Safe to run on any machine.

Expected output ends with:

```
  ✅  All unit tests passed.
```

---

## Step 3 — Run full end-to-end test

```bash
source tests/environment/gs_test_env/bin/activate
cd tests/03_e2e
python run_e2e_test.py
```

This runs the complete pipeline:

1. Downloads PneumoniaMNIST and splits into 2 federated client datasets
2. Runs GSTransformFL via NVFlare simulator (2 simulated clients)
3. Runs 3 rounds of federated CNN training with FedAvg
4. Evaluates global model on shared test set
5. Prints results summary

Expected output ends with:

```
  ✅  E2E test PASSED.
```

The full pipeline takes approximately 35-40 seconds on an NVIDIA L4.

---

## Reference log

A full terminal log from a clean run on NVIDIA L4 (CUDA 11.8) is
provided at:

```
tests/03_e2e_full_run.log
```

Use this as a reference for expected output at each step.

---

## Notes

**Data** — test data is downloaded automatically and never committed
to the repository. Raw and transformed `.npy` files live only in
`tests/03_e2e/data/` which is in `.gitignore`.

**GPU diagnostic** — on first run the package probes your GPU (~2s)
and saves a diagnostic file to the NVFlare workspace. This is expected
behavior and is part of what the test validates.

**Accuracy** — final test accuracy varies slightly between runs
(typically 0.80-0.87) because the GS transform uses random phase
initialization. This is expected.

**CUDA version** — `setup_test_env.sh` automatically detects your
CUDA toolkit version and installs the correct CuPy variant.
If setup fails at the CuPy step, check your CUDA toolkit installation
with `nvcc --version`.
