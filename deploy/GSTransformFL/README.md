# GSTransformFL — Gerchberg-Saxton Single-Channel Transform for NVFlare

A federated learning package that applies a Gerchberg-Saxton (GS) optical
transform to single-channel image data at each client site before training
begins. The transform runs once per client, after standard preprocessing
(normalization, resize), and saves the transformed data to disk for the
downstream training job to consume.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Package Structure](#package-structure)
4. [How It Works](#how-it-works)
5. [Setup](#setup)
6. [Quick Start](#quick-start)
7. [Configuration](#configuration)
8. [Server Override Parameters](#server-override-parameters)
9. [GPU Diagnostics](#gpu-diagnostics)
10. [Error Reporting](#error-reporting)
11. [Model Placeholder](#model-placeholder)
12. [Containerization](#containerization)
13. [Deploying to Multiple Sites](#deploying-to-multiple-sites)
14. [Testing Locally](#testing-locally)
15. [Known Limitations](#known-limitations)

---

## Overview

In federated learning pipelines, each client site holds its own local data
and never shares raw images with the server or other clients. `GSTransformFL`
applies the GS optical transform locally at each site before training,
producing a transformed dataset that is saved to a configurable output path.

Key properties:

- **Single-channel only** — designed for grayscale images (B, H, W).
  Multi-channel data (e.g. RGB) is not supported. Convert to grayscale
  before using this package.
- **One-shot** — runs once before training, not on every epoch.
- **Self-configuring** — automatically probes the GPU on first run and
  saves a diagnostic file to the NVFlare client workspace. Subsequent
  runs on the same machine reuse this file without re-probing.
- **Auto-chunking** — automatically splits large batches into VRAM-safe
  chunks. No manual batch size configuration required.
- **Fault-transparent** — if any client fails, the server collects a
  structured error report showing exactly which client failed, at which
  stage, and why. The job is aborted cleanly rather than silently
  producing partial results.

---

## Requirements

| Dependency   | Version     | Notes                                      |
|--------------|-------------|--------------------------------------------|
| Python       | >= 3.10     |                                            |
| NVFlare      | >= 2.7.0    | Tested on 2.7.2                            |
| CuPy         | >= 12.0.0   | Match to your CUDA version (see below)     |
| NumPy        | >= 1.23.0   |                                            |
| tqdm         | >= 4.65.0   | Optional — progress bar                    |
| ipywidgets   | >= 8.0.0    | Optional — Jupyter notebook progress bar   |

**CuPy CUDA variant** — install the variant matching your CUDA version:

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x
```

Check your CUDA version with `nvidia-smi | head -4`.

---

## Package Structure

```
GSTransformFL/
├── config/
│   ├── config_fed_client.json     # client config — paths resolved from site_vars.sh
│   ├── config_fed_server.json     # server config — do not share with clients
│   └── gs_1ch_job_params.json     # optional server override for iter_count/maskP
│
├── custom/
│   └── gs_1ch/                    # the GS transform package (symlink)
│       ├── core/
│       │   ├── diagnostic.py      # GPU probe, VRAM calculations
│       │   └── transform.py       # GS algorithm + public API
│       ├── executor/
│       │   ├── gs_executor.py     # NVFlare Executor — client-side
│       │   └── policy.py          # DiagnosticPolicy
│       ├── controller/
│       │   └── gs_controller.py   # NVFlare ResponseProcessor — server-side
│       └── reporting/
│           └── error_report.py    # ClientResult envelopes + consolidation
│
├── data/
│   ├── input/                     # place your preprocessed .npy file here
│   └── output/                    # transformed .npy will be written here
│
├── model/                         # place your model files here
│   └── MODEL_PLACEHOLDER.md       # instructions for model integration
│
├── scripts/
│   └── preflight_check.py         # run before deployment to validate setup
│
├── site_vars.sh                   # site-specific environment variables
├── Dockerfile.template            # container template — complete before use
└── README.md                      # this file
```

---

## How It Works

```
Server                                    Clients
──────                                    ───────
BroadcastAndProcess                       GS1chExecutor
    │                                         │
    │── dispatch gs_transform task ──────────▶│
    │                                         │── 1. Run GPU diagnostic
    │                                         │── 2. Load input .npy file
    │                                         │── 3. Run GS transform
    │                                         │      (auto-chunked if needed)
    │                                         │── 4. Save output .npy file
    │                                         │── 5. Return ClientResult envelope
    │◀─────────────── ClientResult ───────────│
    │
    │── GS1chResponseProcessor
    │      collect all ClientResults
    │      consolidate report
    │      abort if any client failed
    │
    ▼
 Job continues to training
 (each client reads from output_path)
```

---

## Setup

These steps must be completed on every client site and the server before
running the job.

### Step 1 — Install dependencies

```bash
# Install CuPy for your CUDA version (check with nvidia-smi | head -4)
pip install cupy-cuda12x       # CUDA 12.x
# pip install cupy-cuda11x     # CUDA 11.x

# Install remaining dependencies
pip install nvflare>=2.7.0 numpy tqdm

# Install GSTransformFL as a package
cd GSTransformFL
pip install -e .
```

### Step 2 — Configure site variables

Edit `site_vars.sh` with your site-specific paths, then activate:

```bash
source site_vars.sh
```

The following variables must be set:

| Variable                  | Description                                    |
|---------------------------|------------------------------------------------|
| `GS_INPUT_PATH`           | Path to your input `.npy` file                 |
| `GS_OUTPUT_PATH`          | Path where transformed `.npy` will be saved    |
| `GS_NVFLARE_WORKSPACE`    | Path to your NVFlare workspace directory       |
| `GS_OVERRIDE_PARAMS_PATH` | Server only — path to override params file     |

### Step 3 — Run pre-flight check

```bash
python scripts/preflight_check.py
```

Fix any failures before proceeding. The pre-flight check validates:
- Python and dependency versions
- GPU availability and CUDA functionality
- Environment variables set correctly
- Input file exists with correct shape and dtype
- Output directory exists and is writable
- Sufficient disk space for output
- NVFlare version compatibility

---

## Quick Start

### 1. Prepare your input data

Each client needs a NumPy array saved as a `.npy` file with shape
`(B, H, W)` and dtype `float32`, where:

- `B` = number of images
- `H` = image height in pixels
- `W` = image width in pixels

Values must be normalized to `[0, 1]` before passing to GSTransformFL.

```python
import numpy as np
data = your_images.astype(np.float32)   # shape (B, H, W)
np.save('GSTransformFL/data/input/images.npy', data)
```

### 2. Set environment variables

```bash
source site_vars.sh
```

### 3. Run pre-flight check

```bash
python scripts/preflight_check.py
```

### 4. Submit the job

```bash
nvflare job submit -j /path/to/GSTransformFL
```

---

## Configuration

All parameters are set in `config/config_fed_client.json`. Paths are
resolved from environment variables set in `site_vars.sh` — do not
hardcode absolute paths in the config file.

| Parameter          | Type  | Default | Description                                              |
|--------------------|-------|---------|----------------------------------------------------------|
| `input_path`       | str   | —       | **Required.** Resolved from `{GS_INPUT_PATH}`           |
| `output_path`      | str   | —       | **Required.** Resolved from `{GS_OUTPUT_PATH}`          |
| `iter_count`       | int   | 50      | Number of GS iterations. Higher = more refined transform |
| `maskP`            | float | 0.0     | Frequency-domain dropout probability. 0.0 = no masking  |
| `auto_chunk`       | bool  | true    | Auto-split batches to fit GPU VRAM. Recommended: true   |
| `verbose`          | bool  | true    | Print progress and timing logs                          |
| `time_budget_warn` | int   | 300     | Seconds above which a time warning is printed           |
| `time_budget_slow` | int   | 1800    | Seconds above which a slow warning is printed           |

### Parameter guidance

**`iter_count`** controls transform quality. 50 is a good default for most
use cases. Lower values (10–20) are faster but produce a less refined
transform. Higher values (100+) give diminishing returns past a certain point.

**`maskP`** introduces stochastic frequency-domain dropout during the
transform. At `0.0` the transform is fully deterministic given the same
random seed. Non-zero values introduce controlled randomness that can
improve generalization in some training scenarios. Typical range: `0.0–0.3`.

> ⚠️ **Important:** `iter_count` and `maskP` should be identical across
> all client sites. If clients use different values, their transformed
> datasets are not comparable, which undermines the federated learning
> objective. Use the server override mechanism (below) to enforce
> consistency.

---

## Server Override Parameters

The server can enforce `iter_count` and `maskP` values across all clients
by using the override file. If this file exists and is configured, its
values take precedence over each client's local `config_fed_client.json`.

### How to use

**Step 1** — Edit the override file at `config/gs_1ch_job_params.json`:

```json
{
  "iter_count": 50,
  "maskP": 0.1
}
```

**Step 2** — Set the path in `site_vars.sh` on the server:

```bash
export GS_OVERRIDE_PARAMS_PATH="${DEPLOY_ROOT}/config/gs_1ch_job_params.json"
```

**Step 3** — Source and submit. The server will inject these values into
the task sent to all clients. Each client logs whether its parameters came
from the server override or its local config:

```
[gs_1ch] iter_count overridden by server: 50
[gs_1ch] maskP overridden by server: 0.1
```

If `GS_OVERRIDE_PARAMS_PATH` is empty (default), all clients use their
own local config values.

---

## GPU Diagnostics

On first run at each client site, GSTransformFL automatically:

1. Detects the GPU model and total VRAM
2. Runs a short performance probe (~2 seconds) to measure transform speed
3. Saves results to `gs_1ch_diagnostic.txt` in the NVFlare client workspace

The diagnostic file is saved to:
```
{GS_NVFLARE_WORKSPACE}/{site_name}/local/gs_1ch_diagnostic.txt
```

On subsequent runs on the same machine the file is reused automatically —
no re-probing occurs. If a significantly different GPU is detected (>20%
VRAM difference or different GPU model), the probe re-runs automatically
to recalibrate.

The diagnostic file is human-readable:

```
gpu_name             = NVIDIA L4
total_vram_gb        = 21.95
safe_vram_bytes      = 20034686156
probe_sec_per_elem   = 2.104e-09
created_at           = 2026-06-01 18:00:40
```

> **Note:** Do not copy diagnostic files between client machines. Each
> machine must generate its own file. Copying a file from a different GPU
> will cause incorrect VRAM estimates and potentially cause out-of-memory
> errors.

---

## Error Reporting

If any client fails, the server prints a consolidated report before
aborting the job. Example:

```
  ╔══════════════════════════════════════════════════╗
  ║         gs_1ch — Client Transform Report         ║
  ╚══════════════════════════════════════════════════╝
  Timestamp  : 2026-06-01 19:00:00
  Clients    : 3 total  |  2 ok  |  1 failed  |  0 skipped
  ──────────────────────────────────────────────────
  ✅  Successful clients:
      • site-1  |  NVIDIA L4  |  (500, 28, 28) → (500, 28, 28)  |  1s
      • site-2  |  NVIDIA T4  |  (500, 28, 28) → (500, 28, 28)  |  2s
  ❌  Failed clients:
      • site-3  |  stage: diagnostic  |  RuntimeError: CUDA not available
  ──────────────────────────────────────────────────
  ⛔  Job aborted — one or more clients failed.
      Fix the errors above and re-submit the job.
```

Errors are attributed to one of four stages:

| Stage        | Meaning                                                  |
|--------------|----------------------------------------------------------|
| `diagnostic` | GPU probe failed — likely no GPU or CUDA driver issue    |
| `io`         | Could not read input file or write output file           |
| `transform`  | GS transform failed — likely out-of-memory               |
| `reporting`  | Could not send result back to server                     |

---

## Model Placeholder

GSTransformFL is model-agnostic. It transforms your input data and saves
the result to `data/output/` — it does not train or load any model itself.

Place your model files in the `model/` directory:

```
GSTransformFL/
└── model/
    ├── model.py            # your model architecture
    ├── weights.pt          # pre-trained weights (if applicable)
    └── MODEL_PLACEHOLDER.md
```

The downstream training job is responsible for:
- Loading the model from `model/`
- Reading transformed data from `data/output/`
- Running federated training

See `model/MODEL_PLACEHOLDER.md` for further guidance.

---

## Containerization

A `Dockerfile.template` is provided as a starting point for containerizing
the full federated pipeline. It is intentionally incomplete — it cannot be
finalized until the model and training framework are determined.

### Current state of the template

- Base image: CUDA 12.4 runtime (placeholder — replace with your framework image)
- GSTransformFL package installed
- Data and workspace mount points defined
- Pre-flight check runs on container start
- NVFlare startup command marked as TODO

### To complete the Dockerfile

Open `Dockerfile.template` and fill in all sections marked `# ── TODO`:

1. Replace the base image with your training framework image
2. Add your model and training framework dependencies
3. Add your model directory
4. Replace the CMD with your full NVFlare startup command

### To build and run

```bash
# Build
docker build -t gstransformfl .

# Run with GPU and site-specific environment
docker run --gpus all \
    -v /your/data/input:/data/input \
    -v /your/data/output:/data/output \
    -v /your/nvflare/workspace:/nvflare_workspace \
    --env-file site.env \
    gstransformfl
```

> **Note:** Do not containerize before the model and training framework
> are finalized. The container will need to be rebuilt once those are
> determined. The template is provided so the containerization groundwork
> is in place when that time comes.

---

## Deploying to Multiple Sites

### What to share

Share the entire `GSTransformFL/` folder with each client site. The server
keeps `config/config_fed_server.json` — do not share it with clients.

Each client site needs:
- The full `GSTransformFL/` folder
- Their own data placed in `data/input/`
- `site_vars.sh` filled in with their specific paths

### Deployment steps

**This package does not handle NVFlare provisioning.** Provisioning —
generating certificates, startup kits, and secure communication channels
between sites — is handled by NVFlare's built-in provisioning tool and
must be completed before deploying this package.

Once provisioning is complete and all sites have their NVFlare startup
kits, follow these steps:

**Step 1 — Distribute the package**

Send the `GSTransformFL/` folder to each client site. Each site places it
in their preferred working directory.

**Step 2 — Prepare input data at each site**

Each site places their preprocessed data in `data/input/`:

```python
import numpy as np
data = your_images.astype(np.float32)   # shape (B, H, W), range [0, 1]
np.save('GSTransformFL/data/input/images.npy', data)
```

**Step 3 — Configure each site**

Each site fills in `site_vars.sh` and runs:

```bash
source site_vars.sh
python scripts/preflight_check.py
```

All checks must pass before proceeding.

**Step 4 — Configure the server**

The server admin fills in `site_vars.sh`. Optionally set
`GS_OVERRIDE_PARAMS_PATH` to enforce consistent `iter_count` and `maskP`
across all sites.

**Step 5 — Submit the job**

Submit from the NVFlare admin console:

```bash
nvflare job submit -j /path/to/GSTransformFL
```

**Step 6 — Monitor**

Check the server log for the consolidated client report. All client
outputs will be at their respective `data/output/` locations.

---

## Testing Locally

A complete local test using the NVFlare simulator and MNIST data is
provided in the parent workspace:

```
NVFlare_test.ipynb
```

This notebook covers environment validation, test data preparation, core
transform testing outside NVFlare, a full 2-client POC simulation, and
output verification. Run it on the target machine before deploying to
confirm the environment is correctly configured.

> **Note:** `NVFlare_test.ipynb` is a development and validation tool.
> It is not part of the `GSTransformFL/` deployment package and should
> not be distributed to client sites.

---

## Known Limitations

**Single-channel only** — GSTransformFL processes `(B, H, W)` arrays. RGB
or multi-channel images `(B, C, H, W)` are not supported. Convert to
grayscale before use. A multi-channel version is planned but not yet
available.

**Data must fit in system RAM** — the executor loads the entire `.npy`
file into system RAM before chunking to GPU. If your dataset is very large
(tens of GB), ensure sufficient system RAM is available.

**NVFlare version** — tested on NVFlare 2.7.2. Earlier versions use
different API paths and are not compatible without modification. The
minimum supported version is 2.7.0.

**Non-deterministic output** — the GS transform uses random phase
initialization. Two runs on identical input will produce different outputs.
This is expected behavior. If reproducibility is required, set a fixed
NumPy random seed before calling the transform.

**Diagnostic file not portable** — the `gs_1ch_diagnostic.txt` file is
machine-specific. Do not copy it between sites or between machines with
different GPUs.

**Containerization incomplete** — the provided `Dockerfile.template` is a
starting point only. It must be completed with the model and training
framework details before use in production.
