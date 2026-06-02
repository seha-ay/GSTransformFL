# gs_1ch — Gerchberg-Saxton Single-Channel Transform for NVFlare

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
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Server Override Parameters](#server-override-parameters)
8. [GPU Diagnostics](#gpu-diagnostics)
9. [Error Reporting](#error-reporting)
10. [Deploying to Multiple Sites](#deploying-to-multiple-sites)
11. [Testing Locally](#testing-locally)
12. [Known Limitations](#known-limitations)

---

## Overview

In federated learning pipelines, each client site holds its own local data
and never shares raw images with the server or other clients. `gs_1ch`
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
gs_1ch/
├── core/
│   ├── diagnostic.py     # GPU probe, diagnostic file I/O, VRAM calculations
│   └── transform.py      # GS algorithm + public gs_transform() function
│
├── executor/
│   ├── gs_executor.py    # NVFlare Executor — client-side task handler
│   └── policy.py         # DiagnosticPolicy — controls non-interactive behavior
│
├── controller/
│   └── gs_controller.py  # NVFlare ResponseProcessor — server-side result collection
│
└── reporting/
    └── error_report.py   # ClientResult envelopes + consolidated error reporting
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

## Quick Start

### 1. Install dependencies

```bash
pip install nvflare>=2.7.0 cupy-cuda12x numpy tqdm
```

### 2. Prepare your input data

Each client needs a NumPy array saved as a `.npy` file with shape
`(B, H, W)` and dtype `float32`, where:

- `B` = number of images
- `H` = image height in pixels
- `W` = image width in pixels

Values should be normalized to `[0, 1]` before passing to gs_1ch.

```python
import numpy as np
data = your_images.astype(np.float32)   # shape (B, H, W)
np.save('/data/client_images.npy', data)
```

### 3. Configure the client

Edit `app/config/config_fed_client.json` on each client site:

```json
{
  "format_version": 2,
  "task_data_filters": [],
  "task_result_filters": [],
  "executors": [
    {
      "tasks": ["gs_transform"],
      "executor": {
        "path": "gs_1ch.executor.gs_executor.GS1chExecutor",
        "args": {
          "input_path"       : "/data/client_images.npy",
          "output_path"      : "/data/client_images_transformed.npy",
          "iter_count"       : 50,
          "maskP"            : 0.0,
          "auto_chunk"       : true,
          "verbose"          : true,
          "time_budget_warn" : 300,
          "time_budget_slow" : 1800
        }
      }
    }
  ]
}
```

### 4. Run the job

Submit the job from the NVFlare server as you would any other NVFlare job.

---

## Configuration

All parameters below are set in `config_fed_client.json` on each client.
`input_path` and `output_path` are the only values that differ between
clients. All other parameters should be consistent across sites — see
[Server Override Parameters](#server-override-parameters) to enforce this.

| Parameter        | Type    | Default | Description                                              |
|------------------|---------|---------|----------------------------------------------------------|
| `input_path`     | str     | —       | **Required.** Path to input `.npy` file `(B, H, W)` float32 |
| `output_path`    | str     | —       | **Required.** Path where transformed `.npy` will be saved |
| `iter_count`     | int     | 50      | Number of GS iterations. Higher = more refined transform  |
| `maskP`          | float   | 0.0     | Frequency-domain dropout probability. 0.0 = no masking   |
| `auto_chunk`     | bool    | true    | Auto-split batches to fit GPU VRAM. Recommended: true    |
| `verbose`        | bool    | true    | Print progress and timing logs                           |
| `time_budget_warn` | int   | 300     | Seconds above which a time warning is printed            |
| `time_budget_slow` | int   | 1800    | Seconds above which a slow warning is printed            |

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
by placing an override file on the server machine. If this file exists,
its values take precedence over each client's local `config_fed_client.json`.

### How to use

**Step 1** — Create the override file on the server machine:

```json
{
  "iter_count": 50,
  "maskP": 0.1
}
```

Save it anywhere accessible on the server, for example:
```
/home/user/gs_1ch_job_params.json
```

**Step 2** — Set the path in `config_fed_server.json`:

```json
"args": {
  "override_params_path": "/home/user/gs_1ch_job_params.json"
}
```

**Step 3** — Submit the job. The server will inject these values into the
task sent to all clients. Each client logs whether its parameters came from
the server override or its local config:

```
[gs_1ch] iter_count overridden by server: 50
[gs_1ch] maskP overridden by server: 0.1
```

If `override_params_path` is `null` (default), all clients use their own
local config values.

---

## GPU Diagnostics

On first run at each client site, `gs_1ch` automatically:

1. Detects the GPU model and total VRAM
2. Runs a short performance probe (~2 seconds) to measure transform speed
3. Saves results to `gs_1ch_diagnostic.txt` in the NVFlare client workspace

The diagnostic file is saved to:
```
{nvflare_workspace}/{site_name}/local/gs_1ch_diagnostic.txt
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

## Deploying to Multiple Sites

### What to share

Share the following folder with each client site:

```
NVFlare/
├── gs_1ch/                  # the package — share this entire folder
└── app/
    └── config/
        └── config_fed_server.json   # server only — do not share with clients
```

Each client site needs:
- The `gs_1ch/` package folder
- Their own `config_fed_client.json` with their specific `input_path`
  and `output_path`

### Deployment steps

**This package does not handle NVFlare provisioning.** Provisioning —
generating certificates, startup kits, and secure communication channels
between sites — is handled by NVFlare's built-in provisioning tool and
must be completed before deploying this package.

Once provisioning is complete and all sites have their NVFlare startup
kits, follow these steps:

**Step 1 — Copy the package to each client site**

Place the `gs_1ch/` folder inside the client's NVFlare `custom/` directory:

```
{nvflare_client_workspace}/
└── startup/
    └── custom/
        └── gs_1ch/          # copy here
```

**Step 2 — Prepare input data at each site**

Each site prepares their own `.npy` file:

```python
import numpy as np
data = your_images.astype(np.float32)   # shape (B, H, W), range [0, 1]
np.save('/path/to/your/images.npy', data)
```

**Step 3 — Configure each site**

Each site edits their `config_fed_client.json` with their own paths:

```json
"input_path"  : "/path/to/your/images.npy",
"output_path" : "/path/to/save/transformed.npy"
```

**Step 4 — Configure the server**

The server admin sets `config_fed_server.json`. Optionally set
`override_params_path` to enforce consistent `iter_count` and `maskP`
across all sites.

**Step 5 — Submit the job**

Submit from the NVFlare admin console as you would any other job:

```bash
nvflare job submit -j /path/to/app
```

**Step 6 — Monitor**

Check the server log for the consolidated client report. All client
outputs will be at their respective configured `output_path` locations.

---

## Testing Locally

A complete local test using the NVFlare simulator and MNIST data is
provided in:

```
NVFlare_test.ipynb
```

This notebook covers environment validation, test data preparation, core
transform testing outside NVFlare, a full 2-client POC simulation, and
output verification. Run it on the target machine before deploying to
confirm the environment is correctly configured.

---

## Known Limitations

**Single-channel only** — `gs_1ch` processes `(B, H, W)` arrays. RGB or
multi-channel images `(B, C, H, W)` are not supported. Convert to
grayscale before use. A multi-channel version (`gs_3ch` or similar) is
planned but not yet available.

**Data must fit input path** — the executor loads the entire `.npy` file
into system RAM before chunking to GPU. If your dataset is very large
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