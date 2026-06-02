#!/bin/bash
# ============================================================
# site_vars.sh
# Site-specific configuration for GSTransformFL deployment.
#
# INSTRUCTIONS:
#   1. Fill in the variables below for your site.
#   2. Run: source site_vars.sh
#   3. Then start NVFlare or submit the job.
#
# This file is different on every client site.
# Never share this file with other sites.
# ============================================================

# ── Deployment root ───────────────────────────────────────────────────────────
# Resolves automatically to the folder where this script lives.
# Do not change this line.
DEPLOY_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"


# ── Input data ────────────────────────────────────────────────────────────────
# Path to your preprocessed image data.
# Must be a .npy file with shape (B, H, W) and dtype float32.
# Values must be normalized to [0, 1] before running GSTransformFL.
# Default points to the data/input/ placeholder folder inside this package.
# Change this to your own path if your data lives elsewhere.
export GS_INPUT_PATH="${DEPLOY_ROOT}/data/input/images.npy"


# ── Output path ───────────────────────────────────────────────────────────────
# Where GSTransformFL will save the transformed images after processing.
# The file will be created automatically — the directory must exist.
# Default points to the data/output/ placeholder folder inside this package.
# Change this to your own path if needed.
export GS_OUTPUT_PATH="${DEPLOY_ROOT}/data/output/images_transformed.npy"


# ── Server override params (server only) ──────────────────────────────────────
# Path to gs_1ch_job_params.json to enforce iter_count and maskP
# across all clients.
# CLIENT SITES : leave this as empty string.
# SERVER ONLY  : set full path to your override file.
#                Example: "${DEPLOY_ROOT}/config/gs_1ch_job_params.json"
export GS_OVERRIDE_PARAMS_PATH=""


# ── NVFlare workspace ─────────────────────────────────────────────────────────
# Path to your NVFlare workspace directory.
# Default creates a workspace folder inside this package.
export GS_NVFLARE_WORKSPACE="${DEPLOY_ROOT}/nvflare_workspace"


# ── Confirm ───────────────────────────────────────────────────────────────────
echo ""
echo "  GSTransformFL environment loaded"
echo "  DEPLOY_ROOT            : ${DEPLOY_ROOT}"
echo "  GS_INPUT_PATH          : ${GS_INPUT_PATH}"
echo "  GS_OUTPUT_PATH         : ${GS_OUTPUT_PATH}"
echo "  GS_OVERRIDE_PARAMS_PATH: ${GS_OVERRIDE_PARAMS_PATH:-not set}"
echo "  GS_NVFLARE_WORKSPACE   : ${GS_NVFLARE_WORKSPACE}"
echo ""