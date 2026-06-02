#!/bin/bash
# ============================================================
# setup_test_env.sh
# Creates a clean virtual environment for GSTransformFL
# end-to-end testing. No existing environment assumptions.
#
# Usage:
#   bash tests/environment/setup_test_env.sh
#
# What it does:
#   1. Creates a fresh venv at tests/environment/gs_test_env
#   2. Installs all dependencies from scratch
#   3. Installs GSTransformFL package in editable mode
#   4. Prints activation instructions
#
# After running:
#   source tests/environment/gs_test_env/bin/activate
# ============================================================

set -e  # exit on any error

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
VENV_DIR="${SCRIPT_DIR}/gs_test_env"

echo ""
echo "  GSTransformFL — Test Environment Setup"
echo "  ────────────────────────────────────────────────────"
echo "  Repo root : ${REPO_ROOT}"
echo "  Venv path : ${VENV_DIR}"
echo "  ────────────────────────────────────────────────────"
echo ""

# ── Check Python version ──────────────────────────────────────────────────────
PYTHON_BIN=$(which python3)
PYTHON_VERSION=$(${PYTHON_BIN} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(${PYTHON_BIN} -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(${PYTHON_BIN} -c "import sys; print(sys.version_info.minor)")

echo "  [1/6] Checking Python version..."
if [ "${PYTHON_MAJOR}" -lt 3 ] || [ "${PYTHON_MINOR}" -lt 10 ]; then
    echo "  ❌  Python ${PYTHON_VERSION} detected — requires >= 3.10"
    exit 1
fi
echo "  ✅  Python ${PYTHON_VERSION}"
echo ""

# ── Check CUDA ────────────────────────────────────────────────────────────────
echo "  [2/6] Checking CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ❌  nvidia-smi not found — GPU required for GSTransformFL"
    exit 1
fi

# Use nvcc toolkit version if available, fall back to nvidia-smi driver version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
else
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
fi
CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1)
CUDA_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f2)

# On systems where driver CUDA > toolkit CUDA, use toolkit version
# Check for actual cuFFT library to determine correct CuPy variant
if find /usr/local/cuda*/lib64 -name "libcufft.so.11*" 2>/dev/null | grep -q .; then
    CUDA_MAJOR=12
elif find /usr/local/cuda*/targets/*/lib -name "libcufft.so.10*" 2>/dev/null | grep -q .; then
    CUDA_MAJOR=11
fi
echo "  ✅  CUDA toolkit major version: ${CUDA_MAJOR}"
echo "  ✅  CUDA ${CUDA_VERSION} detected"
echo ""

# ── Select CuPy variant ───────────────────────────────────────────────────────
if [ "${CUDA_MAJOR}" -ge 12 ]; then
    CUPY_PACKAGE="cupy-cuda12x"
elif [ "${CUDA_MAJOR}" -ge 11 ]; then
    CUPY_PACKAGE="cupy-cuda11x"
else
    echo "  ❌  CUDA ${CUDA_VERSION} not supported — requires CUDA 11.x or 12.x"
    exit 1
fi
echo "  CuPy variant selected: ${CUPY_PACKAGE}"
echo ""

# ── Create venv ───────────────────────────────────────────────────────────────
echo "  [3/6] Creating virtual environment..."
if [ -d "${VENV_DIR}" ]; then
    echo "  ⚠️   Existing venv found at ${VENV_DIR} — removing..."
    rm -rf "${VENV_DIR}"
fi
${PYTHON_BIN} -m venv "${VENV_DIR}"
echo "  ✅  Virtual environment created"
echo ""

# ── Activate and upgrade pip ──────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
echo "  ✅  pip upgraded"
echo ""

# ── Install dependencies ──────────────────────────────────────────────────────
echo "  [4/6] Installing dependencies..."
pip install --quiet \
    "numpy>=1.23.0" \
    "nvflare>=2.7.0" \
    "tqdm>=4.65.0" \
    medmnist \
    torch \
    torchvision \
    matplotlib \
    scikit-learn \
    packaging

echo "  Installing CuPy (${CUPY_PACKAGE})..."
pip install --quiet ${CUPY_PACKAGE}
echo "  ✅  All dependencies installed"
echo ""

# ── Install GSTransformFL package ─────────────────────────────────────────────
echo "  [5/6] Installing GSTransformFL package..."
pip install --quiet -e "${REPO_ROOT}"
echo "  ✅  GSTransformFL installed in editable mode"
echo ""

# ── Verify installation ───────────────────────────────────────────────────────
echo "  [6/6] Verifying installation..."
python -c "import gs_1ch; print('  ✅  gs_1ch importable')"
python -c "import nvflare; print(f'  ✅  nvflare {nvflare.__version__}')"
python -c "import cupy; print(f'  ✅  cupy {cupy.__version__}')"
python -c "import torch; print(f'  ✅  torch {torch.__version__}')"
python -c "import medmnist; print(f'  ✅  medmnist {medmnist.__version__}')"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "  ────────────────────────────────────────────────────"
echo "  ✅  Setup complete."
echo ""
echo "  To activate the test environment:"
echo "    source tests/environment/gs_test_env/bin/activate"
echo ""
echo "  To run the full E2E test:"
echo "    cd tests/03_e2e"
echo "    python run_e2e_test.py"
echo "  ────────────────────────────────────────────────────"
echo ""