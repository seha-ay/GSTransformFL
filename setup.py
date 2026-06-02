# ============================================================
# setup.py
# Makes gs_1ch installable as a proper Python package.
# Run: pip install -e .
# ============================================================

from setuptools import setup, find_packages

setup(
    name             = "gs_1ch",
    version          = "0.1.0",
    description      = "Gerchberg-Saxton single-channel image transform for NVFlare",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = [
        "numpy>=1.23.0",
        "nvflare>=2.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require   = {
        "cuda11": ["cupy-cuda11x>=12.0.0"],
        "cuda12": ["cupy-cuda12x>=12.0.0"],
        "jupyter": ["ipywidgets>=8.0.0"],
    },
    classifiers      = [
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)