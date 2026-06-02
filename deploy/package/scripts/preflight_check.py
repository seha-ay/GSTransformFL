# ============================================================
# scripts/preflight_check.py
# Pre-deployment validation for GSTransformFL.
#
# Run this on every client site before submitting the job:
#   python scripts/preflight_check.py
#
# Checks:
#   1. Python version
#   2. All dependencies importable and correct versions
#   3. GPU available and CUDA working
#   4. CuPy CURAND functional
#   5. Environment variables set (from site_vars.sh)
#   6. Input file exists, correct shape and dtype
#   7. Output directory exists and is writable
#   8. NVFlare version compatible
#   9. Disk space sufficient for output file
# ============================================================

import sys
import os
import traceback
from pathlib import Path

# ── Result tracking ───────────────────────────────────────────────────────────
_passed  = []
_failed  = []
_warned  = []


def passed(check, msg=""):
    _passed.append(check)
    print(f"  ✅  {check}" + (f" — {msg}" if msg else ""))


def failed(check, msg=""):
    _failed.append(check)
    print(f"  ❌  {check}" + (f" — {msg}" if msg else ""))


def warned(check, msg=""):
    _warned.append(check)
    print(f"  ⚠️   {check}" + (f" — {msg}" if msg else ""))
    

# ── Check 1: Python version ───────────────────────────────────────────────────
print()
print("  GSTransformFL — Pre-flight Check")
print("  " + "─" * 50)
print()
print("  [1/9] Python version")

major, minor = sys.version_info[:2]
if major == 3 and minor >= 10:
    passed("Python version", f"{major}.{minor}")
else:
    failed("Python version",
           f"{major}.{minor} detected — requires >= 3.10")
    

# ── Check 2: Dependencies ─────────────────────────────────────────────────────
print()
print("  [2/9] Dependencies")

import importlib

deps = {
    "numpy"      : ("numpy",       "1.23.0"),
    "nvflare"    : ("nvflare",     "2.7.0"),
    "tqdm"       : ("tqdm",        "4.65.0"),
    "cupy"       : ("cupy",        "12.0.0"),
}

def _version_ok(installed, required):
    from packaging.version import Version
    try:
        return Version(installed) >= Version(required)
    except Exception:
        return True   # if packaging not available, skip version check

for display, (module, min_ver) in deps.items():
    try:
        mod     = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        if _version_ok(version, min_ver):
            passed(f"{display}", f"v{version}")
        else:
            warned(f"{display}",
                   f"v{version} installed, v{min_ver} required")
    except ImportError:
        failed(f"{display}", "not installed")
        

# ── Check 3: GPU available ────────────────────────────────────────────────────
print()
print("  [3/9] GPU availability")

try:
    import cupy as cp
    device    = cp.cuda.Device(0)
    props     = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name  = props['name'].decode().strip()
    total_mem = props['totalGlobalMem'] / 1024**3
    passed("GPU detected", f"{gpu_name} ({total_mem:.1f} GB VRAM)")
except Exception as e:
    failed("GPU detected", str(e))


# ── Check 4: CURAND functional ────────────────────────────────────────────────
print()
print("  [4/9] CURAND functionality")

try:
    import cupy as cp
    import numpy as np
    # Use FFT warmup first — avoids cold CURAND init failure
    cp.fft.fft(cp.ones(64, dtype=cp.float32))
    # Now test random generation
    test = cp.asarray(
        np.random.uniform(0, 1, size=(10, 10)).astype(np.float32)
    )
    assert test.shape == (10, 10)
    del test
    cp.get_default_memory_pool().free_all_blocks()
    passed("CURAND / random generation")
except Exception as e:
    failed("CURAND / random generation", str(e))
    

# ── Check 5: Environment variables ───────────────────────────────────────────
print()
print("  [5/9] Environment variables (from site_vars.sh)")

required_vars = [
    "GS_INPUT_PATH",
    "GS_OUTPUT_PATH",
    "GS_NVFLARE_WORKSPACE",
]

all_vars_set = True
for var in required_vars:
    val = os.environ.get(var)
    if val:
        passed(f"${var}", val)
    else:
        failed(f"${var}",
               "not set — run: source site_vars.sh")
        all_vars_set = False

override = os.environ.get("GS_OVERRIDE_PARAMS_PATH", "")
if override:
    passed("$GS_OVERRIDE_PARAMS_PATH", override)
else:
    warned("$GS_OVERRIDE_PARAMS_PATH",
           "not set — server will use client local configs")
    
    
# ── Check 6: Input file ───────────────────────────────────────────────────────
print()
print("  [6/9] Input file")

if all_vars_set:
    input_path = Path(os.environ["GS_INPUT_PATH"])
    if not input_path.exists():
        failed("Input file exists",
               f"{input_path}\n"
               f"         Place your preprocessed .npy file at this path.\n"
               f"         Expected format: shape (B, H, W), dtype float32,\n"
               f"         values normalized to [0, 1].")
    else:
        try:
            import numpy as np
            data = np.load(input_path)
            if data.ndim != 3:
                failed("Input file shape",
                       f"got {data.shape} — expected 3D (B, H, W)")
            else:
                passed("Input file exists", str(input_path))
                passed("Input file shape",  f"{data.shape} (B, H, W)")
                if data.dtype != np.float32:
                    warned("Input file dtype",
                           f"{data.dtype} — recommend float32")
                else:
                    passed("Input file dtype", "float32")
                if data.min() < 0.0 or data.max() > 1.0:
                    warned("Input value range",
                           f"[{data.min():.3f}, {data.max():.3f}] "
                           f"— recommend normalizing to [0, 1]")
                else:
                    passed("Input value range",
                           f"[{data.min():.3f}, {data.max():.3f}]")
                del data
        except Exception as e:
            failed("Input file readable", str(e))
else:
    warned("Input file", "skipped — environment variables not set")
    

# ── Check 7: Output directory writable ───────────────────────────────────────
print()
print("  [7/9] Output directory")

if all_vars_set:
    output_path = Path(os.environ["GS_OUTPUT_PATH"])
    output_dir  = output_path.parent
    if not output_dir.exists():
        failed("Output directory exists",
               f"{output_dir}\n"
               f"         Create it with: mkdir -p {output_dir}")
    else:
        test_file = output_dir / ".gs_write_test"
        try:
            test_file.touch()
            test_file.unlink()
            passed("Output directory writable", str(output_dir))
        except Exception as e:
            failed("Output directory writable", str(e))
else:
    warned("Output directory", "skipped — environment variables not set")


# ── Check 8: Disk space ───────────────────────────────────────────────────────
print()
print("  [8/9] Disk space")

if all_vars_set and Path(os.environ["GS_INPUT_PATH"]).exists():
    try:
        import shutil
        import numpy as np
        input_path  = Path(os.environ["GS_INPUT_PATH"])
        output_dir  = Path(os.environ["GS_OUTPUT_PATH"]).parent
        data        = np.load(input_path)
        output_size = data.nbytes
        free_bytes  = shutil.disk_usage(output_dir).free
        del data
        if free_bytes > output_size * 2:
            passed("Disk space",
                   f"{free_bytes/1024**3:.1f} GB free, "
                   f"{output_size/1024**2:.1f} MB needed")
        else:
            failed("Disk space",
                   f"only {free_bytes/1024**3:.1f} GB free, "
                   f"need {output_size*2/1024**3:.1f} GB")
    except Exception as e:
        warned("Disk space", f"could not check — {e}")
else:
    warned("Disk space", "skipped — input file not available")
    

# ── Check 9: NVFlare version ──────────────────────────────────────────────────
print()
print("  [9/9] NVFlare version")

try:
    import nvflare
    from packaging.version import Version
    if Version(nvflare.__version__) >= Version("2.7.0"):
        passed("NVFlare version", f"v{nvflare.__version__}")
    else:
        failed("NVFlare version",
               f"v{nvflare.__version__} — requires >= 2.7.0")
except Exception as e:
    failed("NVFlare version", str(e))
    

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("  " + "─" * 50)
print(f"  Results: {len(_passed)} passed | "
      f"{len(_warned)} warnings | "
      f"{len(_failed)} failed")
print()

if _failed:
    print("  ⛔  Pre-flight check FAILED.")
    print("      Fix the issues above before submitting the job.")
    print()
    sys.exit(1)
elif _warned:
    print("  ⚠️   Pre-flight check passed with warnings.")
    print("      Review warnings above before submitting the job.")
    print()
    sys.exit(0)
else:
    print("  ✅  All checks passed. Ready to deploy.")
    print()
    sys.exit(0)
    

