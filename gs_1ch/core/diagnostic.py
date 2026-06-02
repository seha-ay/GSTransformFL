# ============================================================
# gs_1ch/core/diagnostic.py
# GPU diagnostic — probe, file I/O, GPU match checking
# Path-injectable: no hardcoded ~/  
# ============================================================

import numpy as np
import cupy as cp
from pathlib import Path
from datetime import datetime

# ── Probe config ──────────────────────────────────────────────────────────────
_PROBE_SHAPE    = (8, 256, 256)
_PROBE_ITER     = 3
_PROBE_REPEATS  = 5

# ── Memory safety fraction ────────────────────────────────────────────────────
_MEM_SAFETY_FRAC    = 0.85

# ── GPU change thresholds ─────────────────────────────────────────────────────
_VRAM_CRITICAL_FRAC = 0.20
_VRAM_MINOR_FRAC    = 0.05


# ── GPU info ──────────────────────────────────────────────────────────────────

def get_gpu_info():
    """Query current GPU properties. Returns dict."""
    device = cp.cuda.Device(0)
    props  = cp.cuda.runtime.getDeviceProperties(device.id)
    try:
        uuid = props.get('uuid', 'unavailable')
        if isinstance(uuid, bytes):
            uuid = uuid.hex()
    except Exception:
        uuid = "unavailable"

    return {
        "gpu_name"        : props['name'].decode().strip(),
        "gpu_uuid"        : str(uuid),
        "total_vram_bytes": int(props['totalGlobalMem']),
        "total_vram_gb"   : props['totalGlobalMem'] / 1024**3,
    }


def safe_vram_bytes(total_vram_bytes):
    return int(total_vram_bytes * _MEM_SAFETY_FRAC)


def estimate_mem_bytes(B, H, W):
    """Peak GPU memory estimate for one gs_1ch call. 64 bytes/element."""
    return B * H * W * 64



# ── Warm-up probe ─────────────────────────────────────────────────────────────

def _flush_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()


def run_probe():
    """
    Time the core GS transform on synthetic data.
    Returns seconds-per-element (float).
    """
    from gs_1ch.core.transform import _gs1ch_core   # local import — avoids circular

    B, H, W = _PROBE_SHAPE
    mock    = np.random.rand(B, H, W).astype(np.float32)

    # Prime CuFFT plan cache — not timed
    _gs1ch_core(cp.asarray(mock), iter_count=1, maskP=0.0)
    _flush_gpu()

    times = []
    for _ in range(_PROBE_REPEATS):
        start = cp.cuda.Event()
        end   = cp.cuda.Event()
        start.record()
        _gs1ch_core(cp.asarray(mock), iter_count=_PROBE_ITER, maskP=0.0)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / 1000.0)
        _flush_gpu()

    median_sec      = float(np.median(times))
    elements        = B * H * W * _PROBE_ITER
    sec_per_element = median_sec / elements
    return sec_per_element


# ── Diagnostic file I/O ───────────────────────────────────────────────────────

def write_diag_file(diag_path: Path, gpu_info: dict, sec_per_element: float):
    """Write diagnostic results to the given path."""
    lines = [
        "# ============================================================",
        "# gs_1ch GPU Diagnostic File",
        "# Generated automatically — do not edit manually.",
        f"# Location: {diag_path}",
        "# ============================================================",
        "",
        f"gpu_name             = {gpu_info['gpu_name']}",
        f"gpu_uuid             = {gpu_info['gpu_uuid']}",
        f"total_vram_gb        = {gpu_info['total_vram_gb']:.2f}",
        f"total_vram_bytes     = {gpu_info['total_vram_bytes']}",
        f"safe_vram_bytes      = {safe_vram_bytes(gpu_info['total_vram_bytes'])}",
        f"probe_sec_per_elem   = {sec_per_element:.12f}",
        f"probe_shape          = {_PROBE_SHAPE}",
        f"probe_iter           = {_PROBE_ITER}",
        f"created_at           = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_diag_file(diag_path: Path):
    """
    Parse diagnostic file into a dict.
    Returns None if file does not exist or is malformed.
    """
    if not diag_path.exists():
        return None
    data = {}
    try:
        with open(diag_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                data[key.strip()] = val.strip()
        data['total_vram_bytes']   = int(data['total_vram_bytes'])
        data['total_vram_gb']      = float(data['total_vram_gb'])
        data['safe_vram_bytes']    = int(data['safe_vram_bytes'])
        data['probe_sec_per_elem'] = float(data['probe_sec_per_elem'])
        return data
    except Exception:
        return None
    
    
    
# ── GPU match check ───────────────────────────────────────────────────────────

def check_gpu_match(diag: dict, gpu_info: dict):
    """
    Compare current GPU against saved diagnostic.

    Returns a dict:
        { "status": "ok" | "minor" | "critical",
          "message": str }
    """
    saved_name = diag.get('gpu_name', '')
    saved_vram = diag.get('total_vram_bytes', 0)
    curr_vram  = gpu_info['total_vram_bytes']
    curr_name  = gpu_info['gpu_name']
    vram_diff  = abs(curr_vram - saved_vram) / max(saved_vram, 1)

    if vram_diff > _VRAM_CRITICAL_FRAC or curr_name != saved_name:
        return {
            "status" : "critical",
            "message": (
                f"GPU mismatch — diagnostic was recorded on "
                f"'{saved_name}' ({diag['total_vram_gb']:.1f} GB), "
                f"current GPU is '{curr_name}' "
                f"({gpu_info['total_vram_gb']:.1f} GB). "
                f"Re-running probe automatically."
            )
        }

    if vram_diff > _VRAM_MINOR_FRAC:
        return {
            "status" : "minor",
            "message": (
                f"Minor GPU VRAM difference detected "
                f"({vram_diff*100:.1f}%). Proceeding with saved diagnostic."
            )
        }

    return {"status": "ok", "message": "GPU matches saved diagnostic."}




# ── Main entry point ──────────────────────────────────────────────────────────

def run_diagnostics(diag_path: Path, policy: dict, logger=None):
    """
    Core diagnostic runner. No interactive prompts.

    Parameters
    ----------
    diag_path   : Path — where to read/write the diagnostic file.
                  Caller (executor) provides this from NVFlare workspace.
    policy      : dict with keys:
                    on_first_run  : "auto"         — run probe, save, continue
                    on_gpu_change : "auto"          — re-run probe, save, continue
                    verbose       : bool
    logger      : optional — NVFlare logger or None (falls back to print)

    Returns
    -------
    diag : dict — parsed diagnostic data
    
    Raises
    ------
    RuntimeError — if probe fails or GPU is completely unavailable
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    verbose = policy.get("verbose", True)
    gpu_info = get_gpu_info()

    # ── Try loading existing file ─────────────────────────────────────────────
    diag = read_diag_file(diag_path)

    if diag is not None:
        match = check_gpu_match(diag, gpu_info)

        if match["status"] == "critical":
            if verbose:
                _log(f"[gs_1ch] {match['message']}")
            diag = None     # will re-run probe below

        elif match["status"] == "minor" and verbose:
            _log(f"[gs_1ch] {match['message']}")

    # ── Run probe if needed ───────────────────────────────────────────────────
    if diag is None:
        if verbose:
            _log(
                f"[gs_1ch] Running GPU probe on {gpu_info['gpu_name']} "
                f"({gpu_info['total_vram_gb']:.1f} GB)..."
            )
        try:
            sec_per_element = run_probe()
        except Exception as e:
            raise RuntimeError(
                f"[gs_1ch] GPU probe failed on {gpu_info['gpu_name']}: {e}"
            ) from e

        write_diag_file(diag_path, gpu_info, sec_per_element)
        diag = read_diag_file(diag_path)

        if verbose:
            _log(
                f"[gs_1ch] Probe complete — "
                f"{sec_per_element:.3e} sec/pixel/iteration. "
                f"Diagnostic saved to {diag_path}"
            )

    if verbose:
        _log(
            f"[gs_1ch] Diagnostic ready — "
            f"GPU: {diag['gpu_name']} ({diag['total_vram_gb']:.1f} GB) | "
            f"safe VRAM: {diag['safe_vram_bytes']/1024**3:.1f} GB | "
            f"probe: {diag['probe_sec_per_elem']:.3e} sec/pixel/iter"
        )

    return diag