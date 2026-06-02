# ============================================================
# gs_1ch/core/transform.py
# GS Single-Channel transform — core algorithm + public API
# ============================================================

import numpy as np
import cupy as cp
import sys
import time

# ── Progress bar ──────────────────────────────────────────────────────────────
_TQDM_NOTEBOOK = False
_TQDM_TEXT     = False

try:
    from tqdm.notebook import tqdm as _tqdm_notebook
    from ipywidgets import IntProgress
    _TQDM_NOTEBOOK = True
except ImportError:
    pass

if not _TQDM_NOTEBOOK:
    try:
        from tqdm import tqdm as _tqdm_text
        _TQDM_TEXT = True
    except ImportError:
        pass
    
    

# ── Progress bar helper ───────────────────────────────────────────────────────

def _get_progress_bar(total, desc):
    if _TQDM_NOTEBOOK:
        return _tqdm_notebook(
            total      = total,
            desc       = desc,
            unit       = "chunk",
            bar_format = "{desc}: {percentage:3.0f}%|{bar}| "
                         "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
    if _TQDM_TEXT:
        return _tqdm_text(
            total      = total,
            desc       = desc,
            unit       = "chunk",
            bar_format = "{desc}: {percentage:3.0f}%|{bar}| "
                         "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )

    class _PrimitiveBar:
        def __init__(self, total, desc):
            self.total   = total
            self.desc    = desc
            self.current = 0
            self._width  = 28
            self._t0     = time.time()

        def update(self, n=1):
            self.current += n
            filled  = int(self._width * self.current / self.total)
            bar     = "█" * filled + "░" * (self._width - filled)
            pct     = int(100 * self.current / self.total)
            elapsed = time.time() - self._t0
            line    = (
                f"\r  {self.desc}: {pct:3d}%|{bar}| "
                f"{self.current}/{self.total} "
                f"[{elapsed:.0f}s elapsed]"
            )
            sys.stdout.write(line.ljust(80))
            sys.stdout.flush()
            if self.current >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()

        def close(self):
            pass

    return _PrimitiveBar(total, desc)



# ── GPU memory flush ──────────────────────────────────────────────────────────

from gs_1ch.core.diagnostic import _flush_gpu


# ── Core algorithm (internal) ─────────────────────────────────────────────────

def _gs1ch_core(data_gpu, iter_count, maskP):
    """
    Raw GS single-channel on a CuPy array already on GPU.
    No chunking, no diagnostics.

    data_gpu : cp.ndarray (B, H, W) float32
    Returns  : cp.ndarray (B, H, W) float32 — still on GPU
    """
    B, H, W    = data_gpu.shape
    rand_phase = cp.asarray(
        np.random.uniform(0.0, 2.0 * np.pi, size=(B, H, W)).astype(np.float32)
    )
    Phi        = cp.exp(1j * rand_phase)

    Zb = None
    for _ in range(iter_count):
        Z   = cp.fft.fft2(data_gpu * Phi, norm='ortho')
        Zn  = Z / (cp.abs(Z) + 1e-12)
        if maskP > 0.0:
            mask = (
                cp.random.random(size=(B, H, W)) >= maskP
            ).astype(cp.float32)
            Zn  = Zn * mask
        Zb  = cp.fft.ifft2(Zn, norm='ortho')
        Phi = cp.exp(1j * cp.angle(Zb))

    return cp.abs(Zb).astype(cp.float32)



# ── Public API ────────────────────────────────────────────────────────────────

def gs_transform(
    data,
    diag,
    iter_count       = 50,
    maskP            = 0.0,
    auto_chunk       = True,
    verbose          = True,
    time_budget_warn = 300,
    time_budget_slow = 1800,
    logger           = None,
):
    """
    Gerchberg-Saxton single-channel transform.

    Parameters
    ----------
    data             : np.ndarray (B, H, W) float32
    diag             : dict — from run_diagnostics(). Caller must provide.
    iter_count       : GS iterations (default 50)
    maskP            : frequency-domain mask probability (default 0.0)
    auto_chunk       : split batch to fit VRAM automatically (default True)
    verbose          : print progress and warnings (default True)
    time_budget_warn : seconds above which → Burdensome warning (default 300)
    time_budget_slow : seconds above which → Slow warning (default 1800)
    logger           : NVFlare logger or None (falls back to print)

    Returns
    -------
    out : np.ndarray (B, H, W) float32 on CPU
    """
    from gs_1ch.core.diagnostic import (
        estimate_mem_bytes, _MEM_SAFETY_FRAC
    )

    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    def _fmt(s):
        if s < 60:   return f"{s:.0f}s"
        if s < 3600: return f"{s/60:.1f} min"
        return f"{s/3600:.1f} hr"

    data    = np.asarray(data, dtype=np.float32)
    B, H, W = data.shape

    # ── VRAM check ────────────────────────────────────────────────────────────
    safe_bytes     = diag['safe_vram_bytes']
    single_img_mem = estimate_mem_bytes(1, H, W)

    if single_img_mem > safe_bytes:
        max_pixels  = int(safe_bytes / 64)
        sq_side     = int(np.sqrt(max_pixels))
        sq_side_p2  = int(2 ** np.floor(np.log2(sq_side)))
        raise MemoryError(
            f"[gs_1ch] Image {H}×{W} requires "
            f"{single_img_mem/1024**3:.1f} GB for a single image — "
            f"exceeds safe VRAM limit of {safe_bytes/1024**3:.1f} GB. "
            f"Resize to {sq_side_p2}×{sq_side_p2} or smaller."
        )

    # ── Chunk size from live free memory ─────────────────────────────────────
    free_bytes = cp.cuda.Device(0).mem_info[0]
    usable     = int(free_bytes * _MEM_SAFETY_FRAC)
    B_chunk    = max(1, int(usable // (H * W * 64)))

    # ── Time estimate ─────────────────────────────────────────────────────────
    spe          = diag['probe_sec_per_elem']
    n_chunks     = int(np.ceil(B / B_chunk))
    total_time_s = spe * min(B_chunk, B) * H * W * iter_count * n_chunks

    if verbose:
        if total_time_s >= time_budget_slow:
            _log(
                f"[gs_1ch] 🔴 Estimated time: {_fmt(total_time_s)} for "
                f"{B:,} images at {H}×{W}. Consider fewer images or "
                f"smaller resolution."
            )
        elif total_time_s >= time_budget_warn:
            _log(
                f"[gs_1ch] ⚠️  Estimated time: {_fmt(total_time_s)} for "
                f"{B:,} images at {H}×{W}."
            )

    # ── Single pass ───────────────────────────────────────────────────────────
    if not auto_chunk or B <= B_chunk:
        if verbose and B > B_chunk:
            _log(
                f"[gs_1ch] ⚠️  auto_chunk=False but dataset ({B:,}) exceeds "
                f"safe single-pass limit ({B_chunk:,}). Attempting anyway."
            )
        if verbose:
            _log(
                f"[gs_1ch] Single pass — {B:,} images at {H}×{W} "
                f"fit within VRAM limit ({B_chunk:,} max). "
                f"iter_count={iter_count}, maskP={maskP}"
            )
        t_start  = time.time()
        data_gpu = cp.asarray(data)
        result   = _gs1ch_core(data_gpu, iter_count, maskP)
        out      = result.get().astype(np.float32)
        _flush_gpu()
        elapsed    = time.time() - t_start
        mins, secs = divmod(int(elapsed), 60)
        time_str   = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        if verbose:
            _log(
                f"[gs_1ch] ✓ Done — {out.shape[0]:,} images in {time_str} "
                f"| output shape: {out.shape}"
            )
        return out

    # ── Chunked pass ──────────────────────────────────────────────────────────
    if verbose:
        _log(
            f"[gs_1ch] Splitting {B:,} images into {n_chunks} batches "
            f"of up to {B_chunk:,} — processing automatically."
        )

    out_chunks  = []
    total_start = time.time()
    pbar        = _get_progress_bar(n_chunks, "  gs_1ch") if verbose else None

    for chunk_idx in range(n_chunks):
        start_b = chunk_idx * B_chunk
        end_b   = min(start_b + B_chunk, B)
        chunk   = data[start_b:end_b]

        _flush_gpu()
        data_gpu = cp.asarray(chunk)
        result   = _gs1ch_core(data_gpu, iter_count, maskP)
        out_chunks.append(result.get().astype(np.float32))
        _flush_gpu()

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    out = np.concatenate(out_chunks, axis=0)

    if verbose:
        elapsed       = time.time() - total_start
        mins, secs    = divmod(int(elapsed), 60)
        time_str      = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        _log(
            f"[gs_1ch] ✓ Done — {out.shape[0]:,} images in {time_str} "
            f"| output shape: {out.shape}"
        )

    return out


