# ============================================================
# gs_1ch/core/preprocessing.py
# Input validation and normalization utilities.
#
# Ensures data is in the correct format (N, H, W) float32
# before passing to gs_transform(). Rejects ambiguous inputs
# with clear error messages rather than silently producing
# wrong results.
# ============================================================

import numpy as np

# ── Supported input shapes ────────────────────────────────────────────────────
# These are the only shapes we will attempt to convert.
# Anything else is rejected with a clear error message.
#
# (N, H, W)       — already correct, no conversion needed
# (N, H, W, 1)    — single channel last, safe to squeeze
# (N, 1, H, W)    — single channel first, safe to squeeze
# (H, W)          — single image, safe to add batch dim
# (1, H, W)       — single image channel first, safe to convert
# (H, W, 1)       — single image channel last, safe to convert

_SUPPORTED_NDIM = {2, 3, 4}

# ── Value range ───────────────────────────────────────────────────────────────
_WARN_RANGE_MAX = 1.0    # values above this trigger a warning
_WARN_RANGE_MIN = 0.0    # values below this trigger a warning



# ── Shape inference ───────────────────────────────────────────────────────────

def _infer_shape(arr: np.ndarray) -> tuple:
    """
    Infer the semantic meaning of each dimension.
    Returns (n, h, w) as integer indices into arr.shape,
    or raises ValueError with a clear message if the shape
    is ambiguous or unsupported.

    Rules:
      ndim=2 : (H, W)         → add batch dim → (1, H, W)
      ndim=3 : (N, H, W)      → correct as-is
      ndim=4 : (N, H, W, 1)   → squeeze last dim
               (N, 1, H, W)   → squeeze dim 1
               (N, H, W, C>1) → rejected — multi-channel
               (N, C>1, H, W) → rejected — multi-channel
    """
    shape = arr.shape
    ndim  = arr.ndim

    # ── 2D: single image (H, W) ───────────────────────────────────────────────
    if ndim == 2:
        return "single_image_hw"

    # ── 3D ────────────────────────────────────────────────────────────────────
    if ndim == 3:
        n, d1, d2 = shape
        # Could be (N, H, W) or (H, W, C) for a single image
        # We assume batch-first convention: (N, H, W)
        # If N==1 and d1==d2 it's unambiguous
        # If all dims equal it's ambiguous — require explicit handling
        if d1 == d2 == n:
            raise ValueError(
                f"[gs_1ch] Ambiguous shape {shape} — all dimensions are equal. "
                f"Cannot determine which axis is batch, height, or width.\n"
                f"  Please reshape explicitly to (N, H, W) before calling "
                f"validate_and_normalize_images()."
            )
        return "nhw"

    # ── 4D ────────────────────────────────────────────────────────────────────
    if ndim == 4:
        n, d1, d2, d3 = shape

        # (N, H, W, 1) — channel last, single channel
        if d3 == 1:
            return "nhwc_single"

        # (N, 1, H, W) — channel first, single channel
        if d1 == 1:
            return "nchw_single"

        # (N, H, W, C>1) — channel last, multi-channel
        if d3 > 1 and d3 <= 4:
            raise ValueError(
                f"[gs_1ch] Multi-channel input detected — shape {shape} "
                f"appears to be (N, H, W, C={d3}).\n"
                f"  gs_1ch processes single-channel (grayscale) images only.\n"
                f"  Convert to grayscale before calling "
                f"validate_and_normalize_images().\n"
                f"  Example: images = images.mean(axis=-1)"
            )

        # (N, C>1, H, W) — channel first, multi-channel
        if d1 > 1 and d1 <= 4:
            raise ValueError(
                f"[gs_1ch] Multi-channel input detected — shape {shape} "
                f"appears to be (N, C={d1}, H, W).\n"
                f"  gs_1ch processes single-channel (grayscale) images only.\n"
                f"  Convert to grayscale before calling "
                f"validate_and_normalize_images().\n"
                f"  Example: images = images.mean(axis=1)"
            )

        raise ValueError(
            f"[gs_1ch] Unrecognized 4D shape {shape}.\n"
            f"  Expected one of:\n"
            f"    (N, H, W, 1) — channel last, single channel\n"
            f"    (N, 1, H, W) — channel first, single channel\n"
            f"  Got: {shape}\n"
            f"  Please reshape explicitly to (N, H, W) before calling "
            f"validate_and_normalize_images()."
        )

    # ── 5D or higher ──────────────────────────────────────────────────────────
    raise ValueError(
        f"[gs_1ch] Unsupported array dimensionality: {ndim}D shape {shape}.\n"
        f"  gs_1ch expects 2D (H, W), 3D (N, H, W), or 4D "
        f"(N, H, W, 1) / (N, 1, H, W) arrays.\n"
        f"  Got {ndim}D array. Please reshape to (N, H, W) explicitly."
    )
    
    
# ── Main validation and normalization function ────────────────────────────────

def validate_and_normalize_images(
    arr,
    normalize    : bool  = True,
    logger                = None,
) -> np.ndarray:
    """
    Validate and normalize an image array for use with gs_transform().

    Accepts the following input shapes and converts to (N, H, W) float32:
      (H, W)        — single grayscale image
      (N, H, W)     — batch of grayscale images
      (N, H, W, 1)  — batch with channel-last single channel
      (N, 1, H, W)  — batch with channel-first single channel

    Rejects with a clear error message:
      (N, H, W, C)  where C > 1  — multi-channel channel-last
      (N, C, H, W)  where C > 1  — multi-channel channel-first
      ndim > 4                    — volumetric or higher-dimensional data
      ambiguous shapes            — all dims equal

    Parameters
    ----------
    arr         : array-like — input image data
    normalize   : bool — if True, normalize values to [0, 1].
                  If values are already in [0, 1], no-op.
                  If values are in [0, 255], divides by 255.
                  If values are in other ranges, raises ValueError.
    logger      : optional — NVFlare logger or None (falls back to print)

    Returns
    -------
    out : np.ndarray — shape (N, H, W), dtype float32, values in [0, 1]

    Raises
    ------
    ValueError  — if shape is unsupported, ambiguous, or multi-channel
    ValueError  — if value range is unexpected after normalization
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # ── Convert to numpy ──────────────────────────────────────────────────────
    arr = np.asarray(arr)

    # ── Infer and convert shape ───────────────────────────────────────────────
    shape_type = _infer_shape(arr)

    if shape_type == "single_image_hw":
        # (H, W) → (1, H, W)
        arr = arr[np.newaxis, ...]
        _log(f"[gs_1ch] Input shape {arr.shape[1:]} — "
             f"added batch dimension → {arr.shape}")

    elif shape_type == "nhw":
        pass  # already correct

    elif shape_type == "nhwc_single":
        # (N, H, W, 1) → (N, H, W)
        original_shape = arr.shape
        arr = arr.squeeze(-1)
        _log(f"[gs_1ch] Squeezed channel-last dim: "
             f"{original_shape} → {arr.shape}")

    elif shape_type == "nchw_single":
        # (N, 1, H, W) → (N, H, W)
        original_shape = arr.shape
        arr = arr.squeeze(1)
        _log(f"[gs_1ch] Squeezed channel-first dim: "
             f"{original_shape} → {arr.shape}")

    # ── Final shape validation ────────────────────────────────────────────────
    if arr.ndim != 3:
        raise ValueError(
            f"[gs_1ch] Shape conversion failed — "
            f"expected 3D (N, H, W) after conversion, got {arr.shape}."
        )

    n, h, w = arr.shape
    if h == 0 or w == 0 or n == 0:
        raise ValueError(
            f"[gs_1ch] Invalid shape {arr.shape} — "
            f"no dimension can be zero."
        )

    if h != w:
        _log(
            f"[gs_1ch] ⚠️  Non-square images detected: {h}×{w}. "
            f"GS transform works on non-square images but square "
            f"inputs are recommended for best results."
        )

    # ── Convert to float32 ────────────────────────────────────────────────────
    arr = arr.astype(np.float32)

    # ── Normalize values ──────────────────────────────────────────────────────
    if normalize:
        vmin = float(arr.min())
        vmax = float(arr.max())

        if vmin >= 0.0 and vmax <= 1.0:
            pass  # already normalized

        elif vmin >= 0.0 and vmax <= 255.0:
            arr = arr / 255.0
            _log(
                f"[gs_1ch] Normalized [0, 255] → [0, 1] "
                f"(divided by 255)."
            )

        elif vmin < 0.0:
            raise ValueError(
                f"[gs_1ch] Unexpected value range [{vmin:.4f}, {vmax:.4f}]. "
                f"Input contains negative values.\n"
                f"  gs_1ch expects values in [0, 1] or [0, 255].\n"
                f"  Please normalize your data before calling "
                f"validate_and_normalize_images()."
            )

        else:
            raise ValueError(
                f"[gs_1ch] Unexpected value range [{vmin:.4f}, {vmax:.4f}]. "
                f"  gs_1ch expects values in [0, 1] or [0, 255].\n"
                f"  Please normalize your data before calling "
                f"validate_and_normalize_images()."
            )

    # ── Final value range check ───────────────────────────────────────────────
    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmin < _WARN_RANGE_MIN or vmax > _WARN_RANGE_MAX:
        raise ValueError(
            f"[gs_1ch] Value range [{vmin:.4f}, {vmax:.4f}] "
            f"out of expected [0, 1] after normalization.\n"
            f"  Please check your input data."
        )

    return arr


# ── Summary helper ────────────────────────────────────────────────────────────

def describe_array(arr, name: str = "array") -> str:
    """
    Return a human-readable description of an array's
    shape, dtype, and value range. Useful for debugging
    data preparation pipelines.

    Example:
        print(describe_array(images, "training data"))
        # training data: shape=(500, 28, 28) dtype=float32
        #   range=[0.000, 1.000] mean=0.482
    """
    arr  = np.asarray(arr)
    desc = (
        f"{name}: shape={arr.shape} dtype={arr.dtype}\n"
        f"  range=[{arr.min():.3f}, {arr.max():.3f}] "
        f"mean={arr.mean():.3f}"
    )
    return desc


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  preprocessing.py — self-test")
    print("  " + "─" * 44)

    passed = []
    failed = []

    def _ok(name):
        passed.append(name)
        print(f"  ✅  {name}")

    def _fail(name, reason):
        failed.append(name)
        print(f"  ❌  {name} — {reason}")

    # (N, H, W) — already correct
    try:
        arr = np.random.rand(10, 28, 28).astype(np.float32)
        out = validate_and_normalize_images(arr)
        assert out.shape == (10, 28, 28)
        _ok("(N, H, W) passthrough")
    except Exception as e:
        _fail("(N, H, W) passthrough", str(e))

    # (N, H, W, 1) — channel last
    try:
        arr = np.random.randint(0, 256, (10, 28, 28, 1), dtype=np.uint8)
        out = validate_and_normalize_images(arr)
        assert out.shape == (10, 28, 28)
        assert out.max() <= 1.0
        _ok("(N, H, W, 1) squeeze + normalize")
    except Exception as e:
        _fail("(N, H, W, 1) squeeze + normalize", str(e))

    # (N, 1, H, W) — channel first
    try:
        arr = np.random.randint(0, 256, (10, 1, 28, 28), dtype=np.uint8)
        out = validate_and_normalize_images(arr)
        assert out.shape == (10, 28, 28)
        assert out.max() <= 1.0
        _ok("(N, 1, H, W) squeeze + normalize")
    except Exception as e:
        _fail("(N, 1, H, W) squeeze + normalize", str(e))

    # (H, W) — single image
    try:
        arr = np.random.rand(28, 28).astype(np.float32)
        out = validate_and_normalize_images(arr)
        assert out.shape == (1, 28, 28)
        _ok("(H, W) single image → (1, H, W)")
    except Exception as e:
        _fail("(H, W) single image", str(e))

    # (N, H, W, 3) — multi-channel should raise
    try:
        arr = np.random.rand(10, 28, 28, 3).astype(np.float32)
        validate_and_normalize_images(arr)
        _fail("(N, H, W, 3) rejection", "Should have raised ValueError")
    except ValueError:
        _ok("(N, H, W, 3) correctly rejected")
    except Exception as e:
        _fail("(N, H, W, 3) rejection", str(e))

    # (N, 3, H, W) — multi-channel should raise
    try:
        arr = np.random.rand(10, 3, 28, 28).astype(np.float32)
        validate_and_normalize_images(arr)
        _fail("(N, 3, H, W) rejection", "Should have raised ValueError")
    except ValueError:
        _ok("(N, 3, H, W) correctly rejected")
    except Exception as e:
        _fail("(N, 3, H, W) rejection", str(e))

    # 5D — should raise
    try:
        arr = np.random.rand(10, 1, 1, 28, 28).astype(np.float32)
        validate_and_normalize_images(arr)
        _fail("5D rejection", "Should have raised ValueError")
    except ValueError:
        _ok("5D array correctly rejected")
    except Exception as e:
        _fail("5D rejection", str(e))

    # Negative values — should raise
    try:
        arr = np.random.rand(10, 28, 28).astype(np.float32) - 0.5
        validate_and_normalize_images(arr)
        _fail("negative values rejection", "Should have raised ValueError")
    except ValueError:
        _ok("Negative values correctly rejected")
    except Exception as e:
        _fail("negative values rejection", str(e))

    # describe_array
    try:
        arr  = np.random.rand(10, 28, 28).astype(np.float32)
        desc = describe_array(arr, "test")
        assert "shape=(10, 28, 28)" in desc
        _ok("describe_array")
    except Exception as e:
        _fail("describe_array", str(e))

    print()
    print("  " + "─" * 44)
    print(f"  {len(passed)} passed | {len(failed)} failed")
    print()
    
    
