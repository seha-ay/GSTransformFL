# ============================================================
# tests/01_unit/test_core_transform.py
# Unit tests for gs_1ch core transform logic.
#
# Tests the algorithm behavior only — no NVFlare, no GPU.
# Safe to run on any machine with numpy installed.
#
# Usage:
#   python tests/01_unit/test_core_transform.py
# ============================================================

import sys
import os
import numpy as np
from pathlib import Path

# ── Add repo root to path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Test tracking ─────────────────────────────────────────────────────────────
_passed = []
_failed = []

def passed(name):
    _passed.append(name)
    print(f"  ✅  {name}")

def failed(name, reason):
    _failed.append(name)
    print(f"  ❌  {name} — {reason}")
    
    
# ── Test 1: Package imports ───────────────────────────────────────────────────
print()
print("  [1] Package imports")
print("  " + "─" * 48)

try:
    from gs_1ch import gs_transform, run_diagnostics
    passed("gs_1ch top-level imports")
except Exception as e:
    failed("gs_1ch top-level imports", str(e))

try:
    from gs_1ch.core.diagnostic import (
        get_gpu_info, read_diag_file, write_diag_file,
        check_gpu_match, estimate_mem_bytes, safe_vram_bytes,
        _MEM_SAFETY_FRAC
    )
    passed("diagnostic module imports")
except Exception as e:
    failed("diagnostic module imports", str(e))

try:
    from gs_1ch.core.transform import _gs1ch_core, gs_transform
    passed("transform module imports")
except Exception as e:
    failed("transform module imports", str(e))

try:
    from gs_1ch.executor.policy import DiagnosticPolicy, DEFAULT_POLICY
    passed("policy module imports")
except Exception as e:
    failed("policy module imports", str(e))

try:
    from gs_1ch.reporting.error_report import (
        ClientResult, make_ok, make_error, make_skipped, consolidate
    )
    passed("reporting module imports")
except Exception as e:
    failed("reporting module imports", str(e))
    
    
# ── Test 2: Diagnostic utilities ─────────────────────────────────────────────
print()
print("  [2] Diagnostic utilities")
print("  " + "─" * 48)

from gs_1ch.core.diagnostic import estimate_mem_bytes, safe_vram_bytes

try:
    mem = estimate_mem_bytes(10, 256, 256)
    assert mem == 10 * 256 * 256 * 64, f"Expected {10*256*256*64}, got {mem}"
    passed("estimate_mem_bytes — correct formula")
except Exception as e:
    failed("estimate_mem_bytes", str(e))

try:
    safe = safe_vram_bytes(16 * 1024**3)
    expected = int(16 * 1024**3 * 0.85)
    assert safe == expected, f"Expected {expected}, got {safe}"
    passed("safe_vram_bytes — 85% of total")
except Exception as e:
    failed("safe_vram_bytes", str(e))

try:
    from gs_1ch.core.diagnostic import check_gpu_match
    diag = {
        'gpu_name'        : 'NVIDIA L4',
        'total_vram_bytes': 23570219008,
        'total_vram_gb'   : 21.95,
    }
    gpu_info = {
        'gpu_name'        : 'NVIDIA L4',
        'total_vram_bytes': 23570219008,
        'total_vram_gb'   : 21.95,
    }
    result = check_gpu_match(diag, gpu_info)
    assert result['status'] == 'ok', f"Expected ok, got {result['status']}"
    passed("check_gpu_match — identical GPU returns ok")
except Exception as e:
    failed("check_gpu_match identical", str(e))

try:
    diag_diff = {
        'gpu_name'        : 'NVIDIA T4',
        'total_vram_bytes': 15843721216,
        'total_vram_gb'   : 14.75,
    }
    result = check_gpu_match(diag_diff, gpu_info)
    assert result['status'] == 'critical', \
        f"Expected critical, got {result['status']}"
    passed("check_gpu_match — different GPU returns critical")
except Exception as e:
    failed("check_gpu_match different", str(e))
    
    
# ── Test 3: Diagnostic file I/O ───────────────────────────────────────────────
print()
print("  [3] Diagnostic file I/O")
print("  " + "─" * 48)

import tempfile
from gs_1ch.core.diagnostic import write_diag_file, read_diag_file

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        diag_path = Path(tmpdir) / "test_diagnostic.txt"
        gpu_info  = {
            'gpu_name'        : 'NVIDIA L4',
            'gpu_uuid'        : 'test-uuid-123',
            'total_vram_bytes': 23570219008,
            'total_vram_gb'   : 21.95,
        }
        write_diag_file(diag_path, gpu_info, sec_per_element=2.104e-9)
        assert diag_path.exists(), "Diagnostic file not created"
        passed("write_diag_file — file created")

        diag = read_diag_file(diag_path)
        assert diag is not None, "read_diag_file returned None"
        assert diag['gpu_name']         == 'NVIDIA L4'
        assert diag['total_vram_bytes'] == 23570219008
        assert abs(diag['probe_sec_per_elem'] - 2.104e-9) < 1e-15
        passed("read_diag_file — values parsed correctly")
except Exception as e:
    failed("diagnostic file I/O", str(e))

try:
    result = read_diag_file(Path("/nonexistent/path/diagnostic.txt"))
    assert result is None, "Expected None for missing file"
    passed("read_diag_file — returns None for missing file")
except Exception as e:
    failed("read_diag_file missing file", str(e))
    
    
# ── Test 4: DiagnosticPolicy ──────────────────────────────────────────────────
print()
print("  [4] DiagnosticPolicy")
print("  " + "─" * 48)

from gs_1ch.executor.policy import DiagnosticPolicy, DEFAULT_POLICY

try:
    policy = DiagnosticPolicy()
    assert policy.on_first_run  == "auto"
    assert policy.on_gpu_change == "auto"
    assert policy.on_oom        == "abort_job"
    assert policy.verbose       == True
    passed("DiagnosticPolicy — default values correct")
except Exception as e:
    failed("DiagnosticPolicy defaults", str(e))

try:
    d = DEFAULT_POLICY.to_dict()
    assert d['on_first_run']  == "auto"
    assert d['on_gpu_change'] == "auto"
    assert d['on_oom']        == "abort_job"
    assert d['verbose']       == True
    passed("DiagnosticPolicy.to_dict — correct keys and values")
except Exception as e:
    failed("DiagnosticPolicy.to_dict", str(e))

try:
    DiagnosticPolicy(on_first_run="invalid_value")
    failed("DiagnosticPolicy invalid value",
           "Should have raised ValueError")
except ValueError:
    passed("DiagnosticPolicy — rejects invalid on_first_run")
except Exception as e:
    failed("DiagnosticPolicy invalid value", str(e))
    
    
# ── Test 5: Error reporting ───────────────────────────────────────────────────
print()
print("  [5] Error reporting")
print("  " + "─" * 48)

from gs_1ch.reporting.error_report import (
    ClientResult, make_ok, make_error, make_skipped,
    consolidate, STATUS_OK, STATUS_ERROR, STATUS_SKIPPED
)

try:
    result = make_ok(
        client_id    = "site-1",
        output_path  = "/data/output/images.npy",
        input_shape  = (500, 28, 28),
        output_shape = (500, 28, 28),
        elapsed_sec  = 1.5,
        gpu_name     = "NVIDIA L4",
        n_chunks     = 1,
    )
    assert result.status      == STATUS_OK
    assert result.client_id   == "site-1"
    assert result.input_shape == (500, 28, 28)
    passed("make_ok — correct fields")
except Exception as e:
    failed("make_ok", str(e))

try:
    err = make_error(
        client_id   = "site-2",
        error_stage = "diagnostic",
        exception   = RuntimeError("CUDA not available"),
    )
    assert err.status      == STATUS_ERROR
    assert err.error_stage == "diagnostic"
    assert err.error_type  == "RuntimeError"
    passed("make_error — correct fields")
except Exception as e:
    failed("make_error", str(e))

try:
    skipped = make_skipped("site-3", "unknown task")
    assert skipped.status == STATUS_SKIPPED
    passed("make_skipped — correct fields")
except Exception as e:
    failed("make_skipped", str(e))

try:
    r1 = make_ok("site-1", "/out/1.npy", (100,28,28),
                 (100,28,28), 1.0, "NVIDIA L4", 1)
    r2 = make_error("site-2", "transform",
                    MemoryError("OOM"))
    summary = consolidate([r1, r2])
    assert summary['abort']        == True
    assert "site-1" in summary['ok_clients']
    assert "site-2" in summary['failed']
    passed("consolidate — abort=True when any client failed")
except Exception as e:
    failed("consolidate abort", str(e))

try:
    r1 = make_ok("site-1", "/out/1.npy", (100,28,28),
                 (100,28,28), 1.0, "NVIDIA L4", 1)
    r2 = make_ok("site-2", "/out/2.npy", (100,28,28),
                 (100,28,28), 1.2, "NVIDIA T4", 1)
    summary = consolidate([r1, r2])
    assert summary['abort'] == False
    passed("consolidate — abort=False when all clients ok")
except Exception as e:
    failed("consolidate no abort", str(e))

try:
    result  = make_ok("site-1", "/out/1.npy", (100,28,28),
                      (100,28,28), 1.0, "NVIDIA L4", 1)
    d       = result.to_dict()
    result2 = ClientResult.from_dict(d)
    assert result2.client_id   == result.client_id
    assert result2.status      == result.status
    assert result2.output_path == result.output_path
    passed("ClientResult serialization round-trip")
except Exception as e:
    failed("ClientResult serialization", str(e))
    
    

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("  " + "─" * 48)
print(f"  Results: {len(_passed)} passed | {len(_failed)} failed")
print()

if _failed:
    print("  ❌  Some unit tests failed:")
    for name in _failed:
        print(f"      • {name}")
    print()
    sys.exit(1)
else:
    print("  ✅  All unit tests passed.")
    print()
    sys.exit(0)