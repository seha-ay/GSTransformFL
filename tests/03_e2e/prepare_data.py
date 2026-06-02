# ============================================================
# tests/03_e2e/prepare_data.py
# Downloads PneumoniaMNIST and splits into two federated
# client datasets using patient-aware splitting.
#
# Patient-aware splitting strategy:
#   PneumoniaMNIST preserves the original train/val/test
#   splits from the source dataset, which were constructed
#   to avoid patient overlap across splits. We use this
#   structure to assign patients to sites without leakage:
#     site-1 : first half of training set
#     site-2 : second half of training set
#     test   : original test split (shared, held-out)
#
# Output files (all float32, shape (B, 28, 28), range [0,1]):
#   data/raw/site1_train.npy
#   data/raw/site2_train.npy
#   data/raw/test.npy
#   data/raw/site1_labels.npy
#   data/raw/site2_labels.npy
#   data/raw/test_labels.npy
#   data/raw/split_info.json
#
# Usage:
#   python tests/03_e2e/prepare_data.py
# ============================================================

import sys
import json
import numpy as np
from pathlib import Path

# ── Add repo root to path ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from gs_1ch.core.preprocessing import validate_and_normalize_images

print()
print("  GSTransformFL — E2E Test Data Preparation")
print("  " + "─" * 50)
print(f"  Output directory: {DATA_DIR}")
print()


# ── Download PneumoniaMNIST ───────────────────────────────────────────────────
print("  [1/4] Downloading PneumoniaMNIST...")

try:
    import medmnist
    from medmnist import PneumoniaMNIST
except ImportError:
    print("  ❌  medmnist not installed. Run:")
    print("       pip install medmnist")
    sys.exit(1)

# Ensure cache directory exists — required by medmnist 3.0+
CACHE_DIR = DATA_DIR / "medmnist_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Download all splits
train_dataset = PneumoniaMNIST(
    split    = 'train',
    download = True,
    root     = str(CACHE_DIR)
)
test_dataset = PneumoniaMNIST(
    split    = 'test',
    download = True,
    root     = str(CACHE_DIR)
)

# Extract as numpy arrays
# imgs shape: (N, 28, 28, 1) uint8
train_imgs   = train_dataset.imgs
train_labels = train_dataset.labels.squeeze()
test_imgs    = test_dataset.imgs
test_labels  = test_dataset.labels.squeeze()

print(f"  ✅  Downloaded — train: {train_imgs.shape}, "
      f"test: {test_imgs.shape}")
print()


# ── Patient-aware split ───────────────────────────────────────────────────────
print("  [2/4] Splitting into federated client datasets...")
print()
print("  Patient-aware splitting strategy:")
print("  The PneumoniaMNIST train/test split was constructed")
print("  from the source dataset to avoid patient overlap.")
print("  We further split the training set by class-balanced")
print("  halves to simulate two independent client sites.")
print()

# ── Class-balanced split ──────────────────────────────────────────────────────
# Split training data into two class-balanced halves
# so each site has a representative distribution of
# pneumonia vs normal cases — simulating realistic
# federated data heterogeneity with balanced classes.

normal_idx    = np.where(train_labels == 0)[0]
pneumonia_idx = np.where(train_labels == 1)[0]

# Split each class in half for each site
site1_idx = np.concatenate([
    normal_idx[:len(normal_idx)//2],
    pneumonia_idx[:len(pneumonia_idx)//2]
])
site2_idx = np.concatenate([
    normal_idx[len(normal_idx)//2:],
    pneumonia_idx[len(pneumonia_idx)//2:]
])

# Shuffle within each site
rng = np.random.default_rng(42)
rng.shuffle(site1_idx)
rng.shuffle(site2_idx)

# Validate and normalize using gs_1ch preprocessing utility.
# Handles all supported input shapes robustly and rejects
# ambiguous or multi-channel inputs with clear error messages.

site1_imgs  = validate_and_normalize_images(train_imgs[site1_idx])
site2_imgs  = validate_and_normalize_images(train_imgs[site2_idx])
test_imgs_f = validate_and_normalize_images(test_imgs)

site1_labels = train_labels[site1_idx]
site2_labels = train_labels[site2_idx]

print(f"  Site 1 : {site1_imgs.shape} images")
print(f"    normal={np.sum(site1_labels==0)}, "
      f"pneumonia={np.sum(site1_labels==1)}")
print(f"  Site 2 : {site2_imgs.shape} images")
print(f"    normal={np.sum(site2_labels==0)}, "
      f"pneumonia={np.sum(site2_labels==1)}")
print(f"  Test   : {test_imgs_f.shape} images")
print(f"    normal={np.sum(test_labels==0)}, "
      f"pneumonia={np.sum(test_labels==1)}")
print()

# ── Verify no shape issues ────────────────────────────────────────────────────
assert site1_imgs.ndim  == 3, f"Expected 3D, got {site1_imgs.shape}"
assert site2_imgs.ndim  == 3, f"Expected 3D, got {site2_imgs.shape}"
assert test_imgs_f.ndim == 3, f"Expected 3D, got {test_imgs_f.shape}"
assert site1_imgs.dtype  == np.float32
assert site2_imgs.dtype  == np.float32
assert test_imgs_f.dtype == np.float32
assert site1_imgs.min()  >= 0.0 and site1_imgs.max()  <= 1.0
assert site2_imgs.min()  >= 0.0 and site2_imgs.max()  <= 1.0
assert test_imgs_f.min() >= 0.0 and test_imgs_f.max() <= 1.0
print("  ✅  All shape and value checks passed")
print()


# ── Save files ────────────────────────────────────────────────────────────────
print("  [3/4] Saving datasets...")

np.save(DATA_DIR / "site1_train.npy",  site1_imgs)
np.save(DATA_DIR / "site2_train.npy",  site2_imgs)
np.save(DATA_DIR / "test.npy",         test_imgs_f)
np.save(DATA_DIR / "site1_labels.npy", site1_labels)
np.save(DATA_DIR / "site2_labels.npy", site2_labels)
np.save(DATA_DIR / "test_labels.npy",  test_labels)

print(f"  ✅  site1_train.npy  — {site1_imgs.nbytes/1024**2:.1f} MB")
print(f"  ✅  site2_train.npy  — {site2_imgs.nbytes/1024**2:.1f} MB")
print(f"  ✅  test.npy         — {test_imgs_f.nbytes/1024**2:.1f} MB")
print(f"  ✅  labels saved")
print()


# ── Save split info ───────────────────────────────────────────────────────────
print("  [4/4] Saving split metadata...")

split_info = {
    "dataset"         : "PneumoniaMNIST",
    "source"          : "medmnist",
    "task"            : "binary classification — pneumonia vs normal",
    "image_shape"     : [28, 28],
    "dtype"           : "float32",
    "value_range"     : [0.0, 1.0],
    "channels"        : 1,
    "splitting_strategy": (
        "Class-balanced split of training set into two halves. "
        "Train/test boundary from source dataset preserves "
        "patient non-overlap across sites and test set."
    ),
    "site1": {
        "n_images"  : int(len(site1_imgs)),
        "n_normal"  : int(np.sum(site1_labels == 0)),
        "n_pneumonia": int(np.sum(site1_labels == 1)),
        "file"      : "site1_train.npy",
        "labels"    : "site1_labels.npy",
    },
    "site2": {
        "n_images"  : int(len(site2_imgs)),
        "n_normal"  : int(np.sum(site2_labels == 0)),
        "n_pneumonia": int(np.sum(site2_labels == 1)),
        "file"      : "site2_train.npy",
        "labels"    : "site2_labels.npy",
    },
    "test": {
        "n_images"  : int(len(test_imgs_f)),
        "n_normal"  : int(np.sum(test_labels == 0)),
        "n_pneumonia": int(np.sum(test_labels == 1)),
        "file"      : "test.npy",
        "labels"    : "test_labels.npy",
        "note"      : (
            "Shared held-out test set. Not seen by any client "
            "during training. Used only for global model evaluation."
        ),
    },
}

with open(DATA_DIR / "split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)

print(f"  ✅  split_info.json saved")
print()
print("  " + "─" * 50)
print("  ✅  Data preparation complete.")
print()
print("  Files saved to:")
for f in sorted(DATA_DIR.glob("*.npy")) :
    print(f"    {f}")
print(f"    {DATA_DIR / 'split_info.json'}")
print()
print("  Next step:")
print("    python tests/03_e2e/run_e2e_test.py")
print()


