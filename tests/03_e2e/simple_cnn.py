# ============================================================
# tests/03_e2e/simple_cnn.py
# Minimal CNN for PneumoniaMNIST binary classification.
#
# Architecture:
#   Conv(1→16) → ReLU → MaxPool
#   Conv(16→32) → ReLU → MaxPool
#   FC(32*5*5→128) → ReLU → Dropout
#   FC(128→2)
#
# Input  : (B, 1, 28, 28) float32
# Output : (B, 2) logits
#
# ~47K parameters — trains in seconds per round on any GPU.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Minimal CNN for binary classification on 28x28 single-channel images.
    Designed for federated learning demonstration — small enough to
    aggregate quickly, large enough to learn meaningful features.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()

        # ── Feature extractor ─────────────────────────────────────────────────
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B,16,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # (B,16,14,14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (B,32,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # (B,32,7,7)
        )

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
# ── Dataset utility ───────────────────────────────────────────────────────────

class NumpyImageDataset(torch.utils.data.Dataset):
    """
    Wraps a (B, H, W) float32 numpy array and label array
    into a PyTorch Dataset.

    Adds channel dimension: (B, H, W) → (B, 1, H, W)
    """

    def __init__(self, images_path: str, labels_path: str):
        import numpy as np
        self.images = np.load(images_path).astype('float32')
        self.labels = np.load(labels_path).astype('int64')

        assert self.images.ndim == 3, \
            f"Expected (B, H, W), got {self.images.shape}"
        assert len(self.images) == len(self.labels), \
            "Images and labels length mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Add channel dim: (H, W) → (1, H, W)
        img   = torch.from_numpy(self.images[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
    
    
# ── Training utilities ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    """
    Train model for one epoch.
    Returns average loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    criterion = nn.CrossEntropyLoss()

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, device):
    """
    Evaluate model on a dataset.
    Returns accuracy and average loss.
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    n_batches     = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits  = model(imgs)
            loss    = criterion(logits, labels)
            preds   = logits.argmax(dim=1)

            total_loss    += loss.item()
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            n_batches     += 1

    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(n_batches, 1)
    return accuracy, avg_loss


def get_model_weights(model):
    """Extract model weights as a dict of numpy arrays."""
    import numpy as np
    return {
        k: v.cpu().numpy().copy()
        for k, v in model.state_dict().items()
    }


def set_model_weights(model, weights):
    """Load model weights from a dict of numpy arrays."""
    import numpy as np
    import torch
    state_dict = {
        k: torch.from_numpy(v.copy())
        for k, v in weights.items()
    }
    model.load_state_dict(state_dict)


def fedavg(weight_list):
    """
    Federated averaging — simple mean of weights across clients.
    weight_list : list of weight dicts (one per client)
    Returns     : averaged weight dict
    """
    import numpy as np
    avg = {}
    for key in weight_list[0].keys():
        avg[key] = np.mean(
            [w[key] for w in weight_list], axis=0
        )
    return avg



# ── Self-test (run directly) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print()
    print("  SimpleCNN — self-test")
    print("  " + "─" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    model = SimpleCNN(num_classes=2).to(device)
    print(f"  Parameters : {model.count_parameters():,}")

    # Forward pass with random input
    x   = torch.randn(4, 1, 28, 28).to(device)
    out = model(x)
    assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"
    print(f"  Forward pass : input {tuple(x.shape)} → output {tuple(out.shape)} ✅")

    # Weight extraction and loading round-trip
    weights  = get_model_weights(model)
    set_model_weights(model, weights)
    print(f"  Weight round-trip : ✅")

    # FedAvg with two identical weight dicts
    avg = fedavg([weights, weights])
    for k in weights:
        import numpy as np
        assert np.allclose(avg[k], weights[k]), f"FedAvg mismatch on {k}"
    print(f"  FedAvg : ✅")

    print()
    print("  ✅  SimpleCNN self-test passed.")
    print()