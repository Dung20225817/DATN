import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    datasets = None  # type: ignore
    transforms = None  # type: ignore


CLASS_NAMES = ["marked", "empty", "erased"]


class SimpleBubbleCNN(nn.Module):  # type: ignore[misc]
    """Tiny CNN for 32x32 bubble classification."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@dataclass
class BubblePrediction:
    label: str
    confidence: float
    probs: Dict[str, float]


class BubbleCellClassifier:
    """Inference wrapper for SimpleBubbleCNN."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = str(model_path or "")
        self.device = device
        self.model = None
        self.ready = False
        self._load()

    def _load(self):
        if torch is None or nn is None:
            return
        if not self.model_path or not os.path.exists(self.model_path):
            return

        try:
            model = SimpleBubbleCNN(num_classes=len(CLASS_NAMES))
            state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            model.load_state_dict(state)
            model.eval()
            model.to(self.device)
            self.model = model
            self.ready = True
        except Exception:
            self.model = None
            self.ready = False

    def predict(self, cell_img: np.ndarray) -> Optional[BubblePrediction]:
        if not self.ready or self.model is None or torch is None:
            return None
        if cell_img is None or cell_img.size == 0:
            return None

        if cell_img.ndim == 3:
            cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            cell_gray = cell_img.copy()

        cell = cv2.resize(cell_gray, (32, 32), interpolation=cv2.INTER_AREA)
        cell = cell.astype(np.float32) / 255.0

        # Keep dark stroke high-response by inverting to "ink map".
        cell = 1.0 - cell
        tens = torch.from_numpy(cell).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tens)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(min(len(CLASS_NAMES), len(probs)))}
        return BubblePrediction(label=CLASS_NAMES[idx], confidence=conf, probs=prob_map)


def train_simple_bubble_classifier(
    dataset_dir: str,
    output_model_path: str,
    epochs: int = 14,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
) -> Dict[str, float]:
    """Train SimpleBubbleCNN from ImageFolder dataset.

    Expected dataset layout:
    dataset_dir/
      marked/
      empty/
      erased/
    """
    if torch is None or nn is None or optim is None or DataLoader is None or datasets is None or transforms is None:
        raise RuntimeError("PyTorch/Torchvision is not available")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ])

    full_ds = datasets.ImageFolder(dataset_dir, transform=tfm)
    if len(full_ds) <= 0:
        raise RuntimeError("Empty dataset")

    expected = set(CLASS_NAMES)
    found = set(full_ds.class_to_idx.keys())
    if not expected.issubset(found):
        raise RuntimeError(f"Dataset classes must include {CLASS_NAMES}, found {sorted(found)}")

    val_len = max(1, int(len(full_ds) * float(val_split)))
    train_len = max(1, len(full_ds) - val_len)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleBubbleCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))

    best_val_acc = 0.0
    history = {"train_loss": 0.0, "val_loss": 0.0, "val_acc": 0.0}

    for _ in range(max(1, int(epochs))):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_steps += 1

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_steps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item())
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == yb).sum().item())
                total += int(yb.numel())
                val_steps += 1

        val_acc = float(correct) / max(1, total)
        history = {
            "train_loss": float(train_loss / max(1, train_steps)),
            "val_loss": float(val_loss / max(1, val_steps)),
            "val_acc": float(val_acc),
        }

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_acc": float(val_acc),
                },
                output_model_path,
            )

    history["best_val_acc"] = float(best_val_acc)
    return history
