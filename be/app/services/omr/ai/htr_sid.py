import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore


DIGIT_LABELS = [str(i) for i in range(10)]


class HandwrittenDigitCNN(nn.Module):  # type: ignore[misc]
    """Tiny handwritten digit recognizer for write-row fallback."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@dataclass
class SidHtrPrediction:
    text: str
    mean_confidence: float
    per_digit_confidence: List[float]


def _segment_write_row(gray_img: np.ndarray, expected_digits: int) -> List[np.ndarray]:
    if gray_img is None or gray_img.size == 0:
        return []

    eq = cv2.equalizeHist(gray_img)
    bw = cv2.adaptiveThreshold(
        eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = bw.shape[:2]
    min_area = max(10, int((H * W) * 0.002))
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area:
            continue
        if h < int(H * 0.30):
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    # If segmentation is noisy, fallback to equal-width splitting.
    if len(boxes) < max(2, expected_digits // 2) or len(boxes) > expected_digits * 2:
        step = max(1, W // max(1, expected_digits))
        crops = []
        for i in range(expected_digits):
            x1 = i * step
            x2 = W if i == expected_digits - 1 else (i + 1) * step
            crops.append(gray_img[:, x1:x2])
        return crops

    # Merge/slice to expected count by center assignment.
    centers = [(b[0] + (b[2] / 2.0), b) for b in boxes]
    crops = []
    for i in range(expected_digits):
        slot_x1 = int((i / expected_digits) * W)
        slot_x2 = int(((i + 1) / expected_digits) * W)
        in_slot = [b for cx, b in centers if slot_x1 <= cx < slot_x2]
        if not in_slot:
            crops.append(gray_img[:, slot_x1:slot_x2])
            continue
        x = min(b[0] for b in in_slot)
        y = min(b[1] for b in in_slot)
        xx = max(b[0] + b[2] for b in in_slot)
        yy = max(b[1] + b[3] for b in in_slot)
        crops.append(gray_img[y:yy, x:xx])

    return crops


class SidHandwritingRecognizer:
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
            model = HandwrittenDigitCNN()
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

    def predict_sequence(self, write_row_gray: np.ndarray, expected_digits: int) -> Optional[SidHtrPrediction]:
        if not self.ready or self.model is None or torch is None:
            return None
        if write_row_gray is None or write_row_gray.size == 0:
            return None

        expected_digits = max(1, int(expected_digits))
        cells = _segment_write_row(write_row_gray, expected_digits=expected_digits)
        if len(cells) <= 0:
            return None

        chars: List[str] = []
        confs: List[float] = []

        for cell in cells[:expected_digits]:
            if cell is None or cell.size == 0:
                chars.append("?")
                confs.append(0.0)
                continue

            patch = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
            patch = patch.astype(np.float32) / 255.0
            patch = 1.0 - patch
            x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)

            idx = int(np.argmax(probs))
            chars.append(DIGIT_LABELS[idx])
            confs.append(float(probs[idx]))

        text = "".join(chars)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return SidHtrPrediction(text=text, mean_confidence=mean_conf, per_digit_confidence=confs)
