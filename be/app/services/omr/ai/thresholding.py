from typing import Tuple

import cv2
import numpy as np


def threshold_weighted_adaptive(
    gray_img: np.ndarray,
    window_size: int = 31,
    k: float = 0.28,
    contrast_weight: float = 0.65,
) -> np.ndarray:
    """Weighted local adaptive binarization robust to shadows.

    Output is binary-inv (marks are white/255), compatible with OMR pixel counting.
    """
    if gray_img is None or gray_img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    w = int(max(15, min(81, window_size)))
    if w % 2 == 0:
        w += 1

    src = gray_img.astype(np.float32)
    mean = cv2.boxFilter(src, ddepth=-1, ksize=(w, w), normalize=True)
    sq_mean = cv2.boxFilter(src * src, ddepth=-1, ksize=(w, w), normalize=True)
    var = np.maximum(sq_mean - (mean * mean), 0.0)
    std = np.sqrt(var)

    std_norm = std / max(1.0, float(np.max(std)))
    cw = float(max(0.0, min(1.0, contrast_weight)))

    # Sauvola-like threshold with contrast weighting.
    # Higher local contrast => lower threshold penalty, preserving faint strokes.
    local_k = float(k) * (1.0 - (cw * std_norm))
    thr_map = mean * (1.0 - local_k)

    bw = np.where(src <= thr_map, 255, 0).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return bw


def threshold_hybrid_shadow_robust(
    gray_img: np.ndarray,
    mode: str = "hybrid",
) -> Tuple[np.ndarray, float]:
    """Global threshold wrapper for OMR.

    Modes:
    - otsu: legacy robust normalization + Otsu
    - weighted_adaptive: weighted local thresholding
    - hybrid: combine otsu and weighted_adaptive to keep faint marks
    """
    if gray_img is None or gray_img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), 0.0

    mode = str(mode or "hybrid").strip().lower()
    if mode not in {"otsu", "weighted_adaptive", "hybrid"}:
        mode = "hybrid"

    # Illumination normalization (same spirit as legacy path).
    k = max(31, (min(gray_img.shape[:2]) // 12) | 1)
    bg = cv2.morphologyEx(
        gray_img,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    norm = cv2.divide(gray_img, bg, scale=255)
    norm = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(norm)

    otsu_val, otsu_bin = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    if mode == "otsu":
        return otsu_bin, float(otsu_val)

    wad = threshold_weighted_adaptive(norm, window_size=35, k=0.26, contrast_weight=0.68)

    if mode == "weighted_adaptive":
        return wad, float(otsu_val)

    # Hybrid keeps marks detected by either path and suppresses tiny noise.
    hybrid = cv2.bitwise_or(otsu_bin, wad)
    hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return hybrid, float(otsu_val)
