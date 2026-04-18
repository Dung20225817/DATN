from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import omr_marker_utils
from .omr_utils import _clip_rect, _order_quad_points, _warp_by_manual_quad


def _safe_float(raw, default=0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _norm_quad_from_points(points: np.ndarray, img_w: int, img_h: int) -> Dict[str, Dict[str, float]]:
    pts = _order_quad_points(np.asarray(points, dtype=np.float32).reshape(4, 2))
    sx = float(max(1, int(img_w) - 1))
    sy = float(max(1, int(img_h) - 1))

    def _pt(pxy: np.ndarray) -> Dict[str, float]:
        return {
            "x": round(float(pxy[0]) / sx, 6),
            "y": round(float(pxy[1]) / sy, 6),
        }

    return {
        "tl": _pt(pts[0]),
        "tr": _pt(pts[1]),
        "br": _pt(pts[2]),
        "bl": _pt(pts[3]),
    }


def _parse_manual_quad(
    crop_tl_x,
    crop_tl_y,
    crop_tr_x,
    crop_tr_y,
    crop_br_x,
    crop_br_y,
    crop_bl_x,
    crop_bl_y,
) -> Optional[List[Tuple[float, float]]]:
    vals = [crop_tl_x, crop_tl_y, crop_tr_x, crop_tr_y, crop_br_x, crop_br_y, crop_bl_x, crop_bl_y]
    if not all(v is not None for v in vals):
        return None

    parsed: List[Tuple[float, float]] = []
    for i in range(0, 8, 2):
        x = _safe_float(vals[i], -1.0)
        y = _safe_float(vals[i + 1], -1.0)

        if x > 1.0 and x <= 100.0:
            x /= 100.0
        if y > 1.0 and y <= 100.0:
            y /= 100.0

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        parsed.append((float(x), float(y)))

    return parsed


def _apply_optional_rect_crop(img_bgr, crop_x, crop_y, crop_w, crop_h):
    vals = [crop_x, crop_y, crop_w, crop_h]
    if not all(v is not None for v in vals):
        return img_bgr, False

    h, w = img_bgr.shape[:2]
    x = _safe_float(crop_x, -1.0)
    y = _safe_float(crop_y, -1.0)
    rw = _safe_float(crop_w, -1.0)
    rh = _safe_float(crop_h, -1.0)

    if max(abs(x), abs(y), abs(rw), abs(rh)) <= 1.5:
        x *= w
        y *= h
        rw *= w
        rh *= h

    x = int(round(x))
    y = int(round(y))
    rw = int(round(rw))
    rh = int(round(rh))
    if rw < 64 or rh < 64:
        return img_bgr, False

    x, y, rw, rh = _clip_rect(x, y, rw, rh, w, h)
    cropped = img_bgr[y : y + rh, x : x + rw]
    if cropped.size == 0:
        return img_bgr, False
    return cropped, True


def _find_page_quad_by_contour(gray_img: np.ndarray) -> Optional[np.ndarray]:
    if gray_img is None or gray_img.size == 0:
        return None

    h, w = gray_img.shape[:2]
    total_area = float(max(1, h * w))

    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        area = float(cv2.contourArea(cnt))
        if area < total_area * 0.18:
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 0.0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_quad_points(pts)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if box is not None and len(box) == 4:
            box = box.astype(np.float32)
            box_area = float(cv2.contourArea(box))
            if box_area >= total_area * 0.16:
                return _order_quad_points(box)

    return None


def _detect_page_quad(gray_img: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    marker_pts = omr_marker_utils._detect_page_corners_from_black_square_markers(gray_img)
    if marker_pts is not None:
        arr = np.asarray(marker_pts, dtype=np.float32).reshape(4, 2)
        arr = np.array([arr[0], arr[1], arr[3], arr[2]], dtype=np.float32)
        return _order_quad_points(arr), "corner-markers"

    contour_pts = _find_page_quad_by_contour(gray_img)
    if contour_pts is not None:
        return contour_pts, "page-contour"

    return None, "none"


def _warp_to_standard_layout(
    img_bgr,
    width_img: int,
    height_img: int,
    a4_warp_w: int,
    a4_warp_h: int,
    manual_quad_norm=None,
):
    src_h, src_w = img_bgr.shape[:2]
    strategy = "resize-only"
    global_warp_used = False
    info: Dict[str, object] = {
        "source": {"width": int(src_w), "height": int(src_h)},
        "target": {"width": int(a4_warp_w), "height": int(a4_warp_h)},
    }

    working = img_bgr

    if manual_quad_norm is not None:
        manual = _warp_by_manual_quad(img_bgr, manual_quad_norm, target_size=(a4_warp_w, a4_warp_h))
        if manual is not None and manual.size > 0:
            working = manual
            strategy = "manual-quad"
            global_warp_used = True
            info["manual_quad"] = list(manual_quad_norm)

    if not global_warp_used:
        gray_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        quad, quad_strategy = _detect_page_quad(gray_src)
        if quad is not None:
            dst = np.array(
                [
                    [0.0, 0.0],
                    [float(a4_warp_w - 1), 0.0],
                    [float(a4_warp_w - 1), float(a4_warp_h - 1)],
                    [0.0, float(a4_warp_h - 1)],
                ],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
            working = cv2.warpPerspective(img_bgr, matrix, (a4_warp_w, a4_warp_h))
            strategy = f"coordinate-global-a4:{quad_strategy}"
            global_warp_used = True
            info["detected_quad"] = _norm_quad_from_points(quad, src_w, src_h)

    interp = cv2.INTER_AREA if working.shape[1] >= width_img else cv2.INTER_CUBIC
    resized = cv2.resize(working, (width_img, height_img), interpolation=interp)
    return resized, strategy, global_warp_used, info


def _binarize(gray_img: np.ndarray, threshold_mode: Optional[str]):
    mode = str(threshold_mode or "otsu").strip().lower()
    if mode not in {"otsu", "weighted_adaptive", "hybrid"}:
        mode = "otsu"

    h, w = gray_img.shape[:2]
    k = max(31, (min(h, w) // 9) | 1)
    bg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    gray_norm = cv2.divide(gray_img, bg, scale=255)
    gray_norm = cv2.GaussianBlur(gray_norm, (3, 3), 0)

    otsu_value, otsu_inv = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_inv = cv2.adaptiveThreshold(
        gray_norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        7,
    )

    if mode == "weighted_adaptive":
        binary_inv = adaptive_inv
    elif mode == "hybrid":
        binary_inv = cv2.bitwise_or(otsu_inv, adaptive_inv)
    else:
        binary_inv = otsu_inv
        mode = "otsu"

    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return gray_norm, binary_inv, {"mode": mode, "otsu_value": float(otsu_value)}
