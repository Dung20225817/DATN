from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import omr_marker_utils
from .omr_utils import _order_quad_points


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

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # xấp xỉ contour thành đa giác ít đỉnh
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_quad_points(pts)

        rect = cv2.minAreaRect(cnt)  # fallback: hình chữ nhật xoay nhỏ nhất bao contour
        box = cv2.boxPoints(rect)  # lấy 4 đỉnh của hình chữ nhật xoay đó
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
):
    src_h, src_w = img_bgr.shape[:2]
    strategy = "resize-only"
    global_warp_used = False
    info: Dict[str, object] = {
        "source": {"width": int(src_w), "height": int(src_h)},
        "target": {"width": int(a4_warp_w), "height": int(a4_warp_h)},
    }

    working = img_bgr

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


def _binarize(gray_img: np.ndarray):
    h, w = gray_img.shape[:2]
    k = max(31, (min(h, w) // 9) | 1)
    bg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    gray_norm = cv2.divide(gray_img, bg, scale=255)
    gray_norm = cv2.GaussianBlur(gray_norm, (3, 3), 0)

    otsu_value, binary_inv = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return gray_norm, binary_inv, {"mode": "otsu", "otsu_value": float(otsu_value)}
