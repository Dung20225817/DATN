# app/services/omr/omr_service.py
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import omr_marker_utils
from .omr_handwriting import (
    _build_handwriting_rois,
    _extract_handwriting_crops,
    _parse_handwriting_config,
)
from .omr_layout import _build_rois_from_anchors, _parse_roi_cfg, _resolve_coordinate_anchors
from .omr_labels import choice_label
from .omr_mcq import (
    _decode_mcq_with_map,
    _detect_q5_start_drift,
    _infer_mcq_geometry_from_markers,
    _mcq_quality,
    _parse_mcq_decode_config,
    build_mcq_roi_from_black_markers,
    refine_mcq_roi,
)
from .omr_numeric import _decode_numeric_columns, _parse_sid_row_offsets
from .omr_preprocess import (
    _binarize,
    _warp_to_standard_layout,
)
from .omr_scoring import _build_answer_compare, _normalize_answer_key
from .omr_visualize import _draw_result_overlay

WIDTH_IMG = 1000
HEIGHT_IMG = 1400
A4_WARP_W = 2480
A4_WARP_H = 3508

def _safe_float(raw, default=0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw, default=0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _safe_bool(raw, default=False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_exam_code_key(raw) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        if len(digits) <= 3:
            return digits.zfill(3)
        return digits

    return text.upper()


def _reject_long_form_low_mcq_top(marker_meta: Dict[str, object], img_h: int) -> bool:
    if not isinstance(marker_meta, dict) or not bool(marker_meta.get("used", False)):
        return False

    y_top = _safe_float(marker_meta.get("y_top_anchor"), -1.0)
    line_h = _safe_float(marker_meta.get("line_h"), float(img_h) * 0.0220)
    if y_top <= 0.0:
        return False

    expected_top = float(img_h) * 0.30
    tolerance = max(2.5 * max(1.0, float(line_h)), float(img_h) * 0.10)
    return y_top > (expected_top + tolerance)


def _validate_mcq_block_bands(
    block_bands: Sequence[Tuple[float, float]],
    mcq_roi: Dict[str, int],
    block_count: int,
    choices: int,
    long_form_mode: bool,
    source: str = "geometry",
) -> Tuple[List[Tuple[float, float]], Dict[str, object]]:
    meta: Dict[str, object] = {
        "used": False,
        "source": str(source),
        "reason": "empty",
        "count": 0,
    }

    if not isinstance(block_bands, (list, tuple)):
        meta["reason"] = "invalid-type"
        return [], meta

    roi_x = float(_safe_int(mcq_roi.get("x"), 0))
    roi_w = float(max(1, _safe_int(mcq_roi.get("w"), 1)))
    roi_right = float(roi_x + roi_w)
    count = max(1, int(block_count))
    choice_count = max(2, int(choices))

    normalized: List[Tuple[float, float]] = []
    for raw in list(block_bands):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        x1 = _safe_float(raw[0], -1.0)
        x2 = _safe_float(raw[1], -1.0)
        if not (math.isfinite(x1) and math.isfinite(x2)):
            continue
        if x2 <= x1:
            continue
        normalized.append((float(x1), float(x2)))

    normalized = sorted(normalized, key=lambda band: float(band[0]))
    meta["count"] = int(len(normalized))
    if len(normalized) < count:
        meta["reason"] = "insufficient-bands"
        return [], meta

    normalized = normalized[:count]
    widths = [float(x2 - x1) for x1, x2 in normalized]
    min_width = max(18.0, 0.012 * roi_w * float(choice_count))
    max_width = max(90.0, 0.32 * roi_w)
    if any(width < min_width or width > max_width for width in widths):
        meta["reason"] = "band-width-out-of-range"
        meta["widths"] = [round(float(width), 4) for width in widths]
        return [], meta

    if len(widths) >= 2:
        median_w = float(np.median(widths))
        if median_w <= 0.0 or max(abs(float(width) - median_w) for width in widths) > (0.55 * median_w):
            meta["reason"] = "band-width-inconsistent"
            meta["widths"] = [round(float(width), 4) for width in widths]
            return [], meta

    slack = 0.06 * roi_w if long_form_mode else 0.10 * roi_w
    if normalized[0][0] < (roi_x - slack) or normalized[-1][1] > (roi_right + slack):
        meta["reason"] = "outside-mcq-roi"
        meta["roi"] = dict(mcq_roi)
        meta["bands"] = [[round(float(a), 4), round(float(b), 4)] for a, b in normalized]
        return [], meta

    overall_span = float(normalized[-1][1] - normalized[0][0])
    min_cover = 0.42 if long_form_mode else 0.30
    max_cover = 0.96 if long_form_mode else 1.05
    if overall_span < (min_cover * roi_w) or overall_span > (max_cover * roi_w):
        meta["reason"] = "overall-span-out-of-range"
        meta["overall_span"] = round(float(overall_span), 4)
        meta["roi_w"] = round(float(roi_w), 4)
        return [], meta

    starts = [float(band[0]) for band in normalized]
    gaps = [starts[idx + 1] - starts[idx] for idx in range(len(starts) - 1)]
    if gaps:
        min_gap = 0.08 * roi_w
        max_gap = 0.40 * roi_w
        if any(gap < min_gap or gap > max_gap for gap in gaps):
            meta["reason"] = "band-gap-out-of-range"
            meta["gaps"] = [round(float(gap), 4) for gap in gaps]
            return [], meta

    meta.update(
        {
            "used": True,
            "reason": "ok",
            "bands": [[round(float(a), 4), round(float(b), 4)] for a, b in normalized],
            "overall_span": round(float(overall_span), 4),
        }
    )
    return normalized, meta


def _derive_mcq_center_bands_from_span(
    left_x: float,
    right_x: float,
    mcq_roi: Dict[str, int],
    block_count: int,
    choices: int,
) -> List[Tuple[float, float]]:
    del mcq_roi
    count = max(1, int(block_count))
    choice_count = max(2, int(choices))
    left = float(left_x)
    right = float(right_x)
    if not (math.isfinite(left) and math.isfinite(right)) or right <= left:
        return []

    raw_band_w = float(right - left) / float(count)
    if raw_band_w < 24.0:
        return []

    bands: List[Tuple[float, float]] = []
    for idx in range(count):
        raw_left = float(left + idx * raw_band_w)
        raw_right = float(left + (idx + 1) * raw_band_w)
        raw_span = max(8.0, float(raw_right - raw_left))
        band_left = float(raw_left + 0.08 * raw_span)
        band_right = float(raw_right - 0.04 * raw_span)
        choice_w = max(4.0, float(band_right - band_left) / float(choice_count))
        center_left = float(band_left + 0.5 * choice_w)
        center_right = float(band_left + (float(choice_count) - 0.5) * choice_w)
        if center_right > center_left:
            bands.append((center_left, center_right))
    return bands


def _infer_long_form_answer_center_bands(
    binary_inv,
    mcq_roi: Dict[str, int],
    top_center_y: float,
    line_h: float,
    block_count: int,
    choices: int,
    img_w: int,
    img_h: int,
    sample_rows: int = 8,
) -> Tuple[List[Tuple[float, float]], Dict[str, object]]:
    meta: Dict[str, object] = {
        "used": False,
        "reason": "init",
        "rows_sampled": 0,
        "rows_used": 0,
    }
    if binary_inv is None or getattr(binary_inv, "size", 0) <= 0:
        meta["reason"] = "invalid-binary"
        return [], meta

    count = max(1, int(block_count))
    choice_count = max(2, int(choices))
    line = max(6.0, float(line_h))
    roi_x = float(_safe_int(mcq_roi.get("x"), 0))
    roi_y = float(_safe_int(mcq_roi.get("y"), 0))
    roi_w = float(max(1, _safe_int(mcq_roi.get("w"), img_w)))
    roi_h = float(max(1, _safe_int(mcq_roi.get("h"), img_h)))

    search_pad_left = max(20.0, 0.035 * roi_w)
    search_pad_right = max(80.0, 0.14 * roi_w)
    sx1 = int(round(max(0.0, roi_x - search_pad_left)))
    sx2 = int(round(min(float(img_w), roi_x + roi_w + search_pad_right)))
    if sx2 <= sx1:
        meta["reason"] = "invalid-search-x"
        return [], meta

    row_limit = max(1, min(int(sample_rows), 12))
    row_bands: List[List[Tuple[float, float]]] = []
    row_debug: List[Dict[str, object]] = []
    min_blob_w = max(15.0, 0.42 * line)
    max_blob_w = max(32.0, 1.55 * line)
    min_blob_h = max(12.0, 0.36 * line)
    max_blob_h = max(28.0, 1.35 * line)

    for row_idx in range(row_limit):
        cy = float(top_center_y + float(row_idx) * line)
        if cy < (roi_y - line) or cy > (roi_y + roi_h + line):
            continue
        y1 = int(round(max(0.0, cy - 0.48 * line)))
        y2 = int(round(min(float(img_h), cy + 0.48 * line)))
        if y2 <= y1:
            continue

        crop = binary_inv[y1:y2, sx1:sx2]
        if crop is None or crop.size == 0:
            continue
        meta["rows_sampled"] = int(meta["rows_sampled"]) + 1

        mask = (crop > 0).astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        centers: List[float] = []
        for label_idx in range(1, int(num_labels)):
            bx = int(stats[label_idx, cv2.CC_STAT_LEFT])
            by = int(stats[label_idx, cv2.CC_STAT_TOP])
            bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if bw < min_blob_w or bw > max_blob_w:
                continue
            if bh < min_blob_h or bh > max_blob_h:
                continue
            if area < 55 or area > 900:
                continue
            fill = float(area) / float(max(1, bw * bh))
            if fill < 0.18 or fill > 0.92:
                continue
            cx = float(sx1 + centroids[label_idx][0])
            if cx < (roi_x - search_pad_left) or cx > (roi_x + roi_w + search_pad_right):
                continue
            centers.append(float(cx))

        centers = sorted(centers)
        if len(centers) < count * choice_count:
            row_debug.append({"row": int(row_idx + 1), "centers": [round(float(v), 2) for v in centers], "accepted": False})
            continue

        diffs = [centers[idx + 1] - centers[idx] for idx in range(len(centers) - 1)]
        step_candidates = [float(val) for val in diffs if 16.0 <= float(val) <= 42.0]
        if not step_candidates:
            row_debug.append({"row": int(row_idx + 1), "centers": [round(float(v), 2) for v in centers], "accepted": False})
            continue
        step = float(np.median(step_candidates))
        split_gap = max(46.0, 1.60 * float(step))

        groups: List[List[float]] = []
        current = [float(centers[0])]
        for val in centers[1:]:
            if float(val) - current[-1] > split_gap:
                groups.append(current)
                current = [float(val)]
            else:
                current.append(float(val))
        groups.append(current)

        row_group_bands: List[Tuple[float, float]] = []
        for group in groups:
            if len(group) < choice_count:
                continue
            if len(group) > choice_count:
                run = [float(v) for v in group[-choice_count:]]
                run_diffs = [run[idx + 1] - run[idx] for idx in range(len(run) - 1)]
                median_run = float(np.median(run_diffs)) if run_diffs else 0.0
                if 16.0 <= median_run <= 42.0 and float(np.std(run_diffs)) <= 8.0:
                    row_group_bands.append((float(run[0]), float(run[-1])))
                    continue
            best_run: Optional[List[float]] = None
            best_score = float("inf")
            for start_idx in range(0, len(group) - choice_count + 1):
                run = [float(v) for v in group[start_idx : start_idx + choice_count]]
                run_diffs = [run[idx + 1] - run[idx] for idx in range(len(run) - 1)]
                if not run_diffs:
                    continue
                median_run = float(np.median(run_diffs))
                if median_run < 16.0 or median_run > 42.0:
                    continue
                score = float(np.std(run_diffs)) + abs(median_run - step) * 0.20
                if score < best_score:
                    best_score = score
                    best_run = run
            if best_run is not None:
                row_group_bands.append((float(best_run[0]), float(best_run[-1])))

        if len(row_group_bands) >= count:
            row_bands.append(row_group_bands[:count])
            row_debug.append(
                {
                    "row": int(row_idx + 1),
                    "centers": [round(float(v), 2) for v in centers],
                    "bands": [[round(float(a), 2), round(float(b), 2)] for a, b in row_group_bands[:count]],
                    "accepted": True,
                }
            )
        else:
            row_debug.append(
                {
                    "row": int(row_idx + 1),
                    "centers": [round(float(v), 2) for v in centers],
                    "groups": [[round(float(v), 2) for v in group] for group in groups],
                    "accepted": False,
                }
            )

    meta["rows_used"] = int(len(row_bands))
    meta["row_debug"] = row_debug[:6]
    if len(row_bands) < max(2, int(math.ceil(0.35 * float(row_limit)))):
        meta["reason"] = "insufficient-valid-rows"
        return [], meta

    final_bands: List[Tuple[float, float]] = []
    for block_idx in range(count):
        lefts = [float(row[block_idx][0]) for row in row_bands if len(row) > block_idx]
        rights = [float(row[block_idx][1]) for row in row_bands if len(row) > block_idx]
        if len(lefts) < 2 or len(rights) < 2:
            meta["reason"] = "insufficient-block-samples"
            return [], meta
        final_bands.append((float(np.median(lefts)), float(np.median(rights))))

    meta.update(
        {
            "used": True,
            "reason": "ok",
            "bands": [[round(float(a), 4), round(float(b), 4)] for a, b in final_bands],
        }
    )
    return final_bands, meta


def _expand_mcq_roi_for_bands(
    mcq_roi: Dict[str, int],
    block_bands: Sequence[Tuple[float, float]],
    img_w: int,
    img_h: int,
    line_h: float,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    meta: Dict[str, object] = {"used": False, "reason": "no-bands"}
    if not block_bands:
        return dict(mcq_roi), meta

    x = _safe_int(mcq_roi.get("x"), 0)
    y = _safe_int(mcq_roi.get("y"), 0)
    w = max(1, _safe_int(mcq_roi.get("w"), img_w))
    h = max(1, _safe_int(mcq_roi.get("h"), img_h))
    roi_right = int(x + w)
    min_x = min(float(a) for a, _ in block_bands)
    max_x = max(float(b) for _, b in block_bands)
    pad = max(28.0, 1.15 * max(6.0, float(line_h)))
    new_x = int(round(max(0.0, min(float(x), min_x - pad))))
    new_right = int(round(min(float(img_w), max(float(roi_right), max_x + pad))))
    if new_x == x and new_right == roi_right:
        meta["reason"] = "unchanged"
        return dict(mcq_roi), meta

    expanded = {
        "x": int(new_x),
        "y": int(max(0, min(y, img_h - 1))),
        "w": int(max(1, new_right - new_x)),
        "h": int(max(1, min(img_h - int(y), h))),
    }
    meta.update(
        {
            "used": True,
            "reason": "expanded",
            "old_roi": dict(mcq_roi),
            "new_roi": dict(expanded),
            "band_min_x": round(float(min_x), 4),
            "band_max_x": round(float(max_x), 4),
        }
    )
    return expanded, meta


def _infer_long_form_row_offsets(
    binary_inv,
    mcq_roi: Dict[str, int],
    block_bands: Sequence[Tuple[float, float]],
    top_center_y: float,
    line_h: float,
    rows_per_block: int,
    choices: int,
    img_w: int,
    img_h: int,
) -> Tuple[List[int], Dict[str, object]]:
    meta: Dict[str, object] = {
        "used": False,
        "source": "none",
        "reason": "skipped",
        "row_y_centers": [],
        "row_offsets_px": [],
    }
    rows = max(1, int(rows_per_block))
    choice_count = max(2, int(choices))
    line = max(6.0, float(line_h))

    if rows != 30:
        meta["reason"] = "unsupported-rows-per-block"
        return [], meta

    fallback_offsets = [int(round(float(idx // 10) * line)) for idx in range(rows)]

    if binary_inv is None or getattr(binary_inv, "size", 0) <= 0 or not block_bands:
        meta.update(
            {
                "used": True,
                "source": "fallback-10-10-10",
                "reason": "missing-binary-or-bands",
                "row_offsets_px": fallback_offsets,
            }
        )
        return fallback_offsets, meta

    first_band = tuple(block_bands[0])
    band_left = float(first_band[0])
    band_right = float(first_band[1])
    if not (math.isfinite(band_left) and math.isfinite(band_right) and band_right > band_left):
        meta.update(
            {
                "used": True,
                "source": "fallback-10-10-10",
                "reason": "invalid-first-band",
                "row_offsets_px": fallback_offsets,
            }
        )
        return fallback_offsets, meta

    x_pad = max(18.0, 0.70 * line)
    sx1 = int(round(max(0.0, band_left - x_pad)))
    sx2 = int(round(min(float(img_w), band_right + x_pad)))
    roi_y = float(_safe_int(mcq_roi.get("y"), 0))
    roi_h = float(max(1, _safe_int(mcq_roi.get("h"), img_h)))
    y_pad = max(18.0, 0.65 * line)
    sy1 = int(round(max(0.0, roi_y - y_pad)))
    sy2 = int(round(min(float(img_h), roi_y + roi_h + y_pad)))
    if sx2 <= sx1 or sy2 <= sy1:
        meta.update(
            {
                "used": True,
                "source": "fallback-10-10-10",
                "reason": "invalid-search-window",
                "row_offsets_px": fallback_offsets,
            }
        )
        return fallback_offsets, meta

    crop = binary_inv[sy1:sy2, sx1:sx2]
    if crop is None or crop.size == 0:
        meta.update(
            {
                "used": True,
                "source": "fallback-10-10-10",
                "reason": "empty-search-window",
                "row_offsets_px": fallback_offsets,
            }
        )
        return fallback_offsets, meta

    mask = (crop > 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_blob_w = max(14.0, 0.40 * line)
    max_blob_w = max(34.0, 1.60 * line)
    min_blob_h = max(10.0, 0.34 * line)
    max_blob_h = max(30.0, 1.40 * line)
    points: List[Tuple[float, float]] = []
    for label_idx in range(1, int(num_labels)):
        bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if bw < min_blob_w or bw > max_blob_w:
            continue
        if bh < min_blob_h or bh > max_blob_h:
            continue
        if area < 45 or area > 900:
            continue
        fill = float(area) / float(max(1, bw * bh))
        if fill < 0.16 or fill > 0.94:
            continue
        cx = float(sx1 + centroids[label_idx][0])
        cy = float(sy1 + centroids[label_idx][1])
        if cx < (band_left - x_pad) or cx > (band_right + x_pad):
            continue
        points.append((float(cx), float(cy)))

    points = sorted(points, key=lambda pt: float(pt[1]))
    clusters: List[List[Tuple[float, float]]] = []
    y_tol = max(7.0, 0.34 * line)
    for pt in points:
        if not clusters or abs(float(pt[1]) - float(np.median([p[1] for p in clusters[-1]]))) > y_tol:
            clusters.append([pt])
        else:
            clusters[-1].append(pt)

    row_centers: List[float] = []
    row_debug: List[Dict[str, object]] = []
    for cluster in clusters:
        xs = sorted(float(pt[0]) for pt in cluster)
        ys = [float(pt[1]) for pt in cluster]
        accepted = len(cluster) >= max(3, choice_count - 1)
        if accepted:
            row_centers.append(float(np.median(ys)))
        if len(row_debug) < 40:
            row_debug.append(
                {
                    "y": round(float(np.median(ys)), 4),
                    "count": int(len(cluster)),
                    "xs": [round(float(x), 2) for x in xs],
                    "accepted": bool(accepted),
                }
            )

    if len(row_centers) >= rows:
        row_centers = row_centers[:rows]
        offsets = [
            int(round(float(row_centers[idx]) - float(top_center_y + idx * line)))
            for idx in range(rows)
        ]
        meta.update(
            {
                "used": True,
                "source": "detected-bubbles",
                "reason": "ok",
                "row_y_centers": [round(float(y), 4) for y in row_centers],
                "row_offsets_px": offsets,
                "row_debug": row_debug,
            }
        )
        return offsets, meta

    meta.update(
        {
            "used": True,
            "source": "fallback-10-10-10",
            "reason": "insufficient-detected-rows",
            "detected_rows": int(len(row_centers)),
            "row_y_centers": [round(float(y), 4) for y in row_centers],
            "row_offsets_px": fallback_offsets,
            "row_debug": row_debug,
        }
    )
    return fallback_offsets, meta


def _parse_numeric_value(value: str, digits: int) -> Optional[List[int]]:
    text = str(value or "").strip()
    if len(text) != int(max(1, digits)):
        return None
    if "?" in text or (not text.isdigit()):
        return None
    return [int(ch) for ch in text]


def _numeric_shift_delta(primary_value: str, alt_value: str, digits: int) -> Optional[int]:
    primary_digits = _parse_numeric_value(primary_value, digits)
    alt_digits = _parse_numeric_value(alt_value, digits)
    if primary_digits is None or alt_digits is None:
        return None

    deltas = {(int(alt) - int(base)) % 10 for base, alt in zip(primary_digits, alt_digits)}
    if len(deltas) != 1:
        return None
    return int(next(iter(deltas)))


def _build_choice_labels(num_choices: int) -> List[str]:
    labels: List[str] = []
    for i in range(max(1, int(num_choices))):
        if i < 26:
            labels.append(chr(ord("A") + i))
        else:
            labels.append(str(i + 1))
    return labels


def _sanitize_filename_part(text: str, fallback: str = "omr") -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip())
    safe = safe.strip("_")
    return safe[:60] if safe else fallback


def generate_omr_template(
    output_folder: str,
    exam_title: str,
    total_questions: int,
    options: int,
    student_id_digits: int,
    rows_per_block: int = 20,
    num_blocks=None,
    exam_code: str = "000",
    info_fields=None,
):
    """Generate blank OMR template image used by /api/omr/template."""
    width_img = 1000
    height_img = 1400

    total_questions = max(1, int(total_questions))
    options = max(2, int(options))
    student_id_digits = max(1, int(student_id_digits))
    rows_per_block = max(1, int(rows_per_block))
    exam_code = str(exam_code or "000").strip()
    if not re.fullmatch(r"\d{3}", exam_code):
        exam_code = "000"
    if not isinstance(info_fields, list):
        info_fields = []
    info_fields = [str(x).strip() for x in info_fields if str(x).strip()]

    if num_blocks is None:
        block_count = max(1, int(math.ceil(total_questions / float(rows_per_block))))
    else:
        block_count = max(1, int(num_blocks))
    if block_count * rows_per_block < total_questions:
        block_count = int(math.ceil(total_questions / float(rows_per_block)))

    os.makedirs(output_folder, exist_ok=True)

    img = np.full((height_img, width_img, 3), 255, dtype=np.uint8)
    black = (0, 0, 0)

    # Four corner markers for alignment.
    marker_w, marker_h = 34, 22
    cv2.rectangle(img, (24, 22), (24 + marker_w, 22 + marker_h), black, -1)
    cv2.rectangle(img, (width_img - 24 - marker_w, 22), (width_img - 24, 22 + marker_h), black, -1)
    cv2.rectangle(img, (24, height_img - 22 - marker_w), (24 + marker_w, height_img - 22), black, -1)
    cv2.rectangle(
        img,
        (width_img - 24 - marker_w, height_img - 22 - marker_w),
        (width_img - 24, height_img - 22),
        black,
        -1,
    )

    title = (exam_title or "OMR Exam").strip()
    cv2.putText(img, title[:80], (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, black, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"Total Questions: {total_questions} | Options: {options}",
        (70, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        black,
        2,
        cv2.LINE_AA,
    )

    # Student ID panel.
    sid_x = int(width_img * 0.085)
    sid_y = int(height_img * 0.165)
    sid_w = int(width_img * 0.255)
    sid_h = int(height_img * 0.225)
    cv2.rectangle(img, (sid_x, sid_y), (sid_x + sid_w, sid_y + sid_h), black, 3)
    cv2.putText(img, "Student ID", (sid_x + 8, sid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, black, 2, cv2.LINE_AA)

    write_row_h = max(24, int(sid_h * 0.16))
    sid_col_edges = np.linspace(sid_x, sid_x + sid_w, student_id_digits + 1, dtype=int)
    for c in range(student_id_digits):
        sx1, sx2 = int(sid_col_edges[c]), int(sid_col_edges[c + 1])
        pad = max(2, int((sx2 - sx1) * 0.18))
        cv2.rectangle(
            img,
            (sx1 + pad, sid_y + 5),
            (sx2 - pad, sid_y + write_row_h - 5),
            (30, 30, 30),
            1,
        )

    bubble_top = sid_y + write_row_h
    sid_row_edges = np.linspace(bubble_top, sid_y + sid_h, 10 + 1, dtype=int)
    for y in sid_row_edges[1:-1]:
        cv2.line(img, (sid_x, int(y)), (sid_x + sid_w, int(y)), (70, 70, 70), 1)
    for x in sid_col_edges[1:-1]:
        cv2.line(img, (int(x), bubble_top), (int(x), sid_y + sid_h), (70, 70, 70), 1)
    cv2.line(img, (sid_x, bubble_top), (sid_x + sid_w, bubble_top), (70, 70, 70), 1)

    for r in range(10):
        y1, y2 = int(sid_row_edges[r]), int(sid_row_edges[r + 1])
        cy = y1 + (y2 - y1) // 2
        cv2.putText(img, str(r), (sid_x - 16, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1, cv2.LINE_AA)
        for c in range(student_id_digits):
            x1, x2 = int(sid_col_edges[c]), int(sid_col_edges[c + 1])
            cx = x1 + (x2 - x1) // 2
            radius = max(5, min(9, int((y2 - y1) * 0.32)))
            cv2.circle(img, (cx, cy), radius, (45, 45, 45), 1)

    # Exam code panel.
    code_x = sid_x + sid_w + int(width_img * 0.045)
    code_y = sid_y
    code_w = int(width_img * 0.12)
    code_h = sid_h
    cv2.rectangle(img, (code_x, code_y), (code_x + code_w, code_y + code_h), black, 3)
    cv2.putText(img, "Ma de", (code_x + 6, code_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, black, 2, cv2.LINE_AA)

    code_write_h = max(24, int(code_h * 0.16))
    code_col_edges = np.linspace(code_x, code_x + code_w, 3 + 1, dtype=int)
    for c in range(3):
        cx1, cx2 = int(code_col_edges[c]), int(code_col_edges[c + 1])
        pad = max(2, int((cx2 - cx1) * 0.18))
        cv2.rectangle(
            img,
            (cx1 + pad, code_y + 5),
            (cx2 - pad, code_y + code_write_h - 5),
            (30, 30, 30),
            1,
        )

    code_bubble_top = code_y + code_write_h
    code_row_edges = np.linspace(code_bubble_top, code_y + code_h, 10 + 1, dtype=int)
    for y in code_row_edges[1:-1]:
        cv2.line(img, (code_x, int(y)), (code_x + code_w, int(y)), (70, 70, 70), 1)
    for x in code_col_edges[1:-1]:
        cv2.line(img, (int(x), code_bubble_top), (int(x), code_y + code_h), (70, 70, 70), 1)
    cv2.line(img, (code_x, code_bubble_top), (code_x + code_w, code_bubble_top), (70, 70, 70), 1)

    for r in range(10):
        y1, y2 = int(code_row_edges[r]), int(code_row_edges[r + 1])
        cy = y1 + (y2 - y1) // 2
        cv2.putText(img, str(r), (code_x - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1, cv2.LINE_AA)
        for c in range(3):
            x1, x2 = int(code_col_edges[c]), int(code_col_edges[c + 1])
            cx = x1 + (x2 - x1) // 2
            radius = max(5, min(9, int((y2 - y1) * 0.32)))
            cv2.circle(img, (cx, cy), radius, (45, 45, 45), 1)

    if info_fields:
        info_x = code_x + code_w + int(width_img * 0.045)
        info_y = sid_y
        info_w = min(int(width_img * 0.36), width_img - info_x - 30)
        info_h = sid_h
        if info_w > 120:
            cv2.rectangle(img, (info_x, info_y), (info_x + info_w, info_y + info_h), black, 3)
            rows = len(info_fields)
            row_h = max(24, int(info_h / max(1, rows)))
            for idx, field_name in enumerate(info_fields):
                y1 = info_y + idx * row_h
                y2 = info_y + info_h if idx == rows - 1 else info_y + (idx + 1) * row_h
                if idx > 0:
                    cv2.line(img, (info_x, y1), (info_x + info_w, y1), (90, 90, 90), 1)
                cv2.putText(
                    img,
                    f"{field_name}:",
                    (info_x + 8, y1 + max(18, min(28, row_h - 6))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (35, 35, 35),
                    1,
                    cv2.LINE_AA,
                )
                line_y = min(y2 - 8, y1 + max(20, row_h - 8))
                cv2.line(img, (info_x + 110, line_y), (info_x + info_w - 8, line_y), (130, 130, 130), 1)

    # MCQ frame.
    mcq_x = int(width_img * 0.075)
    mcq_y = int(height_img * 0.430)
    mcq_w = int(width_img * 0.855)
    mcq_h = int(height_img * 0.470)
    cv2.rectangle(img, (mcq_x, mcq_y), (mcq_x + mcq_w, mcq_y + mcq_h), black, 3)

    block_gap = int(mcq_w * 0.02) if block_count > 1 else 0
    total_gap = block_gap * max(0, block_count - 1)
    usable_w = max(1, mcq_w - total_gap)
    block_w = max(1, int(usable_w / block_count))
    choice_labels = _build_choice_labels(options)

    q_counter = 1
    for block_idx in range(block_count):
        block_start_q = block_idx * rows_per_block
        rows_in_block = min(rows_per_block, total_questions - block_start_q)
        if rows_in_block <= 0:
            break

        bx = block_idx * (block_w + block_gap)
        bx2 = mcq_w if block_idx == block_count - 1 else min(bx + block_w, mcq_w)
        abs_x1 = mcq_x + bx
        abs_x2 = mcq_x + bx2

        cv2.rectangle(img, (abs_x1, mcq_y), (abs_x2, mcq_y + mcq_h), black, 2)

        b_w = max(1, bx2 - bx)
        bubble_left = int(b_w * 0.16)
        bubble_right = int(b_w * 0.95)
        bubble_x1 = abs_x1 + bubble_left
        bubble_x2 = abs_x1 + bubble_right
        cv2.line(img, (bubble_x1, mcq_y), (bubble_x1, mcq_y + mcq_h), (70, 70, 70), 1)

        row_edges = np.linspace(mcq_y, mcq_y + mcq_h, rows_in_block + 1, dtype=int)
        col_edges = np.linspace(bubble_x1, bubble_x2, options + 1, dtype=int)

        for r in range(rows_in_block):
            ry1, ry2 = int(row_edges[r]), int(row_edges[r + 1])
            cy = ry1 + (ry2 - ry1) // 2
            cv2.line(img, (abs_x1, ry1), (abs_x2, ry1), (90, 90, 90), 1)
            cv2.putText(
                img,
                str(q_counter),
                (abs_x1 + 5, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )
            for c in range(options):
                cx1, cx2 = int(col_edges[c]), int(col_edges[c + 1])
                cx = cx1 + (cx2 - cx1) // 2
                radius = max(6, min(10, int((ry2 - ry1) * 0.25)))
                cv2.circle(img, (cx, cy), radius, (45, 45, 45), 1)
                if r == 0:
                    cv2.putText(
                        img,
                        choice_labels[c],
                        (cx - 7, mcq_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (35, 35, 35),
                        1,
                        cv2.LINE_AA,
                    )
            q_counter += 1

        cv2.line(img, (abs_x1, mcq_y + mcq_h), (abs_x2, mcq_y + mcq_h), (90, 90, 90), 1)

    cv2.putText(
        img,
        "Fill bubbles completely with dark pen. One choice per question.",
        (70, height_img - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = _sanitize_filename_part(title, fallback="exam")
    out_name = f"omr_template_{safe_title}_{stamp}.png"
    out_path = os.path.join(output_folder, out_name)
    cv2.imwrite(out_path, img)

    return {
        "success": True,
        "template_image": out_name,
        "mcq_layout": {
            "questions": total_questions,
            "choices": options,
            "rows_per_block": rows_per_block,
            "num_blocks": block_count,
        },
        "student_id_digits": student_id_digits,
    }


def process_omr_exam(
    image_path: str,
    output_folder: str,
    answer_key: list,
    answer_key_by_code: Optional[Dict[str, List[int]]] = None,
    questions=80,
    choices=5,
    rows_per_block=20,
    num_blocks=None,
    student_id_digits=6,
    sid_has_write_row=False,
    profile_sid_roi=None,
    profile_sid_roi_lock=False,
    profile_mcq_roi=None,
    profile_exam_code_roi=None,
    profile_sid_row_offsets=None,
    profile_disable_mcq_rescue=False,
    density_only_scoring=False,
    profile_mcq_decode=None,
    profile_threshold_mode=None,
    profile_corner_markers=None,
    profile_scanner_hint=None,
    profile_page_size_pt=None,
    profile_handwriting_fields=None,
    _internal_retry=False,
):
    del profile_corner_markers
    del profile_scanner_hint
    del _internal_retry

    try:
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            return {"error": "Khong the doc file anh"}

        os.makedirs(output_folder, exist_ok=True)

        questions = max(1, _safe_int(questions, 80))
        choices = max(2, _safe_int(choices, 5))
        rows_per_block = max(1, _safe_int(rows_per_block, 20))
        student_id_digits = max(1, _safe_int(student_id_digits, 6))
        sid_has_write_row = bool(sid_has_write_row)

        if num_blocks is None:
            block_count = max(1, int(math.ceil(float(questions) / float(rows_per_block))))
        else:
            block_count = max(1, _safe_int(num_blocks, 1))
        inferred_block_count = 0

        warnings: List[str] = []
        warning_codes: List[str] = []

        img_input = img_raw

        img_std, warp_strategy, global_warp_used, warp_info = _warp_to_standard_layout(
            img_input,
            width_img=WIDTH_IMG,
            height_img=HEIGHT_IMG,
            a4_warp_w=A4_WARP_W,
            a4_warp_h=A4_WARP_H,
        )

        if not global_warp_used:
            warnings.append("Khong tim thay 4 goc marker ro rang, da fallback ve resize thong thuong.")
            warning_codes.append("COORD_GLOBAL_WARP_FALLBACK")

        gray = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
        gray_norm, binary_inv, threshold_meta = _binarize(gray, profile_threshold_mode)

        sid_roi_cfg = _parse_roi_cfg(profile_sid_roi, WIDTH_IMG, HEIGHT_IMG)
        code_roi_cfg = _parse_roi_cfg(profile_exam_code_roi, WIDTH_IMG, HEIGHT_IMG)
        mcq_roi_cfg = _parse_roi_cfg(profile_mcq_roi, WIDTH_IMG, HEIGHT_IMG)

        markers = omr_marker_utils._extract_black_square_markers_from_gray(
            gray_norm,
            min_area_ratio=0.00002,
            max_area_ratio=0.012,
            max_markers=320,
        )
        mcq_geometry = _infer_mcq_geometry_from_markers(
            markers,
            WIDTH_IMG,
            HEIGHT_IMG,
            rows_per_block=rows_per_block,
            block_count_hint=block_count,
        )

        inferred_block_count = _safe_int(mcq_geometry.get("block_count"), 0)
        geom_bands = mcq_geometry.get("block_bands")
        if inferred_block_count <= 1 and isinstance(geom_bands, (list, tuple)):
            inferred_block_count = max(0, len(list(geom_bands)))
        if num_blocks is None and inferred_block_count >= 2:
            # Accept fiducial inference if:
            # 1. Difference is small (<=1), OR
            # 2. Fiducial count is smaller (fewer blocks → more rows per block),
            #    and dividing questions by inferred count gives integer rows_per_block,
            #    e.g. 120q / 4 blocks = 30 rows (vs default 20 rows with 6 blocks inferred)
            inferred_rows_per_block = int(math.ceil(float(questions) / float(inferred_block_count)))
            current_inferred_rows = int(math.ceil(float(questions) / float(max(1, block_count))))
            fid_count_trusted = (
                block_count <= 1
                or questions <= rows_per_block
                or abs(inferred_block_count - block_count) <= 1
                or (
                    inferred_block_count < block_count
                    and inferred_block_count >= 2
                    and (questions % inferred_block_count == 0 or inferred_rows_per_block <= 40)
                )
            )
            if fid_count_trusted:
                block_count = int(inferred_block_count)
                rows_per_block = inferred_rows_per_block
                warning_codes.append("MCQ_BLOCKS_INFERRED")
                warnings.append(f"Tu dong suy luan so khoi MCQ = {block_count} (rows_per_block={rows_per_block}) tu marker fiducial.")

        long_form_mode = bool(rows_per_block >= 25 or block_count >= 4)
        if long_form_mode and sid_roi_cfg is not None and not bool(profile_sid_roi_lock):
            sid_roi_cfg = None
            warning_codes.append("SID_PROFILE_ROI_IGNORED_LONG_FORM")
            warnings.append("Long-form: bo qua ROI SID trong profile de suy luan theo marker neo.")

        anchors, anchor_fallback_used = _resolve_coordinate_anchors(
            WIDTH_IMG,
            HEIGHT_IMG,
            markers,
            rows_per_block,
            sid_roi_cfg,
            code_roi_cfg,
            mcq_roi_cfg,
            block_count_hint=block_count,
        )

        if anchor_fallback_used:
            warnings.append("Thieu marker neo cuc bo, da fallback ve anchor theo ty le template.")
            warning_codes.append("COORD_ANCHOR_FALLBACK")

        roi_boxes, line_h_anchor = _build_rois_from_anchors(
            anchors,
            WIDTH_IMG,
            HEIGHT_IMG,
            rows_per_block,
            sid_roi_cfg,
            code_roi_cfg,
            mcq_roi_cfg,
            block_count_hint=block_count,
        )

        sid_roi = roi_boxes["sid"]
        code_roi = roi_boxes["code"]
        mcq_roi = roi_boxes["mcq"]

        if sid_roi_cfg is None:
            warning_codes.append("SID_AUTO_ROI_INSET_APPLIED")
            warnings.append("Tu dong thu hep ROI SID vao ben trong anchor tren/duoi va hai canh ben.")

        mcq_block_bands_raw: List[Tuple[float, float]] = []
        if isinstance(geom_bands, (list, tuple)):
            for item in list(geom_bands):
                if not isinstance(item, dict):
                    continue
                x1 = _safe_float(item.get("x_min"), -1.0)
                x2 = _safe_float(item.get("x_max"), -1.0)
                if x2 <= x1:
                    continue
                mcq_block_bands_raw.append((float(x1), float(x2)))
        mcq_block_bands: List[Tuple[float, float]] = list(mcq_block_bands_raw)
        mcq_block_bands, mcq_block_bands_meta = _validate_mcq_block_bands(
            mcq_block_bands,
            mcq_roi,
            block_count,
            choices,
            long_form_mode,
            source="geometry",
        )
        mcq_block_bands_initial_meta = dict(mcq_block_bands_meta)
        mcq_block_bands_revalidated_after_roi = False

        geom_top_center = _safe_float(mcq_geometry.get("top_center_y"), -1.0)
        geom_line_h = _safe_float(mcq_geometry.get("line_h"), -1.0)
        mcq_marker_roi_meta: Dict[str, object] = {"used": False, "reason": "skipped"}
        if mcq_roi_cfg is None:
            marker_roi, mcq_marker_roi_meta = build_mcq_roi_from_black_markers(
                mcq_roi=mcq_roi,
                mcq_geometry=mcq_geometry,
                img_w=WIDTH_IMG,
                img_h=HEIGHT_IMG,
                top_padding_px=5,
                side_padding_px=10,
                bottom_padding_px=15,
            )
            if bool(mcq_marker_roi_meta.get("used", False)):
                if long_form_mode and _reject_long_form_low_mcq_top(mcq_marker_roi_meta, HEIGHT_IMG):
                    mcq_marker_roi_meta = {
                        **mcq_marker_roi_meta,
                        "used": False,
                        "rejected": True,
                        "reason": "long-form-top-fid-too-low",
                    }
                    warnings.append("Long-form: bo qua ROI MCQ theo marker do top fiducial thap bat thuong.")
                    warning_codes.append("MCQ_LONG_FORM_TOP_FID_REJECTED")
                else:
                    mcq_roi = dict(marker_roi)
                    warnings.append("Da dat ROI MCQ theo cac o vuong den cua khung MCQ.")
                    warning_codes.append("MCQ_BLACK_MARKER_ROI_APPLIED")

        mcq_refine_meta: Dict[str, object] = {"used": False, "reason": "skipped"}
        mcq_anchor_height_meta: Dict[str, object] = {"used": False, "reason": "derived-from-mcq-refine"}
        mcq_crop_clean = img_std[
            mcq_roi["y"] : mcq_roi["y"] + mcq_roi["h"],
            mcq_roi["x"] : mcq_roi["x"] + mcq_roi["w"],
        ]
        if mcq_roi_cfg is None:
            mcq_roi, mcq_crop_clean, mcq_refine_meta = refine_mcq_roi(
                source_img=img_std,
                gray_img=gray_norm,
                mcq_roi=mcq_roi,
                top_padding_px=5,
                side_padding_px=10,
                bottom_padding_px=15,
            )

            if bool(mcq_refine_meta.get("used", False)):
                warnings.append("Da refine ROI MCQ bang template matching anchor de loai bo vung chu huong dan.")
                warning_codes.append("MCQ_TEMPLATE_REFINE_APPLIED")

                refined_top = _safe_float(mcq_refine_meta.get("y_anchor_top"), -1.0)
                refined_bottom = _safe_float(mcq_refine_meta.get("y_anchor_bottom"), -1.0)
                refined_line_h = _safe_float(mcq_refine_meta.get("line_h"), -1.0)
                if refined_top > 0.0 and refined_bottom > refined_top:
                    mcq_geometry["top_center_y"] = float(refined_top)
                    mcq_geometry["bottom_center_y"] = float(refined_bottom)
                if refined_line_h > 0.0:
                    roi_line_h_hint = float(mcq_roi["h"]) / max(2.0, float(rows_per_block))
                    min_line_h = 0.60 * float(roi_line_h_hint)
                    max_line_h = 1.35 * float(roi_line_h_hint)
                    if min_line_h <= refined_line_h <= max_line_h:
                        mcq_geometry["line_h"] = float(refined_line_h)
                    else:
                        warning_codes.append("MCQ_REFINE_LINEH_REJECTED")
                        warnings.append("Bo qua line-height tu refine ROI MCQ do vuot nguong hop le.")
                anchor_distance = float(refined_bottom - refined_top)
                mcq_anchor_height_meta = {
                    "used": bool(anchor_distance > 0.0),
                    "reason": "derived-from-refine-mcq-roi",
                    "y_top_anchor": round(float(refined_top), 4),
                    "y_bottom_anchor": round(float(refined_bottom), 4),
                    "anchor_distance": round(float(anchor_distance), 4),
                    "line_h": round(float(refined_line_h), 4) if refined_line_h > 0.0 else None,
                    "match_count": _safe_int(mcq_refine_meta.get("matches"), 0),
                    "top_markers": list(mcq_refine_meta.get("top_row_centers") or []),
                    "bottom_markers": list(mcq_refine_meta.get("bottom_row_centers") or []),
                    "refined_roi": dict(mcq_roi),
                }
            else:
                mcq_anchor_height_meta = {
                    "used": False,
                    "reason": f"mcq-refine-not-used:{mcq_refine_meta.get('reason')}",
                }

        if mcq_roi_cfg is None:
            left_expand_px = max(6, min(20, int(round(0.012 * float(mcq_roi["w"])))))
            if left_expand_px > 0:
                new_x = max(0, int(mcq_roi["x"]) - int(left_expand_px))
                gained = int(mcq_roi["x"]) - int(new_x)
                if gained > 0:
                    mcq_roi["x"] = int(new_x)
                    mcq_roi["w"] = int(min(WIDTH_IMG - int(new_x), int(mcq_roi["w"]) + int(gained)))
                    mcq_crop_clean = img_std[
                        mcq_roi["y"] : mcq_roi["y"] + mcq_roi["h"],
                        mcq_roi["x"] : mcq_roi["x"] + mcq_roi["w"],
                    ]

        handwriting_cfg = _parse_handwriting_config(profile_handwriting_fields)
        handwriting_rois = _build_handwriting_rois(
            anchors,
            sid_roi,
            mcq_roi,
            WIDTH_IMG,
            HEIGHT_IMG,
            handwriting_cfg.get("field_rois"),
            long_form_mode=long_form_mode,
        )

        sid_offsets = _parse_sid_row_offsets(profile_sid_row_offsets, student_id_digits)

        sid_result_with_write = _decode_numeric_columns(
            gray_norm,
            binary_inv,
            sid_roi,
            digits=student_id_digits,
            has_write_row=True,
            row_offsets=sid_offsets,
        )
        sid_result_without_write = _decode_numeric_columns(
            gray_norm,
            binary_inv,
            sid_roi,
            digits=student_id_digits,
            has_write_row=False,
            row_offsets=sid_offsets,
        )

        if bool(sid_has_write_row):
            sid_result = sid_result_with_write
            sid_result_alt = sid_result_without_write
            sid_decode_mode = "with-write-row"
            alt_mode = "without-write-row"
        else:
            sid_result = sid_result_without_write
            sid_result_alt = sid_result_with_write
            sid_decode_mode = "without-write-row"
            alt_mode = "with-write-row"

        primary_value = str(sid_result.get("value") or "")
        alt_value = str(sid_result_alt.get("value") or "")
        primary_ok = sid_result.get("status") == "ok" and "?" not in primary_value
        alt_ok = sid_result_alt.get("status") == "ok" and "?" not in alt_value

        primary_conf = float(sid_result.get("confidence") or 0.0)
        alt_conf = float(sid_result_alt.get("confidence") or 0.0)
        shift_delta = _numeric_shift_delta(primary_value, alt_value, student_id_digits) if (primary_ok and alt_ok) else None

        use_alt = False
        if bool(sid_has_write_row):
            primary_all_zero = primary_value == ("0" * int(max(1, student_id_digits)))
            digit_count = int(max(1, student_id_digits))
            changed_digits = sum(1 for a, b in zip(primary_value, alt_value) if a != b) if (primary_ok and alt_ok) else 0
            if alt_ok and not primary_ok:
                use_alt = True
            elif alt_ok and primary_ok and (alt_conf >= (primary_conf + 0.08)):
                use_alt = True
            elif alt_ok and primary_ok and primary_value.startswith("99") and not alt_value.startswith("99"):
                use_alt = True
            elif alt_ok and primary_ok and shift_delta in (1, 9) and alt_conf >= (primary_conf - 0.15):
                use_alt = True
            elif (
                alt_ok
                and primary_ok
                and shift_delta in (1, 9)
                and changed_digits >= max(2, digit_count - 1)
                and alt_conf >= (primary_conf - 0.35)
            ):
                # If all/most columns shift by exactly one row, prefer the alternate SID mode
                # even when confidence is slightly lower.
                use_alt = True
            elif alt_ok and primary_ok and primary_all_zero and primary_value != alt_value and shift_delta in (1, 9) and alt_conf >= 1.15:
                use_alt = True
        else:
            if alt_ok and not primary_ok:
                use_alt = True

        if use_alt:
            sid_result = sid_result_alt
            sid_decode_mode = f"auto-{alt_mode}"
            warnings.append("SID co dau hieu lech hang theo cot, da tu dong doi che do decode SID.")
            warning_codes.append("SID_AUTO_SWITCH_ROW_MODE")

        code_result = _decode_numeric_columns(
            gray_norm,
            binary_inv,
            code_roi,
            digits=3,
            has_write_row=False,
            row_offsets=None,
        )

        mcq_cfg = _parse_mcq_decode_config(profile_mcq_decode)
        if density_only_scoring:
            mcq_cfg = dict(mcq_cfg)
            mcq_cfg["density_only_scoring"] = True
        mcq_rescue_disabled = bool(profile_disable_mcq_rescue)
        if long_form_mode:
            profile_mcq_decode_raw = profile_mcq_decode if isinstance(profile_mcq_decode, dict) else {}
            mcq_cfg = dict(mcq_cfg)
            if not isinstance(profile_mcq_decode, dict):
                mcq_cfg["min_mark_score"] = float(0.425)
                mcq_cfg["min_mark_density"] = float(0.425)
                mcq_cfg["adaptive_threshold"] = True
                mcq_cfg["soft_mark_floor"] = float(0.30)
                mcq_cfg["noise_center_floor"] = float(0.30)
                mcq_cfg["noise_center_margin"] = float(0.08)
                mcq_cfg["noise_dark_floor"] = float(0.18)
                mcq_cfg["noise_dark_margin"] = float(0.03)
                mcq_cfg["enable_soft_single_mark_rescue"] = True

            # Long-form defaults: local row grid-search + thin-stroke suppression.
            if "local_grid_search" not in profile_mcq_decode_raw:
                mcq_cfg["local_grid_search"] = True
            if "local_grid_shift_ratio" not in profile_mcq_decode_raw:
                mcq_cfg["local_grid_shift_ratio"] = float(0.55)
            if "local_grid_y_shift_ratio" not in profile_mcq_decode_raw:
                mcq_cfg["local_grid_y_shift_ratio"] = float(0.12)
            if "local_anchor_momentum" not in profile_mcq_decode_raw:
                mcq_cfg["local_anchor_momentum"] = float(0.20)
            if "thin_noise_erode_iter" not in profile_mcq_decode_raw:
                mcq_cfg["thin_noise_erode_iter"] = int(1)
            if "thin_noise_kernel" not in profile_mcq_decode_raw:
                mcq_cfg["thin_noise_kernel"] = int(2)
            if "thin_noise_min_component_ratio" not in profile_mcq_decode_raw:
                mcq_cfg["thin_noise_min_component_ratio"] = float(0.012)

            if not mcq_rescue_disabled:
                mcq_rescue_disabled = True
                warning_codes.append("MCQ_RESCUE_DISABLED_LONG_FORM")
                warnings.append("Long-form: tam tat map-rescue de tranh lech top-shift o cac cau dau.")
        geom_top_center = _safe_float(mcq_geometry.get("top_center_y"), -1.0)
        geom_line_h = _safe_float(mcq_geometry.get("line_h"), -1.0)

        anchor_top_y = 0.5 * (anchors["mcq_left_top"][1] + anchors["mcq_right_top"][1])
        anchor_bottom_y = 0.5 * (anchors["mcq_left_bottom"][1] + anchors["mcq_right_bottom"][1])
        anchor_left_x = 0.5 * (anchors["mcq_left_top"][0] + anchors["mcq_left_bottom"][0])
        anchor_right_x = 0.5 * (anchors["mcq_right_top"][0] + anchors["mcq_right_bottom"][0])

        anchor_line_h = (anchor_bottom_y - anchor_top_y) / max(1.0, float(max(2, rows_per_block) - 1))
        roi_line_h = max(8.0, float(mcq_roi["h"]) / max(2.0, float(rows_per_block)))
        geom_line_h_ok = 8.0 <= geom_line_h <= 60.0
        geom_line_h_consistent = geom_line_h_ok and ((0.65 * roi_line_h) <= geom_line_h <= (1.45 * roi_line_h))
        anchor_line_h_consistent = (0.70 * roi_line_h) <= anchor_line_h <= (1.35 * roi_line_h)
        map_line_h_consistent = (0.70 * roi_line_h) <= line_h_anchor <= (1.35 * roi_line_h)

        if geom_line_h_consistent:
            line_h = float(geom_line_h)
        elif anchor_line_h_consistent:
            line_h = float(anchor_line_h)
        elif map_line_h_consistent:
            line_h = float(line_h_anchor)
        else:
            line_h = float(roi_line_h)
        line_h = max(8.0, min(44.0, float(line_h)))

        top_center_y = float(geom_top_center) if geom_top_center > 0.0 else float(anchor_top_y)
        fid_top_y = _safe_float(mcq_geometry.get("fid_top_y"), -1.0)
        if long_form_mode and fid_top_y > 0.0:
            # Long forms commonly expose a marker row above Q1. Start at least ~1 line below it.
            top_center_y = max(float(top_center_y), float(fid_top_y + 0.95 * line_h))
        if fid_top_y > 0.0 and top_center_y > (fid_top_y + 1.8 * line_h):
            top_center_y = float(fid_top_y + line_h)

        top_center_upper_guard = float(mcq_roi["y"] + max(2.2 * line_h, 0.45 * float(mcq_roi["h"])))
        if not ((mcq_roi["y"] - 0.5 * line_h) <= top_center_y <= top_center_upper_guard):
            top_center_y = float(mcq_roi["y"] + 0.5 * line_h)

        geom_left_x = _safe_float(mcq_geometry.get("left_x"), -1.0)
        geom_right_x = _safe_float(mcq_geometry.get("right_x"), -1.0)
        fid_left_x = _safe_float(mcq_geometry.get("fid_left_x"), -1.0)
        fid_right_x = _safe_float(mcq_geometry.get("fid_right_x"), -1.0)
        mcq_answer_band_detection_meta: Dict[str, object] = {"used": False, "reason": "skipped"}
        mcq_roi_expand_meta: Dict[str, object] = {"used": False, "reason": "skipped"}
        mcq_row_offsets_meta: Dict[str, object] = {"used": False, "source": "none", "reason": "skipped"}
        if mcq_block_bands_raw:
            revalidated_bands, revalidated_meta = _validate_mcq_block_bands(
                mcq_block_bands_raw,
                mcq_roi,
                block_count,
                choices,
                long_form_mode,
                source="geometry",
            )
            mcq_block_bands_revalidated_after_roi = True
            if revalidated_bands or not mcq_block_bands:
                mcq_block_bands = list(revalidated_bands)
                mcq_block_bands_meta = dict(revalidated_meta)
        if long_form_mode:
            detected_bands, mcq_answer_band_detection_meta = _infer_long_form_answer_center_bands(
                binary_inv,
                mcq_roi,
                top_center_y,
                line_h,
                block_count,
                choices,
                WIDTH_IMG,
                HEIGHT_IMG,
            )
            detected_bands, detected_bands_meta = _validate_mcq_block_bands(
                detected_bands,
                mcq_roi,
                block_count,
                choices,
                long_form_mode,
                source="detected-bubbles",
            )
            if detected_bands:
                mcq_block_bands = list(detected_bands)
                mcq_block_bands_meta = {
                    **detected_bands_meta,
                    "detection": mcq_answer_band_detection_meta,
                }
                mcq_roi, mcq_roi_expand_meta = _expand_mcq_roi_for_bands(
                    mcq_roi,
                    mcq_block_bands,
                    WIDTH_IMG,
                    HEIGHT_IMG,
                    line_h,
                )
                mcq_block_bands, mcq_block_bands_meta = _validate_mcq_block_bands(
                    mcq_block_bands,
                    mcq_roi,
                    block_count,
                    choices,
                    long_form_mode,
                    source="detected-bubbles",
                )
                if mcq_block_bands:
                    mcq_block_bands_meta = {
                        **mcq_block_bands_meta,
                        "detection": mcq_answer_band_detection_meta,
                        "roi_expand": mcq_roi_expand_meta,
                    }
                    warning_codes.append("MCQ_LONG_FORM_BUBBLE_BANDS_DETECTED")
                    warnings.append("Long-form: detect truc tiep cot dap an MCQ tu bubble centers.")
                    if bool(mcq_roi_expand_meta.get("used", False)):
                        warning_codes.append("MCQ_LONG_FORM_ROI_EXPANDED_FOR_BANDS")
                        warnings.append("Long-form: mo rong ROI MCQ de bao tron block dap an cuoi.")
        mcq_decode_x_source = "anchor-grid"
        if mcq_block_bands:
            band_left_x = float(mcq_block_bands[0][0])
            band_right_x = float(mcq_block_bands[min(len(mcq_block_bands), block_count) - 1][1])
            anchor_left_x = band_left_x
            anchor_right_x = band_right_x
            mcq_decode_x_source = str(mcq_block_bands_meta.get("source") or "block-bands")
        elif long_form_mode:
            roi_w = float(mcq_roi["w"])
            if geom_right_x - geom_left_x >= max(120.0, roi_w * 0.45):
                anchor_left_x = float(geom_left_x)
                if fid_left_x > 0.0 and fid_left_x < anchor_left_x and (anchor_left_x - fid_left_x) <= max(80.0, 0.12 * float(WIDTH_IMG)):
                    anchor_left_x = float(fid_left_x)
                anchor_right_x = float(geom_right_x)
                if fid_right_x > anchor_right_x and (fid_right_x - anchor_right_x) <= max(35.0, 0.04 * float(WIDTH_IMG)):
                    anchor_right_x = float(fid_right_x)
                mcq_decode_x_source = "bubble-span-long-form"
            elif fid_right_x - fid_left_x >= max(120.0, roi_w * 0.35):
                anchor_left_x = float(fid_left_x)
                anchor_right_x = float(fid_right_x)
                mcq_decode_x_source = "fid-span-long-form"
            else:
                anchor_left_x = float(mcq_roi["x"] + 0.06 * roi_w)
                anchor_right_x = float(mcq_roi["x"] + 0.94 * roi_w)
                mcq_decode_x_source = "roi-fallback-long-form"
                warning_codes.append("MCQ_LONG_FORM_X_FALLBACK")
                warnings.append("Long-form: khong du local marker X hop le, fallback sang pham vi ROI rong theo tung block.")
        elif geom_right_x - geom_left_x >= float(mcq_roi["w"]) * 0.45:
            anchor_left_x = float(geom_left_x)
            anchor_right_x = float(geom_right_x)
            mcq_decode_x_source = "bubble-span"
        elif fid_right_x - fid_left_x >= float(mcq_roi["w"]) * 0.45:
            anchor_left_x = float(fid_left_x)
            anchor_right_x = float(fid_right_x)
            mcq_decode_x_source = "fid-span"

        anchor_left_x = max(float(mcq_roi["x"]), float(anchor_left_x))
        anchor_right_x = min(float(mcq_roi["x"] + mcq_roi["w"]), float(anchor_right_x))
        span_x = float(anchor_right_x - anchor_left_x)
        if span_x < (0.55 * float(mcq_roi["w"])) or span_x > (1.02 * float(mcq_roi["w"])):
            anchor_left_x = float(mcq_roi["x"] + 0.06 * mcq_roi["w"])
            anchor_right_x = float(mcq_roi["x"] + 0.94 * mcq_roi["w"])
            mcq_decode_x_source = "roi-guard-fallback"

        if long_form_mode and not mcq_block_bands:
            derived_bands = _derive_mcq_center_bands_from_span(
                anchor_left_x,
                anchor_right_x,
                mcq_roi,
                block_count,
                choices,
            )
            mcq_block_bands, mcq_block_bands_meta = _validate_mcq_block_bands(
                derived_bands,
                mcq_roi,
                block_count,
                choices,
                long_form_mode,
                source=f"derived-{mcq_decode_x_source}",
            )
            if mcq_block_bands:
                warning_codes.append("MCQ_LONG_FORM_BLOCK_BANDS_DERIVED")
                warnings.append("Long-form: dung block bands suy luan tu span MCQ da validate de decode tung cot.")
            else:
                warnings.append("Long-form: khong tao duoc block bands hop le, tiep tuc decode bang span MCQ.")

        if long_form_mode and rows_per_block == 30:
            row_offsets_px, mcq_row_offsets_meta = _infer_long_form_row_offsets(
                binary_inv,
                mcq_roi,
                mcq_block_bands,
                top_center_y,
                line_h,
                rows_per_block,
                choices,
                WIDTH_IMG,
                HEIGHT_IMG,
            )
            if row_offsets_px:
                mcq_cfg = dict(mcq_cfg)
                mcq_cfg["row_offsets_px"] = list(row_offsets_px)
                warning_codes.append("MCQ_LONG_FORM_ROW_OFFSETS_APPLIED")
                if str(mcq_row_offsets_meta.get("source") or "") == "detected-bubbles":
                    warning_codes.append("MCQ_LONG_FORM_ROW_OFFSETS_DETECTED")
                    warnings.append("Long-form: detect truc tiep Y cua 30 dong MCQ va ap dung row offsets.")
                else:
                    warning_codes.append("MCQ_LONG_FORM_ROW_OFFSETS_FALLBACK")
                    warnings.append("Long-form: ap dung row offsets 10-10-10 de bu khoang anchor giua block.")

        mcq_result = _decode_mcq_with_map(
            gray_norm,
            binary_inv,
            mcq_roi,
            questions=questions,
            choices=choices,
            rows_per_block=rows_per_block,
            block_count=block_count,
            left_x=anchor_left_x,
            right_x=anchor_right_x,
            top_center_y=top_center_y,
            line_h=line_h,
            decode_cfg=mcq_cfg,
            top_shift_px=0.0,
            block_bands=mcq_block_bands,
        )
        short_form_bands_locked = bool((not long_form_mode) and mcq_block_bands and bool(mcq_marker_roi_meta.get("used", False)))

        mcq_map_search_meta: Dict[str, object] = {
            "used": False,
            "reason": "skipped",
            "initial_uncertain": int(len(list(mcq_result.get("uncertain_questions") or []))),
        }

        # If marker-anchored ROI is still highly uncertain, search nearby map parameters.
        # This keeps ownership in MCQ module while avoiding hardcoded service-side ROI rewrites.
        initial_uncertain = int(len(list(mcq_result.get("uncertain_questions") or [])))
        search_uncertain_gate = max(4, int(round(0.40 * float(max(1, questions)))))
        if (not mcq_rescue_disabled) and (not short_form_bands_locked) and initial_uncertain >= search_uncertain_gate:
            mcq_map_search_meta = {
                "used": False,
                "reason": "no-better-candidate",
                "initial_uncertain": int(initial_uncertain),
                "gate": int(search_uncertain_gate),
            }

            baseline_quality = float(_mcq_quality(mcq_result))
            baseline_double = int(len(list(mcq_result.get("double_mark_questions") or [])))
            baseline_rank = (int(initial_uncertain), int(baseline_double), float(-baseline_quality))
            baseline_line_h = float(line_h)
            baseline_top_center_y = float(top_center_y)

            best_result = mcq_result
            best_rank = baseline_rank
            best_line_h = float(line_h)
            best_top_shift_px = 0.0
            best_block_bands = list(mcq_block_bands)

            line_scales = [1.00, 0.92, 0.88, 1.08, 0.84, 1.16]
            shift_multipliers = [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]

            candidate_band_sets: List[List[Tuple[float, float]]] = [list(mcq_block_bands)] if mcq_block_bands else [[]]

            tested = 0
            for cand_bands in candidate_band_sets:
                for scale in line_scales:
                    cand_line_h = max(6.0, min(44.0, float(line_h) * float(scale)))
                    for shift_mul in shift_multipliers:
                        cand_shift_px = float(shift_mul) * float(cand_line_h)
                        cand_result = _decode_mcq_with_map(
                            gray_norm,
                            binary_inv,
                            mcq_roi,
                            questions=questions,
                            choices=choices,
                            rows_per_block=rows_per_block,
                            block_count=block_count,
                            left_x=anchor_left_x,
                            right_x=anchor_right_x,
                            top_center_y=top_center_y,
                            line_h=cand_line_h,
                            decode_cfg=mcq_cfg,
                            top_shift_px=cand_shift_px,
                            block_bands=cand_bands,
                        )
                        tested += 1

                        cand_uncertain = int(len(list(cand_result.get("uncertain_questions") or [])))
                        cand_double = int(len(list(cand_result.get("double_mark_questions") or [])))
                        cand_quality = float(_mcq_quality(cand_result))
                        cand_rank = (int(cand_uncertain), int(cand_double), float(-cand_quality))

                        if cand_rank < best_rank:
                            best_rank = cand_rank
                            best_result = cand_result
                            best_line_h = float(cand_line_h)
                            best_top_shift_px = float(cand_shift_px)
                            best_block_bands = list(cand_bands)

            improved = False
            if best_rank < baseline_rank:
                uncertain_gain = int(baseline_rank[0] - best_rank[0])
                if uncertain_gain >= 2:
                    improved = True
                elif uncertain_gain >= 1 and best_rank[1] <= baseline_rank[1]:
                    improved = True
                elif uncertain_gain == 0 and best_rank[1] < baseline_rank[1]:
                    improved = True

            if improved:
                mcq_result = best_result
                line_h = float(best_line_h)
                top_center_y = float(top_center_y + best_top_shift_px)
                top_guard = float(max(3.0, 0.35 * float(line_h)))
                top_center_min = float(mcq_roi["y"] + top_guard)
                top_center_max = float(mcq_roi["y"] + mcq_roi["h"] - top_guard)
                if top_center_min > top_center_max:
                    top_center_min = float(mcq_roi["y"])
                    top_center_max = float(mcq_roi["y"] + mcq_roi["h"])
                top_center_y = max(top_center_min, min(top_center_max, float(top_center_y)))
                mcq_block_bands = list(best_block_bands)

                mcq_map_search_meta = {
                    "used": True,
                    "reason": "improved",
                    "tested": int(tested),
                    "initial_uncertain": int(baseline_rank[0]),
                    "final_uncertain": int(best_rank[0]),
                    "initial_double_mark": int(baseline_rank[1]),
                    "final_double_mark": int(best_rank[1]),
                    "line_h_before": round(float(baseline_line_h), 4),
                    "line_h_after": round(float(best_line_h), 4),
                    "top_center_y_before": round(float(baseline_top_center_y), 4),
                    "top_center_y_after": round(float(top_center_y), 4),
                    "top_shift_px": round(float(best_top_shift_px), 4),
                    "block_bands_used": bool(best_block_bands),
                }
                warning_codes.append("MCQ_MAP_SEARCH_RESCUE")
                warnings.append(
                    "MCQ map rescue da dieu chinh line-height/top-shift (va block bands neu can) de giam uncertain."
                )
            else:
                mcq_map_search_meta = {
                    "used": False,
                    "reason": "no-better-candidate",
                    "tested": int(tested),
                    "initial_uncertain": int(baseline_rank[0]),
                    "final_uncertain": int(best_rank[0]),
                    "initial_double_mark": int(baseline_rank[1]),
                    "final_double_mark": int(best_rank[1]),
                }
        elif short_form_bands_locked:
            mcq_map_search_meta = {
                "used": False,
                "reason": "locked-by-short-form-block-bands",
                "initial_uncertain": int(initial_uncertain),
                "gate": int(search_uncertain_gate),
            }
        elif mcq_rescue_disabled:
            rescue_reason = "disabled-by-profile" if bool(profile_disable_mcq_rescue) else "disabled-by-long-form"
            mcq_map_search_meta = {
                "used": False,
                "reason": rescue_reason,
                "initial_uncertain": int(initial_uncertain),
            }
        else:
            mcq_map_search_meta = {
                "used": False,
                "reason": "below-uncertain-gate",
                "initial_uncertain": int(initial_uncertain),
                "gate": int(search_uncertain_gate),
            }

        drift_suspected = _detect_q5_start_drift(mcq_result, mcq_cfg["min_mark_score"])
        auto_expand = False
        auto_expand_lines = 0

        if drift_suspected:
            warning_codes.append("MCQ_COORD_DRIFT")
            if not mcq_rescue_disabled:
                expand_lines = 4
                expand_px = int(round(float(expand_lines) * float(line_h)))
                retry_roi = dict(mcq_roi)
                if expand_px > 0:
                    y_new = max(0, int(retry_roi["y"]) - int(expand_px))
                    gained = int(retry_roi["y"]) - y_new
                    retry_roi["y"] = int(y_new)
                    retry_roi["h"] = int(min(HEIGHT_IMG - retry_roi["y"], int(retry_roi["h"]) + gained))

                retry_top_center = float(top_center_y - float(expand_lines) * float(line_h))
                if retry_top_center < (retry_roi["y"] + 0.25 * line_h):
                    retry_top_center = float(retry_roi["y"] + 0.5 * line_h)

                retry = _decode_mcq_with_map(
                    gray_norm,
                    binary_inv,
                    retry_roi,
                    questions=questions,
                    choices=choices,
                    rows_per_block=rows_per_block,
                    block_count=block_count,
                    left_x=anchor_left_x,
                    right_x=anchor_right_x,
                    top_center_y=retry_top_center,
                    line_h=line_h,
                    decode_cfg=mcq_cfg,
                    top_shift_px=0.0,
                    block_bands=mcq_block_bands,
                )

                if _mcq_quality(retry) >= _mcq_quality(mcq_result):
                    mcq_result = retry
                    mcq_roi = retry_roi
                    top_center_y = float(retry_top_center)
                    auto_expand = True
                    auto_expand_lines = int(expand_lines)
                    warning_codes.append("MCQ_AUTO_EXPAND_UP")
                    warnings.append("Phat hien roi MCQ co dau hieu bat dau tu cau 5, da tu dong mo rong ROI len 4 dong.")
                else:
                    warnings.append("Phat hien drift roi MCQ nhung thu nghiem mo rong len khong tot hon decode hien tai.")
            else:
                warnings.append("Phat hien drift roi MCQ nhung rescue tu dong dang bi khoa.")

        user_answers = list(mcq_result.get("user_answers") or [])
        if len(user_answers) < questions:
            user_answers.extend([-1] * (questions - len(user_answers)))

        answer_confidences = list(mcq_result.get("answer_confidences") or [])
        if len(answer_confidences) < questions:
            answer_confidences.extend([0.0] * (questions - len(answer_confidences)))

        answer_map = [choice_label(int(val)) for val in user_answers]

        detected_exam_code_raw = str(code_result.get("value") or "").strip()
        detected_exam_code_key = _normalize_exam_code_key(detected_exam_code_raw)

        manual_answer_key = list(answer_key or [])
        selected_answer_key = list(manual_answer_key)
        selected_answer_code: Optional[str] = None
        answer_key_source = "manual-or-file"

        normalized_code_map: Dict[str, List[int]] = {}
        display_code_map: Dict[str, str] = {}
        if isinstance(answer_key_by_code, dict):
            for raw_code, raw_values in answer_key_by_code.items():
                norm_code = _normalize_exam_code_key(raw_code)
                if not norm_code:
                    continue
                values = [int(v) for v in list(raw_values or [])]
                if not values:
                    continue
                normalized_code_map[norm_code] = values
                display_code_map[norm_code] = str(raw_code)

        if normalized_code_map:
            answer_key_source = "assignment-code-map"
            if detected_exam_code_key and detected_exam_code_key in normalized_code_map:
                selected_answer_key = list(normalized_code_map[detected_exam_code_key])
                selected_answer_code = str(display_code_map.get(detected_exam_code_key) or detected_exam_code_raw)
            else:
                selected_answer_key = []
                if detected_exam_code_raw:
                    warnings.append(f"Khong tim thay dap an cho CODE '{detected_exam_code_raw}' trong kho dap an.")
                    warning_codes.append("ANSWER_CODE_NOT_FOUND")

        answer_key_zero = _normalize_answer_key(selected_answer_key, choices)
        compare_questions = min(len(answer_key_zero), len(user_answers), int(questions))

        answer_compare, correct_count, wrong_questions, uncertain_compare, scored_questions = _build_answer_compare(
            user_answers,
            answer_key_zero,
            choices,
            compare_questions,
        )
        graded_questions = int(min(int(questions), int(scored_questions)))

        detection_uncertain_questions = sorted(set(int(q) for q in list(mcq_result.get("uncertain_questions") or [])))
        uncertain_questions = sorted(set(int(q) for q in uncertain_compare))
        double_mark_questions = sorted(set(int(q) for q in list(mcq_result.get("double_mark_questions") or [])))
        if double_mark_questions:
            warning_codes.append("MCQ_DOUBLE_MARK")
            warnings.append(f"Phat hien da to nhieu lua chon o cac cau: {', '.join(str(q) for q in double_mark_questions[:20])}")
        uncertain_count = len([q for q in uncertain_questions if 1 <= q <= questions])

        score = float(correct_count)
        detected_questions = int(questions)
        ungraded_questions = max(0, detected_questions - graded_questions)

        run_tag = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        result_name = f"omr_result_{run_tag}.jpg"
        sid_crop_name = f"omr_sid_{run_tag}.jpg"
        mcq_crop_name = f"omr_mcq_{run_tag}.jpg"
        bubble_name = f"bubble_confidence_{run_tag}.json"

        handwriting_payload = _extract_handwriting_crops(
            img_std,
            handwriting_rois,
            output_folder=output_folder,
            run_tag=run_tag,
            handwriting_cfg=handwriting_cfg,
        )
        warnings.extend(list(handwriting_payload.get("warnings") or []))
        warning_codes.extend(list(handwriting_payload.get("warning_codes") or []))

        sid_crop = img_std[
            sid_roi["y"] : sid_roi["y"] + sid_roi["h"],
            sid_roi["x"] : sid_roi["x"] + sid_roi["w"],
        ]
        if mcq_crop_clean is not None and getattr(mcq_crop_clean, "size", 0) > 0:
            mcq_crop = mcq_crop_clean
        else:
            mcq_crop = img_std[
                mcq_roi["y"] : mcq_roi["y"] + mcq_roi["h"],
                mcq_roi["x"] : mcq_roi["x"] + mcq_roi["w"],
            ]

        img_result = _draw_result_overlay(
            img_std,
            sid_roi,
            code_roi,
            mcq_roi,
            mcq_result.get("rows") or [],
            sid_result.get("value") or "",
            code_result.get("value") or "",
            score,
            graded_questions,
            handwriting_rois,
            answer_compare,
            sid_result.get("selected_cells") or [],
            code_result.get("selected_cells") or [],
        )

        result_path = os.path.join(output_folder, result_name)
        sid_crop_path = os.path.join(output_folder, sid_crop_name)
        mcq_crop_path = os.path.join(output_folder, mcq_crop_name)
        bubble_path = os.path.join(output_folder, bubble_name)

        cv2.imwrite(result_path, img_result)
        cv2.imwrite(sid_crop_path, sid_crop)
        cv2.imwrite(mcq_crop_path, mcq_crop)

        bubble_payload = {
            "pipeline_version": "omr-coordinate-map-v1",
            "image": os.path.basename(image_path),
            "warp_strategy": str(warp_strategy),
            "roi_boxes": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "coordinate_mapping": {
                "global_warp": bool(global_warp_used),
                "global_warp_target": {"width": int(A4_WARP_W), "height": int(A4_WARP_H)},
                "global_warp_info": warp_info,
                "anchor_fallback_used": bool(anchor_fallback_used),
                "mcq_marker_roi": mcq_marker_roi_meta,
                "mcq_refine": mcq_refine_meta,
                "mcq_anchor_height": mcq_anchor_height_meta,
                "inferred_block_count": int(inferred_block_count),
                "decode_x_source": str(mcq_decode_x_source),
                "long_form_block_bands_used": bool(long_form_mode and mcq_block_bands),
                "block_bands": [[round(float(a), 4), round(float(b), 4)] for a, b in mcq_block_bands],
                "block_bands_initial_meta": mcq_block_bands_initial_meta,
                "block_bands_meta": mcq_block_bands_meta,
                "block_bands_final_meta": mcq_block_bands_meta,
                "block_bands_revalidated_after_roi": bool(mcq_block_bands_revalidated_after_roi),
                "answer_band_detection": mcq_answer_band_detection_meta,
                "mcq_roi_expand": mcq_roi_expand_meta,
                "long_form_row_offsets_used": bool(mcq_row_offsets_meta.get("used", False)),
                "row_offsets_source": str(mcq_row_offsets_meta.get("source") or "none"),
                "row_offsets_meta": mcq_row_offsets_meta,
                "line_height_px": round(float(line_h), 4),
                "drift_suspected": bool(drift_suspected),
                "auto_expand_upward": bool(auto_expand),
                "auto_expand_lines": int(auto_expand_lines),
                "mcq_map_search": mcq_map_search_meta,
                "fid_top_y": round(float(_safe_float(mcq_geometry.get("fid_top_y"), 0.0)), 4),
                "fid_bottom_y": round(float(_safe_float(mcq_geometry.get("fid_bottom_y"), 0.0)), 4),
                "anchors": {
                    key: {"x": round(float(val[0]), 4), "y": round(float(val[1]), 4)}
                    for key, val in anchors.items()
                },
            },
            "handwriting": {
                "enabled": bool(handwriting_payload.get("enabled", False)),
                "ocr_engine": str(handwriting_payload.get("ocr_engine") or "disabled"),
                "field_rois": handwriting_rois,
                "values": handwriting_payload.get("values") or {},
                "fields": handwriting_payload.get("fields") or {},
            },
            "rows": mcq_result.get("rows") or [],
        }

        try:
            with open(bubble_path, "w", encoding="utf-8") as fp:
                json.dump(bubble_payload, fp, ensure_ascii=False, indent=2)
        except Exception:
            bubble_name = None
            warnings.append("Khong the ghi telemetry bubble confidence JSON.")
            warning_codes.append("BUBBLE_JSON_WRITE_FAILED")

        warp_layout_score = 0.0
        if global_warp_used:
            warp_layout_score += 0.50
        warp_layout_score += min(0.40, len(markers) / 500.0)
        if not anchor_fallback_used:
            warp_layout_score += 0.10

        roi_strategy = "coordinate-mapping-anchor-grid"
        if sid_roi_cfg is not None or code_roi_cfg is not None or mcq_roi_cfg is not None:
            roi_strategy += " + profile-roi"
        if anchor_fallback_used:
            roi_strategy += " + percent-fallback"

        correct_answers = []
        for i in range(detected_questions):
            if i < len(selected_answer_key):
                correct_answers.append(_safe_int(selected_answer_key[i], -1))
            else:
                correct_answers.append(-1)

        correct_answers_graded = [
            int(item.get("correct", -1))
            for item in answer_compare
            if _safe_int(item.get("correct", -1), -1) >= 0
        ]

        warnings = _dedupe_keep_order(warnings)
        warning_codes = _dedupe_keep_order(warning_codes)

        return {
            "success": True,
            "pipeline_version": "omr-coordinate-map-v1",
            "score": score,
            "student_id": str(sid_result.get("value") or ""),
            "student_id_status": str(sid_result.get("status") or "uncertain"),
            "student_id_confidence": float(sid_result.get("confidence") or 0.0),
            "exam_code": str(code_result.get("value") or ""),
            "exam_code_status": str(code_result.get("status") or "uncertain"),
            "exam_code_confidence": float(code_result.get("confidence") or 0.0),
            "matched_answer_code": selected_answer_code,
            "answer_key_source": answer_key_source,
            "exam_title_detected": None,
            "user_answers": [int(v) for v in user_answers],
            "answer_map": answer_map,
            "answer_confidences": [float(v) for v in answer_confidences],
            "answer_compare": answer_compare,
            "correct_answers": correct_answers,
            "correct_answers_graded": correct_answers_graded,
            "graded_questions": int(graded_questions),
            "answer_key_questions": int(len(answer_key_zero)),
            "detected_questions": int(detected_questions),
            "ungraded_questions": int(ungraded_questions),
            "ungraded_count": int(ungraded_questions),
            "uncertain_questions": uncertain_questions,
            "detection_uncertain_questions": detection_uncertain_questions,
            "uncertain_count": int(uncertain_count),
            "wrong_questions": wrong_questions,
            "double_mark_questions": double_mark_questions,
            "roi_boxes": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "roi_boxes_refined": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "roi_boxes_json": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "handwriting_fields": handwriting_payload.get("values") or {},
            "handwriting": {
                "enabled": bool(handwriting_payload.get("enabled", False)),
                "ocr_engine": str(handwriting_payload.get("ocr_engine") or "disabled"),
                "gpu": bool(handwriting_payload.get("gpu", False)),
                "save_crops": bool(handwriting_payload.get("save_crops", True)),
                "field_rois": handwriting_rois,
                "fields": handwriting_payload.get("fields") or {},
                "crop_images": handwriting_payload.get("crop_images") or {},
                "preprocessed_crop_images": handwriting_payload.get("preprocessed_crop_images") or {},
            },
            "roi_detection": {
                "strategy": roi_strategy,
                "profile_page_size_pt": profile_page_size_pt if isinstance(profile_page_size_pt, dict) else None,
                "marker_count": int(len(markers)),
                "sid_decode_mode": sid_decode_mode,
                "handwriting_enabled": bool(handwriting_payload.get("enabled", False)),
                "handwriting_ocr_engine": str(handwriting_payload.get("ocr_engine") or "disabled"),
                "coordinate_mapping_global_warp": bool(global_warp_used),
                "coordinate_mapping_anchor_fallback": bool(anchor_fallback_used),
                "coordinate_mapping_mcq_map_used": True,
                "coordinate_mapping_mcq_blocks_inferred": int(inferred_block_count),
                "coordinate_mapping_mcq_black_marker_roi_used": bool(mcq_marker_roi_meta.get("used", False)),
                "coordinate_mapping_mcq_black_marker_roi_reason": str(mcq_marker_roi_meta.get("reason") or ""),
                "coordinate_mapping_mcq_template_refine_used": bool(mcq_refine_meta.get("used", False)),
                "coordinate_mapping_mcq_template_refine_reason": str(mcq_refine_meta.get("reason") or ""),
                "coordinate_mapping_mcq_anchor_height_used": bool(mcq_anchor_height_meta.get("used", False)),
                "coordinate_mapping_mcq_anchor_height_reason": str(mcq_anchor_height_meta.get("reason") or ""),
                "coordinate_mapping_mcq_line_height_px": round(float(line_h), 4),
                "coordinate_mapping_mcq_drift_suspected": bool(drift_suspected),
                "coordinate_mapping_mcq_auto_expand_upward": bool(auto_expand),
                "coordinate_mapping_mcq_auto_expand_lines": int(auto_expand_lines),
                "coordinate_mapping_mcq_map_search_used": bool(mcq_map_search_meta.get("used", False)),
                "coordinate_mapping_mcq_map_search_reason": str(mcq_map_search_meta.get("reason") or ""),
                "coordinate_mapping_mcq_q1_y": round(float(top_center_y), 4),
                "coordinate_mapping_mcq_left_x": round(float(anchor_left_x), 4),
                "coordinate_mapping_mcq_right_x": round(float(anchor_right_x), 4),
                "coordinate_mapping_mcq_decode_x_source": str(mcq_decode_x_source),
                "coordinate_mapping_mcq_long_form_block_bands_used": bool(long_form_mode and mcq_block_bands),
                "coordinate_mapping_mcq_block_bands_rejected_reason": str(mcq_block_bands_meta.get("reason") or ""),
                "coordinate_mapping_mcq_block_bands_revalidated_after_roi": bool(mcq_block_bands_revalidated_after_roi),
                "coordinate_mapping_mcq_answer_band_detection_used": bool(mcq_answer_band_detection_meta.get("used", False)),
                "coordinate_mapping_mcq_roi_expanded_for_bands": bool(mcq_roi_expand_meta.get("used", False)),
                "coordinate_mapping_mcq_long_form_row_offsets_used": bool(mcq_row_offsets_meta.get("used", False)),
                "coordinate_mapping_mcq_row_offsets_source": str(mcq_row_offsets_meta.get("source") or "none"),
                "coordinate_mapping_mcq_double_mark_count": int(len(double_mark_questions)),
            },
            "sid_layout": {"digits": int(student_id_digits), "rows": 10},
            "mcq_layout": {
                "questions": int(questions),
                "choices": int(choices),
                "rows_per_block": int(rows_per_block),
                "num_blocks": int(block_count),
                "block_gap_px": 0,
                "block_width_px": int(round(float(mcq_roi["w"]) / max(1.0, float(block_count)))),
            },
            "threshold_info": {
                "method": f"{threshold_meta['mode']} threshold",
                "otsu_value": float(threshold_meta["otsu_value"]),
            },
            "warp_strategy": str(warp_strategy),
            "warp_layout_score": round(float(warp_layout_score), 5),
            "warnings": warnings,
            "warning_codes": warning_codes,
            "bubble_confidence_json": bubble_name,
            "sid_crop_image": sid_crop_name,
            "mcq_crop_image": mcq_crop_name,
            "result_image": result_name,
        }

    except Exception as exc:
        return {"error": f"Loi xu ly OMR: {exc}"}
