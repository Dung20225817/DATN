from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .omr_mcq import _cluster_marker_rows
from .omr_utils import _clip_rect


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


def _parse_roi_cfg(roi_cfg, img_w: int, img_h: int) -> Optional[Dict[str, int]]:
    if not isinstance(roi_cfg, dict):
        return None

    x = _safe_float(roi_cfg.get("x"), -1.0)
    y = _safe_float(roi_cfg.get("y"), -1.0)
    w = _safe_float(roi_cfg.get("w"), -1.0)
    h = _safe_float(roi_cfg.get("h"), -1.0)

    if w <= 0.0 or h <= 0.0:
        return None

    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x *= float(img_w)
        y *= float(img_h)
        w *= float(img_w)
        h *= float(img_h)

    xi = int(round(x))
    yi = int(round(y))
    wi = int(round(w))
    hi = int(round(h))

    if wi < 8 or hi < 8:
        return None

    xi, yi, wi, hi = _clip_rect(xi, yi, wi, hi, img_w, img_h)
    return {"x": int(xi), "y": int(yi), "w": int(wi), "h": int(hi)}


def _pick_anchor_marker(markers, target_xy, img_w: int, img_h: int, max_norm_dist: float = 0.15):
    if not markers:
        return None

    tx = float(target_xy[0]) * float(img_w)
    ty = float(target_xy[1]) * float(img_h)
    diag = max(1.0, math.hypot(float(img_w), float(img_h)))

    best = None
    best_score = -1e9
    for marker in markers:
        cx = _safe_float(marker.get("cx"), -1.0)
        cy = _safe_float(marker.get("cy"), -1.0)
        if cx < 0.0 or cy < 0.0:
            continue

        dist = math.hypot(cx - tx, cy - ty) / diag
        if dist > float(max_norm_dist):
            continue

        area_ratio = _safe_float(marker.get("area"), 0.0) / max(1.0, float(img_w * img_h))
        fill = _safe_float(marker.get("fill"), 0.0)
        score = (1.0 - dist) * 2.0 + area_ratio * 650.0 + fill * 0.12
        if score > best_score:
            best_score = score
            best = marker

    if best is None:
        return None

    return float(best.get("cx", 0.0)), float(best.get("cy", 0.0))


def _default_anchor_percent(rows_per_block: int):
    mcq_top = 0.575
    line_h = 0.0152
    mcq_bottom = min(0.94, mcq_top + (max(2, int(rows_per_block)) - 1) * line_h)

    return {
        "sid_top": (0.57, 0.04),
        "sid_bottom": (0.57, 0.26),
        "code_top": (0.80, 0.04),
        "code_bottom": (0.80, 0.25),
        "mcq_left_top": (0.20, mcq_top),
        "mcq_left_bottom": (0.20, mcq_bottom),
        "mcq_right_top": (0.77, mcq_top),
        "mcq_right_bottom": (0.77, mcq_bottom),
    }


def _resolve_coordinate_anchors(
    img_w: int,
    img_h: int,
    markers,
    rows_per_block: int,
    sid_roi_cfg,
    code_roi_cfg,
    mcq_roi_cfg,
):
    fallback_used = False
    expected = _default_anchor_percent(rows_per_block)

    if sid_roi_cfg is not None:
        expected["sid_top"] = (
            (sid_roi_cfg["x"] + sid_roi_cfg["w"] * 0.52) / float(img_w),
            (sid_roi_cfg["y"] + sid_roi_cfg["h"] * 0.05) / float(img_h),
        )
        expected["sid_bottom"] = (
            (sid_roi_cfg["x"] + sid_roi_cfg["w"] * 0.52) / float(img_w),
            (sid_roi_cfg["y"] + sid_roi_cfg["h"] * 0.95) / float(img_h),
        )

    if code_roi_cfg is not None:
        expected["code_top"] = (
            (code_roi_cfg["x"] + code_roi_cfg["w"] * 0.50) / float(img_w),
            (code_roi_cfg["y"] + code_roi_cfg["h"] * 0.05) / float(img_h),
        )
        expected["code_bottom"] = (
            (code_roi_cfg["x"] + code_roi_cfg["w"] * 0.50) / float(img_w),
            (code_roi_cfg["y"] + code_roi_cfg["h"] * 0.95) / float(img_h),
        )

    if mcq_roi_cfg is not None:
        expected["mcq_left_top"] = (
            (mcq_roi_cfg["x"] + mcq_roi_cfg["w"] * 0.05) / float(img_w),
            (mcq_roi_cfg["y"] + mcq_roi_cfg["h"] * 0.05) / float(img_h),
        )
        expected["mcq_left_bottom"] = (
            (mcq_roi_cfg["x"] + mcq_roi_cfg["w"] * 0.05) / float(img_w),
            (mcq_roi_cfg["y"] + mcq_roi_cfg["h"] * 0.95) / float(img_h),
        )
        expected["mcq_right_top"] = (
            (mcq_roi_cfg["x"] + mcq_roi_cfg["w"] * 0.95) / float(img_w),
            (mcq_roi_cfg["y"] + mcq_roi_cfg["h"] * 0.05) / float(img_h),
        )
        expected["mcq_right_bottom"] = (
            (mcq_roi_cfg["x"] + mcq_roi_cfg["w"] * 0.95) / float(img_w),
            (mcq_roi_cfg["y"] + mcq_roi_cfg["h"] * 0.95) / float(img_h),
        )

    candidate_markers = list(markers or [])
    square_markers = []
    for marker in candidate_markers:
        area = _safe_float(marker.get("area"), 0.0)
        size = _safe_float(marker.get("size"), 0.0)
        fill = _safe_float(marker.get("fill"), 0.0)
        vertices = _safe_int(marker.get("vertices"), 0)
        aspect = _safe_float(marker.get("w"), 1.0) / max(1.0, _safe_float(marker.get("h"), 1.0))
        if area < 40.0 or size < 6.0:
            continue
        if not (0.72 <= aspect <= 1.40):
            continue
        if vertices < 4 or vertices > 6:
            continue
        if fill < 0.58:
            continue
        square_markers.append(marker)
    if len(square_markers) >= 12:
        candidate_markers = square_markers

    fiducial_markers = [
        marker
        for marker in candidate_markers
        if _safe_float(marker.get("fill"), 0.0) >= 0.86
        and _safe_float(marker.get("size"), 0.0) >= 11.0
        and _safe_float(marker.get("circularity"), 1.0) <= 0.90
    ]

    anchors: Dict[str, Tuple[float, float]] = {}

    for name in ("sid_top", "sid_bottom", "code_top", "code_bottom"):
        target = expected[name]
        max_dist = 0.085 if name.startswith("sid") else 0.075
        picked = _pick_anchor_marker(fiducial_markers, target, img_w, img_h, max_norm_dist=max_dist)
        if picked is None:
            picked = _pick_anchor_marker(candidate_markers, target, img_w, img_h, max_norm_dist=max_dist)
        if picked is None:
            fallback_used = True
            anchors[name] = (float(target[0] * img_w), float(target[1] * img_h))
        else:
            anchors[name] = (float(picked[0]), float(picked[1]))

    exp_top_y = float(expected["mcq_left_top"][1] * img_h)
    exp_bottom_y = float(expected["mcq_left_bottom"][1] * img_h)
    exp_span = max(80.0, float(exp_bottom_y - exp_top_y))

    mcq_rows = _cluster_marker_rows(
        candidate_markers,
        min_x=float(img_w) * 0.12,
        max_x=float(img_w) * 0.90,
        min_y=float(img_h) * 0.46,
        max_y=float(img_h) * 0.96,
        y_tol=6.0,
    )
    mcq_rows = [
        row
        for row in mcq_rows
        if row.get("count", 0) >= 4
        and (float(row.get("x_max", 0.0)) - float(row.get("x_min", 0.0))) >= (float(img_w) * 0.48)
    ]

    top_row = None
    bottom_row = None
    if mcq_rows:
        top_row = min(mcq_rows, key=lambda row: abs(float(row.get("cy", 0.0)) - exp_top_y))
        bottom_candidates = [
            row
            for row in mcq_rows
            if float(row.get("cy", 0.0)) > (float(top_row.get("cy", 0.0)) + 0.70 * exp_span)
        ]
        if bottom_candidates:
            target_bottom = float(top_row.get("cy", 0.0)) + exp_span
            bottom_row = min(bottom_candidates, key=lambda row: abs(float(row.get("cy", 0.0)) - target_bottom))

    if top_row is not None and bottom_row is not None:
        anchors["mcq_left_top"] = (float(top_row["x_min"]), float(top_row["cy"]))
        anchors["mcq_right_top"] = (float(top_row["x_max"]), float(top_row["cy"]))
        anchors["mcq_left_bottom"] = (float(bottom_row["x_min"]), float(bottom_row["cy"]))
        anchors["mcq_right_bottom"] = (float(bottom_row["x_max"]), float(bottom_row["cy"]))
    else:
        for name in ("mcq_left_top", "mcq_left_bottom", "mcq_right_top", "mcq_right_bottom"):
            target = expected[name]
            picked = _pick_anchor_marker(candidate_markers, target, img_w, img_h, max_norm_dist=0.09)
            if picked is None:
                fallback_used = True
                anchors[name] = (float(target[0] * img_w), float(target[1] * img_h))
            else:
                anchors[name] = (float(picked[0]), float(picked[1]))

    return anchors, fallback_used


def _build_rois_from_anchors(
    anchors,
    img_w: int,
    img_h: int,
    rows_per_block: int,
    sid_roi_cfg,
    code_roi_cfg,
    mcq_roi_cfg,
):
    if sid_roi_cfg is None:
        sid_top = anchors["sid_top"]
        sid_bottom = anchors["sid_bottom"]
        sid_span = max(120.0, float(abs(sid_bottom[1] - sid_top[1])))
        sid_digit_diam = max(10.0, sid_span / 10.0)
        sid_w = int(round(img_w * 0.20))
        sid_h = int(round(max(220.0, sid_span - img_h * 0.020)))
        sid_x = int(round(sid_top[0] + img_w * 0.004))
        sid_y = int(round(min(sid_top[1], sid_bottom[1]) + img_h * 0.010))

        sid_bottom_target = float(max(sid_top[1], sid_bottom[1]) + 0.50 * sid_digit_diam)
        sid_h = max(int(sid_h), int(round(sid_bottom_target - float(sid_y))))

        sid_x = max(0, min(sid_x, max(0, img_w - sid_w - 1)))
        sid_x, sid_y, sid_w, sid_h = _clip_rect(sid_x, sid_y, sid_w, sid_h, img_w, img_h)
        sid_roi = {"x": sid_x, "y": sid_y, "w": sid_w, "h": sid_h}
    else:
        sid_roi = dict(sid_roi_cfg)

    if code_roi_cfg is None:
        code_top = anchors["code_top"]
        code_bottom = anchors["code_bottom"]
        code_span = max(120.0, float(abs(code_bottom[1] - code_top[1])))
        code_digit_diam = max(10.0, code_span / 10.0)
        code_w = int(round(img_w * 0.10))
        code_h = int(round(max(200.0, code_span + img_h * 0.010)))
        code_x = int(round(code_top[0] + img_w * 0.008))
        code_y = int(round(min(code_top[1], code_bottom[1]) + img_h * 0.006))

        code_bottom_target = float(max(code_top[1], code_bottom[1]) + 0.50 * code_digit_diam)
        code_h = max(int(code_h), int(round(code_bottom_target - float(code_y))))

        code_x = max(0, min(code_x, max(0, img_w - code_w - 1)))
        code_x, code_y, code_w, code_h = _clip_rect(code_x, code_y, code_w, code_h, img_w, img_h)
        code_roi = {"x": code_x, "y": code_y, "w": code_w, "h": code_h}
    else:
        code_roi = dict(code_roi_cfg)

    if sid_roi_cfg is None and code_roi_cfg is None:
        min_gap = max(8, int(round(img_w * 0.010)))
        sid_right = int(sid_roi["x"] + sid_roi["w"])
        code_left = int(code_roi["x"])

        if sid_right + min_gap > code_left:
            target_sid_w = int(code_left - min_gap - sid_roi["x"])
            if target_sid_w >= 120:
                sid_roi["w"] = target_sid_w
            else:
                shifted_code_x = int(min(img_w - int(code_roi["w"]), max(0, sid_right + min_gap)))
                code_roi["x"] = shifted_code_x
                sid_roi["w"] = max(120, int(code_roi["x"] - min_gap - sid_roi["x"]))

            sid_roi["x"], sid_roi["y"], sid_roi["w"], sid_roi["h"] = _clip_rect(
                sid_roi["x"], sid_roi["y"], sid_roi["w"], sid_roi["h"], img_w, img_h
            )
            code_roi["x"], code_roi["y"], code_roi["w"], code_roi["h"] = _clip_rect(
                code_roi["x"], code_roi["y"], code_roi["w"], code_roi["h"], img_w, img_h
            )

    left_top = anchors["mcq_left_top"]
    left_bottom = anchors["mcq_left_bottom"]
    right_top = anchors["mcq_right_top"]
    right_bottom = anchors["mcq_right_bottom"]

    top_y = float(0.5 * (left_top[1] + right_top[1]))
    bottom_y = float(0.5 * (left_bottom[1] + right_bottom[1]))
    span_y = max(12.0, float(bottom_y - top_y))
    line_h = span_y / max(1.0, float(max(2, int(rows_per_block)) - 1))

    if mcq_roi_cfg is None:
        left_x = float(0.5 * (left_top[0] + left_bottom[0]))
        right_x = float(0.5 * (right_top[0] + right_bottom[0]))

        default_left = float(img_w) * 0.18
        default_right = float(img_w) * 0.80
        default_top = float(img_h) * 0.57
        default_bottom = min(
            float(img_h) * 0.94,
            default_top + (max(2, int(rows_per_block)) - 1) * (float(img_h) * 0.0152),
        )

        if not (float(img_w) * 0.12 <= left_x <= float(img_w) * 0.35):
            left_x = default_left
        if not (float(img_w) * 0.65 <= right_x <= float(img_w) * 0.90):
            right_x = default_right
        if (right_x - left_x) < (float(img_w) * 0.45) or (right_x - left_x) > (float(img_w) * 0.72):
            left_x = default_left
            right_x = default_right

        if not (float(img_h) * 0.55 <= top_y <= float(img_h) * 0.66):
            top_y = default_top

        expected_span = max(
            80.0,
            (max(2, int(rows_per_block)) - 1) * (float(img_h) * 0.0152),
        )
        measured_span = float(bottom_y - top_y)
        if measured_span < (0.80 * expected_span) or measured_span > (1.10 * expected_span):
            bottom_y = top_y + expected_span

        if not ((top_y + float(img_h) * 0.12) <= bottom_y <= (float(img_h) * 0.96)):
            bottom_y = default_bottom

        span_y = max(12.0, float(bottom_y - top_y))
        line_h = span_y / max(1.0, float(max(2, int(rows_per_block)) - 1))
        line_h = max(12.0, min(44.0, float(line_h)))

        mcq_x = int(round(left_x - 0.40 * line_h))
        mcq_w = int(round((right_x - left_x) + 0.80 * line_h))
        mcq_y = int(round(top_y - 0.45 * line_h))
        mcq_h = int(round(span_y + 0.90 * line_h))
        mcq_x, mcq_y, mcq_w, mcq_h = _clip_rect(mcq_x, mcq_y, mcq_w, mcq_h, img_w, img_h)
        mcq_roi = {"x": mcq_x, "y": mcq_y, "w": mcq_w, "h": mcq_h}
    else:
        mcq_roi = dict(mcq_roi_cfg)
        line_h = max(12.0, min(44.0, float(mcq_roi["h"]) / max(1.0, float(max(2, int(rows_per_block))))))

    return {"sid": sid_roi, "code": code_roi, "mcq": mcq_roi}, float(line_h)
