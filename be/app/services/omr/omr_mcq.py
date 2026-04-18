from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .omr_labels import choice_label


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


def _clip_rect_local(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return int(x), int(y), int(w), int(h)


def _pick_anchor_template(search_gray: np.ndarray) -> Optional[np.ndarray]:
    if search_gray is None or search_gray.size == 0:
        return None

    h, w = search_gray.shape[:2]
    if h < 20 or w < 20:
        return None

    top_h = max(40, min(h, int(round(0.58 * h))))
    top_band = search_gray[:top_h, :]
    blur = cv2.GaussianBlur(top_band, (3, 3), 0)
    _, band_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    band_inv = cv2.morphologyEx(band_inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(band_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = max(24, int(round(0.00012 * float(w * top_h))))
    max_area = max(min_area + 1, int(round(0.0060 * float(w * top_h))))
    best = None
    best_score = -1e9

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        if area < min_area or area > max_area:
            continue

        aspect = float(bw) / max(1.0, float(bh))
        if not (0.60 <= aspect <= 1.45):
            continue

        contour_area = float(cv2.contourArea(cnt))
        fill_ratio = contour_area / max(1.0, area)
        if fill_ratio < 0.50:
            continue

        x_center_norm = (float(x) + 0.5 * float(bw)) / max(1.0, float(w))
        y_norm = (float(y) + 0.5 * float(bh)) / max(1.0, float(top_h))
        square_size = min(float(bw), float(bh)) / max(1.0, float(min(w, top_h)))
        center_bias = 1.0 - abs(x_center_norm - 0.5)

        score = (2.0 * fill_ratio) + (1.1 * square_size) + (0.45 * center_bias) - (0.38 * y_norm)
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    if best is None:
        return None

    bx, by, bw, bh = best
    pad = max(2, int(round(0.22 * float(max(bw, bh)))))
    x1, y1, tw, th = _clip_rect_local(bx - pad, by - pad, bw + (2 * pad), bh + (2 * pad), w, top_h)
    template = top_band[y1 : y1 + th, x1 : x1 + tw]
    if template.size == 0:
        return None

    return template


def _cluster_match_rows(matches: List[Dict[str, float]], y_tol: float) -> List[Dict[str, object]]:
    if not matches:
        return []

    rows: List[Dict[str, object]] = []
    for item in sorted(matches, key=lambda m: float(m["cy"])):
        cy = float(item["cy"])
        hit = None
        for row in rows:
            if abs(cy - float(row["cy"])) <= float(y_tol):
                hit = row
                break

        if hit is None:
            rows.append({"cy": cy, "points": [item]})
        else:
            hit_points = list(hit["points"])
            hit_points.append(item)
            hit["points"] = hit_points
            hit["cy"] = float(np.mean([float(p["cy"]) for p in hit_points]))

    out = []
    for row in rows:
        points = list(row["points"])
        if len(points) < 2:
            continue
        out.append(
            {
                "cy": float(row["cy"]),
                "count": int(len(points)),
                "points": points,
            }
        )
    return out


def _pick_three_markers_on_row(
    row_points: Sequence[Dict[str, float]],
    x_left: float,
    x_right: float,
) -> Optional[List[Dict[str, float]]]:
    points = [p for p in list(row_points or []) if isinstance(p, dict)]
    if len(points) < 3:
        return None

    span = max(40.0, float(x_right - x_left))
    # Three expected fiducials around question columns 1, 18, 35.
    expected_x = [
        float(x_left + 0.14 * span),
        float(x_left + 0.50 * span),
        float(x_left + 0.86 * span),
    ]

    used: set[int] = set()
    selected: List[Dict[str, float]] = []
    for target_x in expected_x:
        candidates = []
        for idx, item in enumerate(points):
            if idx in used:
                continue
            cx = float(item.get("cx", -1.0))
            if cx < 0.0:
                continue
            score = float(item.get("score", 0.0))
            candidates.append((abs(cx - target_x), -score, idx))
        if not candidates:
            continue
        candidates.sort(key=lambda t: (float(t[0]), float(t[1])))
        best_idx = int(candidates[0][2])
        used.add(best_idx)
        selected.append(points[best_idx])

    if len(selected) < 3:
        sorted_pts = sorted(points, key=lambda p: float(p.get("cx", 0.0)))
        gap_min = max(8.0, 0.10 * span)
        deduped: List[Dict[str, float]] = []
        for item in sorted_pts:
            cx = float(item.get("cx", 0.0))
            if not deduped or abs(cx - float(deduped[-1].get("cx", 0.0))) >= gap_min:
                deduped.append(item)
        if len(deduped) >= 3:
            mid = int(len(deduped) // 2)
            selected = [deduped[0], deduped[mid], deduped[-1]]

    if len(selected) != 3:
        return None

    selected = sorted(selected, key=lambda p: float(p.get("cx", 0.0)))
    row_span = float(selected[-1].get("cx", 0.0) - selected[0].get("cx", 0.0))
    if row_span < max(90.0, 0.38 * span):
        return None
    return selected


def refine_mcq_roi(
    source_img: np.ndarray,
    gray_img: np.ndarray,
    mcq_roi: Dict[str, int],
    top_padding_px: int = 5,
    side_padding_px: int = 10,
    bottom_padding_px: int = 15,
) -> Tuple[Dict[str, int], np.ndarray, Dict[str, object]]:
    """Refine MCQ ROI by 3-top-fiducial template matching and clean crop output."""
    meta: Dict[str, object] = {
        "used": False,
        "reason": "init",
        "y_anchor_top": None,
        "y_anchor_bottom": None,
        "line_h": None,
        "matches": 0,
        "top_row_centers": [],
        "bottom_row_centers": [],
    }

    if gray_img is None or gray_img.size == 0 or source_img is None or source_img.size == 0:
        meta["reason"] = "invalid-image"
        return dict(mcq_roi), source_img, meta

    img_h, img_w = gray_img.shape[:2]
    x, y, w, h = _clip_rect_local(
        _safe_int(mcq_roi.get("x"), 0),
        _safe_int(mcq_roi.get("y"), 0),
        _safe_int(mcq_roi.get("w"), img_w),
        _safe_int(mcq_roi.get("h"), img_h),
        img_w,
        img_h,
    )

    base_roi = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    base_crop = source_img[y : y + h, x : x + w]
    if base_crop.size == 0 or w < 80 or h < 120:
        meta["reason"] = "roi-too-small"
        return base_roi, base_crop, meta

    top_pad = max(22, int(round(0.34 * float(h))))
    bottom_pad = max(24, int(round(0.14 * float(h))))
    sx1, sy1, sw, sh = _clip_rect_local(x, y - top_pad, w, h + top_pad + bottom_pad, img_w, img_h)
    search_gray = gray_img[sy1 : sy1 + sh, sx1 : sx1 + sw]
    if search_gray.size == 0:
        meta["reason"] = "empty-search-window"
        return base_roi, base_crop, meta

    template = _pick_anchor_template(search_gray)
    if template is None:
        meta["reason"] = "template-not-found"
        return base_roi, base_crop, meta

    th, tw = template.shape[:2]
    if th < 8 or tw < 8:
        meta["reason"] = "template-too-small"
        return base_roi, base_crop, meta

    result = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
    if result is None or result.size == 0:
        meta["reason"] = "template-match-empty"
        return base_roi, base_crop, meta

    threshold_levels = [0.80, 0.74, 0.68, 0.62, 0.56]
    matches: List[Dict[str, float]] = []

    blur = cv2.GaussianBlur(search_gray, (3, 3), 0)
    _, search_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    min_dist = max(6.0, 0.65 * float(max(1, min(th, tw))))
    for threshold in threshold_levels:
        ys, xs = np.where(result >= float(threshold))
        if len(xs) <= 0:
            continue

        candidates = sorted(
            [
                {
                    "x": float(xi),
                    "y": float(yi),
                    "score": float(result[yi, xi]),
                }
                for yi, xi in zip(ys, xs)
            ],
            key=lambda m: float(m["score"]),
            reverse=True,
        )

        picked_local: List[Dict[str, float]] = []
        for cand in candidates:
            cx = float(cand["x"]) + (0.5 * float(tw))
            cy = float(cand["y"]) + (0.5 * float(th))

            overlap = False
            for done in picked_local:
                if math.hypot(cx - float(done["cx"]), cy - float(done["cy"])) < float(min_dist):
                    overlap = True
                    break
            if overlap:
                continue

            px1, py1, pw, ph = _clip_rect_local(
                _safe_int(cand["x"], 0),
                _safe_int(cand["y"], 0),
                int(tw),
                int(th),
                sw,
                sh,
            )
            patch = search_inv[py1 : py1 + ph, px1 : px1 + pw]
            if patch.size == 0:
                continue

            fill_ratio = float(cv2.countNonZero(patch)) / float(max(1, patch.shape[0] * patch.shape[1]))
            if fill_ratio < 0.36:
                continue

            picked_local.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "score": float(cand["score"]),
                }
            )

            if len(picked_local) >= 220:
                break

        if len(picked_local) >= 9:
            matches = picked_local
            break

    if len(matches) < 6:
        meta["reason"] = "insufficient-matches"
        return base_roi, base_crop, meta

    global_matches = [
        {
            "cx": float(sx1 + m["cx"]),
            "cy": float(sy1 + m["cy"]),
            "score": float(m["score"]),
        }
        for m in matches
    ]

    row_tol = max(4.0, 0.32 * float(min(th, tw)))
    rows = _cluster_match_rows(global_matches, y_tol=row_tol)
    if len(rows) < 2:
        meta["reason"] = "insufficient-anchor-rows"
        return base_roi, base_crop, meta

    rows = sorted(rows, key=lambda row: float(row["cy"]))
    candidate_rows = []
    min_row_span = max(90.0, 0.42 * float(w))
    for row in rows:
        pts = list(row.get("points") or [])
        if len(pts) < 3:
            continue
        xs = [float(p.get("cx", 0.0)) for p in pts]
        if not xs:
            continue
        if (max(xs) - min(xs)) < min_row_span:
            continue
        candidate_rows.append(row)

    if len(candidate_rows) < 2:
        meta["reason"] = "no-row-with-3-fiducials"
        return base_roi, base_crop, meta

    top_row = candidate_rows[0]
    top_markers = _pick_three_markers_on_row(top_row.get("points") or [], float(x), float(x + w))
    if top_markers is None:
        meta["reason"] = "top-row-not-3-markers"
        return base_roi, base_crop, meta

    min_bottom_gap = max(80.0, 0.22 * float(sh))
    bottom_markers = None
    bottom_row = None
    for row in reversed(candidate_rows[1:]):
        if float(row.get("cy", 0.0)) - float(top_row.get("cy", 0.0)) < min_bottom_gap:
            continue
        maybe = _pick_three_markers_on_row(row.get("points") or [], float(x), float(x + w))
        if maybe is None:
            continue
        bottom_markers = maybe
        bottom_row = row
        break

    if bottom_markers is None or bottom_row is None:
        meta["reason"] = "bottom-row-not-found"
        return base_roi, base_crop, meta

    y_anchor_top = float(np.mean([float(p.get("cy", 0.0)) for p in top_markers]))
    y_anchor_bottom = float(np.mean([float(p.get("cy", 0.0)) for p in bottom_markers]))
    if y_anchor_bottom <= y_anchor_top:
        meta["reason"] = "invalid-anchor-order"
        return base_roi, base_crop, meta

    # Grid projection from marker rows: 16 intervals between top and bottom marker rows.
    line_h = (y_anchor_bottom - y_anchor_top) / 16.0
    if not (6.0 <= line_h <= 72.0):
        meta["reason"] = "line-height-out-of-range"
        return base_roi, base_crop, meta

    # Hard top boundary: start below top marker row to avoid instruction text overlap.
    y_top = float(y_anchor_top + max(0, int(top_padding_px)))

    # Bottom safety margin: keep at least +15 px under last marker-row center.
    y_bottom = float(y_anchor_bottom + max(15.0, float(bottom_padding_px)))
    y_bottom = max(y_bottom, float(y_anchor_bottom + 0.45 * line_h + 15.0))

    expand_lr = max(0, int(side_padding_px))
    desired_w = int(w + (2 * expand_lr))
    rx1 = int(max(0, x - expand_lr))
    rx2 = int(min(img_w, x + w + expand_lr))
    cur_w = int(max(1, rx2 - rx1))
    if desired_w <= img_w and cur_w < desired_w:
        deficit = int(desired_w - cur_w)
        if rx1 == 0 and rx2 < img_w:
            rx2 = int(min(img_w, rx2 + deficit))
        elif rx2 == img_w and rx1 > 0:
            rx1 = int(max(0, rx1 - deficit))

    ry1 = int(round(max(0.0, y_top)))
    ry2 = int(round(min(float(img_h), y_bottom)))

    ry1 = max(0, min(ry1, img_h - 1))
    ry2 = max(ry1 + 1, min(ry2, img_h))
    if (ry2 - ry1) < 120:
        meta["reason"] = "refined-height-too-small"
        return base_roi, base_crop, meta

    refined_roi = {
        "x": int(rx1),
        "y": int(ry1),
        "w": int(max(1, rx2 - rx1)),
        "h": int(max(1, ry2 - ry1)),
    }
    refined_crop = source_img[ry1:ry2, rx1:rx2]
    if refined_crop.size == 0:
        meta["reason"] = "empty-refined-crop"
        return base_roi, base_crop, meta

    meta.update(
        {
            "used": True,
            "reason": "ok",
            "y_anchor_top": round(float(y_anchor_top), 4),
            "y_anchor_bottom": round(float(y_anchor_bottom), 4),
            "line_h": round(float(line_h), 4),
            "matches": int(len(global_matches)),
            "top_row_centers": [
                {
                    "x": round(float(item["cx"]), 3),
                    "y": round(float(item["cy"]), 3),
                }
                for item in top_markers
            ],
            "bottom_row_centers": [
                {
                    "x": round(float(item["cx"]), 3),
                    "y": round(float(item["cy"]), 3),
                }
                for item in bottom_markers
            ],
            "refined_roi": dict(refined_roi),
        }
    )

    return refined_roi, refined_crop, meta


def _cluster_marker_rows(markers, min_x: float, max_x: float, min_y: float, max_y: float, y_tol: float = 6.0):
    pool = []
    for marker in list(markers or []):
        cx = _safe_float(marker.get("cx"), -1.0)
        cy = _safe_float(marker.get("cy"), -1.0)
        if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
            continue
        pool.append(marker)

    if not pool:
        return []

    pool.sort(key=lambda m: _safe_float(m.get("cy"), 0.0))
    rows = []
    cur = [pool[0]]
    for marker in pool[1:]:
        cy = _safe_float(marker.get("cy"), 0.0)
        cur_center = float(np.mean([_safe_float(m.get("cy"), 0.0) for m in cur]))
        if abs(cy - cur_center) <= float(y_tol):
            cur.append(marker)
        else:
            rows.append(cur)
            cur = [marker]
    if cur:
        rows.append(cur)

    out = []
    for group in rows:
        xs = [_safe_float(m.get("cx"), 0.0) for m in group]
        ys = [_safe_float(m.get("cy"), 0.0) for m in group]
        out.append(
            {
                "cy": float(np.mean(ys)),
                "x_min": float(min(xs)),
                "x_max": float(max(xs)),
                "count": int(len(group)),
                "markers": group,
            }
        )
    return out


def _infer_mcq_geometry_from_markers(markers, img_w: int, img_h: int) -> Dict[str, object]:
    rows = _cluster_marker_rows(
        markers,
        min_x=float(img_w) * 0.10,
        max_x=float(img_w) * 0.90,
        min_y=float(img_h) * 0.44,
        max_y=float(img_h) * 0.98,
        y_tol=6.0,
    )
    if not rows:
        return {}

    def _dedupe_x(xs: Sequence[float], min_gap: float) -> List[float]:
        out: List[float] = []
        for val in sorted(float(v) for v in xs):
            if not out or abs(val - out[-1]) >= float(min_gap):
                out.append(float(val))
        return out

    fid_rows = []
    bubble_rows = []

    for row in rows:
        row_markers = list(row.get("markers") or [])
        row_count = int(row.get("count", 0))
        row_x_min = float(row.get("x_min", 0.0))
        row_x_max = float(row.get("x_max", 0.0))
        row_span = float(row_x_max - row_x_min)
        row_cy = float(row.get("cy", 0.0))

        dark_large = []
        for marker in row_markers:
            if (
                _safe_float(marker.get("fill"), 0.0) >= 0.86
                and _safe_float(marker.get("area"), 0.0) >= 360.0
                and _safe_float(marker.get("size"), 0.0) >= 15.0
            ):
                dark_large.append(marker)

        if len(dark_large) >= 2 and row_span >= (float(img_w) * 0.34) and row_count <= 8:
            xs = [_safe_float(m.get("cx"), -1.0) for m in dark_large]
            ys = [_safe_float(m.get("cy"), -1.0) for m in dark_large]
            xs = [x for x in xs if x >= 0.0]
            ys = [y for y in ys if y >= 0.0]
            if len(xs) >= 2 and ys:
                fid_rows.append(
                    {
                        "cy": float(np.mean(ys)),
                        "xs": xs,
                        "x_min": float(min(xs)),
                        "x_max": float(max(xs)),
                        "count": int(len(xs)),
                    }
                )

        if row_count >= 8 and row_span >= (float(img_w) * 0.50):
            xs = sorted(
                _safe_float(marker.get("cx"), -1.0)
                for marker in row_markers
                if _safe_float(marker.get("cx"), -1.0) >= 0.0
            )
            bubble_rows.append(
                {
                    "cy": row_cy,
                    "x_min": row_x_min,
                    "x_max": row_x_max,
                    "count": row_count,
                    "xs": xs,
                }
            )

    out: Dict[str, object] = {}

    fid_top_xs: List[float] = []
    if fid_rows:
        fid_rows = sorted(fid_rows, key=lambda row: float(row.get("cy", 0.0)))
        fid_top = fid_rows[0]
        fid_bottom = fid_rows[-1]
        fid_top_xs = _dedupe_x(fid_top.get("xs") or [], min_gap=float(img_w) * 0.08)
        if 2 <= len(fid_top_xs) <= 6:
            out["block_count"] = float(len(fid_top_xs))
        out["fid_top_y"] = float(fid_top.get("cy", 0.0))
        out["fid_bottom_y"] = float(fid_bottom.get("cy", 0.0))
        out["fid_left_x"] = float(
            np.median(
                [
                    float(fid_top.get("x_min", 0.0)),
                    float(fid_bottom.get("x_min", 0.0)),
                ]
            )
        )
        out["fid_right_x"] = float(
            np.median(
                [
                    float(fid_top.get("x_max", 0.0)),
                    float(fid_bottom.get("x_max", 0.0)),
                ]
            )
        )

    if bubble_rows:
        bubble_rows = sorted(bubble_rows, key=lambda row: float(row.get("cy", 0.0)))
        top_bubble = bubble_rows[0]
        bottom_bubble = bubble_rows[-1]
        diffs = []
        for idx in range(len(bubble_rows) - 1):
            dy = float(bubble_rows[idx + 1]["cy"] - bubble_rows[idx]["cy"])
            if 12.0 <= dy <= 65.0:
                diffs.append(dy)
        if diffs:
            out["line_h"] = float(np.median(diffs))

        sample_rows = bubble_rows[: min(4, len(bubble_rows))]
        if len(bubble_rows) > 4:
            sample_rows += bubble_rows[-min(4, len(bubble_rows)) :]
        if sample_rows:
            out["left_x"] = float(np.median([float(row["x_min"]) for row in sample_rows]))
            out["right_x"] = float(np.median([float(row["x_max"]) for row in sample_rows]))

        band_rows = bubble_rows[: min(4, len(bubble_rows))]
        band_groups: List[List[float]] = []
        if not band_groups:
            bucket_w = max(8.0, float(img_w) * 0.010)
            x_buckets: Dict[int, Dict[str, object]] = {}
            for ridx, row in enumerate(band_rows):
                row_xs = _dedupe_x(row.get("xs") or [], min_gap=float(img_w) * 0.018)
                if fid_top_xs:
                    row_xs = [
                        float(x)
                        for x in row_xs
                        if min(abs(float(x) - float(fx)) for fx in fid_top_xs) > (float(img_w) * 0.030)
                    ]
                for x in row_xs:
                    key = int(round(float(x) / bucket_w))
                    bucket = x_buckets.get(key)
                    if bucket is None:
                        x_buckets[key] = {"sum": float(x), "count": 1, "rows": {int(ridx)}}
                    else:
                        bucket["sum"] = float(bucket["sum"]) + float(x)
                        bucket["count"] = int(bucket["count"]) + 1
                        rows = set(bucket.get("rows") or set())
                        rows.add(int(ridx))
                        bucket["rows"] = rows

            min_row_hits = max(2, int(math.ceil(len(band_rows) * 0.5)))
            merged_xs = []
            for bucket in x_buckets.values():
                row_hits = len(set(bucket.get("rows") or set()))
                if row_hits < min_row_hits:
                    continue
                merged_xs.append(float(bucket["sum"]) / max(1, int(bucket["count"])))

            merged_xs = sorted(merged_xs)
            if len(merged_xs) >= 8:
                diffs = [merged_xs[i + 1] - merged_xs[i] for i in range(len(merged_xs) - 1)]
                small_diffs = [d for d in diffs if 8.0 <= d <= 80.0]
                if small_diffs:
                    split_gap = max(58.0, float(np.median(small_diffs)) * 2.1)
                    groups: List[List[float]] = []
                    current: List[float] = [merged_xs[0]]
                    for val in merged_xs[1:]:
                        if (val - current[-1]) > split_gap:
                            groups.append(current)
                            current = [val]
                        else:
                            current.append(val)
                    if current:
                        groups.append(current)

                    band_groups = [grp for grp in groups if len(grp) >= 3]

        if 2 <= len(band_groups) <= 6:
            out["block_bands"] = [
                {
                    "x_min": float(min(grp)),
                    "x_max": float(max(grp)),
                }
                for grp in band_groups
            ]
            if "block_count" not in out or len(band_groups) >= int(_safe_int(out.get("block_count"), 0)):
                out["block_count"] = float(len(band_groups))

        out["top_center_y"] = float(top_bubble.get("cy", 0.0))
        out["bottom_center_y"] = float(bottom_bubble.get("cy", 0.0))

    return out


def build_mcq_roi_from_black_markers(
    mcq_roi: Dict[str, int],
    mcq_geometry: Dict[str, object],
    img_w: int,
    img_h: int,
    top_padding_px: int = 5,
    side_padding_px: int = 10,
    bottom_padding_px: int = 15,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    """Build MCQ ROI directly from detected black frame markers."""
    x, y, w, h = _clip_rect_local(
        _safe_int(mcq_roi.get("x"), 0),
        _safe_int(mcq_roi.get("y"), 0),
        _safe_int(mcq_roi.get("w"), img_w),
        _safe_int(mcq_roi.get("h"), img_h),
        int(img_w),
        int(img_h),
    )
    base_roi = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    meta: Dict[str, object] = {
        "used": False,
        "reason": "init",
        "y_top_anchor": None,
        "y_bottom_anchor": None,
        "x_left_anchor": None,
        "x_right_anchor": None,
        "line_h": None,
        "x_source": None,
        "y_source": None,
    }

    if not isinstance(mcq_geometry, dict):
        meta["reason"] = "invalid-geometry"
        return base_roi, meta

    y_top_fid = _safe_float(mcq_geometry.get("fid_top_y"), -1.0)
    y_bottom_fid = _safe_float(mcq_geometry.get("fid_bottom_y"), -1.0)
    fid_y_span = float(y_bottom_fid - y_top_fid)
    fid_y_valid = y_top_fid > 0.0 and y_bottom_fid > y_top_fid and fid_y_span >= max(60.0, 0.20 * float(h))

    if fid_y_valid:
        y_top_anchor = float(y_top_fid)
        y_bottom_anchor = float(y_bottom_fid)
        y_source = "fid"
    else:
        y_top_anchor = _safe_float(mcq_geometry.get("top_center_y"), -1.0)
        y_bottom_anchor = _safe_float(mcq_geometry.get("bottom_center_y"), -1.0)
        y_source = "bubble-centers"

    if y_top_anchor <= 0.0 or y_bottom_anchor <= y_top_anchor:
        meta["reason"] = "missing-anchor-y"
        return base_roi, meta

    x_source = "fallback-roi"
    x_left_anchor = -1.0
    x_right_anchor = -1.0

    raw_bands = mcq_geometry.get("block_bands")
    if isinstance(raw_bands, (list, tuple)):
        band_ranges: List[Tuple[float, float]] = []
        for item in list(raw_bands):
            if not isinstance(item, dict):
                continue
            bx1 = _safe_float(item.get("x_min"), -1.0)
            bx2 = _safe_float(item.get("x_max"), -1.0)
            if bx2 > bx1 > 0.0:
                band_ranges.append((float(bx1), float(bx2)))
        if band_ranges:
            x_left_anchor = float(min(val[0] for val in band_ranges))
            x_right_anchor = float(max(val[1] for val in band_ranges))
            x_source = "block-bands"

    if x_right_anchor <= x_left_anchor:
        x_left_geom = _safe_float(mcq_geometry.get("left_x"), -1.0)
        x_right_geom = _safe_float(mcq_geometry.get("right_x"), -1.0)
        if x_right_geom > x_left_geom:
            x_left_anchor = float(x_left_geom)
            x_right_anchor = float(x_right_geom)
            x_source = "bubble-span"

    if x_right_anchor <= x_left_anchor and fid_y_valid:
        x_left_fid = _safe_float(mcq_geometry.get("fid_left_x"), -1.0)
        x_right_fid = _safe_float(mcq_geometry.get("fid_right_x"), -1.0)
        if x_right_fid > x_left_fid:
            x_left_anchor = float(x_left_fid)
            x_right_anchor = float(x_right_fid)
            x_source = "fid"

    if x_right_anchor <= x_left_anchor:
        x_left_anchor = float(x)
        x_right_anchor = float(x + w)
        x_source = "fallback-roi"

    anchor_span_y = float(y_bottom_anchor - y_top_anchor)
    line_h_from_span = anchor_span_y / 16.0
    line_h_geom = _safe_float(mcq_geometry.get("line_h"), -1.0)
    if line_h_geom > 0.0 and line_h_from_span > 0.0 and (0.45 * line_h_from_span) <= line_h_geom <= (1.80 * line_h_from_span):
        line_h = float(line_h_geom)
    elif line_h_from_span > 0.0:
        line_h = float(line_h_from_span)
    else:
        line_h = max(8.0, float(h) / 20.0)
    line_h = max(6.0, min(44.0, float(line_h)))

    y_top = float(y_top_anchor + max(0, int(top_padding_px)))
    y_bottom = float(y_bottom_anchor + max(float(bottom_padding_px), 0.45 * float(line_h) + 15.0))

    x1 = float(x_left_anchor - max(0, int(side_padding_px)))
    x2 = float(x_right_anchor + max(0, int(side_padding_px)))

    rx1 = int(round(max(0.0, x1)))
    rx2 = int(round(min(float(img_w), x2)))
    ry1 = int(round(max(0.0, y_top)))
    ry2 = int(round(min(float(img_h), y_bottom)))

    min_w = max(120, int(round(0.55 * float(w))))
    min_h = max(120, int(round(anchor_span_y * 1.02)))

    if (rx2 - rx1) < min_w:
        cx = 0.5 * float(x_left_anchor + x_right_anchor)
        half = 0.5 * float(min_w)
        rx1 = int(round(max(0.0, cx - half)))
        rx2 = int(round(min(float(img_w), cx + half)))

    if (ry2 - ry1) < min_h:
        ry2 = int(round(min(float(img_h), float(ry1 + min_h))))

    if rx2 <= rx1 or ry2 <= ry1:
        meta["reason"] = "invalid-refined-rect"
        return base_roi, meta

    refined = {
        "x": int(rx1),
        "y": int(ry1),
        "w": int(max(1, rx2 - rx1)),
        "h": int(max(1, ry2 - ry1)),
    }

    meta.update(
        {
            "used": True,
            "reason": "ok",
            "y_top_anchor": round(float(y_top_anchor), 4),
            "y_bottom_anchor": round(float(y_bottom_anchor), 4),
            "x_left_anchor": round(float(x_left_anchor), 4),
            "x_right_anchor": round(float(x_right_anchor), 4),
            "line_h": round(float(line_h), 4),
            "x_source": str(x_source),
            "y_source": str(y_source),
            "refined_roi": dict(refined),
        }
    )
    return refined, meta


def _extract_cell(gray_img, binary_inv, x1, y1, x2, y2, inner_ratio: float = 0.78):
    h, w = gray_img.shape[:2]

    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))

    g = gray_img[y1:y2, x1:x2]
    b = binary_inv[y1:y2, x1:x2]
    if g.size == 0 or b.size == 0:
        return None, None

    ratio = max(0.45, min(1.0, float(inner_ratio)))
    if ratio < 0.999:
        ch, cw = g.shape[:2]
        ih = max(2, int(round(ch * ratio)))
        iw = max(2, int(round(cw * ratio)))
        sy = max(0, (ch - ih) // 2)
        sx = max(0, (cw - iw) // 2)
        g = g[sy : sy + ih, sx : sx + iw]
        b = b[sy : sy + ih, sx : sx + iw]

    if g.size == 0 or b.size == 0:
        return None, None

    return g, b


def _cell_score(gray_cell, binary_cell) -> float:
    if gray_cell is None or binary_cell is None:
        return 0.0
    if gray_cell.size == 0 or binary_cell.size == 0:
        return 0.0

    area = float(max(1, gray_cell.shape[0] * gray_cell.shape[1]))
    fill_ratio = float(cv2.countNonZero(binary_cell)) / area

    mean_val = float(np.mean(gray_cell))
    p25 = float(np.percentile(gray_cell, 25))
    dark_mean = max(0.0, 1.0 - (mean_val / 255.0))
    dark_p25 = max(0.0, 1.0 - (p25 / 255.0))

    return float(0.55 * fill_ratio + 0.30 * dark_mean + 0.15 * dark_p25)


def _cell_density(binary_cell) -> float:
    if binary_cell is None or getattr(binary_cell, "size", 0) <= 0:
        return 0.0
    area = float(max(1, binary_cell.shape[0] * binary_cell.shape[1]))
    return float(cv2.countNonZero(binary_cell)) / area


def _parse_mcq_decode_config(profile_mcq_decode):
    cfg = profile_mcq_decode if isinstance(profile_mcq_decode, dict) else {}

    min_mark = _safe_float(
        cfg.get("min_mark_density", cfg.get("min_mark_score", cfg.get("marked_threshold", 0.55))),
        0.55,
    )
    min_margin = _safe_float(cfg.get("min_margin", 0.05), 0.05)
    min_conf_ratio = _safe_float(cfg.get("min_conf_ratio", 1.10), 1.10)
    double_mark_gap = _safe_float(cfg.get("double_mark_gap", 0.08), 0.08)

    min_mark = max(0.50, min(0.60, min_mark))
    min_margin = max(0.01, min(0.40, min_margin))
    min_conf_ratio = max(1.0, min(4.0, min_conf_ratio))
    double_mark_gap = max(0.01, min(0.30, double_mark_gap))

    soft_mark_floor = _safe_float(cfg.get("soft_mark_floor", 0.38), 0.38)
    soft_margin = _safe_float(cfg.get("soft_margin", 0.06), 0.06)
    soft_conf_ratio = _safe_float(cfg.get("soft_conf_ratio", 1.18), 1.18)
    soft_mark_floor = max(0.30, min(0.52, soft_mark_floor))
    soft_margin = max(0.02, min(0.22, soft_margin))
    soft_conf_ratio = max(1.02, min(2.50, soft_conf_ratio))
    enable_soft_single_mark_rescue = bool(cfg.get("enable_soft_single_mark_rescue", True))

    raw_offsets = cfg.get("row_offsets_px")
    if not isinstance(raw_offsets, (list, tuple)):
        raw_offsets = cfg.get("row_offsets")

    row_offsets_px: List[int] = []
    if isinstance(raw_offsets, (list, tuple)):
        for val in raw_offsets:
            row_offsets_px.append(_safe_int(val, 0))

    return {
        "min_mark_score": float(min_mark),
        "min_mark_density": float(min_mark),
        "min_margin": float(min_margin),
        "min_conf_ratio": float(min_conf_ratio),
        "double_mark_gap": float(double_mark_gap),
        "adaptive_threshold": bool(cfg.get("adaptive_threshold", False)),
        "soft_mark_floor": float(soft_mark_floor),
        "soft_margin": float(soft_margin),
        "soft_conf_ratio": float(soft_conf_ratio),
        "enable_soft_single_mark_rescue": bool(enable_soft_single_mark_rescue),
        "row_offsets_px": row_offsets_px,
    }


def _decode_mcq_with_map(
    gray_img,
    binary_inv,
    roi,
    questions: int,
    choices: int,
    rows_per_block: int,
    block_count: int,
    left_x: float,
    right_x: float,
    top_center_y: float,
    line_h: float,
    decode_cfg,
    top_shift_px: float = 0.0,
    block_bands: Optional[Sequence[Tuple[float, float]]] = None,
):
    x = float(roi["x"])
    y = float(roi["y"])
    w = float(roi["w"])
    h = float(roi["h"])

    line_h = max(6.0, float(line_h))
    block_count = max(1, int(block_count))
    choices = max(2, int(choices))

    left_x = max(x, float(left_x))
    right_x = min(x + w, float(right_x))
    if right_x - left_x < 100.0:
        left_x = x
        right_x = x + w

    span_x = float(right_x - left_x)
    band_w = span_x / float(block_count)
    if band_w < 30.0:
        band_w = max(30.0, w / float(block_count))
        left_x = x

    normalized_bands: List[Tuple[float, float]] = []
    if isinstance(block_bands, (list, tuple)):
        for item in list(block_bands):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            bx1 = float(item[0])
            bx2 = float(item[1])
            if choices > 1 and (bx2 - bx1) >= 8.0:
                step = float(bx2 - bx1) / float(max(1, choices - 1))
                bx1 -= 0.55 * step
                bx2 += 0.55 * step
            bx1 = max(x, min(x + w, float(bx1)))
            bx2 = max(x, min(x + w, float(bx2)))
            if bx2 - bx1 >= 24.0:
                normalized_bands.append((float(min(bx1, bx2)), float(max(bx1, bx2))))
    has_custom_bands = len(normalized_bands) >= int(block_count)

    min_mark = float(decode_cfg["min_mark_density"])
    min_margin = float(decode_cfg["min_margin"])
    min_conf_ratio = float(decode_cfg["min_conf_ratio"])
    double_mark_gap = float(decode_cfg["double_mark_gap"])
    adaptive_threshold = bool(decode_cfg["adaptive_threshold"])
    soft_mark_floor = float(decode_cfg.get("soft_mark_floor", 0.38))
    soft_margin = float(decode_cfg.get("soft_margin", 0.06))
    soft_conf_ratio = float(decode_cfg.get("soft_conf_ratio", 1.18))
    enable_soft_single_mark_rescue = bool(decode_cfg.get("enable_soft_single_mark_rescue", True))
    row_offsets_px = list(decode_cfg["row_offsets_px"])

    user_answers: List[int] = []
    answer_confidences: List[float] = []
    uncertain_questions: List[int] = []
    double_mark_questions: List[int] = []
    rows_payload: List[Dict[str, object]] = []

    total_questions = max(1, int(questions))

    for q_idx in range(total_questions):
        q_num = int(q_idx + 1)
        block_idx = int(q_idx // rows_per_block)
        row_idx = int(q_idx % rows_per_block)

        if block_idx >= block_count:
            user_answers.append(-1)
            answer_confidences.append(0.0)
            uncertain_questions.append(q_num)
            rows_payload.append(
                {
                    "question": q_num,
                    "block": int(block_idx + 1),
                    "row": int(row_idx + 1),
                    "scores": [],
                    "best_score": 0.0,
                    "second_score": 0.0,
                    "margin": 0.0,
                    "selected": -1,
                    "selected_label": "-",
                    "threshold": round(min_mark, 4),
                    "uncertain": True,
                    "cell_boxes": [],
                }
            )
            continue

        row_shift = 0
        if row_idx < len(row_offsets_px):
            row_shift = int(row_offsets_px[row_idx])

        cy = float(top_center_y + row_idx * line_h + top_shift_px + row_shift)

        if has_custom_bands:
            raw_left = float(normalized_bands[block_idx][0])
            raw_right = float(normalized_bands[block_idx][1])
        else:
            raw_left = float(left_x + block_idx * band_w)
            raw_right = float(left_x + (block_idx + 1) * band_w)
        raw_span = max(8.0, float(raw_right - raw_left))

        if has_custom_bands:
            band_left = raw_left
            band_right = raw_right
        else:
            band_left = raw_left + 0.08 * raw_span
            band_right = raw_right - 0.04 * raw_span
        if (band_right - band_left) < (0.60 * raw_span):
            band_left = raw_left
            band_right = raw_right

        choice_w = max(6.0, float((band_right - band_left) / max(1, choices)))

        y1 = cy - 0.44 * line_h
        y2 = cy + 0.44 * line_h
        y1 = max(y, y1)
        y2 = min(y + h, y2)
        if y2 <= y1:
            y1 = max(y, min((y + h) - 2.0, cy - 0.36 * line_h))
            y2 = min(y + h, y1 + max(2.0, 0.72 * line_h))
        if y2 <= y1:
            y1 = max(y, min((y + h) - 2.0, cy))
            y2 = min(y + h, y1 + 2.0)

        choice_scores: List[float] = []
        density_scores: List[float] = []
        cell_boxes: List[List[int]] = []
        x_inset = 0.08 if has_custom_bands else 0.14
        for c_idx in range(int(choices)):
            x1 = band_left + c_idx * choice_w + x_inset * choice_w
            x2 = band_left + (c_idx + 1) * choice_w - x_inset * choice_w
            x1 = max(x, x1)
            x2 = min(x + w, x2)

            if x2 <= x1 or y2 <= y1:
                choice_scores.append(0.0)
                density_scores.append(0.0)
                cell_boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
                continue

            cell_gray, cell_bin = _extract_cell(gray_img, binary_inv, x1, y1, x2, y2, inner_ratio=0.78)
            density = _cell_density(cell_bin)
            darkness = _cell_score(cell_gray, cell_bin)
            score = float(0.90 * density + 0.10 * darkness)
            density_scores.append(float(density))
            choice_scores.append(float(score))
            cell_boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])

        density_arr = np.asarray(density_scores, dtype=np.float32)
        if density_arr.size <= 0:
            best_idx = -1
            best_score = 0.0
            second_score = 0.0
            row_mean = 0.0
        else:
            best_idx = int(np.argmax(density_arr))
            best_score = float(density_arr[best_idx])
            row_mean = float(np.mean(density_arr))
            if density_arr.size >= 2:
                two_best = np.partition(density_arr, -2)[-2:]
                second_score = float(np.min(two_best))
            else:
                second_score = 0.0

        dynamic_mark = float(min_mark)
        if adaptive_threshold and density_arr.size > 0:
            row_mean = float(np.mean(density_arr))
            dynamic_mark = max(min_mark * 0.95, min(0.60, row_mean + 0.06))

        margin = float(best_score - second_score)
        conf_ratio = float(best_score / max(1e-6, second_score))

        soft_threshold = max(
            float(soft_mark_floor),
            float(row_mean + 0.08),
        )
        soft_rescue_ok = (
            bool(enable_soft_single_mark_rescue)
            and (best_idx >= 0)
            and (best_score >= soft_threshold)
            and (margin >= soft_margin)
            and (conf_ratio >= soft_conf_ratio)
        )

        marked_indices = [int(i) for i, val in enumerate(density_scores) if float(val) >= dynamic_mark]
        is_double_mark = len(marked_indices) >= 2
        resolved_by_highest = False
        resolved_by_soft = False

        if best_idx < 0 or best_score < dynamic_mark:
            if (not is_double_mark) and soft_rescue_ok:
                selected = int(best_idx)
                resolved_by_soft = True
            else:
                selected = -1
                uncertain_questions.append(q_num)
        elif is_double_mark:
            if margin >= max(min_margin, double_mark_gap) and conf_ratio >= min_conf_ratio:
                selected = int(best_idx)
                resolved_by_highest = True
            else:
                selected = -1
                uncertain_questions.append(q_num)
                double_mark_questions.append(q_num)
        else:
            if margin < min_margin and conf_ratio < min_conf_ratio:
                selected = -1
                uncertain_questions.append(q_num)
            else:
                selected = int(best_idx)

        user_answers.append(int(selected))
        answer_confidences.append(round(float(best_score), 4))

        rows_payload.append(
            {
                "question": q_num,
                "block": int(block_idx + 1),
                "row": int(row_idx + 1),
                "scores": [round(float(s), 5) for s in density_scores],
                "best_score": round(float(best_score), 5),
                "second_score": round(float(second_score), 5),
                "margin": round(float(margin), 5),
                "selected": int(selected),
                "selected_label": choice_label(int(selected)),
                "threshold": round(float(dynamic_mark), 5),
                "soft_threshold": round(float(soft_threshold), 5),
                "uncertain": bool(selected < 0),
                "double_mark": bool(is_double_mark and selected < 0),
                "multi_mark_count": int(len(marked_indices)),
                "resolved_by_highest": bool(resolved_by_highest),
                "resolved_by_soft": bool(resolved_by_soft),
                "cell_boxes": cell_boxes,
            }
        )

    return {
        "user_answers": user_answers,
        "answer_confidences": answer_confidences,
        "uncertain_questions": uncertain_questions,
        "double_mark_questions": sorted(set(int(q) for q in double_mark_questions)),
        "rows": rows_payload,
        "line_h": float(line_h),
        "left_x": float(left_x),
        "right_x": float(right_x),
        "top_center_y": float(top_center_y),
    }


def _detect_q5_start_drift(mcq_result, min_mark_score: float) -> bool:
    rows = list(mcq_result.get("rows") or [])
    if len(rows) < 5:
        return False

    first_count = min(4, max(1, len(rows) - 1))
    next_count = min(4, len(rows) - first_count)
    first_rows = rows[:first_count]
    next_rows = rows[first_count : first_count + next_count]

    weak_first = 0
    for row in first_rows:
        if int(row.get("selected", -1)) < 0 or _safe_float(row.get("best_score"), 0.0) < float(min_mark_score):
            weak_first += 1

    strong_next = 0
    for row in next_rows:
        if int(row.get("selected", -1)) >= 0 and _safe_float(row.get("best_score"), 0.0) >= float(min_mark_score):
            strong_next += 1

    first_selected = [int(row.get("selected", -1)) for row in first_rows]
    first_margins = [_safe_float(row.get("margin"), 0.0) for row in first_rows]
    left_bias = sum(1 for sel in first_selected if sel == 0) >= 3
    weak_margin = (float(np.mean(first_margins)) if first_margins else 0.0) < 0.035
    next_confident = sum(1 for row in next_rows if int(row.get("selected", -1)) >= 0) >= 2

    if left_bias and weak_margin and next_confident:
        return True

    if next_rows:
        next_margins = [_safe_float(row.get("margin"), 0.0) for row in next_rows]
        next_margin_mean = float(np.mean(next_margins)) if next_margins else 0.0
    else:
        next_margin_mean = 0.0

    first_non_uncertain = [sel for sel in first_selected if sel >= 0]
    first_unique = len(set(first_non_uncertain)) if first_non_uncertain else 0
    low_diversity = first_unique <= 1 and len(first_non_uncertain) >= 2

    if len(rows) < 8:
        return bool((weak_first >= max(2, first_count - 1)) and low_diversity and (next_margin_mean > 0.010))

    return bool(weak_first >= 3 and strong_next >= 3)


def _mcq_quality(mcq_result) -> float:
    rows = list(mcq_result.get("rows") or [])
    if not rows:
        return 0.0

    confident = 0
    first4 = 0
    margin_sum = 0.0
    for idx, row in enumerate(rows):
        selected = int(row.get("selected", -1))
        margin = _safe_float(row.get("margin"), 0.0)
        margin_sum += margin
        if selected >= 0:
            confident += 1
            if idx < 4:
                first4 += 1

    return confident * 10.0 + first4 * 6.0 + margin_sum * 70.0
