import math

import cv2
import numpy as np


def _extract_black_square_markers(
    binary_img,
    min_area_ratio=0.00002,
    max_area_ratio=0.02,
    min_fill_ratio=0.40,
    max_markers=220,
):
    """Extract black square-like connected components from a binary inverted image."""
    if binary_img is None or binary_img.size == 0:
        return []

    h, w = binary_img.shape[:2]
    total_area = float(max(1, h * w))

    prep = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    prep = cv2.morphologyEx(prep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(prep, connectivity=8)
    min_area = max(12, int(total_area * float(min_area_ratio)))
    max_area = max(min_area + 1, int(total_area * float(max_area_ratio)))

    markers = []
    for idx in range(1, int(num)):
        x, y, bw, bh, area = stats[idx]
        area = int(area)
        if area < min_area or area > max_area:
            continue
        if bw < 4 or bh < 4:
            continue

        aspect = float(bw) / max(1.0, float(bh))
        if not (0.62 <= aspect <= 1.55):
            continue

        rect_area = float(max(1, bw * bh))
        fill_ratio = float(area) / rect_area
        if fill_ratio < float(min_fill_ratio):
            continue

        comp_mask = (labels[y:y + bh, x:x + bw] == int(idx)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnt = max(cnts, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(cnt))
        if contour_area < float(area) * 0.55:
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
        vertex_count = int(len(approx))
        if vertex_count < 4 or vertex_count > 8:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / max(1.0, hull_area)
        if solidity < 0.82:
            continue

        # Reject near-circular blobs (often filled answer bubbles) when square evidence is weak.
        circularity = float((4.0 * math.pi * contour_area) / max(1.0, (peri * peri)))
        if circularity > 0.90 and fill_ratio < 0.92 and vertex_count > 5:
            continue

        cx = float(centroids[idx][0])
        cy = float(centroids[idx][1])
        size = float(0.5 * (bw + bh))

        markers.append(
            {
                "id": int(idx),
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "cx": float(cx),
                "cy": float(cy),
                "area": int(area),
                "fill": float(fill_ratio),
                "size": float(size),
                "vertices": int(vertex_count),
                "solidity": float(solidity),
                "circularity": float(circularity),
            }
        )

    markers.sort(key=lambda m: (m["area"], m["fill"]), reverse=True)
    if len(markers) > int(max_markers):
        markers = markers[: int(max_markers)]
    return markers


def _extract_black_square_markers_from_gray(gray_img, min_area_ratio=0.00002, max_area_ratio=0.02, max_markers=220):
    """Detect black square-like markers from grayscale image with illumination normalization."""
    if gray_img is None or gray_img.size == 0:
        return []

    h, w = gray_img.shape[:2]
    k = max(31, (min(h, w) // 10) | 1)
    bg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray_img, bg, scale=255)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)

    _, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _extract_black_square_markers(
        bin_inv,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_fill_ratio=0.40,
        max_markers=max_markers,
    )


def _merge_square_marker_lists(primary_markers, secondary_markers, img_w, img_h, max_markers=280):
    """Merge marker lists from multiple detector passes and deduplicate by center proximity."""
    merged = []

    def _push(marker):
        if not isinstance(marker, dict):
            return

        try:
            cx = float(marker.get("cx", 0.0))
            cy = float(marker.get("cy", 0.0))
            size = float(marker.get("size", 0.0))
        except Exception:
            return

        if size <= 0:
            return

        best_idx = -1
        best_dist = 1e9
        for i, cur in enumerate(merged):
            csize = float(cur.get("size", 0.0))
            dist = math.hypot(cx - float(cur.get("cx", 0.0)), cy - float(cur.get("cy", 0.0)))
            merge_dist = max(4.0, min(size, csize) * 0.60)
            if dist <= merge_dist and dist < best_dist:
                best_idx = i
                best_dist = dist

        if best_idx >= 0:
            cur = merged[best_idx]
            cur_score = float(cur.get("area", 0.0)) * (0.8 + float(cur.get("fill", 0.0)))
            new_score = float(marker.get("area", 0.0)) * (0.8 + float(marker.get("fill", 0.0)))
            if new_score > cur_score:
                merged[best_idx] = dict(marker)
            return

        merged.append(dict(marker))

    for m in list(primary_markers or []):
        _push(m)
    for m in list(secondary_markers or []):
        _push(m)

    for i, m in enumerate(merged):
        m["id"] = int(i + 1)
        m["cx"] = float(max(0.0, min(float(img_w - 1), float(m.get("cx", 0.0)))))
        m["cy"] = float(max(0.0, min(float(img_h - 1), float(m.get("cy", 0.0)))))

    merged.sort(key=lambda m: (float(m.get("area", 0.0)), float(m.get("fill", 0.0))), reverse=True)
    if len(merged) > int(max_markers):
        merged = merged[: int(max_markers)]
    return merged


def _detect_page_corners_from_black_square_markers(gray_img):
    """Find page corners from four large black square markers near sheet corners."""
    if gray_img is None or gray_img.size == 0:
        return None

    h, w = gray_img.shape[:2]
    markers = _extract_black_square_markers_from_gray(
        gray_img,
        min_area_ratio=0.00008,
        max_area_ratio=0.025,
        max_markers=260,
    )
    if len(markers) < 4:
        return None

    def _pick(cands, tx, ty):
        if not cands:
            return None
        best = None
        best_score = -1e9
        for m in cands:
            dx = float(m["cx"]) - float(tx)
            dy = float(m["cy"]) - float(ty)
            dist_norm = (dx * dx + dy * dy) / max(1.0, float(w * h))
            area_norm = float(m["area"]) / max(1.0, float(w * h))
            score = area_norm * 1000.0 + float(m["fill"]) * 40.0 - dist_norm * 230.0
            if score > best_score:
                best_score = score
                best = m
        return best

    corner_w = int(w * 0.30)
    corner_h = int(h * 0.30)
    tl = _pick([m for m in markers if m["cx"] <= corner_w and m["cy"] <= corner_h], 0.0, 0.0)
    tr = _pick([m for m in markers if m["cx"] >= (w - corner_w) and m["cy"] <= corner_h], float(w), 0.0)
    bl = _pick([m for m in markers if m["cx"] <= corner_w and m["cy"] >= (h - corner_h)], 0.0, float(h))
    br = _pick([m for m in markers if m["cx"] >= (w - corner_w) and m["cy"] >= (h - corner_h)], float(w), float(h))

    if tl is None or tr is None or bl is None or br is None:
        return None

    pad = int(max(3, min(10, 0.25 * np.median([tl["size"], tr["size"], bl["size"], br["size"]]))))
    tl_pt = [max(0, int(tl["x"]) - pad), max(0, int(tl["y"]) - pad)]
    tr_pt = [min(w - 1, int(tr["x"] + tr["w"] + pad)), max(0, int(tr["y"]) - pad)]
    bl_pt = [max(0, int(bl["x"]) - pad), min(h - 1, int(bl["y"] + bl["h"] + pad))]
    br_pt = [min(w - 1, int(br["x"] + br["w"] + pad)), min(h - 1, int(br["y"] + br["h"] + pad))]

    return np.float32([tl_pt, tr_pt, bl_pt, br_pt])


def _build_vertical_marker_pairs(marker_list, img_w, img_h, top_max_ratio=0.60):
    """Build vertical marker pairs (top and bottom anchors) from marker list."""
    if not marker_list:
        return []

    max_x_diff = max(10.0, float(img_w) * 0.04)
    min_dy = max(24.0, float(img_h) * 0.06)
    max_dy = float(img_h) * 0.38

    pairs = []
    markers = sorted(marker_list, key=lambda m: float(m.get("cy", 0.0)))
    for i in range(len(markers)):
        a = markers[i]
        for j in range(i + 1, len(markers)):
            b = markers[j]
            dy = float(b.get("cy", 0.0)) - float(a.get("cy", 0.0))
            if dy < min_dy:
                continue
            if dy > max_dy:
                break

            if float(a.get("cy", 0.0)) > float(img_h) * float(top_max_ratio):
                continue

            x_diff = abs(float(a.get("cx", 0.0)) - float(b.get("cx", 0.0)))
            if x_diff > max_x_diff:
                continue

            area_a = float(a.get("area", 1.0))
            area_b = float(b.get("area", 1.0))
            area_sim = min(area_a, area_b) / max(1.0, max(area_a, area_b))
            if area_sim < 0.35:
                continue

            size = 0.5 * (float(a.get("size", 0.0)) + float(b.get("size", 0.0)))
            score = area_sim * 2.1 + (dy / max(1.0, float(img_h))) * 3.5 - (x_diff / max_x_diff) * 1.4
            pairs.append(
                {
                    "top": a,
                    "bottom": b,
                    "x": float(0.5 * (float(a.get("cx", 0.0)) + float(b.get("cx", 0.0)))),
                    "cx": float(0.5 * (float(a.get("cx", 0.0)) + float(b.get("cx", 0.0)))),
                    "dy": float(dy),
                    "size": float(size),
                    "score": float(score),
                }
            )

    pairs.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
    return pairs
