# app/services/omr_service.py
import cv2
import numpy as np
import os
import math
import re
from datetime import datetime
from app.services.omr import until # Import file until.py cùng thư mục

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

_EASYOCR_READER = None


def _put_text_unicode(img_bgr, text, org, font_size=32, color=(0, 0, 0)):
    """Draw unicode text with PIL when available; fallback to OpenCV otherwise."""
    if Image is None or ImageDraw is None or ImageFont is None:
        cv2.putText(img_bgr, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
        return img_bgr

    try:
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = None
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
        for fp in font_candidates:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
        draw.text((int(org[0]), int(org[1])), str(text), fill=(int(color[2]), int(color[1]), int(color[0])), font=font)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img_bgr, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
        return img_bgr

def _clip_rect(x, y, w, h, max_w, max_h):
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return x, y, w, h


def _order_quad_points(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _warp_by_manual_quad(img_bgr, norm_points, target_size=None):
    """Perspective-warp by 4 normalized points (TL,TR,BR,BL-ish), returns None if invalid."""
    if norm_points is None or len(norm_points) != 4:
        return None

    h, w = img_bgr.shape[:2]
    pts = []
    for px, py in norm_points:
        x = max(0.0, min(1.0, float(px))) * max(1, w - 1)
        y = max(0.0, min(1.0, float(py))) * max(1, h - 1)
        pts.append([x, y])

    src = _order_quad_points(np.array(pts, dtype=np.float32))

    width_top = np.linalg.norm(src[1] - src[0])
    width_bottom = np.linalg.norm(src[2] - src[3])
    target_w = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(src[3] - src[0])
    height_right = np.linalg.norm(src[2] - src[1])
    target_h = int(max(height_left, height_right))

    if target_w < 100 or target_h < 100:
        return None

    if target_size is not None:
        tw = max(100, int(target_size[0]))
        th = max(100, int(target_size[1]))
    else:
        tw = target_w
        th = target_h

    dst = np.array(
        [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, matrix, (tw, th))
    return warped


def _split_grid(binary_img, rows: int, cols: int, inner_ratio: float = 1.0):
    """Split binary ROI to rows x cols cells and return filled-pixel matrix."""
    h, w = binary_img.shape[:2]
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)
    pixel_val = np.zeros((rows, cols), dtype=np.float32)
    inner_ratio = max(0.35, min(1.0, float(inner_ratio)))

    for r in range(rows):
        y1, y2 = row_edges[r], row_edges[r + 1]
        for c in range(cols):
            x1, x2 = col_edges[c], col_edges[c + 1]
            cell = binary_img[y1:y2, x1:x2]
            if inner_ratio < 0.999 and cell.size > 0:
                ch, cw = cell.shape[:2]
                ih = max(3, int(ch * inner_ratio))
                iw = max(3, int(cw * inner_ratio))
                sy = max(0, (ch - ih) // 2)
                sx = max(0, (cw - iw) // 2)
                cell = cell[sy:sy + ih, sx:sx + iw]
            pixel_val[r, c] = float(cv2.countNonZero(cell))
    return pixel_val


def _threshold_omr(gray_img):
    """Build robust binary image for bubble counting with Otsu + light morphology."""
    # Shadow/illumination normalization improves robustness on real camera photos.
    k = max(31, (min(gray_img.shape[:2]) // 12) | 1)
    bg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray_img, bg, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    otsu_val, otsu_bin = cv2.threshold(
        norm,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened, float(otsu_val)


def _largest_active_run(signal_1d, active_threshold):
    active = signal_1d > active_threshold
    best = None
    start = None

    for i, is_on in enumerate(active):
        if is_on and start is None:
            start = i
        if (not is_on) and start is not None:
            seg = (start, i)
            if best is None or (seg[1] - seg[0]) > (best[1] - best[0]):
                best = seg
            start = None

    if start is not None:
        seg = (start, len(active))
        if best is None or (seg[1] - seg[0]) > (best[1] - best[0]):
            best = seg

    return best


def _refine_bubble_grid_roi(
    block_bin,
    left_ratio=0.10,
    right_ratio=0.97,
    top_ratio=0.03,
    bottom_ratio=0.98,
    refine_vertical=True,
    refine_horizontal=True,
):
    """Use projection profile to isolate the densest bubble area in a block."""
    h, w = block_bin.shape[:2]
    if h < 20 or w < 20:
        return block_bin, (0, 0)

    # Vertical projection (find bubble columns)
    col_sum = np.sum(block_bin > 0, axis=0).astype(np.float32)
    if np.max(col_sum) > 0:
        col_sum = col_sum / np.max(col_sum)
    col_mean = float(np.mean(col_sum))
    col_peak = float(np.max(col_sum))
    col_th = col_mean + 0.28 * max(0.0, (col_peak - col_mean))
    col_run = _largest_active_run(col_sum, col_th) if refine_horizontal else None

    # Horizontal projection (find question rows)
    row_sum = np.sum(block_bin > 0, axis=1).astype(np.float32)
    if np.max(row_sum) > 0:
        row_sum = row_sum / np.max(row_sum)
    row_mean = float(np.mean(row_sum))
    row_peak = float(np.max(row_sum))
    row_th = row_mean + 0.20 * max(0.0, (row_peak - row_mean))
    row_run = _largest_active_run(row_sum, row_th) if refine_vertical else None

    x1 = 0 if not refine_horizontal else int(w * left_ratio)
    x2 = w if not refine_horizontal else int(w * right_ratio)
    y1 = 0 if not refine_vertical else int(h * top_ratio)
    y2 = h if not refine_vertical else int(h * bottom_ratio)

    if col_run is not None:
        cx1, cx2 = col_run
        if (cx2 - cx1) >= int(w * 0.35):
            pad = int(0.03 * w)
            x1 = max(0, cx1 - pad)
            x2 = min(w, cx2 + pad)

    if row_run is not None:
        ry1, ry2 = row_run
        if (ry2 - ry1) >= int(h * 0.55):
            pad = int(0.02 * h)
            y1 = max(0, ry1 - pad)
            y2 = min(h, ry2 + pad)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return block_bin, (0, 0)

    return block_bin[y1:y2, x1:x2], (x1, y1)


def _find_rect_in_region(binary_img, x, y, w, h, min_area=1000, min_aspect=0.3, max_aspect=10.0):
    """Find the largest rectangle-like bounding box in a cropped region."""
    x, y, w, h = _clip_rect(x, y, w, h, binary_img.shape[1], binary_img.shape[0])
    roi = binary_img[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    kernel = np.ones((3, 3), np.uint8)
    prep = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(prep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area:
            continue
        aspect = bw / max(1.0, float(bh))
        if not (min_aspect <= aspect <= max_aspect):
            continue
        if area > best_area:
            best_area = area
            best = (x + bx, y + by, bw, bh)

    return best


def _fallback_sid_roi(w, h):
    return _clip_rect(
        x=w * 0.085,
        y=h * 0.165,
        w=w * 0.255,
        h=h * 0.225,
        max_w=w,
        max_h=h,
    )


def _fallback_mcq_roi(w, h):
    return _clip_rect(
        x=w * 0.075,
        y=h * 0.430,
        w=w * 0.855,
        h=h * 0.470,
        max_w=w,
        max_h=h,
    )


def _fallback_exam_code_roi(w, h):
    sid_x, sid_y, sid_w, sid_h = _fallback_sid_roi(w, h)
    return _clip_rect(
        x=sid_x + sid_w + int(w * 0.045),
        y=sid_y,
        w=w * 0.12,
        h=sid_h,
        max_w=w,
        max_h=h,
    )


def _normalize_ocr_title(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = re.sub(r"[^0-9A-Za-zÀ-ỹ\s_-]", "", cleaned)
    cleaned = re.sub(r"\btotal\s+questions\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\boptions?\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -_")
    return cleaned[:100]


def _detect_printed_title(img_bgr):
    """Detect printed exam title in header area using easyocr (if available)."""
    if easyocr is None:
        return None

    global _EASYOCR_READER
    try:
        if _EASYOCR_READER is None:
            _EASYOCR_READER = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    except Exception:
        return None

    h, w = img_bgr.shape[:2]
    y1 = max(0, int(h * 0.035))
    y2 = min(h, int(h * 0.09))
    x1 = max(0, int(w * 0.06))
    x2 = min(w, int(w * 0.94))
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        results = _EASYOCR_READER.readtext(bw, detail=1)
    except Exception:
        return None

    if not results:
        return None

    line_items = []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box = item[0]
        txt = str(item[1]).strip()
        if not txt:
            continue
        ys = [float(p[1]) for p in box] if box else [0.0]
        xs = [float(p[0]) for p in box] if box else [0.0]
        y_mid = float(sum(ys) / max(1, len(ys)))
        x_min = float(min(xs)) if xs else 0.0
        line_items.append((y_mid, x_min, txt))

    if not line_items:
        return None

    min_y = min(y for y, _, _ in line_items)
    top_band = [it for it in line_items if it[0] <= (min_y + 18.0)]
    top_band.sort(key=lambda x: x[1])
    text = _normalize_ocr_title(" ".join(t for _, _, t in top_band))
    return text or None


def _detect_form_boxes(binary_img):
    """Detect SID/MCQ by spatial constraints with block morphology."""
    h, w = binary_img.shape[:2]
    total_area = float(w * h)

    sid_fallback = _fallback_sid_roi(w, h)
    mcq_fallback = _fallback_mcq_roi(w, h)

    # 1) Suppress tiny speckles from bubbles/noise.
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # 2) Block-morphology: merge internal whitespace so each table becomes one big component.
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    table_blob = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, block_kernel, iterations=2)
    table_blob = cv2.dilate(table_blob, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(table_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, bw, bh = cv2.boundingRect(approx)
        rect_area = float(bw * bh)
        if rect_area <= 0:
            continue

        area_ratio = rect_area / total_area
        if area_ratio < 0.008:
            continue

        # Ignore near full-page outer border when scan includes paper edge.
        if area_ratio > 0.82:
            continue

        # Ignore border-like component touching almost all image edges.
        edge_touch = (x <= 4) + (y <= 4) + ((x + bw) >= (w - 4)) + ((y + bh) >= (h - 4))
        if edge_touch >= 3:
            continue

        contour_area = float(cv2.contourArea(cnt))
        extent = contour_area / max(1.0, rect_area)
        if extent < 0.25:
            continue

        candidates.append({
            "rect": (x, y, bw, bh),
            "area": rect_area,
            "aspect": bw / max(1.0, float(bh)),
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
        })

    mcq_rect = mcq_fallback
    sid_rect = sid_fallback

    # Spatial filter for SID: vertical rectangle in top-left half.
    sid_candidates = [
        c for c in candidates
        if c["h"] > c["w"]
        and c["y"] < int(h * 0.52)
        and c["x"] < int(w * 0.52)
        and (c["area"] / total_area) <= 0.14
        and (c["area"] / total_area) >= 0.008
    ]

    if sid_candidates:
        # Prioritize top-left placement and vertical shape, not raw area.
        sid_candidates.sort(
            key=lambda c: (
                c["x"] + c["y"],
                abs((c["h"] / max(1.0, float(c["w"]))) - 1.2),
                -c["area"],
            )
        )
        sid_rect = sid_candidates[0]["rect"]
    else:
        sid_candidate = _find_rect_in_region(
            table_blob,
            x=int(w * 0.03),
            y=int(h * 0.06),
            w=int(w * 0.45),
            h=int(h * 0.40),
            min_area=int(total_area * 0.01),
            min_aspect=0.60,
            max_aspect=1.60,
        )
        if sid_candidate is not None:
            sid_rect = sid_candidate

    # Spatial filter for MCQ: horizontal rectangle in lower half, choose largest area.
    mcq_candidates = [
        c for c in candidates
        if c["w"] > c["h"]
        and c["y"] >= (h // 2)
    ]

    if mcq_candidates:
        mcq_candidates.sort(key=lambda c: c["area"], reverse=True)
        mcq_rect = mcq_candidates[0]["rect"]

    # Guard: MCQ box should be larger than SID; if violated, keep fallback MCQ.
    if (mcq_rect[2] * mcq_rect[3]) < (sid_rect[2] * sid_rect[3]):
        mcq_rect = mcq_fallback

    return sid_rect, mcq_rect


def _tighten_rect_to_table(binary_img, rect, min_active_ratio=0.08, pad=3):
    """Trim a coarse ROI to tight table borders using row/column activity projection."""
    h, w = binary_img.shape[:2]
    x, y, rw, rh = _clip_rect(rect[0], rect[1], rect[2], rect[3], w, h)
    roi = binary_img[y:y + rh, x:x + rw]
    if roi.size == 0:
        return x, y, rw, rh

    # Use long-line extraction to emphasize table borders and suppress bubbles/noise.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, rw // 10), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, rh // 10)))
    hlines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, hk, iterations=1)
    vlines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vk, iterations=1)
    mask = cv2.bitwise_or(hlines, vlines)

    row_proj = np.mean(mask > 0, axis=1).astype(np.float32)
    col_proj = np.mean(mask > 0, axis=0).astype(np.float32)

    if row_proj.size < 2 or col_proj.size < 2:
        return x, y, rw, rh

    row_th = max(min_active_ratio, float(np.percentile(row_proj, 70)) * 0.5)
    col_th = max(min_active_ratio, float(np.percentile(col_proj, 70)) * 0.5)

    row_idx = np.where(row_proj >= row_th)[0]
    col_idx = np.where(col_proj >= col_th)[0]

    if len(row_idx) == 0 or len(col_idx) == 0:
        return x, y, rw, rh

    y1 = max(0, int(row_idx[0]) - pad)
    y2 = min(rh - 1, int(row_idx[-1]) + pad)
    x1 = max(0, int(col_idx[0]) - pad)
    x2 = min(rw - 1, int(col_idx[-1]) + pad)

    tw = x2 - x1 + 1
    th = y2 - y1 + 1
    if tw < int(rw * 0.55) or th < int(rh * 0.55):
        return x, y, rw, rh

    return x + x1, y + y1, tw, th


def _expand_rect_asymmetric(img_shape, rect, left=0, top=0, right=0, bottom=0):
    """Expand rect by different margins on each side, clipped to image bounds."""
    h, w = img_shape[:2]
    x, y, rw, rh = rect

    nx = max(0, x - int(left))
    ny = max(0, y - int(top))
    nr = min(w, x + rw + int(right))
    nb = min(h, y + rh + int(bottom))

    return _clip_rect(nx, ny, nr - nx, nb - ny, w, h)


def _contract_rect_asymmetric(img_shape, rect, left=0, top=0, right=0, bottom=0):
    """Contract rect by different margins on each side, clipped and guarded."""
    h, w = img_shape[:2]
    x, y, rw, rh = rect

    nx = x + max(0, int(left))
    ny = y + max(0, int(top))
    nr = x + rw - max(0, int(right))
    nb = y + rh - max(0, int(bottom))

    if nr - nx < 20 or nb - ny < 20:
        return _clip_rect(x, y, rw, rh, w, h)
    return _clip_rect(nx, ny, nr - nx, nb - ny, w, h)


def _pick_marked_with_flags(
    pixel_val,
    min_conf_ratio=1.12,
    min_peak_factor=1.35,
    blank_floor=45.0,
    blank_std_factor=0.9,
    min_peak_strength=1.25,
):
    """Pick marked option per row and flag blank/multi-mark uncertain rows."""
    indices = []
    statuses = []
    confidences = []

    for r in range(pixel_val.shape[0]):
        row = pixel_val[r].astype(np.float32)
        sorted_idx = np.argsort(row)
        top1_idx = int(sorted_idx[-1])
        top1 = float(row[top1_idx])
        top2 = float(row[int(sorted_idx[-2])]) if row.size > 1 else 0.0
        mean_val = float(np.mean(row))
        std_val = float(np.std(row))

        # Adaptive blank threshold per row.
        blank_threshold = max(float(blank_floor), mean_val + float(blank_std_factor) * std_val)
        peak_strength = top1 / max(1.0, mean_val)
        conf = top1 / max(1.0, top2)
        confidences.append(round(conf, 4))

        if top1 <= max(blank_threshold, mean_val * min_peak_factor):
            indices.append(-1)
            statuses.append("blank")
            continue

        if conf < min_conf_ratio:
            indices.append(-1)
            statuses.append("multiple")
            continue

        # Additional protection for weak peaks even if conf ratio barely passes.
        if peak_strength < float(min_peak_strength):
            indices.append(-1)
            statuses.append("blank")
            continue

        indices.append(top1_idx)
        statuses.append("ok")

    return indices, statuses, confidences


def _trim_mcq_block_top_header(block_bin, max_trim_ratio=0.16):
    """Trim top decorative/header rows (e.g., A/B/C labels) before splitting question rows."""
    h, w = block_bin.shape[:2]
    if h < 30 or w < 30:
        return block_bin, 0

    row_density = np.mean(block_bin > 0, axis=1).astype(np.float32)
    if np.max(row_density) <= 0:
        return block_bin, 0

    smooth = cv2.GaussianBlur(row_density.reshape(-1, 1), (1, 9), 0).reshape(-1)
    th = max(0.015, float(np.percentile(smooth, 65)) * 0.55)

    # Ignore first rows containing thick border/labels; find a stable active run.
    min_scan = max(2, int(h * 0.02))
    active = (smooth >= th).astype(np.uint8)
    start = None
    run = 0
    for i in range(min_scan, len(active)):
        if active[i] == 1:
            if start is None:
                start = i
                run = 1
            else:
                run += 1
            if run >= 3:
                break
        else:
            start = None
            run = 0

    if start is None:
        return block_bin, 0

    max_trim = int(h * max_trim_ratio)
    if start > max_trim:
        return block_bin, 0

    trim = max(0, start - 2)
    if trim < 2 or trim >= h - 5:
        return block_bin, 0

    return block_bin[trim:, :], trim


def _build_choice_labels(num_choices: int):
    """Build option labels: A..Z then fallback to numeric labels."""
    labels = []
    for i in range(max(1, int(num_choices))):
        if i < 26:
            labels.append(chr(ord("A") + i))
        else:
            labels.append(str(i + 1))
    return labels


def _sanitize_filename_part(text: str, fallback: str = "omr"):
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
    """Generate a blank OMR template image that matches the current grading layout assumptions."""
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

    # Four corner markers for robust page alignment.
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

    # Header
    title = (exam_title or "OMR Exam").strip()
    img = _put_text_unicode(img, title[:80], (70, 50), font_size=34, color=black)
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

    # Student ID box (top-left, vertical table like current detector assumption).
    sid_x = int(width_img * 0.085)
    sid_y = int(height_img * 0.165)
    sid_w = int(width_img * 0.255)
    sid_h = int(height_img * 0.225)
    cv2.rectangle(img, (sid_x, sid_y), (sid_x + sid_w, sid_y + sid_h), black, 3)
    cv2.putText(img, "Student ID", (sid_x + 8, sid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, black, 2, cv2.LINE_AA)

    # Top write row: blank squares for handwritten ID.
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

    # Bubble region for 10 digits (0..9), with thinner circle strokes.
    bubble_top = sid_y + write_row_h
    bubble_h = sid_h - write_row_h
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

    # Exam code panel (3 digits 0..9), placed next to SID.
    code_x = sid_x + sid_w + int(width_img * 0.045)
    code_y = sid_y
    code_w = int(width_img * 0.12)
    code_h = sid_h
    cv2.rectangle(img, (code_x, code_y), (code_x + code_w, code_y + code_h), black, 3)
    img = _put_text_unicode(img, "Mã đề", (code_x + 6, code_y - 34), font_size=26, color=black)

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

    # Optional student information table next to exam code.
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
                img = _put_text_unicode(
                    img,
                    f"{field_name}:",
                    (info_x + 8, y1 + 6),
                    font_size=20,
                    color=(35, 35, 35),
                )
                line_y = min(y2 - 8, y1 + max(20, row_h - 8))
                cv2.line(img, (info_x + 110, line_y), (info_x + info_w - 8, line_y), (130, 130, 130), 1)

    # Main MCQ frame (lower half, horizontal large rectangle).
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


def _detect_page_corners_from_markers(gray_img):
    """Detect page corners from 4 corner markers with shadow-robust preprocessing."""
    h, w = gray_img.shape[:2]
    total_area = float(w * h)

    # Normalize illumination so cast shadows are less likely to dominate thresholding.
    k = max(31, (min(h, w) // 10) | 1)
    bg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray_img, bg, scale=255)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)

    _, bin_inv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)

    # Corner-specific detector to avoid selecting center blobs caused by shadows/text.
    corner_w = int(w * 0.24)
    corner_h = int(h * 0.24)
    min_blob = max(18, int(total_area * 0.00003))
    max_blob = int(total_area * 0.02)

    def _pick_corner_marker(x1, y1, x2, y2, cx_target, cy_target, aspect_min=0.45, aspect_max=2.6):
        roi_bin = bin_inv[y1:y2, x1:x2]
        roi_gray = norm[y1:y2, x1:x2]
        if roi_bin.size == 0:
            return None

        num, labels, stats, _ = cv2.connectedComponentsWithStats(roi_bin, connectivity=8)
        best = None
        best_score = -1e9

        for idx in range(1, num):
            rx, ry, rw, rh, area = stats[idx]
            if area < min_blob or area > max_blob:
                continue
            aspect = rw / max(1.0, float(rh))
            if not (float(aspect_min) <= aspect <= float(aspect_max)):
                continue

            rect_area = float(rw * rh)
            fill_ratio = float(area) / max(1.0, rect_area)
            if fill_ratio < 0.38:
                continue

            comp_mask = (labels[ry:ry + rh, rx:rx + rw] == idx)
            if not np.any(comp_mask):
                continue
            comp_pixels = roi_gray[ry:ry + rh, rx:rx + rw][comp_mask]
            darkness = 255.0 - float(np.mean(comp_pixels))

            cx = x1 + rx + rw / 2.0
            cy = y1 + ry + rh / 2.0
            dist2 = (cx - cx_target) ** 2 + (cy - cy_target) ** 2

            # Prefer dark, compact components close to true paper corner.
            score = darkness * 1.2 + fill_ratio * 120.0 - dist2 / max(1.0, (w * h) * 0.15)
            if score > best_score:
                best_score = score
                best = (x1 + rx, y1 + ry, rw, rh)

        return best

    # Template markers: top are horizontal rectangles, bottom are near-squares.
    tl = _pick_corner_marker(0, 0, corner_w, corner_h, 0.0, 0.0, aspect_min=0.95, aspect_max=2.9)
    tr = _pick_corner_marker(w - corner_w, 0, w, corner_h, float(w), 0.0, aspect_min=0.95, aspect_max=2.9)
    bl = _pick_corner_marker(0, h - corner_h, corner_w, h, 0.0, float(h), aspect_min=0.70, aspect_max=1.55)
    br = _pick_corner_marker(w - corner_w, h - corner_h, w, h, float(w), float(h), aspect_min=0.70, aspect_max=1.55)

    # Fallback to global contour strategy when one or more corner pools fail.
    if tl is None or tr is None or bl is None or br is None:
        contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            rect_area = float(bw * bh)
            if rect_area < total_area * 0.00012 or rect_area > total_area * 0.02:
                continue

            aspect = bw / max(1.0, float(bh))
            if not (0.45 <= aspect <= 2.6):
                continue

            contour_area = float(cv2.contourArea(cnt))
            extent = contour_area / max(1.0, rect_area)
            if extent < 0.30:
                continue

            near_left = x < int(w * 0.28)
            near_right = (x + bw) > int(w * 0.72)
            near_top = y < int(h * 0.28)
            near_bottom = (y + bh) > int(h * 0.72)
            if not ((near_left or near_right) and (near_top or near_bottom)):
                continue

            candidates.append((x, y, bw, bh))

        if candidates:
            def pick(cands, target_x, target_y):
                if not cands:
                    return None
                return min(
                    cands,
                    key=lambda r: ((r[0] + r[2] / 2.0 - target_x) ** 2 + (r[1] + r[3] / 2.0 - target_y) ** 2),
                )

            if tl is None:
                tl = pick([c for c in candidates if c[0] < int(w * 0.35) and c[1] < int(h * 0.35)], 0, 0)
            if tr is None:
                tr = pick([c for c in candidates if (c[0] + c[2]) > int(w * 0.65) and c[1] < int(h * 0.35)], w, 0)
            if bl is None:
                bl = pick([c for c in candidates if c[0] < int(w * 0.35) and (c[1] + c[3]) > int(h * 0.65)], 0, h)
            if br is None:
                br = pick([c for c in candidates if (c[0] + c[2]) > int(w * 0.65) and (c[1] + c[3]) > int(h * 0.65)], w, h)

    if tl is None or tr is None or bl is None or br is None:
        return None

    pad = 5
    tl_pt = [max(0, tl[0] - pad), max(0, tl[1] - pad)]
    tr_pt = [min(w - 1, tr[0] + tr[2] + pad), max(0, tr[1] - pad)]
    bl_pt = [max(0, bl[0] - pad), min(h - 1, bl[1] + bl[3] + pad)]
    br_pt = [min(w - 1, br[0] + br[2] + pad), min(h - 1, br[1] + br[3] + pad)]

    return np.float32([tl_pt, tr_pt, bl_pt, br_pt])


def _select_page_like_contour(rect_con, img_w, img_h):
    """Select a contour likely to be the full sheet page, not an inner table."""
    total_area = float(img_w * img_h)
    best = None
    best_area = 0.0

    for cnt in rect_con:
        approx = until.getCornerPoints(cnt)
        if approx is None or approx.size == 0:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        area = float(bw * bh)
        area_ratio = area / max(1.0, total_area)
        aspect = bw / max(1.0, float(bh))

        # Full page should be very large and near portrait ratio.
        if area_ratio < 0.55:
            continue
        if not (0.55 <= aspect <= 1.05):
            continue

        if area > best_area:
            best_area = area
            best = approx

    return best


def _detect_page_corners_from_outer_border(gray_img):
    """Fallback page-corner detection from outer border for tilted/captured photos."""
    h, w = gray_img.shape[:2]
    total_area = float(h * w)

    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edge = cv2.Canny(blur, 40, 120)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < total_area * 0.30:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = np.float32([p[0] for p in approx])
            return until.reorder(pts)

        # If contour is not exactly 4 points, approximate with minAreaRect box.
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.float32(box)
        bw = float(np.linalg.norm(box[0] - box[1]))
        bh = float(np.linalg.norm(box[1] - box[2]))
        if bw <= 1 or bh <= 1:
            continue
        area_ratio = (bw * bh) / max(1.0, total_area)
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if area_ratio < 0.35:
            continue
        if aspect > 2.0:
            continue
        return until.reorder(box)

    return None


def _auto_crop_document_region(img_bgr):
    """Auto-crop region likely containing the exam sheet when photo has many surrounding objects."""
    h, w = img_bgr.shape[:2]
    total = float(h * w)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 45, 140)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, None

    best_rect = None
    best_score = -1.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < total * 0.15:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        rect_area = float(bw * bh)
        if rect_area <= 0:
            continue
        area_ratio = rect_area / total
        if area_ratio < 0.22:
            continue

        aspect = bw / max(1.0, float(bh))
        # Accept portrait-ish sheet and mild perspective deformation.
        if not (0.55 <= aspect <= 1.25):
            continue

        extent = area / rect_area
        # Score prefers large, compact, sheet-like region.
        score = area_ratio * 1.8 + max(0.0, min(extent, 1.0)) * 0.7 - abs(aspect - 0.75) * 0.35
        if score > best_score:
            best_score = score
            best_rect = (x, y, bw, bh)

    if best_rect is None:
        return img_bgr, None

    x, y, bw, bh = best_rect
    pad_x = max(8, int(bw * 0.03))
    pad_y = max(8, int(bh * 0.03))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    if x2 - x1 < 80 or y2 - y1 < 80:
        return img_bgr, None
    return img_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)


def _rect_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    ua = float(max(1, aw * ah) + max(1, bw * bh) - inter)
    return inter / ua


def _score_warp_layout(warp_bgr):
    """Heuristic quality score for selecting the best perspective-warp candidate."""
    h, w = warp_bgr.shape[:2]
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    marker_ok = _detect_page_corners_from_markers(gray) is not None

    th, _ = _threshold_omr(gray)
    sid_rect, mcq_rect = _detect_form_boxes(th)
    sid_rect = _tighten_rect_to_table(th, sid_rect, min_active_ratio=0.08, pad=1)
    mcq_rect = _tighten_rect_to_table(th, mcq_rect, min_active_ratio=0.06, pad=2)

    fsid = _fallback_sid_roi(w, h)
    fmcq = _fallback_mcq_roi(w, h)
    sid_iou = _rect_iou(sid_rect, fsid)
    mcq_iou = _rect_iou(mcq_rect, fmcq)

    sid_area_ratio = (sid_rect[2] * sid_rect[3]) / max(1.0, float(w * h))
    mcq_area_ratio = (mcq_rect[2] * mcq_rect[3]) / max(1.0, float(w * h))

    size_ok = 0.0
    if 0.04 <= sid_area_ratio <= 0.16:
        size_ok += 0.6
    if 0.28 <= mcq_area_ratio <= 0.56:
        size_ok += 0.8

    return (
        (2.2 if marker_ok else 0.0)
        + sid_iou * 1.8
        + mcq_iou * 2.6
        + size_ok
    )


def _norm_quad_from_points(points, img_w, img_h):
    ordered = _order_quad_points(np.asarray(points, dtype=np.float32))
    ow = max(1.0, float(img_w - 1))
    oh = max(1.0, float(img_h - 1))
    tl, tr, br, bl = ordered
    return {
        "tl": {"x": float(np.clip(tl[0] / ow, 0.0, 1.0)), "y": float(np.clip(tl[1] / oh, 0.0, 1.0))},
        "tr": {"x": float(np.clip(tr[0] / ow, 0.0, 1.0)), "y": float(np.clip(tr[1] / oh, 0.0, 1.0))},
        "br": {"x": float(np.clip(br[0] / ow, 0.0, 1.0)), "y": float(np.clip(br[1] / oh, 0.0, 1.0))},
        "bl": {"x": float(np.clip(bl[0] / ow, 0.0, 1.0)), "y": float(np.clip(bl[1] / oh, 0.0, 1.0))},
    }


def suggest_omr_crop_quad(image_path: str):
    """Suggest initial manual crop quadrilateral from current OMR CV pipeline."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Không thể đọc file ảnh"}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _score_points(points):
        ordered = _order_quad_points(np.asarray(points, dtype=np.float32).reshape(4, 2))
        dst = np.array([[0, 0], [999, 0], [999, 1399], [0, 1399]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(img, matrix, (1000, 1400))
        return float(_score_warp_layout(warped))

    candidates = []

    marker_orig = _detect_page_corners_from_markers(gray)
    if marker_orig is not None:
        candidates.append({
            "name": "marker-original",
            "points": np.asarray(marker_orig, dtype=np.float32).reshape(4, 2),
        })

    border_orig = _detect_page_corners_from_outer_border(gray)
    if border_orig is not None:
        candidates.append({
            "name": "outer-border-original",
            "points": np.asarray(border_orig, dtype=np.float32).reshape(4, 2),
        })

    cropped, crop_rect = _auto_crop_document_region(img)
    if crop_rect is not None:
        cx, cy, cw, ch = crop_rect
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        marker_crop = _detect_page_corners_from_markers(gray_crop)
        if marker_crop is not None:
            mapped = np.asarray(marker_crop, dtype=np.float32).reshape(4, 2)
            mapped[:, 0] += float(cx)
            mapped[:, 1] += float(cy)
            candidates.append({"name": "marker-autocrop", "points": mapped})

        border_crop = _detect_page_corners_from_outer_border(gray_crop)
        if border_crop is not None:
            mapped = np.asarray(border_crop, dtype=np.float32).reshape(4, 2)
            mapped[:, 0] += float(cx)
            mapped[:, 1] += float(cy)
            candidates.append({"name": "outer-border-autocrop", "points": mapped})

        rect_pts = np.array(
            [
                [float(cx), float(cy)],
                [float(cx + cw), float(cy)],
                [float(cx + cw), float(cy + ch)],
                [float(cx), float(cy + ch)],
            ],
            dtype=np.float32,
        )
        candidates.append({"name": "autocrop-rect", "points": rect_pts})

    if candidates:
        best = None
        best_score = -1e9
        for cand in candidates:
            try:
                s = _score_points(cand["points"])
            except Exception:
                s = -1e9
            if s > best_score:
                best_score = s
                best = cand

        if best is not None:
            return {
                "success": True,
                "strategy": best["name"],
                "quad": _norm_quad_from_points(best["points"], w, h),
            }

    # Default full-ish frame when nothing is detected.
    return {
        "success": True,
        "strategy": "default",
        "quad": {
            "tl": {"x": 0.08, "y": 0.08},
            "tr": {"x": 0.92, "y": 0.08},
            "br": {"x": 0.92, "y": 0.92},
            "bl": {"x": 0.08, "y": 0.92},
        },
    }


def process_omr_exam(
    image_path: str,
    output_folder: str,
    answer_key: list,
    questions=80,
    choices=5,
    rows_per_block=20,
    num_blocks=None,
    student_id_digits=6,
    sid_has_write_row=False,
    crop_x=None,
    crop_y=None,
    crop_w=None,
    crop_h=None,
    crop_tl_x=None,
    crop_tl_y=None,
    crop_tr_x=None,
    crop_tr_y=None,
    crop_br_x=None,
    crop_br_y=None,
    crop_bl_x=None,
    crop_bl_y=None,
    _internal_retry=False,
):
    """
    Xử lý ảnh trắc nghiệm OMR.
    
    Args:
        image_path: Đường dẫn ảnh đầu vào.
        output_folder: Thư mục để lưu ảnh kết quả.
        answer_key: List đáp án đúng (ví dụ: [1, 0, 2, 1, 4]).
        questions: Số câu hỏi.
        choices: Số lựa chọn.
        
    Returns:
        dict: Kết quả chấm (score, grading array, output_image_path)
    """
    
    # Cấu hình kích thước ảnh chuẩn hóa (sheet dọc)
    widthImg = 1000
    heightImg = 1400
    
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Không thể đọc file ảnh"}

    manual_quad_warp = None
    manual_quad_requested = False
    manual_quad_locked = False
    manual_quad_warning = None

    # Optional manual 4-corner crop (normalized points from UI).
    quad_vals = [crop_tl_x, crop_tl_y, crop_tr_x, crop_tr_y, crop_br_x, crop_br_y, crop_bl_x, crop_bl_y]
    manual_quad_requested = all(v is not None for v in quad_vals)
    if manual_quad_requested:
        try:
            quad_points = [
                (float(crop_tl_x), float(crop_tl_y)),
                (float(crop_tr_x), float(crop_tr_y)),
                (float(crop_br_x), float(crop_br_y)),
                (float(crop_bl_x), float(crop_bl_y)),
            ]
            warped_by_quad = _warp_by_manual_quad(img, quad_points, target_size=(widthImg, heightImg))
            if warped_by_quad is not None:
                manual_quad_warp = warped_by_quad
                manual_quad_locked = True
            else:
                manual_quad_warning = "Vùng cắt thủ công không hợp lệ, hệ thống đã fallback về nhận diện tự động."
        except Exception:
            manual_quad_warning = "Không đọc được vùng cắt thủ công, hệ thống đã fallback về nhận diện tự động."

    # Optional manual crop from UI (normalized ratios in [0..1]).
    if crop_x is not None and crop_y is not None and crop_w is not None and crop_h is not None:
        ih, iw = img.shape[:2]
        try:
            rx = float(crop_x)
            ry = float(crop_y)
            rw = float(crop_w)
            rh = float(crop_h)
            rx = max(0.0, min(0.98, rx))
            ry = max(0.0, min(0.98, ry))
            rw = max(0.02, min(1.0 - rx, rw))
            rh = max(0.02, min(1.0 - ry, rh))
            x = int(rx * iw)
            y = int(ry * ih)
            w = int(rw * iw)
            h = int(rh * ih)
            x, y, w, h = _clip_rect(x, y, w, h, iw, ih)
            if w > 40 and h > 40:
                img = img[y:y + h, x:x + w].copy()
        except Exception:
            pass

    # --- 1. Tiền xử lý (Preprocessing) ---
    # Step A: auto-crop likely sheet region for cluttered photos.
    # Keep original image as fallback when crop accidentally trims corner markers.
    orig_img = img.copy()
    if manual_quad_locked:
        # Respect manual crop: do not run another automatic document crop.
        img = cv2.resize(img, (widthImg, heightImg))
        orig_for_fallback = img.copy()
        crop_rect = (0, 0, widthImg, heightImg)
    else:
        img, crop_rect = _auto_crop_document_region(img)
        img = cv2.resize(img, (widthImg, heightImg))
        orig_for_fallback = cv2.resize(orig_img, (widthImg, heightImg))
    imgContours = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # Canny Edge Detection
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # --- 2. Tìm Contours ---
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # Tìm hình chữ nhật
    rectCon = until.rectContours(contours)

    # --- 3. Warp Perspective (Biến đổi góc nhìn) ---
    imgWarpColored = img.copy()
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    warp_strategy = "none"

    warp_candidates = []
    warp_lookup = {}

    def _add_warp_candidate(name, base_img, corners):
        if corners is None:
            return
        matrix = cv2.getPerspectiveTransform(np.float32(corners), pt2)
        warped = cv2.warpPerspective(base_img, matrix, (widthImg, heightImg))
        quality = _score_warp_layout(warped) if bool(sid_has_write_row) else 0.0
        warp_candidates.append({"name": name, "warp": warped, "quality": float(quality)})
        warp_lookup[str(name)] = warped

    def _add_ordered_warp_candidate(name, base_img, corners):
        if corners is None:
            return
        src = _order_quad_points(np.asarray(corners, dtype=np.float32).reshape(4, 2))
        dst = np.array(
            [[0, 0], [widthImg - 1, 0], [widthImg - 1, heightImg - 1], [0, heightImg - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(base_img, matrix, (widthImg, heightImg))
        quality = _score_warp_layout(warped) if bool(sid_has_write_row) else 0.0
        warp_candidates.append({"name": name, "warp": warped, "quality": float(quality)})
        warp_lookup[str(name)] = warped

    if manual_quad_locked and manual_quad_warp is not None:
        imgWarpColored = manual_quad_warp.copy()
        warp_strategy = "manual-quad-locked"
        warp_lookup[warp_strategy] = imgWarpColored.copy()
    else:
        # Keep a no-perspective baseline candidate for already aligned template screenshots.
        identity_quality = _score_warp_layout(img) if bool(sid_has_write_row) else 0.0
        warp_candidates.append({"name": "identity-crop", "warp": img.copy(), "quality": float(identity_quality)})
        warp_lookup["identity-crop"] = img.copy()

        marker_crop = _detect_page_corners_from_markers(imgGray)
        _add_warp_candidate("marker-crop", img, marker_crop)

        marker_orig = _detect_page_corners_from_markers(cv2.cvtColor(orig_for_fallback, cv2.COLOR_BGR2GRAY))
        _add_warp_candidate("marker-orig", orig_for_fallback, marker_orig)
        _add_ordered_warp_candidate("marker-orig-ordered", orig_for_fallback, marker_orig)

        pageContour = _select_page_like_contour(rectCon, widthImg, heightImg)
        if pageContour is not None and pageContour.size != 0:
            _add_warp_candidate("page-contour", img, until.reorder(pageContour))

        outer_crop = _detect_page_corners_from_outer_border(imgGray)
        _add_warp_candidate("outer-border-crop", img, outer_crop)

        outer_orig = _detect_page_corners_from_outer_border(cv2.cvtColor(orig_for_fallback, cv2.COLOR_BGR2GRAY))
        _add_warp_candidate("outer-border-orig", orig_for_fallback, outer_orig)

        if manual_quad_warp is not None:
            manual_quality = _score_warp_layout(manual_quad_warp) if bool(sid_has_write_row) else 0.0
            # Conservative bias: avoid degrading results when user keeps default suggested quad.
            manual_quality = float(manual_quality) - 0.35
            warp_candidates.append({"name": "manual-quad", "warp": manual_quad_warp, "quality": float(manual_quality)})
            warp_lookup["manual-quad"] = manual_quad_warp

        if warp_candidates:
            if bool(sid_has_write_row):
                warp_candidates.sort(key=lambda x: x["quality"], reverse=True)
                best = warp_candidates[0]
            else:
                best = warp_candidates[0]
            imgWarpColored = best["warp"]
            warp_strategy = str(best["name"])
            warp_lookup[warp_strategy] = imgWarpColored.copy()

    # In template-like sheets (has write row), missing markers should not hard-fail.
    # Keep grading via contour/fallback pipeline and return a warning to the client.
    marker_warning = None
    if bool(sid_has_write_row) and not manual_quad_locked:
        warpGrayCheck = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        marker_check = _detect_page_corners_from_markers(warpGrayCheck)
        if marker_check is None:
            # Attempt rescue warp from outer border when marker-based warp is unreliable.
            rescue_base = img
            rescue_corners = _detect_page_corners_from_outer_border(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if rescue_corners is None:
                rescue_base = orig_for_fallback
                rescue_corners = _detect_page_corners_from_outer_border(cv2.cvtColor(orig_for_fallback, cv2.COLOR_BGR2GRAY))

            if rescue_corners is not None:
                rescue_matrix = cv2.getPerspectiveTransform(np.float32(rescue_corners), pt2)
                rescue_warp = cv2.warpPerspective(rescue_base, rescue_matrix, (widthImg, heightImg))
                rescue_gray = cv2.cvtColor(rescue_warp, cv2.COLOR_BGR2GRAY)
                rescue_marker = _detect_page_corners_from_markers(rescue_gray)
                if rescue_marker is not None:
                    imgWarpColored = rescue_warp
                    warp_strategy = "outer-border-rescue"

            marker_warning = (
                "Không nhận diện đủ 4 marker góc (2 ô vuông + 2 hình chữ nhật). "
                "Hệ thống đã dùng chế độ fallback để tiếp tục chấm; nên bật Manual Crop để tăng độ chính xác."
            )

    # --- 4. Apply Threshold ---
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh, otsu_value = _threshold_omr(imgWarpGray)

    # --- 5. ROI theo form mới (auto-calibration + fallback) ---
    h, w = imgThresh.shape[:2]

    # Khu vực SBD (trên trái) và khu đáp án lớn được detect như 2 khung riêng.
    sid_rect, mcq_rect = _detect_form_boxes(imgThresh)

    use_template_geometry = bool(sid_has_write_row) and (warp_strategy == "identity-crop")
    if use_template_geometry:
        sid_rect = _fallback_sid_roi(w, h)
        mcq_rect = _fallback_mcq_roi(w, h)

    # If marker warp succeeded but contour box is clearly implausible, fallback to template geometry.
    if warp_strategy.startswith("marker"):
        fsid = _fallback_sid_roi(w, h)
        fmcq = _fallback_mcq_roi(w, h)

        sid_bad = (
            sid_rect[2] < int(w * 0.17)
            or sid_rect[3] < int(h * 0.18)
            or sid_rect[0] > int(w * 0.20)
            or sid_rect[1] > int(h * 0.32)
        )
        mcq_bad = (
            mcq_rect[2] < int(w * 0.82)
            or mcq_rect[3] < int(h * 0.44)
            or mcq_rect[0] > int(w * 0.10)
            or mcq_rect[1] < int(h * 0.34)
        )

        if sid_bad:
            sid_rect = fsid
        if mcq_bad:
            mcq_rect = fmcq
    if not use_template_geometry:
        sid_rect = _tighten_rect_to_table(imgThresh, sid_rect, min_active_ratio=0.08, pad=1)
        mcq_rect = _tighten_rect_to_table(imgThresh, mcq_rect, min_active_ratio=0.06, pad=2)

        # SID contour is often slightly oversized due header text/border thickness; contract lightly.
        sid_rect = _contract_rect_asymmetric(
            imgThresh.shape,
            sid_rect,
            left=max(3, int(sid_rect[2] * 0.04)),
            top=max(3, int(sid_rect[3] * 0.04)),
            right=max(2, int(sid_rect[2] * 0.035)),
            bottom=max(8, int(sid_rect[3] * 0.150)),
        )

        # Guard SID from touching MCQ header labels (A/B/C/D): keep SID bottom above MCQ top.
        sid_x0, sid_y0, sid_w0, sid_h0 = sid_rect
        sid_bottom_guard = mcq_rect[1] - max(20, int(h * 0.020))
        if (sid_y0 + sid_h0) > sid_bottom_guard:
            sid_h0 = max(20, sid_bottom_guard - sid_y0)
            sid_rect = _clip_rect(sid_x0, sid_y0, sid_w0, sid_h0, w, h)

        # Empirical correction: MCQ top/left/right tend to be under-cropped after tighten.
        # Keep bottom almost unchanged because it is already aligned well.
        mcq_expand_left = max(7, int(mcq_rect[2] * 0.024))
        mcq_expand_right = max(6, int(mcq_rect[2] * 0.020))
        mcq_expand_top = max(2, int(mcq_rect[3] * 0.008))
        mcq_expand_bottom = max(5, int(mcq_rect[3] * 0.016))
        mcq_rect = _expand_rect_asymmetric(
            imgThresh.shape,
            mcq_rect,
            left=mcq_expand_left,
            top=mcq_expand_top,
            right=mcq_expand_right,
            bottom=mcq_expand_bottom,
        )

    sid_x, sid_y, sid_w, sid_h = sid_rect

    sid_roi = imgThresh[sid_y:sid_y + sid_h, sid_x:sid_x + sid_w]
    if bool(sid_has_write_row):
        # Ignore the top handwritten ID row when extracting filled bubbles.
        sid_head_h = max(1, int(sid_roi.shape[0] * 0.16))
        sid_roi = sid_roi[sid_head_h:, :]
    student_id_digits = max(1, int(student_id_digits))
    # _split_grid returns [rows(0..9), cols(digit positions)] => transpose to [digit, 0..9].
    sid_pixels = _split_grid(sid_roi, rows=10, cols=student_id_digits, inner_ratio=0.78).T
    sid_indices, sid_status, sid_conf = _pick_marked_with_flags(sid_pixels)

    # Local adaptive threshold on SID ROI improves weak strokes under shadows.
    sid_gray_roi = imgWarpGray[sid_y:sid_y + sid_h, sid_x:sid_x + sid_w]
    if bool(sid_has_write_row):
        sid_gray_roi = sid_gray_roi[sid_head_h:, :]
    sid_local = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(sid_gray_roi)
    sid_local = cv2.adaptiveThreshold(
        sid_local,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    sid_local = cv2.morphologyEx(sid_local, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    sid_pixels_local = _split_grid(sid_local, rows=10, cols=student_id_digits, inner_ratio=0.62).T
    sid_idx_local, sid_st_local, sid_cf_local = _pick_marked_with_flags(
        sid_pixels_local,
        min_conf_ratio=1.01,
        min_peak_factor=1.04,
        blank_floor=12.0,
        blank_std_factor=0.25,
        min_peak_strength=1.02,
    )

    # Prefer local-adaptive digit if global decode is uncertain.
    for i in range(len(sid_status)):
        if sid_status[i] != "ok" and sid_st_local[i] == "ok":
            sid_indices[i] = sid_idx_local[i]
            sid_status[i] = sid_st_local[i]
            sid_conf[i] = sid_cf_local[i]
    if any(s != "ok" for s in sid_status):
        sid_indices2, sid_status2, sid_conf2 = _pick_marked_with_flags(
            sid_pixels,
            min_conf_ratio=1.03,
            min_peak_factor=1.08,
            blank_floor=20.0,
            blank_std_factor=0.40,
            min_peak_strength=1.05,
        )
        for i in range(len(sid_status)):
            if sid_status[i] != "ok" and sid_status2[i] == "ok":
                sid_indices[i] = sid_indices2[i]
                sid_status[i] = sid_status2[i]
                sid_conf[i] = sid_conf2[i]

    # Shadow artifact guard for SID: bottom rows (8/9) can dominate weak columns.
    # If a high digit is selected with low confidence, prefer the runner-up when it is much lower.
    for i in range(len(sid_indices)):
        if sid_status[i] != "ok":
            continue
        d = int(sid_indices[i])
        c = float(sid_conf[i]) if i < len(sid_conf) else 0.0
        if d >= 7 and c < 1.08:
            col = sid_pixels[i].astype(np.float32)
            order = np.argsort(col)
            alt = int(order[-2]) if col.size > 1 else d
            if alt <= 4:
                sid_indices[i] = alt
                sid_conf[i] = max(c, 1.0)

    # Final SID refinement from cropped SID panel with adaptive threshold.
    sid_crop_gray = cv2.cvtColor(imgWarpColored[sid_y:sid_y + sid_h, sid_x:sid_x + sid_w], cv2.COLOR_BGR2GRAY)
    if bool(sid_has_write_row):
        sid_crop_gray = sid_crop_gray[sid_head_h:, :]
    sid_crop_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(sid_crop_gray)
    sid_crop_bin = cv2.adaptiveThreshold(
        sid_crop_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    sid_crop_bin = cv2.morphologyEx(sid_crop_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    sid_crop_pixels = _split_grid(sid_crop_bin, rows=10, cols=student_id_digits, inner_ratio=0.68).T
    sid_idx_crop, sid_st_crop, sid_cf_crop = _pick_marked_with_flags(
        sid_crop_pixels,
        min_conf_ratio=1.01,
        min_peak_factor=1.04,
        blank_floor=12.0,
        blank_std_factor=0.25,
        min_peak_strength=1.02,
    )

    # Trust crop-adaptive decode when it is complete and current decode is low-confidence.
    if all(s == "ok" for s in sid_st_crop):
        if any(s != "ok" for s in sid_status) or (len(sid_conf) > 0 and min(sid_conf) < 1.07):
            sid_indices = sid_idx_crop
            sid_status = sid_st_crop
            sid_conf = sid_cf_crop

    # Extremely close tie: if confidence is very low, allow runner-up lower digit.
    if sid_crop_pixels is not None and sid_crop_pixels.size > 0:
        for i in range(min(len(sid_indices), sid_crop_pixels.shape[0])):
            if sid_status[i] != "ok":
                continue
            conf_i = float(sid_conf[i]) if i < len(sid_conf) else 0.0
            if conf_i >= 1.02:
                continue
            col = sid_crop_pixels[i].astype(np.float32)
            if col.size < 2:
                continue
            order = np.argsort(col)
            top_idx = int(order[-1])
            sec_idx = int(order[-2])
            top_val = float(col[top_idx])
            sec_val = float(col[sec_idx])
            ratio = top_val / max(1.0, sec_val)
            if int(sid_indices[i]) == top_idx and sec_idx < top_idx and ratio < 1.03:
                sid_indices[i] = sec_idx
                sid_conf[i] = max(conf_i, 1.0)

    # Suppress common blank-sheet artifact: many SID columns collapse to the same lower row (typically 7/8/9).
    sid_valid = [int(v) for v in sid_indices if int(v) >= 0]
    if bool(sid_has_write_row) and sid_valid:
        freq = {}
        for d in sid_valid:
            freq[d] = freq.get(d, 0) + 1
        mode_digit = max(freq, key=freq.get)
        mode_count = freq.get(mode_digit, 0)
        top_vals = []
        for i in range(min(len(sid_indices), sid_pixels.shape[0])):
            d = int(sid_indices[i])
            if d >= 0 and d < sid_pixels.shape[1]:
                top_vals.append(float(sid_pixels[i, d]))
        median_top = float(np.median(top_vals)) if top_vals else 0.0

        max_top = max(top_vals) if top_vals else 0.0
        collapse_pattern = mode_count >= max(5, int(student_id_digits * 0.75)) and len(freq) <= 2
        low_strength_pattern = mode_count >= max(3, int(student_id_digits * 0.55)) and median_top < 140.0 and max_top < 220.0
        if collapse_pattern or low_strength_pattern:
            sid_indices = [-1 for _ in sid_indices]
            sid_status = ["blank" for _ in sid_status]
            sid_conf = [1.0 for _ in sid_conf]

    student_id = "".join(str(int(d)) if d >= 0 else "?" for d in sid_indices)

    # Decode exam code (3 digits) from its dedicated bubble box.
    code_x, code_y, code_w, code_h = _fallback_exam_code_roi(w, h)
    code_roi = imgThresh[code_y:code_y + code_h, code_x:code_x + code_w]
    code_head_h = max(1, int(code_roi.shape[0] * 0.16)) if code_roi.size > 0 else 1
    if code_roi.size > 0:
        code_roi = code_roi[code_head_h:, :]
    code_pixels = _split_grid(code_roi, rows=10, cols=3, inner_ratio=0.78).T if code_roi.size > 0 else np.zeros((3, 10), dtype=np.float32)
    code_idx, code_status, code_conf = _pick_marked_with_flags(code_pixels)
    exam_code_detected = "".join(str(int(d)) if d >= 0 else "?" for d in code_idx)

    # OCR printed title (best effort, may be None depending on model availability).
    exam_title_detected = _detect_printed_title(imgWarpColored)

    # SID rescue across alternative warp candidates.
    if bool(sid_has_write_row) and not manual_quad_locked and (not use_template_geometry) and len(warp_lookup) > 0:
        def _sid_repeat_ratio(indices):
            vals = [int(v) for v in indices if int(v) >= 0]
            if not vals:
                return 1.0
            freq = {}
            for v in vals:
                freq[v] = freq.get(v, 0) + 1
            return max(freq.values()) / float(len(vals))

        def _decode_sid_from_warp_quick(warp_img):
            gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
            sid_fx, sid_fy, sid_fw, sid_fh = _fallback_sid_roi(widthImg, heightImg)
            sid_gray = gray[sid_fy:sid_fy + sid_fh, sid_fx:sid_fx + sid_fw]
            if sid_gray.size == 0:
                return None
            if bool(sid_has_write_row):
                sid_head = max(1, int(sid_gray.shape[0] * 0.16))
                sid_gray = sid_gray[sid_head:, :]

            sid_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(sid_gray)
            sid_bin = cv2.adaptiveThreshold(
                sid_eq,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                7,
            )
            sid_bin = cv2.morphologyEx(sid_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
            sid_px = _split_grid(sid_bin, rows=10, cols=student_id_digits, inner_ratio=0.68).T
            sid_idx, sid_st, sid_cf = _pick_marked_with_flags(
                sid_px,
                min_conf_ratio=1.01,
                min_peak_factor=1.04,
                blank_floor=12.0,
                blank_std_factor=0.25,
                min_peak_strength=1.02,
            )
            return sid_idx, sid_st, sid_cf

        current_uncertain = sum(1 for s in sid_status if s != "ok")
        current_min_conf = min(sid_conf) if len(sid_conf) > 0 else 0.0
        current_repeat = _sid_repeat_ratio(sid_indices)

        sid_alt_best = None
        sid_alt_score = None

        for cand_name in ["marker-orig", "marker-orig-ordered", "outer-border-orig", "marker-crop", "outer-border-crop", "page-contour"]:
            cand_warp = warp_lookup.get(cand_name)
            if cand_warp is None:
                continue
            decoded = _decode_sid_from_warp_quick(cand_warp)
            if decoded is None:
                continue
            idx_alt, st_alt, cf_alt = decoded
            alt_uncertain = sum(1 for s in st_alt if s != "ok")
            alt_min_conf = min(cf_alt) if len(cf_alt) > 0 else 0.0
            alt_repeat = _sid_repeat_ratio(idx_alt)
            # Lower is better: uncertain first, then confidence penalty, then repeated-digit penalty.
            alt_score = alt_uncertain * 3.0 + max(0.0, 1.12 - float(alt_min_conf)) * 4.0 + max(0.0, alt_repeat - 0.55) * 2.0
            if sid_alt_score is None or alt_score < sid_alt_score:
                sid_alt_score = float(alt_score)
                sid_alt_best = (idx_alt, st_alt, cf_alt)

        if sid_alt_best is not None:
            alt_idx, alt_st, alt_cf = sid_alt_best
            alt_uncertain = sum(1 for s in alt_st if s != "ok")
            alt_min_conf = min(alt_cf) if len(alt_cf) > 0 else 0.0
            alt_repeat = _sid_repeat_ratio(alt_idx)

            current_score = current_uncertain * 3.0 + max(0.0, 1.12 - float(current_min_conf)) * 4.0 + max(0.0, current_repeat - 0.55) * 2.0
            new_score = alt_uncertain * 3.0 + max(0.0, 1.12 - float(alt_min_conf)) * 4.0 + max(0.0, alt_repeat - 0.55) * 2.0

            if new_score + 0.2 < current_score:
                sid_indices = alt_idx
                sid_status = alt_st
                sid_conf = alt_cf
                student_id = "".join(str(int(d)) if d >= 0 else "?" for d in sid_indices)

    # Khu vực đáp án: hỗ trợ layout động theo cau hinh.
    mcq_x, mcq_y, mcq_w, mcq_h = mcq_rect

    questions = max(1, int(questions))
    choices = max(2, int(choices))
    rows_per_block = max(1, int(rows_per_block))
    if num_blocks is None:
        block_count = max(1, int(math.ceil(questions / float(rows_per_block))))
    else:
        block_count = max(1, int(num_blocks))
    # Guard: ensure enough rows to cover all questions.
    if block_count * rows_per_block < questions:
        block_count = int(math.ceil(questions / float(rows_per_block)))

    mcq_roi = imgThresh[mcq_y:mcq_y + mcq_h, mcq_x:mcq_x + mcq_w]
    block_gap = int(mcq_w * 0.02) if block_count > 1 else 0
    total_gap = block_gap * max(0, block_count - 1)
    usable_w = max(1, mcq_w - total_gap)
    block_w = max(1, int(usable_w / block_count))

    user_answers = []
    answer_map = {}
    uncertain_questions = []
    answer_confidences = []
    question_visual_meta = []

    for block_idx in range(block_count):
        block_start_q = block_idx * rows_per_block
        rows_in_block = min(rows_per_block, questions - block_start_q)
        if rows_in_block <= 0:
            break

        bx = block_idx * (block_w + block_gap)
        if block_idx == block_count - 1:
            bx2 = mcq_w
        else:
            bx2 = min(bx + block_w, mcq_w)
        block = mcq_roi[:, bx:bx2]

        # Bỏ cột số thứ tự câu bên trái, chỉ giữ vùng bubble A-E
        bw = block.shape[1]
        bubble_left = int(bw * 0.16)
        bubble_right = int(bw * 0.95)
        bubble_grid = block[:, bubble_left:bubble_right]

        # Keep vertical alignment fixed per block to avoid row-index drift.
        bubble_grid, (refine_x, refine_y) = _refine_bubble_grid_roi(
            bubble_grid,
            refine_vertical=False,
            refine_horizontal=False,
        )

        # Remove top header text row so question-1 aligns to row-1 (avoid one-row drift).
        if use_template_geometry:
            header_trim = 0
        else:
            bubble_grid, header_trim = _trim_mcq_block_top_header(bubble_grid, max_trim_ratio=0.16)
        refine_y += int(header_trim)

        block_pixels = _split_grid(bubble_grid, rows=rows_in_block, cols=choices, inner_ratio=0.72)
        block_indices, block_status, block_conf = _pick_marked_with_flags(
            block_pixels,
            min_conf_ratio=1.07,
            min_peak_factor=1.18,
            blank_floor=30.0,
            blank_std_factor=0.55,
            min_peak_strength=1.12,
        )

        gh, gw = bubble_grid.shape[:2]
        row_edges = np.linspace(0, gh, rows_in_block + 1, dtype=int)
        col_edges = np.linspace(0, gw, choices + 1, dtype=int)

        for row_idx, ans_idx in enumerate(block_indices):
            q_num = block_start_q + row_idx + 1
            user_answers.append(int(ans_idx))
            answer_map[str(q_num)] = int(ans_idx)
            answer_confidences.append(float(block_conf[row_idx]))

            row_y1 = int(row_edges[row_idx])
            row_y2 = int(row_edges[row_idx + 1])
            cy = mcq_y + refine_y + row_y1 + (row_y2 - row_y1) // 2
            option_centers = []
            for c in range(choices):
                cx1 = int(col_edges[c])
                cx2 = int(col_edges[c + 1])
                cx = mcq_x + bx + bubble_left + refine_x + cx1 + (cx2 - cx1) // 2
                option_centers.append((cx, cy))

            question_visual_meta.append({
                "question": q_num,
                "selected": int(ans_idx),
                "status": block_status[row_idx],
                "option_centers": option_centers,
                "row_center": (mcq_x + bx + refine_x, cy),
            })

            if block_status[row_idx] != "ok":
                uncertain_questions.append({
                    "question": q_num,
                    "status": block_status[row_idx],
                })

    # Adaptive fallback for hard photos: rerun MCQ decode on marker-orig warp + local threshold.
    # Besides uncertain rows, also guard against over-biased answer distributions from brittle warps.
    if bool(sid_has_write_row):
        marker_orig = _detect_page_corners_from_markers(cv2.cvtColor(orig_for_fallback, cv2.COLOR_BGR2GRAY))
        if marker_orig is not None:
            m_alt = cv2.getPerspectiveTransform(np.float32(marker_orig), pt2)
            warp_alt = cv2.warpPerspective(orig_for_fallback, m_alt, (widthImg, heightImg))
            gray_alt = cv2.cvtColor(warp_alt, cv2.COLOR_BGR2GRAY)
            clahe_alt = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            norm_alt = clahe_alt.apply(gray_alt)
            bin_alt = cv2.adaptiveThreshold(
                norm_alt,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                41,
                9,
            )
            bin_alt = cv2.morphologyEx(bin_alt, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

            ax, ay, aw, ah = _fallback_mcq_roi(widthImg, heightImg)
            mcq_alt = bin_alt[ay:ay + ah, ax:ax + aw]

            block_gap_alt = int(aw * 0.02) if block_count > 1 else 0
            total_gap_alt = block_gap_alt * max(0, block_count - 1)
            usable_w_alt = max(1, aw - total_gap_alt)
            block_w_alt = max(1, int(usable_w_alt / block_count))

            alt_answers = []
            alt_statuses = []
            alt_conf = []

            for block_idx in range(block_count):
                block_start_q = block_idx * rows_per_block
                rows_in_block = min(rows_per_block, questions - block_start_q)
                if rows_in_block <= 0:
                    break

                bx = block_idx * (block_w_alt + block_gap_alt)
                bx2 = aw if block_idx == block_count - 1 else min(bx + block_w_alt, aw)
                block = mcq_alt[:, bx:bx2]

                bw = block.shape[1]
                bubble_left = int(bw * 0.16)
                bubble_right = int(bw * 0.95)
                bubble_grid = block[:, bubble_left:bubble_right]
                bubble_grid, _ = _refine_bubble_grid_roi(
                    bubble_grid,
                    refine_vertical=False,
                    refine_horizontal=False,
                )
                if not use_template_geometry:
                    bubble_grid, _ = _trim_mcq_block_top_header(bubble_grid, max_trim_ratio=0.16)

                block_pixels = _split_grid(bubble_grid, rows=rows_in_block, cols=choices, inner_ratio=0.68)
                block_indices, block_status, block_cf = _pick_marked_with_flags(
                    block_pixels,
                    min_conf_ratio=1.03,
                    min_peak_factor=1.06,
                    blank_floor=16.0,
                    blank_std_factor=0.30,
                    min_peak_strength=1.03,
                )

                alt_answers.extend([int(x) for x in block_indices])
                alt_statuses.extend([str(x) for x in block_status])
                alt_conf.extend([float(x) for x in block_cf])

            alt_uncertain_count = sum(1 for s in alt_statuses if s != "ok")

            def _answer_dominance(ans_list):
                vals = [int(x) for x in ans_list if int(x) >= 0]
                if not vals:
                    return 1.0
                freq = {}
                for v in vals:
                    freq[v] = freq.get(v, 0) + 1
                return max(freq.values()) / float(len(vals))

            primary_uncertain = len(uncertain_questions)
            primary_dom = _answer_dominance(user_answers)
            alt_dom = _answer_dominance(alt_answers)

            choose_alt = False
            if len(alt_answers) == len(user_answers):
                if alt_uncertain_count < primary_uncertain:
                    choose_alt = True
                elif warp_strategy == "page-contour" and alt_uncertain_count <= primary_uncertain:
                    choose_alt = True
                elif primary_dom >= 0.65 and alt_dom <= 0.50 and alt_uncertain_count <= primary_uncertain + 1:
                    choose_alt = True

            if choose_alt:
                user_answers = alt_answers
                answer_confidences = alt_conf
                answer_map = {str(i + 1): int(a) for i, a in enumerate(user_answers)}
                uncertain_questions = [
                    {"question": i + 1, "status": alt_statuses[i]}
                    for i in range(len(alt_statuses))
                    if alt_statuses[i] != "ok"
                ]
                for i in range(min(len(question_visual_meta), len(alt_answers))):
                    question_visual_meta[i]["selected"] = int(alt_answers[i])
                    question_visual_meta[i]["status"] = str(alt_statuses[i])

    # Secondary decode on current warp using grayscale-adaptive threshold.
    # This helps when global binarization collapses many rows into the same option.
    if bool(sid_has_write_row) and len(user_answers) > 0:
        mcq_gray_now = imgWarpGray[mcq_y:mcq_y + mcq_h, mcq_x:mcq_x + mcq_w]
        if mcq_gray_now.size > 0:
            clahe_now = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(mcq_gray_now)
            mcq_bin_now = cv2.adaptiveThreshold(
                clahe_now,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                35,
                8,
            )
            mcq_bin_now = cv2.morphologyEx(mcq_bin_now, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

            sec_answers = []
            sec_statuses = []
            sec_conf = []

            for block_idx in range(block_count):
                block_start_q = block_idx * rows_per_block
                rows_in_block = min(rows_per_block, questions - block_start_q)
                if rows_in_block <= 0:
                    break

                bx = block_idx * (block_w + block_gap)
                bx2 = mcq_w if block_idx == block_count - 1 else min(bx + block_w, mcq_w)
                block = mcq_bin_now[:, bx:bx2]

                bw = block.shape[1]
                bubble_left = int(bw * 0.16)
                bubble_right = int(bw * 0.95)
                bubble_grid = block[:, bubble_left:bubble_right]
                bubble_grid, _ = _refine_bubble_grid_roi(
                    bubble_grid,
                    refine_vertical=False,
                    refine_horizontal=False,
                )
                if not use_template_geometry:
                    bubble_grid, _ = _trim_mcq_block_top_header(bubble_grid, max_trim_ratio=0.16)

                px = _split_grid(bubble_grid, rows=rows_in_block, cols=choices, inner_ratio=0.70)
                idx2, st2, cf2 = _pick_marked_with_flags(
                    px,
                    min_conf_ratio=1.03,
                    min_peak_factor=1.10,
                    blank_floor=18.0,
                    blank_std_factor=0.35,
                    min_peak_strength=1.05,
                )

                sec_answers.extend([int(x) for x in idx2])
                sec_statuses.extend([str(x) for x in st2])
                sec_conf.extend([float(x) for x in cf2])

            def _ans_dominance(ans_list):
                vals = [int(x) for x in ans_list if int(x) >= 0]
                if not vals:
                    return 1.0
                freq = {}
                for v in vals:
                    freq[v] = freq.get(v, 0) + 1
                return max(freq.values()) / float(len(vals))

            if len(sec_answers) == len(user_answers):
                pri_unc = len(uncertain_questions)
                sec_unc = sum(1 for s in sec_statuses if s != "ok")
                pri_dom = _ans_dominance(user_answers)
                sec_dom = _ans_dominance(sec_answers)

                pri_score = pri_unc * 2.0 + max(0.0, pri_dom - 0.45) * 10.0
                sec_score = sec_unc * 2.0 + max(0.0, sec_dom - 0.45) * 10.0

                if sec_score + 0.2 < pri_score:
                    user_answers = sec_answers
                    answer_confidences = sec_conf
                    answer_map = {str(i + 1): int(a) for i, a in enumerate(user_answers)}
                    uncertain_questions = [
                        {"question": i + 1, "status": sec_statuses[i]}
                        for i in range(len(sec_statuses))
                        if sec_statuses[i] != "ok"
                    ]
                    for i in range(min(len(question_visual_meta), len(sec_answers))):
                        question_visual_meta[i]["selected"] = int(sec_answers[i])
                        question_visual_meta[i]["status"] = str(sec_statuses[i])

    # Horizontal-shift rescue when one option dominates too heavily (often left-boundary drift).
    if bool(sid_has_write_row) and len(user_answers) > 0:
        def _ans_dominance_local(ans_list):
            vals = [int(x) for x in ans_list if int(x) >= 0]
            if not vals:
                return 1.0
            freq = {}
            for v in vals:
                freq[v] = freq.get(v, 0) + 1
            return max(freq.values()) / float(len(vals))

        base_dom = _ans_dominance_local(user_answers)
        if base_dom >= 0.62:
            shift_best = None
            shift_best_score = None

            shift_left_ratios = [0.18, 0.22, 0.26, 0.30]
            shift_right_ratios = [0.95, 0.96, 0.97]
            for left_ratio in shift_left_ratios:
                for right_ratio in shift_right_ratios:
                    if right_ratio - left_ratio < 0.50:
                        continue
                    trial_answers = []
                    trial_statuses = []
                    trial_conf = []

                    for block_idx in range(block_count):
                        block_start_q = block_idx * rows_per_block
                        rows_in_block = min(rows_per_block, questions - block_start_q)
                        if rows_in_block <= 0:
                            break

                        bx = block_idx * (block_w + block_gap)
                        bx2 = mcq_w if block_idx == block_count - 1 else min(bx + block_w, mcq_w)
                        block = mcq_roi[:, bx:bx2]

                        bw = block.shape[1]
                        bubble_left = int(bw * left_ratio)
                        bubble_right = int(bw * right_ratio)
                        if bubble_right - bubble_left < int(max(20, bw * 0.40)):
                            continue
                        bubble_grid = block[:, bubble_left:bubble_right]
                        bubble_grid, _ = _refine_bubble_grid_roi(
                            bubble_grid,
                            refine_vertical=False,
                            refine_horizontal=False,
                        )
                        if not use_template_geometry:
                            bubble_grid, _ = _trim_mcq_block_top_header(bubble_grid, max_trim_ratio=0.16)

                        px = _split_grid(bubble_grid, rows=rows_in_block, cols=choices, inner_ratio=0.72)
                        idx3, st3, cf3 = _pick_marked_with_flags(
                            px,
                            min_conf_ratio=1.03,
                            min_peak_factor=1.08,
                            blank_floor=20.0,
                            blank_std_factor=0.35,
                            min_peak_strength=1.04,
                        )
                        trial_answers.extend([int(x) for x in idx3])
                        trial_statuses.extend([str(x) for x in st3])
                        trial_conf.extend([float(x) for x in cf3])

                    if len(trial_answers) != len(user_answers):
                        continue

                    tr_unc = sum(1 for s in trial_statuses if s != "ok")
                    tr_dom = _ans_dominance_local(trial_answers)
                    tr_score = tr_unc * 2.0 + max(0.0, tr_dom - 0.50) * 8.0
                    if shift_best_score is None or tr_score < shift_best_score:
                        shift_best_score = float(tr_score)
                        shift_best = (trial_answers, trial_statuses, trial_conf, tr_unc, tr_dom)

            if shift_best is not None:
                tr_answers, tr_statuses, tr_conf, tr_unc, tr_dom = shift_best
                cur_unc = len(uncertain_questions)
                cur_score = cur_unc * 2.0 + max(0.0, base_dom - 0.50) * 8.0
                new_score = tr_unc * 2.0 + max(0.0, tr_dom - 0.50) * 8.0
                if new_score + 0.2 < cur_score:
                    user_answers = tr_answers
                    answer_confidences = tr_conf
                    answer_map = {str(i + 1): int(a) for i, a in enumerate(user_answers)}
                    uncertain_questions = [
                        {"question": i + 1, "status": tr_statuses[i]}
                        for i in range(len(tr_statuses))
                        if tr_statuses[i] != "ok"
                    ]
                    for i in range(min(len(question_visual_meta), len(tr_answers))):
                        question_visual_meta[i]["selected"] = int(tr_answers[i])
                        question_visual_meta[i]["status"] = str(tr_statuses[i])

    # --- 6. Chấm điểm ---
    detected_questions = len(user_answers)
    grading_questions = min(detected_questions, len(answer_key), int(questions))
    if grading_questions <= 0:
        return {"error": "Không có đáp án hợp lệ để chấm."}

    grading = []
    for i in range(grading_questions):
        user_ans = int(user_answers[i])
        if user_ans < 0:
            grading.append(0)
        else:
            grading.append(1 if int(answer_key[i]) == user_ans else 0)

    score = (sum(grading) / grading_questions) * 100.0

    # Guarded self-retry: when current warp is weak, retry once with marker-original manual quad.
    if bool(sid_has_write_row) and (not bool(_internal_retry)) and (not manual_quad_locked):
        cur_sid_uncertain = sum(1 for s in sid_status if s != "ok")
        cur_sid_min_conf = min(sid_conf) if len(sid_conf) > 0 else 0.0
        cur_mcq_uncertain = len(uncertain_questions)

        should_retry = False
        if warp_strategy in ("marker-orig-ordered", "page-contour"):
            if cur_sid_uncertain > 0 or cur_mcq_uncertain > 0 or cur_sid_min_conf < 1.04:
                should_retry = True

        if should_retry:
            raw_h, raw_w = orig_img.shape[:2]
            marker_raw = _detect_page_corners_from_markers(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY))
            if marker_raw is not None:
                marker_raw = _order_quad_points(np.asarray(marker_raw, dtype=np.float32).reshape(4, 2))
                rw = max(1.0, float(raw_w - 1))
                rh = max(1.0, float(raw_h - 1))
                retry_res = process_omr_exam(
                    image_path=image_path,
                    output_folder=output_folder,
                    answer_key=answer_key,
                    questions=questions,
                    choices=choices,
                    rows_per_block=rows_per_block,
                    num_blocks=num_blocks,
                    student_id_digits=student_id_digits,
                    sid_has_write_row=sid_has_write_row,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    crop_tl_x=float(marker_raw[0, 0] / rw),
                    crop_tl_y=float(marker_raw[0, 1] / rh),
                    crop_tr_x=float(marker_raw[1, 0] / rw),
                    crop_tr_y=float(marker_raw[1, 1] / rh),
                    crop_br_x=float(marker_raw[2, 0] / rw),
                    crop_br_y=float(marker_raw[2, 1] / rh),
                    crop_bl_x=float(marker_raw[3, 0] / rw),
                    crop_bl_y=float(marker_raw[3, 1] / rh),
                    _internal_retry=True,
                )
                if isinstance(retry_res, dict) and bool(retry_res.get("success")):
                    r_sid_status = retry_res.get("student_id_status", [])
                    r_sid_conf = retry_res.get("student_id_confidence", [])
                    r_sid_uncertain = sum(1 for s in r_sid_status if s != "ok")
                    r_sid_min_conf = min(r_sid_conf) if len(r_sid_conf) > 0 else 0.0
                    r_mcq_uncertain = int(retry_res.get("uncertain_count", 9999))

                    cur_score = cur_sid_uncertain * 3.0 + cur_mcq_uncertain * 2.0 + max(0.0, 1.05 - cur_sid_min_conf) * 4.0
                    ret_score = r_sid_uncertain * 3.0 + r_mcq_uncertain * 2.0 + max(0.0, 1.05 - r_sid_min_conf) * 4.0

                    if ret_score + 0.05 < cur_score:
                        retry_warnings = list(retry_res.get("warnings") or [])
                        retry_warnings.append("Đã tự động retry bằng marker-original để cải thiện độ ổn định.")
                        retry_res["warnings"] = retry_warnings
                        return retry_res

    # --- 7. Vẽ kết quả lên ảnh warped ---
    imgResult = imgWarpColored.copy()
    choice_labels = _build_choice_labels(choices)
    sidCrop = imgWarpColored[sid_y:sid_y + sid_h, sid_x:sid_x + sid_w].copy()
    mcqCrop = imgWarpColored[mcq_y:mcq_y + mcq_h, mcq_x:mcq_x + mcq_w].copy()

    # Draw student-id ROI
    cv2.rectangle(imgResult, (sid_x, sid_y), (sid_x + sid_w, sid_y + sid_h), (255, 128, 0), 2)
    cv2.putText(
        imgResult,
        f"Student ID: {student_id}",
        (sid_x, max(20, sid_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 128, 0),
        2,
    )

    # Draw exam-code ROI for user verification.
    cv2.rectangle(imgResult, (code_x, code_y), (code_x + code_w, code_y + code_h), (0, 128, 255), 2)
    cv2.putText(
        imgResult,
        f"Exam code: {exam_code_detected}",
        (code_x, max(20, code_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 128, 255),
        2,
    )

    # Draw block boundaries
    cv2.rectangle(imgResult, (mcq_x, mcq_y), (mcq_x + mcq_w, mcq_y + mcq_h), (0, 255, 255), 2)
    cv2.putText(
        imgResult,
        f"Score: {score:.2f}% ({sum(grading)}/{grading_questions})",
        (mcq_x, max(20, mcq_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 0),
        2,
    )

    cv2.putText(
        imgResult,
        f"Uncertain marks: {len(uncertain_questions)} | Otsu: {otsu_value:.1f}",
        (mcq_x, min(h - 20, mcq_y + mcq_h + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 140, 255),
        2,
    )

    # Draw selected vs correct answers directly on the sheet.
    answer_compare = []
    wrong_questions = []

    for i, meta in enumerate(question_visual_meta):
        q_num = int(meta["question"])
        selected_idx = int(meta["selected"])
        status = str(meta["status"])
        option_centers = meta["option_centers"]
        row_anchor = meta["row_center"]

        correct_idx = int(answer_key[i]) if i < len(answer_key) else -1

        selected_label = choice_labels[selected_idx] if 0 <= selected_idx < len(choice_labels) else "?"
        correct_label = choice_labels[correct_idx] if 0 <= correct_idx < len(choice_labels) else "-"
        is_correct = (selected_idx >= 0 and correct_idx >= 0 and selected_idx == correct_idx and status == "ok")

        answer_compare.append({
            "question": q_num,
            "selected": selected_idx,
            "selected_label": selected_label,
            "correct": correct_idx,
            "correct_label": correct_label,
            "status": status,
            "is_correct": bool(is_correct),
        })

        if not is_correct and correct_idx >= 0:
            wrong_questions.append(q_num)

        # Draw question number near each row
        cv2.putText(
            imgResult,
            str(q_num),
            (int(row_anchor[0]) - 18, int(row_anchor[1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (80, 80, 80),
            1,
        )

        # Draw question number on MCQ crop (local coordinates)
        local_row_x = int(row_anchor[0]) - mcq_x
        local_row_y = int(row_anchor[1]) - mcq_y
        cv2.putText(
            mcqCrop,
            str(q_num),
            (local_row_x - 18, local_row_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (80, 80, 80),
            1,
        )

        # Draw correct answer marker (green ring)
        if 0 <= correct_idx < len(option_centers):
            cc = option_centers[correct_idx]
            cv2.circle(imgResult, cc, 8, (0, 180, 0), 2)
            cv2.circle(mcqCrop, (cc[0] - mcq_x, cc[1] - mcq_y), 8, (0, 180, 0), 2)

        # Draw selected answer marker
        if 0 <= selected_idx < len(option_centers):
            sc = option_centers[selected_idx]
            if is_correct:
                cv2.circle(imgResult, sc, 6, (0, 200, 0), cv2.FILLED)
                cv2.circle(mcqCrop, (sc[0] - mcq_x, sc[1] - mcq_y), 6, (0, 200, 0), cv2.FILLED)
            else:
                cv2.circle(imgResult, sc, 6, (0, 0, 255), cv2.FILLED)
                cv2.circle(mcqCrop, (sc[0] - mcq_x, sc[1] - mcq_y), 6, (0, 0, 255), cv2.FILLED)
        elif status != "ok":
            cv2.putText(
                imgResult,
                "?",
                (int(row_anchor[0]) + 2, int(row_anchor[1]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 140, 255),
                2,
            )
            cv2.putText(
                mcqCrop,
                "?",
                (local_row_x + 2, local_row_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 140, 255),
                2,
            )

    cv2.putText(
        sidCrop,
        f"SID: {student_id}",
        (8, min(sid_h - 8, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 128, 0),
        2,
    )

    cv2.putText(
        mcqCrop,
        f"Score: {score:.2f}% | Uncertain: {len(uncertain_questions)}",
        (8, min(mcq_h - 8, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 200, 0),
        2,
    )

    # --- 8. Lưu ảnh kết quả ---
    base_name = os.path.basename(image_path)
    base_stem, _ = os.path.splitext(base_name)
    result_name = f"graded_{base_name}"
    result_path = os.path.join(output_folder, result_name)
    sid_crop_name = f"sid_crop_{base_stem}.png"
    sid_crop_path = os.path.join(output_folder, sid_crop_name)
    mcq_crop_name = f"mcq_crop_{base_stem}.png"
    mcq_crop_path = os.path.join(output_folder, mcq_crop_name)

    cv2.imwrite(result_path, imgResult)
    cv2.imwrite(sid_crop_path, sidCrop)
    cv2.imwrite(mcq_crop_path, mcqCrop)

    warnings = []
    if manual_quad_warning:
        warnings.append(manual_quad_warning)
    if marker_warning:
        warnings.append(marker_warning)

    return {
        "success": True,
        "pipeline_version": "omr-warp-select-v3",
        "score": score,
        "student_id": student_id,
        "student_id_status": sid_status,
        "student_id_confidence": sid_conf,
        "exam_code": exam_code_detected,
        "exam_code_status": code_status,
        "exam_code_confidence": code_conf,
        "exam_title_detected": exam_title_detected,
        "user_answers": user_answers,
        "answer_map": answer_map,
        "answer_confidences": answer_confidences,
        "answer_compare": answer_compare,
        "correct_answers": answer_key[:grading_questions],
        "graded_questions": grading_questions,
        "detected_questions": detected_questions,
        "uncertain_questions": uncertain_questions,
        "uncertain_count": len(uncertain_questions),
        "wrong_questions": wrong_questions,
        "roi_boxes": {
            "student_id": {"x": sid_x, "y": sid_y, "w": sid_w, "h": sid_h},
            "exam_code": {"x": code_x, "y": code_y, "w": code_w, "h": code_h},
            "mcq": {"x": mcq_x, "y": mcq_y, "w": mcq_w, "h": mcq_h},
        },
        "roi_detection": {
            "strategy": "block-morphology-close-25x25 + spatial-filters",
            "sid_rule": "vertical && top-half && left-half",
            "mcq_rule": "horizontal && lower-half && largest-area",
        },
        "sid_layout": {
            "digits": student_id_digits,
            "rows": 10,
        },
        "mcq_layout": {
            "questions": questions,
            "choices": choices,
            "rows_per_block": rows_per_block,
            "num_blocks": block_count,
            "block_gap_px": block_gap,
            "block_width_px": block_w,
        },
        "threshold_info": {
            "method": "otsu_binary_inv + morph_close_open",
            "otsu_value": otsu_value,
        },
        "warp_strategy": warp_strategy,
        "warnings": warnings,
        "sid_crop_image": sid_crop_name,
        "mcq_crop_image": mcq_crop_name,
        "result_image": result_name
    }