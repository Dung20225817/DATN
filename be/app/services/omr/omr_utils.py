import cv2
import numpy as np


def _clip_rect(x, y, w, h, max_w, max_h):
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return x, y, w, h


def _bool_from_any(raw, default=False):
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _float_from_any(raw, default, low=None, high=None):
    try:
        val = float(raw)
    except Exception:
        val = float(default)
    if low is not None:
        val = max(float(low), val)
    if high is not None:
        val = min(float(high), val)
    return float(val)


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


def _split_grid_with_row_edges(
    binary_img,
    row_edges,
    cols: int,
    inner_ratio: float = 1.0,
):
    """Split binary ROI using explicit non-uniform row edges."""
    fallback_rows = 1
    if isinstance(row_edges, (list, tuple, np.ndarray)):
        try:
            fallback_rows = max(1, int(len(row_edges) - 1))
        except Exception:
            fallback_rows = 1

    if not isinstance(row_edges, (list, tuple, np.ndarray)):
        return _split_grid(binary_img, rows=fallback_rows, cols=cols, inner_ratio=inner_ratio)

    h, w = binary_img.shape[:2]
    edges = np.asarray(row_edges, dtype=np.int32).reshape(-1)
    if edges.size < 2:
        return _split_grid(binary_img, rows=fallback_rows, cols=cols, inner_ratio=inner_ratio)

    edges = np.clip(edges, 0, h)
    edges[0] = 0
    edges[-1] = h

    valid = True
    for i in range(1, int(edges.size)):
        if edges[i] <= edges[i - 1]:
            valid = False
            break
    if not valid:
        edges = np.linspace(0, h, fallback_rows + 1, dtype=np.int32)

    rows = max(1, int(edges.size - 1))
    col_edges = np.linspace(0, w, max(1, int(cols)) + 1, dtype=np.int32)
    pixel_val = np.zeros((rows, max(1, int(cols))), dtype=np.float32)
    inner_ratio = max(0.35, min(1.0, float(inner_ratio)))

    for r in range(rows):
        y1, y2 = int(edges[r]), int(edges[r + 1])
        if y2 <= y1:
            continue
        for c in range(int(cols)):
            x1, x2 = int(col_edges[c]), int(col_edges[c + 1])
            if x2 <= x1:
                continue
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


def _extract_grid_cell(
    img,
    row_edges,
    col_edges,
    row_idx,
    col_idx,
    row_shift=0,
    inner_ratio=0.84,
):
    """Extract one grid cell patch with optional row shift and center crop."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if row_idx < 0 or col_idx < 0 or row_idx + 1 >= len(row_edges) or col_idx + 1 >= len(col_edges):
        return None

    y1 = int(row_edges[row_idx]) + int(row_shift)
    y2 = int(row_edges[row_idx + 1]) + int(row_shift)
    y1 = max(0, min(max(0, h - 2), y1))
    y2 = max(y1 + 1, min(h, y2))

    x1 = int(col_edges[col_idx])
    x2 = int(col_edges[col_idx + 1])
    x1 = max(0, min(max(0, w - 2), x1))
    x2 = max(x1 + 1, min(w, x2))

    cell = img[y1:y2, x1:x2]
    if cell.size == 0:
        return None

    ratio = max(0.45, min(1.0, float(inner_ratio)))
    if ratio < 0.999:
        ch, cw = cell.shape[:2]
        ih = max(3, int(ch * ratio))
        iw = max(3, int(cw * ratio))
        sy = max(0, (ch - ih) // 2)
        sx = max(0, (cw - iw) // 2)
        cell = cell[sy:sy + ih, sx:sx + iw]

    return cell


def _split_grid_darkness(gray_img, rows: int, cols: int, inner_ratio: float = 1.0):
    """Split grayscale ROI to rows x cols and return darkness score matrix (higher is darker)."""
    h, w = gray_img.shape[:2]
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)
    darkness = np.zeros((rows, cols), dtype=np.float32)
    inner_ratio = max(0.35, min(1.0, float(inner_ratio)))

    for r in range(rows):
        y1, y2 = int(row_edges[r]), int(row_edges[r + 1])
        cell_h = max(1, y2 - y1)
        y_pad = int((1.0 - inner_ratio) * cell_h * 0.5)
        iy1 = min(max(0, y1 + y_pad), h)
        iy2 = min(max(0, y2 - y_pad), h)
        if iy2 <= iy1:
            iy1, iy2 = y1, y2

        for c in range(cols):
            x1, x2 = int(col_edges[c]), int(col_edges[c + 1])
            cell_w = max(1, x2 - x1)
            x_pad = int((1.0 - inner_ratio) * cell_w * 0.5)
            ix1 = min(max(0, x1 + x_pad), w)
            ix2 = min(max(0, x2 - x_pad), w)
            if ix2 <= ix1:
                ix1, ix2 = x1, x2

            cell = gray_img[iy1:iy2, ix1:ix2]
            if cell.size == 0:
                darkness[r, c] = 0.0
                continue

            p25 = float(np.percentile(cell, 25))
            mean_val = float(np.mean(cell))
            darkness[r, c] = max(0.0, 255.0 - (0.65 * p25 + 0.35 * mean_val))

    return darkness


def _split_grid_darkness_with_row_edges(gray_img, row_edges, cols: int, inner_ratio: float = 1.0):
    """Split grayscale ROI with non-uniform row edges and return darkness matrix."""
    fallback_rows = 1
    if isinstance(row_edges, (list, tuple, np.ndarray)):
        try:
            fallback_rows = max(1, int(len(row_edges) - 1))
        except Exception:
            fallback_rows = 1

    if not isinstance(row_edges, (list, tuple, np.ndarray)):
        return _split_grid_darkness(gray_img, rows=fallback_rows, cols=cols, inner_ratio=inner_ratio)

    h, w = gray_img.shape[:2]
    edges = np.asarray(row_edges, dtype=np.int32).reshape(-1)
    if edges.size < 2:
        return _split_grid_darkness(gray_img, rows=fallback_rows, cols=cols, inner_ratio=inner_ratio)

    edges = np.clip(edges, 0, h)
    edges[0] = 0
    edges[-1] = h
    valid = True
    for i in range(1, int(edges.size)):
        if edges[i] <= edges[i - 1]:
            valid = False
            break
    if not valid:
        return _split_grid_darkness(gray_img, rows=fallback_rows, cols=cols, inner_ratio=inner_ratio)

    rows = max(1, int(edges.size - 1))
    col_edges = np.linspace(0, w, max(1, int(cols)) + 1, dtype=np.int32)
    darkness = np.zeros((rows, max(1, int(cols))), dtype=np.float32)
    ratio = max(0.35, min(1.0, float(inner_ratio)))

    for r in range(rows):
        y1 = int(edges[r])
        y2 = int(edges[r + 1])
        if y2 <= y1:
            continue
        cell_h = max(1, y2 - y1)
        y_pad = int((1.0 - ratio) * cell_h * 0.5)
        iy1 = min(max(0, y1 + y_pad), h)
        iy2 = min(max(0, y2 - y_pad), h)
        if iy2 <= iy1:
            iy1, iy2 = y1, y2

        for c in range(int(cols)):
            x1 = int(col_edges[c])
            x2 = int(col_edges[c + 1])
            if x2 <= x1:
                continue
            cell_w = max(1, x2 - x1)
            x_pad = int((1.0 - ratio) * cell_w * 0.5)
            ix1 = min(max(0, x1 + x_pad), w)
            ix2 = min(max(0, x2 - x_pad), w)
            if ix2 <= ix1:
                ix1, ix2 = x1, x2

            cell = gray_img[iy1:iy2, ix1:ix2]
            if cell.size == 0:
                darkness[r, c] = 0.0
                continue
            p25 = float(np.percentile(cell, 25))
            mean_val = float(np.mean(cell))
            darkness[r, c] = max(0.0, 255.0 - (0.65 * p25 + 0.35 * mean_val))

    return darkness


def _build_row_edges_from_center_bounds(top_center, bottom_center, rows, max_h):
    """Build non-uniform row edges from two anchor centers (first row and last row)."""
    try:
        rows = max(1, int(rows))
        max_h = max(1, int(max_h))
        top_center = float(top_center)
        bottom_center = float(bottom_center)
    except Exception:
        return None

    if rows == 1:
        return np.array([0, max_h], dtype=np.int32)

    min_span = max(18.0, float(max_h) * 0.40)
    if (bottom_center - top_center) < min_span:
        return None

    pitch = (bottom_center - top_center) / max(1.0, float(rows - 1))
    if pitch < 2.5:
        return None

    edges = np.zeros((rows + 1,), dtype=np.float32)
    edges[0] = top_center - (0.5 * pitch)
    for i in range(1, rows):
        edges[i] = top_center + ((float(i) - 0.5) * pitch)
    edges[-1] = bottom_center + (0.5 * pitch)

    edges = np.clip(edges, 0.0, float(max_h))
    edges[0] = 0.0
    edges[-1] = float(max_h)
    out = np.round(edges).astype(np.int32)

    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    if out[-1] > max_h:
        out[-1] = max_h
    for i in range(out.size - 2, -1, -1):
        if out[i] >= out[i + 1]:
            out[i] = out[i + 1] - 1

    if out[0] != 0:
        out[0] = 0
    if out[-1] != max_h:
        out[-1] = max_h

    valid = True
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            valid = False
            break
    if not valid:
        return None

    return out


def _shift_row_edges(row_edges, trim_top, max_h):
    """Shift row edges after trimming top pixels and keep monotonic constraints."""
    if row_edges is None:
        return None
    if not isinstance(row_edges, (list, tuple, np.ndarray)):
        return None
    try:
        trim_top = int(trim_top)
        max_h = int(max_h)
    except Exception:
        return None

    arr = np.asarray(row_edges, dtype=np.int32).reshape(-1)
    if arr.size < 2:
        return None

    shifted = arr - int(trim_top)
    shifted = np.clip(shifted, 0, max_h)
    shifted[0] = 0
    shifted[-1] = max_h

    for i in range(1, shifted.size):
        if shifted[i] <= shifted[i - 1]:
            shifted[i] = shifted[i - 1] + 1
    if shifted[-1] > max_h:
        shifted[-1] = max_h
    for i in range(shifted.size - 2, -1, -1):
        if shifted[i] >= shifted[i + 1]:
            shifted[i] = shifted[i + 1] - 1

    if shifted[0] != 0:
        shifted[0] = 0
    if shifted[-1] != max_h:
        shifted[-1] = max_h

    valid = True
    for i in range(1, shifted.size):
        if shifted[i] <= shifted[i - 1]:
            valid = False
            break
    if not valid:
        return None
    return shifted
