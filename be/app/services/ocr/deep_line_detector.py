"""
Deep Learning-based Text Line Detector
Dùng CRAFT (craft-text-detector) để phát hiện text region,
gộp thành dòng bằng Linear Regression baseline, crop từng dòng cho VietOCR.

Cài đặt: pip install craft-text-detector
"""

import cv2
import numpy as np
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _boxes_to_xyxy(boxes) -> List[Tuple[int, int, int, int]]:
    """
    Chuyển đổi danh sách polygon / mảng (N,2) → tuple (x1, y1, x2, y2).
    Hỗ trợ: flat list [x1,y1,x2,y2,...], numpy (4,2), list of [x,y] points.
    """
    rects = []
    for b in boxes:
        pts = np.array(b, dtype=np.float32).reshape(-1, 2)
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)
        if x2 > x1 and y2 > y1:
            rects.append((int(x1), int(y1), int(x2), int(y2)))
    return rects


def _group_boxes_into_lines(
    rects: List[Tuple[int, int, int, int]],
    img_h: int,
    height_scale: float = 0.6,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Gộp các bounding box thành dòng text dùng Linear Regression baseline.

    Thuật toán:
    - Sắp xếp box theo y_center (top-to-bottom).
    - Với mỗi box mới, ước tính đường baseline của từng dòng hiện có
      bằng Linear Regression (np.polyfit) trên tọa độ (x_center, y_center)
      của các box đã thuộc dòng đó.
    - Nếu |y_center_box - y_dự_đoán| < tolerance → gán vào dòng đó.
    - Tolerance = max(chiều cao box hiện tại, chiều cao TB dòng) × height_scale.
    - Trong mỗi dòng: sắp xếp lại box theo x1 (trái → phải) cho VietOCR.

    height_scale: hệ số nhân với chiều cao box để tính tolerance.
                  Tăng lên nếu chữ nghiêng nhiều (thử 0.7-0.9).
    """
    if not rects:
        return []

    def _yc(r): return (r[1] + r[3]) / 2.0
    def _xc(r): return (r[0] + r[2]) / 2.0
    def _h(r):  return max(r[3] - r[1], 1)

    # Sắp xếp top-to-bottom theo y_center
    rects = sorted(rects, key=_yc)

    lines: List[List] = []

    for rect in rects:
        rc_y = _yc(rect)
        rc_x = _xc(rect)
        rc_h = _h(rect)

        best_line = None
        best_dist = float('inf')

        for line in lines:
            avg_h = sum(_h(r) for r in line) / len(line)
            tolerance = max(avg_h, rc_h) * height_scale

            xs = [_xc(r) for r in line]
            ys = [_yc(r) for r in line]

            if len(line) >= 2:
                # Linear regression: predicted baseline y tại vị trí x của box mới
                a, b = np.polyfit(xs, ys, 1)
                predicted_y = a * rc_x + b
            else:
                predicted_y = ys[0]

            dist = abs(rc_y - predicted_y)
            if dist < tolerance and dist < best_dist:
                best_dist = dist
                best_line = line

        if best_line is not None:
            best_line.append(rect)
        else:
            lines.append([rect])

    # Sắp xếp box trong mỗi dòng theo x1 (trái → phải) để VietOCR đọc đúng thứ tự
    for line in lines:
        line.sort(key=lambda r: r[0])

    # Tính slope (hệ số góc baseline) cho mỗi dòng bằng linear regression
    # Slope dùng để xoay deskew khi crop → trả về List[Tuple[List, float]]
    result = []
    for line in lines:
        xs = [_xc(r) for r in line]
        ys = [_yc(r) for r in line]
        if len(line) >= 2:
            a, _ = np.polyfit(xs, ys, 1)
            slope = float(a)
        else:
            slope = 0.0
        result.append((line, slope))

    return result  # List[Tuple[List[rect], float(slope)]]


def _crop_line_deskew(
    img: np.ndarray,
    line_boxes: List[Tuple[int, int, int, int]],
    slope: float = 0.0,
    padding: int = 5,
    y_safe_top: int = 0,
    y_safe_bottom: int = -1,
) -> np.ndarray:
    """
    Crop dòng text với Perspective Deskew dựa trên slope baseline.

    Thuật toán:
    1. Tính union bbox + safe padding (giống axis-aligned cũ).
    2. Tính góc nghiêng: angle = degrees(arctan(slope)).
    3. Xoay toàn bộ ảnh quanh tâm vùng crop bằng -angle
       (BORDER_REPLICATE tránh viền đen).
    4. Crop vùng đó từ ảnh đã xoay → text nằm ngang.

    Args:
        slope        : hệ số góc baseline từ linear regression (y = slope*x + b).
        padding      : pixel đệm mỗi phía.
        y_safe_top   : giới hạn trên để không lấn sang dòng liền trên.
        y_safe_bottom: giới hạn dưới để không lấn sang dòng liền dưới.
    """
    h, w = img.shape[:2]
    if y_safe_bottom < 0:
        y_safe_bottom = h

    # Union bbox với safe vertical padding
    x1 = max(0,             min(b[0] for b in line_boxes) - padding)
    y1 = max(y_safe_top,    min(b[1] for b in line_boxes) - padding)
    x2 = min(w,             max(b[2] for b in line_boxes) + padding)
    y2 = min(y_safe_bottom, max(b[3] for b in line_boxes) + padding)

    # Góc nghiêng của dòng (trong hệ tọa độ ảnh, y tăng xuống dưới)
    # slope > 0 → text nghiêng XUỐNG-phải; slope < 0 → nghiêng LÊN-phải
    skew_deg = float(np.degrees(np.arctan(slope)))

    # Nếu nghiêng không đáng kể (< 1.0°) → crop thẳng, tránh noise từ ít box
    if abs(skew_deg) < 1.0:
        return img[y1:y2, x1:x2]

    # OpenCV getRotationMatrix2D: góc DƯƠNG = xoay CCW (ngược chiều kim đồng hồ)
    # trên màn hình (gốc tọa độ góc trên-trái, Y tăng xuống dưới).
    #
    # Để leveling:
    #   slope > 0 (nghiêng xuống-phải) → cần xoay CCW  → angle = +skew_deg (dương)
    #   slope < 0 (nghiêng lên-phải)   → cần xoay CW   → angle = +skew_deg (âm)
    # ⟹ luôn dùng +skew_deg (KHÔNG phủ định)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), skew_deg, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# CRAFT detector — via craft-text-detector
# ─────────────────────────────────────────────────────────────────────────────

class CRAFTLineDetector:
    """
    Phát hiện dòng text bằng CRAFT (Character Region Awareness For Text).

    CRAFT phát hiện vùng ký tự và vùng liên kết giữa các ký tự,
    trả về bounding box ở cấp độ từ/ký tự. Các box được gộp thành
    dòng bằng _group_boxes_into_lines.

    Yêu cầu: pip install craft-text-detector
    """

    def __init__(self, device: str = 'cpu'):
        try:
            # Patch: torchvision >= 0.13 removed model_urls from vgg module.
            # craft-text-detector still references it — add it back before import.
            import torchvision.models.vgg as _vgg
            if not hasattr(_vgg, 'model_urls'):
                _vgg.model_urls = {
                    'vgg11': '', 'vgg13': '', 'vgg16': '', 'vgg19': '',
                    'vgg11_bn': '', 'vgg13_bn': '', 'vgg16_bn': '', 'vgg19_bn': '',
                }
            from craft_text_detector import Craft
        except ImportError:
            raise RuntimeError(
                "craft-text-detector chưa được cài đặt. "
                "Chạy: pip install craft-text-detector"
            )
        cuda = (device == 'cuda')
        print(f"   [CRAFT] Loading (device={device})...")
        self._craft = Craft(
            output_dir=None,
            crop_type='poly',
            cuda=cuda,
        )
        print(f"   [CRAFT] [OK] Loaded")

    def detect_lines(self, image_path: str) -> list:
        """
        Returns: list[numpy.ndarray] — mỗi phần tử là 1 dòng text (BGR).
        """
        # Patch: craft_text_detector.predict uses np.array(polys) on inhomogeneous
        # polygon lists — fails under NumPy 2.x. Swap the module's `np` reference
        # with a proxy that catches ValueError and retries with dtype=object.
        import craft_text_detector.predict as _predict
        import craft_text_detector.craft_utils as _cu
        import numpy as _np_real

        class _SafeNP:
            """Proxy that wraps numpy; falls back to dtype=object on inhomogeneous arrays."""
            def __getattr__(self, name):
                return getattr(_np_real, name)

            def array(self, obj, *args, **kwargs):
                try:
                    return _np_real.array(obj, *args, **kwargs)
                except ValueError:
                    kwargs['dtype'] = object
                    return _np_real.array(obj, *args, **kwargs)

        _orig_predict_np = _predict.np
        _predict.np = _SafeNP()

        def _patched_adjust(polys, ratio_w, ratio_h, ratio_net=2):
            if len(polys) > 0:
                for k in range(len(polys)):
                    if polys[k] is not None:
                        polys[k] = _np_real.array(polys[k], dtype=_np_real.float32)
                        polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
            return polys

        _orig_adjust = _cu.adjustResultCoordinates
        _cu.adjustResultCoordinates = _patched_adjust
        try:
            prediction = self._craft.detect_text(image_path)
        finally:
            _predict.np = _orig_predict_np
            _cu.adjustResultCoordinates = _orig_adjust  # restore always

        raw_boxes = prediction.get('boxes', [])

        if not raw_boxes or len(raw_boxes) == 0:
            print("   [CRAFT] Không phát hiện text region nào")
            return []

        img = cv2.imread(image_path)
        boxes = _boxes_to_xyxy(raw_boxes)

        # _group_boxes_into_lines trả về List[Tuple[List[rect], float(slope)]]
        line_groups = _group_boxes_into_lines(boxes, img.shape[0])

        # Sắp xếp dòng top-to-bottom theo y trung bình của envelope
        line_groups.sort(
            key=lambda item: (min(r[1] for r in item[0]) + max(r[3] for r in item[0])) / 2
        )

        # Tính safe y-limits cho từng dòng để padding không lấn sang dòng liền kề
        n = len(line_groups)
        h_img = img.shape[0]
        y_limits = []
        for i, (line, _slope) in enumerate(line_groups):
            line_y1 = min(r[1] for r in line)
            line_y2 = max(r[3] for r in line)
            if i > 0:
                prev_y2 = max(r[3] for r in line_groups[i - 1][0])
                safe_top = (prev_y2 + line_y1) // 2
            else:
                safe_top = 0
            if i < n - 1:
                next_y1 = min(r[1] for r in line_groups[i + 1][0])
                safe_bottom = (line_y2 + next_y1) // 2
            else:
                safe_bottom = h_img
            y_limits.append((safe_top, safe_bottom))

        crops = [
            _crop_line_deskew(
                img, line, slope,
                padding=5,
                y_safe_top=lim[0],
                y_safe_bottom=lim[1],
            )
            for (line, slope), lim in zip(line_groups, y_limits)
        ]
        crops = [c for c in crops if c.shape[0] > 8 and c.shape[1] > 8]

        angles = [round(float(np.degrees(np.arctan(s))), 1) for _, s in line_groups]
        print(f"   [CRAFT] {len(raw_boxes)} regions -> {len(crops)} lines | angles: {angles}")
        return crops
