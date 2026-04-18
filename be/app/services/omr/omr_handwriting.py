from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .omr_utils import _clip_rect

HANDWRITING_FIELDS = ["truong", "ho_ten", "lop", "mon"]
_HANDWRITING_READER_CACHE: Dict[Tuple[str, bool], object] = {}
_LABEL_WIDTH_RATIO = 0.18
_OCR_TARGET_HEIGHT = 128
_OCR_OK_CONFIDENCE = 0.60
_RANDOM_NUMERIC_RE = re.compile(r"^\d{6,}$")
_HAS_DIGIT_RE = re.compile(r"\d")
_LABEL_LEAK_TOKENS = {
    "truong",
    "hoten",
    "ho",
    "ten",
    "lop",
    "mon",
    "monthi",
    "ngaysinh",
    "ngay",
    "nam",
    "thi",
    "chuky",
    "chu",
    "ky",
}
_SUBJECT_HINT_TOKENS = {
    "toan",
    "ly",
    "li",
    "hoa",
    "sinh",
    "van",
    "anh",
    "su",
    "dia",
    "tin",
    "gdcd",
    "congnghe",
    "nhat",
    "han",
    "phap",
    "duc",
}
_COMMON_VN_SURNAME_TOKENS = {
    "nguyen",
    "tran",
    "le",
    "pham",
    "hoang",
    "huynh",
    "phan",
    "vu",
    "vo",
    "dang",
    "bui",
    "do",
    "ho",
    "ngo",
    "duong",
    "ly",
}
_OCR_REFUSAL_HINTS = {
    "sorry",
    "cannot assist",
    "can not assist",
    "can't assist",
    "unable to assist",
    "i cannot",
    "i can not",
    "i'm unable",
    "i am unable",
    "cannot help",
    "can't help",
    "khong the",
    "khong ho tro",
    "xin loi",
    "toi khong the",
    "toi khong ho tro",
}


def _safe_float(raw, default=0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


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


def _dedupe_keep_order(items) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in list(items or []):
        text = str(raw or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _compact_text(text: str) -> str:
    return re.sub(r"[\s\-\._:/,]+", "", str(text or ""))


def _normalize_ascii_token(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFD", raw)
    no_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", "", no_marks)


def _normalize_ascii_words(text: str) -> List[str]:
    words: List[str] = []
    for part in re.split(r"\s+", str(text or "").strip()):
        token = _normalize_ascii_token(part)
        if token:
            words.append(token)
    return words


def _looks_like_label_leak(text: str) -> bool:
    words = _normalize_ascii_words(text)
    if not words:
        return False
    if len(words) == 1 and words[0] in _LABEL_LEAK_TOKENS:
        return True
    joined = "".join(words)
    return joined in _LABEL_LEAK_TOKENS


def _looks_like_valid_class(text: str) -> bool:
    compact = _normalize_ascii_token(text)
    if not compact:
        return False
    has_alpha = any(ch.isalpha() for ch in compact)
    has_digit = any(ch.isdigit() for ch in compact)
    return bool(has_alpha and has_digit and len(compact) >= 4)


def _looks_like_valid_subject(text: str) -> bool:
    words = _normalize_ascii_words(text)
    if not words:
        return False
    joined = "".join(words)
    if joined in _SUBJECT_HINT_TOKENS:
        return True
    for token in words:
        if token in _SUBJECT_HINT_TOKENS:
            return True
    if len(joined) <= 2 and joined in {"ly", "li", "van", "anh", "su", "dia", "tin"}:
        return True
    return False


def _looks_like_ocr_refusal(text: str) -> bool:
    norm = _normalize_ascii_token(text)
    if not norm:
        return False

    compact_text = str(text or "").strip().lower()
    for hint in _OCR_REFUSAL_HINTS:
        if hint in compact_text:
            return True

    # Handle punctuation-stripped replies where words are compacted.
    compact_hints = [
        "sorryicantassist",
        "sorryicannotassist",
        "unabletoassist",
        "cannotassist",
        "cantassist",
        "khongthe",
        "khonghotro",
        "xintloi",
        "xinloi",
    ]
    return any(hint in norm for hint in compact_hints)


def _looks_like_plausible_person_name(text: str) -> bool:
    words = _normalize_ascii_words(text)
    if len(words) < 2:
        return False
    if len(words) > 5:
        return False
    if any(len(token) <= 1 for token in words):
        return False
    return words[0] in _COMMON_VN_SURNAME_TOKENS


def _is_random_numeric_text(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False
    return bool(_RANDOM_NUMERIC_RE.fullmatch(compact))


def _estimate_text_confidence(
    text: str,
    ink_ratio: float,
    line_ratio: float,
    largest_component_ratio: float = 0.0,
) -> float:
    clean = str(text or "").strip()
    if not clean:
        return 0.0

    compact = _compact_text(clean)
    if not compact:
        return 0.0

    alnum_chars = [ch for ch in compact if ch.isalnum()]
    if not alnum_chars:
        return 0.0

    total = float(len(alnum_chars))
    alpha_count = sum(1 for ch in alnum_chars if ch.isalpha())
    digit_count = sum(1 for ch in alnum_chars if ch.isdigit())

    alpha_ratio = alpha_count / total
    digit_ratio = digit_count / total
    diversity = len(set(ch.lower() for ch in alnum_chars)) / float(max(1, min(len(alnum_chars), 12)))

    ink_term = _clamp(float(ink_ratio) * 16.0, 0.0, 1.0)
    line_term = 1.0 - _clamp(float(line_ratio) * 5.0, 0.0, 1.0)
    mix_term = 1.0 - _clamp(digit_ratio * 1.25, 0.0, 1.0)
    component_penalty = _clamp((float(largest_component_ratio) - 0.35) * 1.55, 0.0, 0.45)

    score = 0.40 * ink_term + 0.25 * line_term + 0.20 * diversity + 0.15 * mix_term
    if alpha_ratio >= 0.50:
        score += 0.10
    if len(alnum_chars) <= 5:
        score -= 0.12
    if len(alnum_chars) <= 3:
        score -= 0.10
    score -= component_penalty
    if _is_random_numeric_text(clean):
        score -= 0.25

    return round(_clamp(score, 0.0, 0.95), 6)


def _remove_horizontal_guide_lines(binary_inv, dotted_mask):
    if binary_inv is None or getattr(binary_inv, "size", 0) <= 0:
        return binary_inv, np.zeros((0, 0), dtype=np.uint8)

    cleaned = binary_inv.copy()
    removed_mask = np.zeros_like(binary_inv)
    if dotted_mask is None or getattr(dotted_mask, "size", 0) <= 0:
        return cleaned, removed_mask

    h, w = binary_inv.shape[:2]
    line_mask = np.zeros_like(binary_inv)

    lines = cv2.HoughLinesP(
        dotted_mask,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(10, int(round(w * 0.08))),
        minLineLength=max(12, int(round(w * 0.25))),
        maxLineGap=max(3, int(round(w * 0.04))),
    )

    if lines is not None:
        for segment in lines:
            x1, y1, x2, y2 = [int(v) for v in segment[0]]
            dx = abs(int(x2) - int(x1))
            dy = abs(int(y2) - int(y1))
            if dx < max(8, int(round(w * 0.08))):
                continue
            if dy > max(2, int(round(0.06 * max(1, dx)))):
                continue
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

    if cv2.countNonZero(line_mask) <= 0:
        line_mask = dotted_mask.copy()

    up = np.zeros_like(binary_inv)
    down = np.zeros_like(binary_inv)
    up[1:, :] = binary_inv[:-1, :]
    down[:-1, :] = binary_inv[1:, :]
    vertical_support = cv2.bitwise_or(up, down)
    vertical_support = cv2.dilate(vertical_support, np.ones((3, 1), np.uint8), iterations=1)

    removable = cv2.bitwise_and(line_mask, cv2.bitwise_not(vertical_support))
    removable = cv2.bitwise_and(removable, dotted_mask)
    if cv2.countNonZero(removable) > 0:
        removed_mask = cv2.bitwise_or(removed_mask, removable)
        cleaned = cv2.subtract(cleaned, removable)

    return cleaned, removed_mask


def _trim_strong_vertical_borders(binary_inv):
    if binary_inv is None or getattr(binary_inv, "size", 0) <= 0:
        return binary_inv

    h, w = binary_inv.shape[:2]
    if w < 20 or h < 8:
        return binary_inv

    col_density = np.mean(binary_inv > 0, axis=0)
    max_trim = max(1, int(round(w * 0.16)))

    left_trim = 0
    for idx in range(min(max_trim, w)):
        if col_density[idx] >= 0.83:
            left_trim += 1
        elif left_trim >= 2 and col_density[idx] >= 0.70:
            left_trim += 1
        else:
            break

    right_trim = 0
    for idx in range(w - 1, max(-1, w - max_trim - 1), -1):
        if col_density[idx] >= 0.83:
            right_trim += 1
        elif right_trim >= 2 and col_density[idx] >= 0.70:
            right_trim += 1
        else:
            break

    x1 = left_trim if left_trim >= 2 else 0
    x2 = w - right_trim if right_trim >= 2 else w
    if x2 - x1 < 16:
        return binary_inv

    return binary_inv[:, x1:x2]


def _largest_component_ratio(binary_inv) -> float:
    if binary_inv is None or getattr(binary_inv, "size", 0) <= 0:
        return 0.0

    ink_pixels = int(cv2.countNonZero(binary_inv))
    if ink_pixels <= 0:
        return 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    if num_labels <= 1:
        return 0.0

    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    return round(float(largest) / float(max(1, ink_pixels)), 6)


def _parse_roi_cfg(roi_cfg, img_w: int, img_h: int):
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


def _parse_handwriting_config(profile_handwriting_fields) -> Dict[str, object]:
    cfg = profile_handwriting_fields if isinstance(profile_handwriting_fields, dict) else {}

    engine = str(cfg.get("ocr_engine") or "vietocr_transformer").strip().lower()
    if engine not in {"vietocr_transformer", "internvl", "openai_gpt4o_mini"}:
        engine = "vietocr_transformer"

    parsed_rois: Dict[str, Dict[str, int]] = {}
    raw_rois = cfg.get("field_rois") if isinstance(cfg.get("field_rois"), dict) else {}
    for key in HANDWRITING_FIELDS:
        roi = _parse_roi_cfg(raw_rois.get(key), 1000, 1400)
        if roi is not None:
            parsed_rois[key] = roi

    return {
        "enabled": _safe_bool(cfg.get("enabled"), False),
        "ocr_engine": engine,
        "gpu": _safe_bool(cfg.get("gpu"), False),
        "save_crops": _safe_bool(cfg.get("save_crops"), True),
        "field_rois": parsed_rois,
    }


def _build_handwriting_rois(anchors, sid_roi, mcq_roi, img_w: int, img_h: int, cfg_field_rois) -> Dict[str, Dict[str, int]]:
    del anchors

    sid_x = float(_safe_float(sid_roi.get("x"), 0.0))
    sid_y = float(_safe_float(sid_roi.get("y"), 0.0))
    sid_w = float(max(1.0, _safe_float(sid_roi.get("w"), 1.0)))
    sid_h = float(max(1.0, _safe_float(sid_roi.get("h"), 1.0)))

    mcq_x = float(_safe_float(mcq_roi.get("x"), 0.0))
    mcq_y = float(_safe_float(mcq_roi.get("y"), 0.0))
    mcq_w = float(max(1.0, _safe_float(mcq_roi.get("w"), 1.0)))

    # Relative-frame layout: derive handwriting band from SID and MCQ spacing.
    sid_step = max(14.0, min(64.0, sid_h / 10.0))

    text_left = float(mcq_x + 0.06 * mcq_w)
    text_right = float(sid_x - max(16.0, 0.03 * mcq_w))
    if (text_right - text_left) < max(160.0, 0.16 * float(img_w)):
        text_left = float(max(0.18 * float(img_w), sid_x - max(360.0, 0.42 * float(img_w))))
        text_right = float(min(float(img_w) * 0.64, sid_x - 12.0))

    text_left = max(0.0, min(float(img_w) - 24.0, text_left))
    text_right = max(text_left + 80.0, min(float(img_w), text_right))
    full_w = max(120.0, float(text_right - text_left))
    LABEL_WIDTH_PX = max(18.0, _LABEL_WIDTH_RATIO * float(full_w))

    def _shift_right_for_label(rect: Dict[str, float], offset_px: float) -> Dict[str, float]:
        width = float(rect["w"])
        shift = min(float(offset_px), max(0.0, width - 48.0))
        return {
            "x": float(rect["x"] + shift),
            "y": float(rect["y"]),
            "w": float(max(48.0, width - shift)),
            "h": float(rect["h"]),
        }

    field_h = max(28.0, min(64.0, 1.02 * sid_step))
    name_h = max(field_h + 12.0, 1.48 * field_h)
    row_tops = [
        float(sid_y + 0.04 * sid_h),
        float(sid_y + 0.26 * sid_h),
        float(sid_y + 0.60 * sid_h),
        float(sid_y + 0.72 * sid_h),
    ]

    max_text_bottom = float(min(float(mcq_y) - 10.0, float(img_h) * 0.60))
    if (row_tops[-1] + field_h) > max_text_bottom:
        shift = (row_tops[-1] + field_h) - max_text_bottom
        row_tops = [float(val - shift) for val in row_tops]

    min_top = float(img_h) * 0.03
    row_tops = [max(min_top, float(val)) for val in row_tops]

    lop_width = max(96.0, 0.46 * full_w)
    mon_width = max(78.0, 0.34 * full_w)
    lop_x = text_left + 0.14 * full_w
    mon_x = text_left + 0.58 * full_w

    defaults = {
        "truong": _shift_right_for_label(
            {"x": text_left, "y": row_tops[0], "w": full_w, "h": field_h},
            LABEL_WIDTH_PX,
        ),
        "ho_ten": _shift_right_for_label(
            {
                "x": text_left,
                "y": row_tops[1] - 0.08 * sid_step,
                "w": full_w,
                "h": name_h,
            },
            LABEL_WIDTH_PX * 0.70,
        ),
        "lop": _shift_right_for_label(
            {"x": lop_x, "y": row_tops[2], "w": lop_width, "h": field_h},
            LABEL_WIDTH_PX * 0.35,
        ),
        "mon": _shift_right_for_label(
            {
                "x": mon_x,
                "y": row_tops[3],
                "w": mon_width,
                "h": field_h,
            },
            LABEL_WIDTH_PX * 0.30,
        ),
    }

    out: Dict[str, Dict[str, int]] = {}
    cfg_rois = cfg_field_rois if isinstance(cfg_field_rois, dict) else {}
    for key in HANDWRITING_FIELDS:
        roi = cfg_rois.get(key)
        if not isinstance(roi, dict):
            roi = defaults[key]

        x = int(round(_safe_float(roi.get("x"), 0.0)))
        y = int(round(_safe_float(roi.get("y"), 0.0)))
        w = int(round(_safe_float(roi.get("w"), 0.0)))
        h = int(round(_safe_float(roi.get("h"), 0.0)))

        x, y, w, h = _clip_rect(x, y, w, h, img_w, img_h)
        out[key] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    return out


def _preprocess_handwriting_crop(crop_bgr):
    if crop_bgr is None or getattr(crop_bgr, "size", 0) <= 0:
        return None, {"ink_ratio": 0.0, "line_suppressed_ratio": 0.0}

    hsv = None
    if len(crop_bgr.shape) == 3:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr.copy()

    h, w = gray.shape[:2]
    if h < 6 or w < 12:
        return gray, {"ink_ratio": 0.0, "line_suppressed_ratio": 0.0}

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    k = max(15, (min(h, w) // 2) | 1)
    bg = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(blur, bg, scale=255)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)

    min_dim = max(3, min(h, w))
    block_size = min(35, min_dim if (min_dim % 2 == 1) else (min_dim - 1))
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1

    binary_inv = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )

    color_ink_ratio = 0.0
    if hsv is not None:
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        blue_purple = cv2.inRange(h_ch, 88, 165)
        sat_mask = cv2.inRange(s_ch, 48, 255)
        dark_enough = cv2.inRange(v_ch, 20, 230)

        color_ink_mask = cv2.bitwise_and(blue_purple, sat_mask)
        color_ink_mask = cv2.bitwise_and(color_ink_mask, dark_enough)
        area = float(max(1, h * w))
        color_ink_ratio = float(cv2.countNonZero(color_ink_mask)) / area

        if color_ink_ratio < 0.0025:
            color_ink_mask = cv2.bitwise_and(sat_mask, dark_enough)
            color_ink_ratio = float(cv2.countNonZero(color_ink_mask)) / area

        if color_ink_ratio >= 0.001:
            color_ink_mask = cv2.morphologyEx(color_ink_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
            color_ink_mask = cv2.dilate(color_ink_mask, np.ones((2, 2), np.uint8), iterations=1)

            low_sat_dark = cv2.bitwise_and(cv2.inRange(s_ch, 0, 55), cv2.inRange(v_ch, 0, 200))
            low_sat_dark = cv2.morphologyEx(low_sat_dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

            binary_inv = cv2.subtract(binary_inv, low_sat_dark)
            binary_inv = cv2.bitwise_or(binary_inv, color_ink_mask)

    line_kernel_w = max(15, int(round(float(w) * 0.11)))
    line_kernel_w = min(line_kernel_w, max(15, w - 1))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_kernel_w, 1))
    dotted_mask = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    dotted_mask = cv2.dilate(dotted_mask, np.ones((1, 3), np.uint8), iterations=1)

    cleaned_inv, removed_line_mask = _remove_horizontal_guide_lines(binary_inv, dotted_mask)
    cleaned_inv = cv2.morphologyEx(cleaned_inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    cleaned_inv = _trim_strong_vertical_borders(cleaned_inv)

    area = float(max(1, cleaned_inv.shape[0] * cleaned_inv.shape[1]))
    ink_ratio = float(cv2.countNonZero(cleaned_inv)) / area
    line_ratio = float(cv2.countNonZero(removed_line_mask)) / area
    largest_ratio = _largest_component_ratio(cleaned_inv)

    ocr_gray = cv2.bitwise_not(cleaned_inv)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    ocr_gray = clahe.apply(ocr_gray)

    if ocr_gray.shape[0] < _OCR_TARGET_HEIGHT:
        scale = float(_OCR_TARGET_HEIGHT) / max(1.0, float(ocr_gray.shape[0]))
        target_w = max(12, int(round(float(ocr_gray.shape[1]) * scale)))
        ocr_gray = cv2.resize(ocr_gray, (target_w, int(_OCR_TARGET_HEIGHT)), interpolation=cv2.INTER_LANCZOS4)

    return ocr_gray, {
        "ink_ratio": round(float(ink_ratio), 6),
        "line_suppressed_ratio": round(float(line_ratio), 6),
        "color_ink_ratio": round(float(color_ink_ratio), 6),
        "largest_component_ratio": round(float(largest_ratio), 6),
    }


def _clean_handwriting_text(raw_text) -> str:
    text = str(raw_text or "").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines).strip()


def _get_handwriting_reader(ocr_engine: str, gpu: bool):
    engine = str(ocr_engine or "vietocr_transformer").strip().lower()
    if engine not in {"vietocr_transformer", "internvl", "openai_gpt4o_mini"}:
        engine = "vietocr_transformer"

    key = (engine, bool(gpu))
    cached = _HANDWRITING_READER_CACHE.get(key)
    if cached is not None:
        return cached

    if engine == "vietocr_transformer":
        from app.services.ocr.reader import OCRReader2

        reader = OCRReader2(gpu=bool(gpu), model_type="handwritten")
    elif engine == "internvl":
        from app.services.ocr.reader import OCRReaderInternVL

        reader = OCRReaderInternVL(gpu=bool(gpu))
    else:
        from app.services.ocr.reader import OCRReaderOpenAI4oMini

        reader = OCRReaderOpenAI4oMini()

    _HANDWRITING_READER_CACHE[key] = reader
    return reader


def _run_handwriting_ocr(img_bgr, field_rois, output_folder: str, run_tag: str, handwriting_cfg):
    cfg = handwriting_cfg if isinstance(handwriting_cfg, dict) else {}

    payload = {
        "enabled": bool(cfg.get("enabled", False)),
        "ocr_engine": str(cfg.get("ocr_engine") or "vietocr_transformer"),
        "gpu": bool(cfg.get("gpu", False)),
        "save_crops": bool(cfg.get("save_crops", True)),
        "field_rois": field_rois if isinstance(field_rois, dict) else {},
        "values": {key: "" for key in HANDWRITING_FIELDS},
        "fields": {},
        "crop_images": {},
        "preprocessed_crop_images": {},
        "warnings": [],
        "warning_codes": [],
    }

    if not payload["enabled"]:
        for key in HANDWRITING_FIELDS:
            payload["fields"][key] = {
                "text": "",
                "status": "disabled",
                "roi": payload["field_rois"].get(key),
                "ink_ratio": 0.0,
                "line_suppressed_ratio": 0.0,
                "confidence": 0.0,
            }
        return payload

    try:
        reader = _get_handwriting_reader(payload["ocr_engine"], payload["gpu"])
    except Exception as exc:
        payload["warnings"].append(f"Khoi tao OCR viet tay that bai: {exc}")
        payload["warning_codes"].append("HANDWRITING_OCR_INIT_FAILED")
        for key in HANDWRITING_FIELDS:
            payload["fields"][key] = {
                "text": "",
                "status": "ocr-init-failed",
                "roi": payload["field_rois"].get(key),
                "ink_ratio": 0.0,
                "line_suppressed_ratio": 0.0,
                "confidence": 0.0,
            }
        return payload

    for key in HANDWRITING_FIELDS:
        roi = payload["field_rois"].get(key)
        field_meta = {
            "text": "",
            "status": "missing-roi",
            "roi": roi,
            "ink_ratio": 0.0,
            "line_suppressed_ratio": 0.0,
            "color_ink_ratio": 0.0,
            "largest_component_ratio": 0.0,
            "confidence": 0.0,
        }

        if not isinstance(roi, dict):
            payload["fields"][key] = field_meta
            continue

        x = int(roi["x"])
        y = int(roi["y"])
        w = int(roi["w"])
        h = int(roi["h"])
        crop = img_bgr[y : y + h, x : x + w]
        if crop is None or crop.size <= 0:
            field_meta["status"] = "empty-roi"
            payload["fields"][key] = field_meta
            continue

        prep_img, prep_meta = _preprocess_handwriting_crop(crop)
        field_meta["ink_ratio"] = float(prep_meta.get("ink_ratio", 0.0))
        field_meta["line_suppressed_ratio"] = float(prep_meta.get("line_suppressed_ratio", 0.0))
        field_meta["color_ink_ratio"] = float(prep_meta.get("color_ink_ratio", 0.0))
        field_meta["largest_component_ratio"] = float(prep_meta.get("largest_component_ratio", 0.0))

        if prep_img is None or getattr(prep_img, "size", 0) <= 0:
            field_meta["status"] = "preprocess-failed"
            payload["fields"][key] = field_meta
            payload["warnings"].append(f"Tien xu ly OCR viet tay that bai cho truong '{key}'.")
            payload["warning_codes"].append("HANDWRITING_PREPROCESS_FAILED")
            continue

        raw_text = ""
        try:
            raw_text = reader.predict(prep_img)
        except Exception as exc:
            field_meta["status"] = "ocr-error"
            payload["fields"][key] = field_meta
            payload["warnings"].append(f"OCR viet tay bi loi tai truong '{key}': {exc}")
            payload["warning_codes"].append("HANDWRITING_OCR_RUNTIME_FAILED")
            continue

        text = _clean_handwriting_text(raw_text)
        field_meta["text"] = text
        confidence = _estimate_text_confidence(
            text,
            field_meta.get("ink_ratio", 0.0),
            field_meta.get("line_suppressed_ratio", 0.0),
            field_meta.get("largest_component_ratio", 0.0),
        )
        field_meta["confidence"] = float(confidence)
        has_digits = bool(_HAS_DIGIT_RE.search(text or ""))
        random_numeric = _is_random_numeric_text(text)
        is_refusal = _looks_like_ocr_refusal(text)
        label_leak = _looks_like_label_leak(text)
        class_like = _looks_like_valid_class(text)
        subject_like = _looks_like_valid_subject(text)
        person_name_like = _looks_like_plausible_person_name(text)
        word_count = len(_normalize_ascii_words(text))

        if is_refusal:
            field_meta["status"] = "ocr-refusal"
            field_meta["confidence"] = 0.0
            payload["warning_codes"].append("HANDWRITING_OCR_REFUSAL")
            payload["warnings"].append(f"OCR tu choi trich xuat truong '{key}'.")
        elif key == "ho_ten" and has_digits:
            field_meta["status"] = "junk-detected"
            payload["warning_codes"].append("HANDWRITING_JUNK_DETECTED")
            payload["warnings"].append("OCR ho_ten co ky tu so, danh dau junk-detected.")
        elif text and label_leak:
            field_meta["status"] = "junk-detected"
            payload["warning_codes"].append("HANDWRITING_LABEL_LEAK")
        elif text and confidence > _OCR_OK_CONFIDENCE and not random_numeric:
            if key in {"truong", "ho_ten"} and word_count < 2:
                field_meta["status"] = "low-confidence"
            elif key == "ho_ten" and not person_name_like:
                field_meta["status"] = "low-confidence"
            elif key == "lop" and not class_like:
                field_meta["status"] = "low-confidence"
            elif key == "mon" and not subject_like:
                field_meta["status"] = "low-confidence"
            else:
                field_meta["status"] = "ok"
        elif text and random_numeric:
            field_meta["status"] = "junk-detected"
            payload["warning_codes"].append("HANDWRITING_JUNK_DETECTED")
        elif float(field_meta["ink_ratio"]) < 0.003:
            field_meta["status"] = "blank"
        elif text:
            field_meta["status"] = "low-confidence"
        else:
            field_meta["status"] = "empty"

        payload["fields"][key] = field_meta
        payload["values"][key] = text

        if payload["save_crops"]:
            crop_name = f"omr_hw_{key}_{run_tag}.jpg"
            prep_name = f"omr_hw_{key}_{run_tag}_prep.jpg"
            try:
                cv2.imwrite(os.path.join(output_folder, crop_name), crop)
                cv2.imwrite(os.path.join(output_folder, prep_name), prep_img)
                payload["crop_images"][key] = crop_name
                payload["preprocessed_crop_images"][key] = prep_name
            except Exception:
                payload["warnings"].append(f"Khong the luu crop OCR viet tay cho truong '{key}'.")
                payload["warning_codes"].append("HANDWRITING_CROP_WRITE_FAILED")

    payload["warnings"] = _dedupe_keep_order(payload["warnings"])
    payload["warning_codes"] = _dedupe_keep_order(payload["warning_codes"])
    return payload
