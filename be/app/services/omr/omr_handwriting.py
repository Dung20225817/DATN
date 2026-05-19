from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .omr_utils import _clip_rect

HANDWRITING_FIELDS = ["truong", "ho_ten", "lop", "mon"]
_HANDWRITING_CROP_FIELDS = ["truong", "ho_ten_1", "ho_ten_2", "lop", "mon"]
_HANDWRITING_ROI_KEYS = ["truong", "ho_ten", "ho_ten_1", "ho_ten_2", "lop", "mon"]


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

    parsed_rois: Dict[str, Dict[str, int]] = {}
    raw_rois = cfg.get("field_rois") if isinstance(cfg.get("field_rois"), dict) else {}
    for key in _HANDWRITING_ROI_KEYS:
        roi = _parse_roi_cfg(raw_rois.get(key), 1000, 1400)
        if roi is not None:
            parsed_rois[key] = roi

    return {
        "enabled": _safe_bool(cfg.get("enabled"), True),
        "save_crops": _safe_bool(cfg.get("save_crops"), True),
        "field_rois": parsed_rois,
    }


def _build_handwriting_rois(anchors, sid_roi, mcq_roi, img_w: int, img_h: int, cfg_field_rois) -> Dict[str, Dict[str, int]]:
    del anchors

    sid_x = float(_safe_float(sid_roi.get("x"), 0.0))
    sid_y = float(_safe_float(sid_roi.get("y"), 0.0))
    sid_h = float(max(1.0, _safe_float(sid_roi.get("h"), 1.0)))

    mcq_x = float(_safe_float(mcq_roi.get("x"), 0.0))
    mcq_w = float(max(1.0, _safe_float(mcq_roi.get("w"), 1.0)))

    text_left = float(mcq_x + 0.06 * mcq_w)
    text_right = float(sid_x - max(8.0, 0.018 * mcq_w))
    if (text_right - text_left) < max(160.0, 0.16 * float(img_w)):
        text_left = float(max(0.18 * float(img_w), sid_x - max(360.0, 0.42 * float(img_w))))
        text_right = float(min(float(img_w) * 0.66, sid_x - 8.0))

    text_left = max(0.0, min(float(img_w) - 24.0, text_left - max(2.0, 0.004 * mcq_w)))
    text_right = max(text_left + 80.0, min(float(img_w), text_right))
    full_w = max(120.0, float(text_right - text_left))

    # Keep offsets conservative to avoid truncating left handwriting strokes.
    base_offset_x = max(8.0, 0.11 * float(full_w))
    offset_multiplier = {
        "truong": 1.10,
        "ho_ten_1": 0.95,
        "ho_ten_2": 0.12,
        "lop": 0.36,
        "mon": 0.36,
    }

    def _apply_hard_x_offset(field_key: str, rect: Dict[str, float]) -> Dict[str, float]:
        width = float(max(1.0, _safe_float(rect.get("w"), 1.0)))
        target_shift = float(base_offset_x) * float(offset_multiplier.get(field_key, 1.0))
        shift = min(float(target_shift), max(0.0, width - 48.0))
        return {
            "x": float(_safe_float(rect.get("x"), 0.0) + shift),
            "y": float(_safe_float(rect.get("y"), 0.0)),
            "w": float(max(48.0, width - shift)),
            "h": float(max(8.0, _safe_float(rect.get("h"), 8.0))),
        }

    hw_step = float(sid_h) / 8.0
    field_h = hw_step * 1.05
    top_lift = max(8.0, min(28.0, 0.58 * hw_step))
    row_tops = [float(sid_y) - top_lift + idx * hw_step for idx in range(8)]

    line2_left_extra = max(8.0, 0.06 * float(full_w))
    lop_mon_extra_h = max(8.0, 0.34 * hw_step)
    right_trim_px = max(8.0, 0.038 * float(full_w))
    left_label_cut_px = {
        "truong": max(30.0, 0.11 * float(full_w)),
        "ho_ten_1": max(30.0, 0.11 * float(full_w)),
        "ho_ten_2": max(8.0, 0.03 * float(full_w)),
        "lop": max(52.0, 0.19 * float(full_w)),
        "mon": max(64.0, 0.23 * float(full_w)),
    }

    defaults = {
        # Row mapping per current template.
        "truong": {"x": text_left, "y": row_tops[0], "w": full_w, "h": field_h},
        "ho_ten_1": {"x": text_left, "y": row_tops[1], "w": full_w, "h": field_h},
        "ho_ten_2": {
            "x": max(0.0, text_left - line2_left_extra),
            "y": row_tops[2],
            "w": min(float(img_w), full_w + line2_left_extra),
            "h": field_h,
        },
        "lop": {"x": text_left, "y": row_tops[4], "w": full_w, "h": field_h},
        "mon": {"x": text_left, "y": row_tops[5], "w": full_w, "h": field_h},
    }

    out: Dict[str, Dict[str, int]] = {}
    cfg_rois = cfg_field_rois if isinstance(cfg_field_rois, dict) else {}
    legacy_name_roi = cfg_rois.get("ho_ten") if isinstance(cfg_rois.get("ho_ten"), dict) else None

    def _split_legacy_name_roi(split_key: str):
        if not isinstance(legacy_name_roi, dict):
            return None
        x0 = float(_safe_float(legacy_name_roi.get("x"), 0.0))
        y0 = float(_safe_float(legacy_name_roi.get("y"), 0.0))
        w0 = float(max(1.0, _safe_float(legacy_name_roi.get("w"), 1.0)))
        h0 = float(max(8.0, _safe_float(legacy_name_roi.get("h"), 8.0)))
        first_h = max(16.0, h0 * 0.5)
        second_h = max(16.0, h0 - first_h)
        if split_key == "ho_ten_1":
            return {"x": x0, "y": y0, "w": w0, "h": first_h}
        return {"x": x0, "y": y0 + first_h, "w": w0, "h": second_h}

    for key in _HANDWRITING_CROP_FIELDS:
        roi = cfg_rois.get(key)
        if not isinstance(roi, dict) and key in {"ho_ten_1", "ho_ten_2"}:
            roi = _split_legacy_name_roi(key)
        if not isinstance(roi, dict):
            roi = defaults[key]

        roi = _apply_hard_x_offset(key, roi)

        if key == "ho_ten_2":
            # Line 2 has no printed label, so restore extra left span.
            restore_left = max(3.0, 0.015 * float(full_w))
            roi = {
                "x": float(_safe_float(roi.get("x"), 0.0) - restore_left),
                "y": float(_safe_float(roi.get("y"), 0.0)),
                "w": float(max(48.0, _safe_float(roi.get("w"), 48.0) + restore_left)),
                "h": float(max(8.0, _safe_float(roi.get("h"), 8.0))),
            }

        if key in {"lop", "mon"}:
            roi = {
                "x": float(_safe_float(roi.get("x"), 0.0)),
                "y": float(_safe_float(roi.get("y"), 0.0)),
                "w": float(max(48.0, _safe_float(roi.get("w"), 48.0))),
                "h": float(max(8.0, _safe_float(roi.get("h"), 8.0) + lop_mon_extra_h)),
            }

        width = float(max(1.0, _safe_float(roi.get("w"), 1.0)))
        left_cut = min(float(left_label_cut_px.get(key, 0.0)), max(0.0, width - 44.0))
        roi = {
            "x": float(_safe_float(roi.get("x"), 0.0) + left_cut),
            "y": float(_safe_float(roi.get("y"), 0.0)),
            "w": float(max(44.0, width - left_cut)),
            "h": float(max(8.0, _safe_float(roi.get("h"), 8.0))),
        }

        # Per request: shift left margin right by 10px for Truong and Ho ten.
        if key in {"truong", "ho_ten_1", "ho_ten_2"}:
            width = float(max(1.0, _safe_float(roi.get("w"), 1.0)))
            left_nudge = min(10.0, max(0.0, width - 44.0))
            roi = {
                "x": float(_safe_float(roi.get("x"), 0.0) + left_nudge),
                "y": float(_safe_float(roi.get("y"), 0.0)),
                "w": float(max(44.0, width - left_nudge)),
                "h": float(max(8.0, _safe_float(roi.get("h"), 8.0))),
            }

        width = float(max(1.0, _safe_float(roi.get("w"), 1.0)))
        trim_px = min(float(right_trim_px), max(0.0, width - 44.0))
        roi = {
            "x": float(_safe_float(roi.get("x"), 0.0)),
            "y": float(_safe_float(roi.get("y"), 0.0)),
            "w": float(max(44.0, width - trim_px)),
            "h": float(max(8.0, _safe_float(roi.get("h"), 8.0))),
        }

        x = int(round(_safe_float(roi.get("x"), 0.0)))
        y = int(round(_safe_float(roi.get("y"), 0.0)))
        w = int(round(_safe_float(roi.get("w"), 0.0)))
        h = int(round(_safe_float(roi.get("h"), 0.0)))

        x, y, w, h = _clip_rect(x, y, w, h, img_w, img_h)
        out[key] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    return out


def _trim_ink_bounding_box(crop_bgr):
    if crop_bgr is None or getattr(crop_bgr, "size", 0) <= 0:
        return crop_bgr

    if len(crop_bgr.shape) == 2:
        work_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    else:
        work_bgr = crop_bgr.copy()

    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_adapt = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    ink_mask = cv2.bitwise_or(mask_otsu, mask_adapt)
    ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    ink_mask = cv2.dilate(ink_mask, np.ones((2, 2), np.uint8), iterations=1)

    points = cv2.findNonZero(ink_mask)
    if points is None:
        h, w = work_bgr.shape[:2]
        if w <= 20:
            return work_bgr
        shrink = max(6, int(round(w * 0.08)))
        x1 = min(shrink, max(0, w - 8))
        x2 = max(x1 + 8, w - shrink)
        x2 = min(w, x2)
        if x2 <= x1:
            return work_bgr
        return work_bgr[:, x1:x2].copy()

    x, y, box_w, box_h = cv2.boundingRect(points)
    pad = max(5, min(10, int(round(0.03 * max(box_w, box_h)))))

    h, w = work_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + box_w + pad)
    y2 = min(h, y + box_h + pad)

    if x2 <= x1 or y2 <= y1:
        return work_bgr
    return work_bgr[y1:y2, x1:x2].copy()


def _merge_roi_rects(rois) -> Dict[str, int] | None:
    points: List[Tuple[float, float, float, float]] = []
    for roi in list(rois or []):
        if not isinstance(roi, dict):
            continue
        x = float(_safe_float(roi.get("x"), -1.0))
        y = float(_safe_float(roi.get("y"), -1.0))
        w = float(_safe_float(roi.get("w"), 0.0))
        h = float(_safe_float(roi.get("h"), 0.0))
        if w <= 0.0 or h <= 0.0:
            continue
        points.append((x, y, x + w, y + h))

    if not points:
        return None

    x1 = min(p[0] for p in points)
    y1 = min(p[1] for p in points)
    x2 = max(p[2] for p in points)
    y2 = max(p[3] for p in points)

    return {
        "x": int(round(x1)),
        "y": int(round(y1)),
        "w": int(round(max(1.0, x2 - x1))),
        "h": int(round(max(1.0, y2 - y1))),
    }


def _build_output_field_rois(field_rois) -> Dict[str, Dict[str, int] | None]:
    rois = field_rois if isinstance(field_rois, dict) else {}
    name_roi = _merge_roi_rects([rois.get("ho_ten_1"), rois.get("ho_ten_2")])
    if name_roi is None and isinstance(rois.get("ho_ten"), dict):
        name_roi = rois.get("ho_ten")

    return {
        "truong": rois.get("truong") if isinstance(rois.get("truong"), dict) else None,
        "ho_ten": name_roi,
        "lop": rois.get("lop") if isinstance(rois.get("lop"), dict) else None,
        "mon": rois.get("mon") if isinstance(rois.get("mon"), dict) else None,
    }


def _crop_from_roi(img_bgr, roi):
    if not isinstance(roi, dict) or img_bgr is None or getattr(img_bgr, "size", 0) <= 0:
        return None

    img_h, img_w = img_bgr.shape[:2]
    x = int(round(_safe_float(roi.get("x"), 0.0)))
    y = int(round(_safe_float(roi.get("y"), 0.0)))
    w = int(round(_safe_float(roi.get("w"), 0.0)))
    h = int(round(_safe_float(roi.get("h"), 0.0)))

    x, y, w, h = _clip_rect(x, y, w, h, int(img_w), int(img_h))
    if w <= 0 or h <= 0:
        return None

    crop = img_bgr[y : y + h, x : x + w]
    if crop is None or crop.size <= 0:
        return None

    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return crop


def _concat_name_lines(crop_1, crop_2):
    valid = [img for img in [crop_1, crop_2] if img is not None and getattr(img, "size", 0) > 0]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    h1, w1 = valid[0].shape[:2]
    h2, w2 = valid[1].shape[:2]
    target_h = max(h1, h2)

    def _pad_bottom(img, target):
        h, w = img.shape[:2]
        if h >= target:
            return img
        pad = target - h
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    c1 = _pad_bottom(valid[0], target_h)
    c2 = _pad_bottom(valid[1], target_h)
    return cv2.hconcat([c1, c2])


def _extract_handwriting_crops(img_bgr, field_rois, output_folder: str, run_tag: str, handwriting_cfg):
    cfg = handwriting_cfg if isinstance(handwriting_cfg, dict) else {}
    source_rois = dict(field_rois) if isinstance(field_rois, dict) else {}
    output_rois = _build_output_field_rois(source_rois)

    payload = {
        "enabled": bool(cfg.get("enabled", False)),
        "ocr_engine": "disabled",
        "gpu": False,
        "save_crops": bool(cfg.get("save_crops", True)),
        "field_rois": source_rois,
        "values": {key: "" for key in HANDWRITING_FIELDS},
        "fields": {},
        "crop_images": {},
        "preprocessed_crop_images": {},
        "warnings": [],
        "warning_codes": [],
    }

    def _field_meta(status: str, roi):
        return {
            "text": "",
            "status": status,
            "roi": roi,
        }

    if not payload["enabled"]:
        for key in HANDWRITING_FIELDS:
            payload["fields"][key] = _field_meta("disabled", output_rois.get(key))
        return payload

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception:
        payload["warnings"].append("Khong the tao thu muc luu crop handwriting.")
        payload["warning_codes"].append("HANDWRITING_OUTPUT_DIR_FAILED")

    for key in ["truong", "lop", "mon"]:
        roi = source_rois.get(key)
        crop = _crop_from_roi(img_bgr, roi)
        if crop is None:
            status = "missing-roi" if not isinstance(roi, dict) else "empty-roi"
            payload["fields"][key] = _field_meta(status, output_rois.get(key))
            continue

        trimmed = _trim_ink_bounding_box(crop)
        payload["fields"][key] = _field_meta("ok", output_rois.get(key))

        if payload["save_crops"]:
            out_name = f"omr_hw_{key}_{run_tag}.jpg"
            out_path = os.path.join(output_folder, out_name)
            try:
                cv2.imwrite(out_path, trimmed)
                payload["crop_images"][key] = out_name
            except Exception:
                payload["warnings"].append(f"Khong the luu crop handwriting cho truong '{key}'.")
                payload["warning_codes"].append("HANDWRITING_CROP_WRITE_FAILED")

    roi_1 = source_rois.get("ho_ten_1")
    roi_2 = source_rois.get("ho_ten_2")
    crop_1 = _crop_from_roi(img_bgr, roi_1)
    crop_2 = _crop_from_roi(img_bgr, roi_2)
    merged_name_crop = _concat_name_lines(crop_1, crop_2)

    if merged_name_crop is None:
        has_any_roi = isinstance(roi_1, dict) or isinstance(roi_2, dict)
        status = "empty-roi" if has_any_roi else "missing-roi"
        payload["fields"]["ho_ten"] = _field_meta(status, output_rois.get("ho_ten"))
    else:
        trimmed_name = _trim_ink_bounding_box(merged_name_crop)
        payload["fields"]["ho_ten"] = _field_meta("ok", output_rois.get("ho_ten"))

        if payload["save_crops"]:
            out_name = f"omr_hw_ho_ten_{run_tag}.jpg"
            out_path = os.path.join(output_folder, out_name)
            try:
                cv2.imwrite(out_path, trimmed_name)
                payload["crop_images"]["ho_ten"] = out_name
            except Exception:
                payload["warnings"].append("Khong the luu crop handwriting cho truong 'ho_ten'.")
                payload["warning_codes"].append("HANDWRITING_CROP_WRITE_FAILED")

    payload["warnings"] = _dedupe_keep_order(payload["warnings"])
    payload["warning_codes"] = _dedupe_keep_order(payload["warning_codes"])
    return payload
