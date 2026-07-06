from __future__ import annotations

import os
from typing import Dict, List

import cv2
import numpy as np

from .omr_utils import _clip_rect

_HANDWRITING_ROI_KEYS = ["ho_ten"]
_LEGACY_HO_TEN_KEYS = ["ho_ten_1", "ho_ten_2"]


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
        if not text or text in seen:
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


def _merge_roi_rects(rois: List[Dict[str, int]], img_w: int, img_h: int):
    valid = []
    for roi in list(rois or []):
        if not isinstance(roi, dict):
            continue
        parsed = _parse_roi_cfg(roi, img_w, img_h)
        if parsed is not None:
            valid.append(parsed)

    if not valid:
        return None

    x1 = min(int(roi["x"]) for roi in valid)
    y1 = min(int(roi["y"]) for roi in valid)
    x2 = max(int(roi["x"]) + int(roi["w"]) for roi in valid)
    y2 = max(int(roi["y"]) + int(roi["h"]) for roi in valid)
    x, y, w, h = _clip_rect(x1, y1, x2 - x1, y2 - y1, img_w, img_h)
    if w <= 0 or h <= 0:
        return None
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _resolve_ho_ten_roi(field_rois, img_w: int, img_h: int):
    rois = field_rois if isinstance(field_rois, dict) else {}

    ho_ten = _parse_roi_cfg(rois.get("ho_ten"), img_w, img_h)
    if ho_ten is not None:
        return ho_ten

    legacy_rois = [rois.get(key) for key in _LEGACY_HO_TEN_KEYS]
    return _merge_roi_rects(legacy_rois, img_w, img_h)


def _infer_short_form_ho_ten_roi(anchors, sid_roi, mcq_roi, img_w: int, img_h: int):
    del anchors

    if not isinstance(sid_roi, dict):
        return None

    # The configured A4 20/40/50 forms keep the handwritten information panel
    # immediately to the left of the student-id grid, with MCQ starting low.
    if isinstance(mcq_roi, dict):
        mcq_top = _safe_float(mcq_roi.get("y"), 0.0)
        if mcq_top < float(img_h) * 0.42:
            return None

    sid_x = _safe_float(sid_roi.get("x"), 0.0)
    sid_y = _safe_float(sid_roi.get("y"), 0.0)
    sid_w = _safe_float(sid_roi.get("w"), 0.0)
    sid_h = _safe_float(sid_roi.get("h"), 0.0)
    if sid_w <= 0.0 or sid_h <= 0.0:
        return None

    x = int(round(max(float(img_w) * 0.24, sid_x - float(img_w) * 0.285)))
    y = int(round(sid_y + sid_h * 0.18))
    w = int(round(min(float(img_w) * 0.30, sid_x - x - float(img_w) * 0.008)))
    h = int(round(max(float(img_h) * 0.052, sid_h * 0.28)))

    x, y, w, h = _clip_rect(x, y, w, h, img_w, img_h)
    if w < 40 or h < 20:
        return None
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _parse_handwriting_config(profile_handwriting_fields) -> Dict[str, object]:
    cfg = profile_handwriting_fields if isinstance(profile_handwriting_fields, dict) else {}

    parsed_rois: Dict[str, Dict[str, int]] = {}
    raw_rois = cfg.get("field_rois") if isinstance(cfg.get("field_rois"), dict) else {}
    roi = _resolve_ho_ten_roi(raw_rois, 1000, 1400)
    if roi is not None:
        parsed_rois["ho_ten"] = roi

    return {
        "enabled": _safe_bool(cfg.get("enabled"), True),
        "save_crops": _safe_bool(cfg.get("save_crops"), True),
        "field_rois": parsed_rois,
    }


def _build_handwriting_rois(
    anchors,
    sid_roi,
    mcq_roi,
    img_w: int,
    img_h: int,
    cfg_field_rois,
    long_form_mode: bool = False,
) -> Dict[str, Dict[str, int]]:
    cfg_rois = cfg_field_rois if isinstance(cfg_field_rois, dict) else {}
    parsed = _resolve_ho_ten_roi(cfg_rois, img_w, img_h)
    if parsed is None and not bool(long_form_mode):
        parsed = _infer_short_form_ho_ten_roi(anchors, sid_roi, mcq_roi, img_w, img_h)
    if parsed is None:
        return {}
    return {"ho_ten": parsed}


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


def _extract_handwriting_crops(img_bgr, field_rois, output_folder: str, run_tag: str, handwriting_cfg):
    cfg = handwriting_cfg if isinstance(handwriting_cfg, dict) else {}
    source_rois = field_rois if isinstance(field_rois, dict) else {}
    img_h, img_w = img_bgr.shape[:2] if img_bgr is not None and getattr(img_bgr, "size", 0) > 0 else (1400, 1000)
    ho_ten_roi = _resolve_ho_ten_roi(source_rois, int(img_w), int(img_h))

    payload = {
        "enabled": bool(cfg.get("enabled", False)),
        "ocr_engine": "disabled",
        "gpu": False,
        "save_crops": bool(cfg.get("save_crops", True)),
        "field_rois": {"ho_ten": ho_ten_roi} if isinstance(ho_ten_roi, dict) else {},
        "values": {"ho_ten": ""},
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
        payload["fields"]["ho_ten"] = _field_meta("disabled", ho_ten_roi)
        return payload

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception:
        payload["warnings"].append("Khong the tao thu muc luu crop handwriting.")
        payload["warning_codes"].append("HANDWRITING_OUTPUT_DIR_FAILED")

    crop = _crop_from_roi(img_bgr, ho_ten_roi)
    if crop is None:
        status = "missing-roi" if not isinstance(ho_ten_roi, dict) else "empty-roi"
        payload["fields"]["ho_ten"] = _field_meta(status, ho_ten_roi)
    else:
        trimmed_name = _trim_ink_bounding_box(crop)
        payload["fields"]["ho_ten"] = _field_meta("ok", ho_ten_roi)

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
