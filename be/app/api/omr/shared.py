# Shared helpers for the OMR API routers.
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
import json
import re
from copy import deepcopy
from typing import Optional, List, Tuple, Dict, Any

class UnicodeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")

JSONResponse = UnicodeJSONResponse
import os
import shutil
from datetime import datetime
import zipfile
from urllib.parse import quote
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

# Import service vừa viết
from app.core.paths import OMR_ANSWER_KEYS_DIR, OMR_DATA_DIR, OMR_DIR, OMR_PROFILE_DIR, OMR_TEMPLATE_DIR
from app.services.omr.answer_keys import (
    extract_answer_key_text,
    normalize_choice_token,
    parse_answer_key_from_text,
)
from app.services.omr.omr_service import process_omr_exam, generate_omr_template, suggest_omr_crop_quad
from app.db.models.omr import OMRTest, OMRAssignment
from app.db.session import get_db

BASE_OMR_DIR = str(OMR_DIR)
BASE_OMR_ANSWER_KEY_DIR = str(OMR_ANSWER_KEYS_DIR)
BASE_OMR_TEMPLATE_DIR = str(OMR_TEMPLATE_DIR)
BASE_OMR_DATA_DIR = str(OMR_DATA_DIR)
BASE_OMR_PROFILE_DIR = str(OMR_PROFILE_DIR)
os.makedirs(BASE_OMR_DIR, exist_ok=True)
os.makedirs(BASE_OMR_ANSWER_KEY_DIR, exist_ok=True)
os.makedirs(BASE_OMR_TEMPLATE_DIR, exist_ok=True)
os.makedirs(BASE_OMR_DATA_DIR, exist_ok=True)
os.makedirs(BASE_OMR_PROFILE_DIR, exist_ok=True)

INFO_FIELD_WHITELIST = ["Tên", "Lớp", "Môn thi", "Lớp thi", "Năm học"]



def _omr_sidecar_path(omrid: int) -> str:
    return os.path.join(BASE_OMR_TEMPLATE_DIR, f"omr_test_{int(omrid)}.json")

def _write_omr_sidecar(omrid: int, payload: dict) -> None:
    path = _omr_sidecar_path(omrid)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _read_omr_sidecar(omrid: int) -> dict:
    path = _omr_sidecar_path(omrid)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _delete_omr_sidecar_and_template(omrid: int) -> None:
    data = _read_omr_sidecar(omrid)
    image_name = str(data.get("template_image") or "").strip()
    if image_name:
        image_path = os.path.join(BASE_OMR_TEMPLATE_DIR, os.path.basename(image_name))
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass
    sidecar = _omr_sidecar_path(omrid)
    if os.path.exists(sidecar):
        try:
            os.remove(sidecar)
        except Exception:
            pass

def _normalize_exam_name_key(name: str) -> str:
    value = re.sub(r"\s+", " ", str(name or "").strip())
    return value.lower()

def _check_duplicate_template_identity(
    db: Session,
    uid: int,
    omr_name: str,
    omr_code: str,
    student_id_digits: int,
) -> None:
    """Reject duplicate (exam name, exam code, SID digits) to avoid answer-key ambiguity."""
    recs = (
        db.query(OMRTest)
        .filter(OMRTest.uuid == int(uid), OMRTest.omr_code == str(omr_code))
        .all()
    )
    key_name = _normalize_exam_name_key(omr_name)
    target_sid = int(student_id_digits)

    for rec in recs:
        if _normalize_exam_name_key(rec.omr_name) != key_name:
            continue
        sidecar = _read_omr_sidecar(rec.omrid)
        stored_sid = sidecar.get("student_id_digits")
        if stored_sid is None:
            raise HTTPException(
                status_code=400,
                detail="Đề trùng tên + mã đề đã tồn tại (không đủ metadata SID). Vui lòng đổi Tên bài thi hoặc Mã đề.",
            )
        try:
            if int(stored_sid) == target_sid:
                raise HTTPException(
                    status_code=400,
                    detail="Đề đã tồn tại với cùng Tên bài thi, Mã đề và Số chữ số SID.",
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Đề trùng tên + mã đề đã tồn tại (SID metadata lỗi). Vui lòng đổi thông tin đề.",
            )

def _parse_info_fields(raw: str) -> List[str]:
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
    except Exception:
        return []

    cleaned = []
    for item in parsed:
        value = str(item).strip()
        if value in INFO_FIELD_WHITELIST and value not in cleaned:
            cleaned.append(value)
    return cleaned

def _normalize_omr_code(code: str) -> str:
    code = str(code or "").strip()
    if not re.fullmatch(r"\d{3}", code):
        raise HTTPException(status_code=400, detail="Mã đề phải gồm đúng 3 chữ số (0-9)")
    return code

def _validate_uid(uid: Optional[int]) -> int:
    if uid is None or int(uid) <= 0:
        raise HTTPException(status_code=400, detail="Thiếu uid hợp lệ")
    return int(uid)

def _static_omr_url(file_name: Optional[str]) -> Optional[str]:
    safe_name = os.path.basename(str(file_name or "").strip())
    if not safe_name:
        return None
    return f"/static/omr/{quote(safe_name)}"

def _inject_handwriting_crop_urls(result: Dict[str, Any]) -> None:
    if not isinstance(result, dict):
        return

    handwriting = result.get("handwriting")
    if not isinstance(handwriting, dict):
        return

    crop_images = handwriting.get("crop_images")
    if not isinstance(crop_images, dict) or not crop_images:
        crop_images = handwriting.get("preprocessed_crop_images")
    if not isinstance(crop_images, dict) or not crop_images:
        return

    crop_images_url: Dict[str, str] = {}
    for key, value in crop_images.items():
        url = _static_omr_url(str(value) if value is not None else None)
        if url:
            crop_images_url[str(key)] = url

    if crop_images_url:
        handwriting["crop_images_url"] = crop_images_url

    ho_ten_url = crop_images_url.get("ho_ten")
    if ho_ten_url:
        result["ho_ten_crop_url"] = ho_ten_url

def _normalize_assignment_last_result(last_result: Any) -> Any:
    if not isinstance(last_result, dict):
        return last_result

    _inject_handwriting_crop_urls(last_result)

    grade_records = last_result.get("grade_records")
    if not isinstance(grade_records, list):
        return last_result

    for item in grade_records:
        if not isinstance(item, dict):
            continue

        data = item.get("data")
        if isinstance(data, dict):
            _inject_handwriting_crop_urls(data)
            ho_ten_url = data.get("ho_ten_crop_url")
            if isinstance(ho_ten_url, str) and ho_ten_url.strip():
                item["ho_ten_crop_url"] = ho_ten_url

    return last_result

def _safe_profile_code(raw: str) -> str:
    code = re.sub(r"[^a-z0-9_-]+", "-", str(raw or "").strip().lower())
    code = re.sub(r"-{2,}", "-", code).strip("-")
    return code[:80]

def _profile_path(code: str) -> str:
    safe = _safe_profile_code(code)
    return os.path.join(BASE_OMR_PROFILE_DIR, f"{safe}.json")

def _is_omr_sample_file(name: str) -> bool:
    lower = str(name or "").lower()
    if lower.endswith(".json") or lower.endswith(".html"):
        return False
    return lower.endswith((".pdf", ".png", ".jpg", ".jpeg", ".webp"))

def _list_omr_sample_files() -> List[str]:
    items: List[str] = []
    for name in os.listdir(BASE_OMR_DATA_DIR):
        abs_path = os.path.join(BASE_OMR_DATA_DIR, name)
        if os.path.isfile(abs_path) and _is_omr_sample_file(name):
            items.append(name)
    items.sort(key=lambda x: x.lower())
    return items

def _guess_questions_from_name(name: str) -> int:
    nums = re.findall(r"(\d{1,3})", str(name or ""))
    if not nums:
        return 40
    for token in nums:
        val = int(token)
        if 10 <= val <= 200:
            return val
    return max(1, int(nums[0]))

def _sanitize_bool_flag(raw: Any) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw))

    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None

def _sanitize_norm_rect(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x", 0))
        y = float(raw.get("y", 0))
        w = float(raw.get("w", 0))
        h = float(raw.get("h", 0))
    except Exception:
        return None

    x = max(0.0, min(0.98, x))
    y = max(0.0, min(0.98, y))
    w = max(0.01, min(1.0 - x, w))
    h = max(0.01, min(1.0 - y, h))
    return {"x": round(x, 6), "y": round(y, 6), "w": round(w, 6), "h": round(h, 6)}

def _merge_norm_rects(rects: list) -> Optional[dict]:
    valid = [rect for rect in rects if isinstance(rect, dict)]
    if not valid:
        return None

    x1 = min(float(rect["x"]) for rect in valid)
    y1 = min(float(rect["y"]) for rect in valid)
    x2 = max(float(rect["x"]) + float(rect["w"]) for rect in valid)
    y2 = max(float(rect["y"]) + float(rect["h"]) for rect in valid)
    return _sanitize_norm_rect({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})

def _sanitize_ho_ten_roi(field_rois_raw: Any) -> Optional[dict]:
    if not isinstance(field_rois_raw, dict):
        return None

    direct = _sanitize_norm_rect(field_rois_raw.get("ho_ten"))
    if direct is not None:
        return direct

    legacy_rects = [
        _sanitize_norm_rect(field_rois_raw.get("ho_ten_1")),
        _sanitize_norm_rect(field_rois_raw.get("ho_ten_2")),
    ]
    return _merge_norm_rects([rect for rect in legacy_rects if rect is not None])

def _sanitize_sid_row_offsets(raw: Any) -> Optional[list]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None

    out = []
    for item in list(raw)[:20]:
        try:
            val = int(item)
        except Exception:
            val = 0
        out.append(max(-4, min(4, val)))

    return out if out else None

def _sanitize_mcq_row_offsets(
    raw: Any,
    *,
    max_items: int = 240,
    min_shift: int = -80,
    max_shift: int = 80,
) -> Optional[list]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None

    out = []
    for item in list(raw)[: int(max_items)]:
        try:
            val = int(item)
        except Exception:
            val = 0
        out.append(max(int(min_shift), min(int(max_shift), val)))

    return out if out else None

def _sanitize_mcq_decode(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    out: Dict[str, Any] = {}

    mode = str(raw.get("threshold_mode") or "").strip().lower()
    if mode in {"otsu", "adaptive"}:
        out["threshold_mode"] = mode

    float_fields = {
        "bubble_left_ratio": (0.05, 0.60),
        "bubble_right_ratio": (0.40, 0.99),
        "inner_ratio": (0.35, 1.00),
        "header_trim_ratio": (0.00, 0.40),
        "min_mark_density": (0.50, 0.60),
        "double_mark_gap": (0.01, 0.30),
        "min_conf_ratio": (1.00, 2.50),
        "min_peak_factor": (1.00, 3.00),
        "blank_floor": (0.00, 500.00),
        "blank_std_factor": (0.00, 2.00),
        "min_peak_strength": (1.00, 3.00),
        "adaptive_c": (-30.00, 30.00),
    }
    for key, (low, high) in float_fields.items():
        if key not in raw:
            continue
        try:
            val = float(raw.get(key))
        except Exception:
            continue
        out[key] = round(max(low, min(high, val)), 6)

    int_fields = {
        "adaptive_block_size": (15, 71),
        "adaptive_open_kernel": (0, 5),
    }
    for key, (low, high) in int_fields.items():
        if key not in raw:
            continue
        try:
            val = int(raw.get(key))
        except Exception:
            continue
        if key == "adaptive_block_size" and val % 2 == 0:
            val += 1
        out[key] = max(low, min(high, val))

    for key in ["refine_vertical", "refine_horizontal"]:
        if key not in raw:
            continue
        parsed = _sanitize_bool_flag(raw.get(key))
        if parsed is not None:
            out[key] = bool(parsed)

    row_offsets = _sanitize_mcq_row_offsets(raw.get("row_offsets_px"), max_items=240, min_shift=-80, max_shift=80)
    if row_offsets is not None:
        out["row_offsets_px"] = row_offsets

    return out or None

def _sanitize_threshold_mode(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    mode = str(raw).strip().lower()
    if mode in {"otsu", "weighted_adaptive", "hybrid"}:
        return mode
    return None

def _sanitize_quad(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    out = {}
    for key in ["tl", "tr", "br", "bl"]:
        point = raw.get(key)
        if not isinstance(point, dict):
            return None
        try:
            px = float(point.get("x", 0.0))
            py = float(point.get("y", 0.0))
        except Exception:
            return None
        out[key] = {
            "x": round(max(0.0, min(1.0, px)), 6),
            "y": round(max(0.0, min(1.0, py)), 6),
        }
    return out

def _sanitize_corner_markers(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    out = {}
    for key in ["tl", "tr", "br", "bl"]:
        box = raw.get(key)
        if not isinstance(box, dict):
            return None
        norm = _sanitize_norm_rect(box)
        if norm is None:
            return None
        try:
            cx = float(box.get("cx", norm["x"] + (norm["w"] / 2)))
            cy = float(box.get("cy", norm["y"] + (norm["h"] / 2)))
        except Exception:
            return None
        norm["cx"] = round(max(0.0, min(1.0, cx)), 6)
        norm["cy"] = round(max(0.0, min(1.0, cy)), 6)
        out[key] = norm

    return out

def _sanitize_scanner_hint(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    out = {}
    if "min_dark_ratio" in raw:
        try:
            out["min_dark_ratio"] = round(max(0.01, min(0.8, float(raw.get("min_dark_ratio")))), 6)
        except Exception:
            pass
    if "min_center_luma" in raw:
        try:
            out["min_center_luma"] = round(max(1.0, min(255.0, float(raw.get("min_center_luma")))), 6)
        except Exception:
            pass
    if "min_marker_contrast" in raw:
        try:
            out["min_marker_contrast"] = round(max(1.0, min(255.0, float(raw.get("min_marker_contrast")))), 6)
        except Exception:
            pass
    if "sample_size_norm" in raw:
        try:
            out["sample_size_norm"] = round(max(0.005, min(0.4, float(raw.get("sample_size_norm")))), 6)
        except Exception:
            pass

    return out or None

def _sanitize_page_size_pt(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    try:
        width = float(raw.get("width", 0.0))
        height = float(raw.get("height", 0.0))
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return {"width": round(width, 3), "height": round(height, 3)}

def _sanitize_handwriting_fields(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    out: Dict[str, Any] = {}

    enabled = _sanitize_bool_flag(raw.get("enabled"))
    if enabled is not None:
        out["enabled"] = bool(enabled)

    save_crops = _sanitize_bool_flag(raw.get("save_crops"))
    if save_crops is not None:
        out["save_crops"] = bool(save_crops)

    field_rois_raw = raw.get("field_rois") if isinstance(raw.get("field_rois"), dict) else {}
    field_rois: Dict[str, dict] = {}
    rect = _sanitize_ho_ten_roi(field_rois_raw)
    if rect is not None:
        field_rois["ho_ten"] = rect
    if field_rois:
        out["field_rois"] = field_rois

    return out or None

def _default_profile(sample_file: str) -> dict:
    code = _safe_profile_code(os.path.splitext(sample_file)[0])
    return {
        "code": code,
        "title": os.path.splitext(sample_file)[0],
        "sample_file": sample_file,
        "default_questions": _guess_questions_from_name(sample_file),
        "total_points": 10,
        "num_choices": 4,
        "rows_per_block": 20,
        "num_blocks": None,
        "student_id_digits": 6,
        "sid_has_write_row": True,
        "strategy": {
            "crop_quad": None,
            "sid_roi": None,
            "mcq_roi": None,
            "exam_code_roi": None,
            "handwriting_fields": None,
            "corner_markers": None,
            "scanner_hint": None,
            "page_size_pt": None,
            "threshold_mode": None,
        },
    }

def _read_profile_by_code(code: str) -> Optional[dict]:
    if not code:
        return None
    path = _profile_path(code)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _find_sample_name_by_code(code: str) -> Optional[str]:
    target = _safe_profile_code(code)
    for name in _list_omr_sample_files():
        if _safe_profile_code(os.path.splitext(name)[0]) == target:
            return name
    return None

def _resolve_profile(code: Optional[str]) -> Optional[dict]:
    code = _safe_profile_code(code or "")
    if not code:
        return None
    saved = _read_profile_by_code(code)
    if saved:
        return saved
    sample = _find_sample_name_by_code(code)
    if sample:
        return _default_profile(sample)
    return None

def _extract_form_profile_code(last_result: Any) -> Optional[str]:
    if not isinstance(last_result, dict):
        return None
    meta = last_result.get("__meta__")
    if not isinstance(meta, dict):
        return None
    code = _safe_profile_code(str(meta.get("form_profile_code") or ""))
    return code or None

def _with_assignment_meta(base_result: Any, form_profile_code: Optional[str]) -> Optional[dict]:
    payload = dict(base_result) if isinstance(base_result, dict) else {}
    meta = payload.get("__meta__")
    if not isinstance(meta, dict):
        meta = {}
    code = _safe_profile_code(form_profile_code or "")
    if code:
        meta["form_profile_code"] = code
        payload["__meta__"] = meta
    elif "__meta__" in payload:
        meta.pop("form_profile_code", None)
        if meta:
            payload["__meta__"] = meta
        else:
            payload.pop("__meta__", None)
    return payload or None

def _build_runtime_config(
    profile: Optional[dict],
    *,
    num_questions: int,
    num_choices: int,
    rows_per_block: int,
    num_blocks: Optional[int],
    student_id_digits: int,
    sid_has_write_row: bool,
    crop_x: Optional[float] = None,
    crop_y: Optional[float] = None,
    crop_w: Optional[float] = None,
    crop_h: Optional[float] = None,
    crop_tl_x: Optional[float] = None,
    crop_tl_y: Optional[float] = None,
    crop_tr_x: Optional[float] = None,
    crop_tr_y: Optional[float] = None,
    crop_br_x: Optional[float] = None,
    crop_br_y: Optional[float] = None,
    crop_bl_x: Optional[float] = None,
    crop_bl_y: Optional[float] = None,
) -> dict:
    p = profile or {}
    strategy = p.get("strategy") if isinstance(p.get("strategy"), dict) else {}

    manual_quad_requested = all(
        v is not None
        for v in [crop_tl_x, crop_tl_y, crop_tr_x, crop_tr_y, crop_br_x, crop_br_y, crop_bl_x, crop_bl_y]
    )
    if (not manual_quad_requested) and isinstance(strategy.get("crop_quad"), dict):
        quad = strategy.get("crop_quad")
        try:
            crop_tl_x = float(quad.get("tl", {}).get("x"))
            crop_tl_y = float(quad.get("tl", {}).get("y"))
            crop_tr_x = float(quad.get("tr", {}).get("x"))
            crop_tr_y = float(quad.get("tr", {}).get("y"))
            crop_br_x = float(quad.get("br", {}).get("x"))
            crop_br_y = float(quad.get("br", {}).get("y"))
            crop_bl_x = float(quad.get("bl", {}).get("x"))
            crop_bl_y = float(quad.get("bl", {}).get("y"))
        except Exception:
            crop_tl_x = crop_tl_y = crop_tr_x = crop_tr_y = crop_br_x = crop_br_y = crop_bl_x = crop_bl_y = None

    parsed_disable_rescue = _sanitize_bool_flag(strategy.get("disable_mcq_rescue"))

    return {
        "num_questions": max(1, int(p.get("default_questions") or num_questions)),
        "num_choices": max(2, int(p.get("num_choices") or num_choices)),
        "rows_per_block": max(1, int(p.get("rows_per_block") or rows_per_block)),
        "num_blocks": int(p.get("num_blocks")) if p.get("num_blocks") not in (None, "", "null") else num_blocks,
        "student_id_digits": max(1, int(p.get("student_id_digits") or student_id_digits)),
        "sid_has_write_row": bool(p.get("sid_has_write_row") if "sid_has_write_row" in p else sid_has_write_row),
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "crop_tl_x": crop_tl_x,
        "crop_tl_y": crop_tl_y,
        "crop_tr_x": crop_tr_x,
        "crop_tr_y": crop_tr_y,
        "crop_br_x": crop_br_x,
        "crop_br_y": crop_br_y,
        "crop_bl_x": crop_bl_x,
        "crop_bl_y": crop_bl_y,
        "profile_sid_roi": _sanitize_norm_rect(strategy.get("sid_roi")),
        "profile_mcq_roi": _sanitize_norm_rect(strategy.get("mcq_roi")),
        "profile_exam_code_roi": _sanitize_norm_rect(strategy.get("exam_code_roi")),
        "profile_sid_roi_lock": bool(_sanitize_bool_flag(strategy.get("sid_roi_lock"))),
        "profile_sid_row_offsets": _sanitize_sid_row_offsets(strategy.get("sid_row_offsets")),
        "profile_disable_mcq_rescue": bool(parsed_disable_rescue) if parsed_disable_rescue is not None else False,
        "profile_mcq_decode": _sanitize_mcq_decode(strategy.get("mcq_decode")),
        "profile_threshold_mode": _sanitize_threshold_mode(strategy.get("threshold_mode")),
        "profile_corner_markers": _sanitize_corner_markers(strategy.get("corner_markers")),
        "profile_scanner_hint": _sanitize_scanner_hint(strategy.get("scanner_hint")),
        "profile_page_size_pt": _sanitize_page_size_pt(strategy.get("page_size_pt")),
        "profile_handwriting_fields": _sanitize_handwriting_fields(strategy.get("handwriting_fields")),
    }

def _resolve_answer_key_from_omr_test(db: Session, uid: int, omr_test_id: int) -> Tuple[List[int], str, int]:
    record = (
        db.query(OMRTest)
        .filter(OMRTest.omrid == int(omr_test_id), OMRTest.uuid == int(uid))
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề OMR đã lưu")

    stored = record.omr_answer
    if not isinstance(stored, list) or not stored:
        raise HTTPException(status_code=400, detail="Đề OMR đã lưu không có đáp án hợp lệ")

    try:
        answer_key = [int(x) for x in stored]
    except Exception:
        raise HTTPException(status_code=400, detail="Định dạng đáp án trong DB không hợp lệ")

    return answer_key, f"omr_test:{record.omrid}", len(answer_key)

def _normalize_title_for_match(text: Optional[str]) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"[^0-9a-zà-ỹ\s_-]", "", raw)
    return raw

def _auto_match_omr_test(db: Session, uid: int, exam_code: Optional[str], exam_title: Optional[str]) -> Optional[OMRTest]:
    query = db.query(OMRTest).filter(OMRTest.uuid == int(uid))

    code = str(exam_code or "").strip()
    title_norm = _normalize_title_for_match(exam_title)

    code_candidates = []
    if re.fullmatch(r"\d{3}", code):
        code_candidates = query.filter(OMRTest.omr_code == code).order_by(OMRTest.omrid.desc()).all()
        if len(code_candidates) == 1:
            return code_candidates[0]

    if code_candidates and title_norm:
        for rec in code_candidates:
            rec_title = _normalize_title_for_match(rec.omr_name)
            if rec_title and (title_norm in rec_title or rec_title in title_norm):
                return rec
        return code_candidates[0]

    if code_candidates:
        return code_candidates[0]

    if title_norm:
        title_candidates = query.order_by(OMRTest.omrid.desc()).all()
        for rec in title_candidates:
            rec_title = _normalize_title_for_match(rec.omr_name)
            if rec_title and (title_norm in rec_title or rec_title in title_norm):
                return rec

    return None

async def _resolve_answer_key_auto_from_sheet(
    db: Session,
    uid: int,
    file_location: str,
    num_questions: int,
    num_choices: int,
    rows_per_block: int,
    num_blocks: Optional[int],
    student_id_digits: int,
    sid_has_write_row: bool,
    crop_x: Optional[float] = None,
    crop_y: Optional[float] = None,
    crop_w: Optional[float] = None,
    crop_h: Optional[float] = None,
    crop_tl_x: Optional[float] = None,
    crop_tl_y: Optional[float] = None,
    crop_tr_x: Optional[float] = None,
    crop_tr_y: Optional[float] = None,
    crop_br_x: Optional[float] = None,
    crop_br_y: Optional[float] = None,
    crop_bl_x: Optional[float] = None,
    crop_bl_y: Optional[float] = None,
):
    detect_probe = await run_in_threadpool(
        process_omr_exam,
        image_path=file_location,
        output_folder=BASE_OMR_DIR,
        answer_key=[0] * max(1, int(num_questions)),
        questions=max(1, int(num_questions)),
        choices=num_choices,
        rows_per_block=rows_per_block,
        num_blocks=num_blocks,
        student_id_digits=student_id_digits,
        sid_has_write_row=sid_has_write_row,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
        crop_tl_x=crop_tl_x,
        crop_tl_y=crop_tl_y,
        crop_tr_x=crop_tr_x,
        crop_tr_y=crop_tr_y,
        crop_br_x=crop_br_x,
        crop_br_y=crop_br_y,
        crop_bl_x=crop_bl_x,
        crop_bl_y=crop_bl_y,
    )
    if "error" in detect_probe:
        raise HTTPException(status_code=400, detail=detect_probe.get("error", "Không thể nhận diện đề từ ảnh"))

    exam_code = detect_probe.get("exam_code")
    exam_title = detect_probe.get("exam_title_detected")
    matched = _auto_match_omr_test(db=db, uid=uid, exam_code=exam_code, exam_title=exam_title)
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy đề phù hợp trong DB (code={exam_code or '-'}, title={exam_title or '-'})",
        )

    answer_key, answer_source, parsed_questions = _resolve_answer_key_from_omr_test(
        db=db,
        uid=uid,
        omr_test_id=matched.omrid,
    )
    return answer_key, answer_source, parsed_questions, matched, detect_probe

def _serialize_assignment(record: OMRAssignment) -> dict:
    form_profile_code = _extract_form_profile_code(record.last_result)
    last_result_payload = deepcopy(record.last_result) if isinstance(record.last_result, dict) else record.last_result
    last_result_payload = _normalize_assignment_last_result(last_result_payload)
    return {
        "aid": record.aid,
        "title": record.title,
        "created_at_raw": record.created_at_raw,
        "created_at_label": record.created_at_label,
        "question_count": int(record.question_count or 0),
        "total_points": int(record.total_points or 0),
        "graded_count": int(record.graded_count or 0),
        "answer_sets": record.answer_sets if isinstance(record.answer_sets, list) else [],
        "active_code": record.active_code,
        "last_result": last_result_payload,
        "form_profile_code": form_profile_code,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }

def _resolve_shared_answer_key(
    answers: str,
    answer_key_file: Optional[UploadFile],
    num_choices: int,
    num_questions: int,
):
    """Resolve answer key from file or text input and return (answer_key, source, parsed_num_questions)."""
    answer_key = None
    answer_source = "manual"

    if answer_key_file is not None and answer_key_file.filename:
        lower_name = answer_key_file.filename.lower()
        if not (
            lower_name.endswith(".doc")
            or lower_name.endswith(".docx")
            or lower_name.endswith(".pdf")
            or lower_name.endswith(".txt")
        ):
            raise HTTPException(status_code=400, detail="File đáp án chỉ hỗ trợ .doc/.docx/.pdf/.txt")

        safe_answer_name = os.path.basename(answer_key_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        answer_file_name = f"answer_key_{timestamp}_{safe_answer_name}"
        answer_file_path = os.path.join(BASE_OMR_ANSWER_KEY_DIR, answer_file_name)
        with open(answer_file_path, "wb") as buffer:
            shutil.copyfileobj(answer_key_file.file, buffer)

        try:
            answer_text = extract_answer_key_text(answer_file_path)
            # File dap an: cho phep auto detect 1-5 hoac 0-4.
            answer_key = parse_answer_key_from_text(answer_text, num_choices, assume_one_based=None)
            answer_source = "file"
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
    elif answers.strip():
        try:
            # Nhap tay: su dung chuan one-based theo UI (A=1..E=5), backend quy doi ve zero-based.
            answer_key = parse_answer_key_from_text(answers, num_choices, assume_one_based=True)
        except ValueError:
            raise HTTPException(status_code=400, detail="Format đáp án không hợp lệ. Hỗ trợ A-E hoặc 1-5 (ngăn cách bởi dấu phẩy).")
    else:
        raise HTTPException(status_code=400, detail="Cần nhập chuỗi đáp án hoặc upload file đáp án")

    if not answer_key:
        raise HTTPException(status_code=400, detail="Không tìm thấy đáp án hợp lệ")

    parsed_questions = int(num_questions)
    if answer_source == "file":
        parsed_questions = len(answer_key)
    elif len(answer_key) != int(num_questions):
        raise HTTPException(
            status_code=400,
            detail=f"Số lượng đáp án ({len(answer_key)}) không khớp với số câu hỏi ({num_questions}).",
        )

    return answer_key, answer_source, parsed_questions

def _parse_assignment_answer_token(token: Any, num_choices: int) -> int:
    text = str(token or "").strip()
    if not text:
        return -1

    idx = normalize_choice_token(text, num_choices, assume_one_based=True)
    if idx is None:
        idx = normalize_choice_token(text, num_choices, assume_one_based=False)
    if idx is None:
        return -1
    return int(idx)

def _build_assignment_answer_key_map(record: OMRAssignment, num_choices: int) -> Tuple[Dict[str, List[int]], Optional[str]]:
    answer_sets = record.answer_sets if isinstance(record.answer_sets, list) else []
    out: Dict[str, List[int]] = {}

    for item in answer_sets:
        if not isinstance(item, dict):
            continue

        code = str(item.get("code") or "").strip()
        answers_raw = item.get("answers")
        if not code or not isinstance(answers_raw, list):
            continue

        parsed = [_parse_assignment_answer_token(token, num_choices) for token in answers_raw]
        while parsed and parsed[-1] < 0:
            parsed.pop()
        if parsed:
            out[code] = parsed

    active_code = str(record.active_code or "").strip() or None
    return out, active_code

# Transitional export surface while endpoint modules are separated from the shared helpers.
__all__ = [name for name in globals() if not name.startswith("__")]

