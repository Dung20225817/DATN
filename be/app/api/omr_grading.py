# app/api/omr_grading.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Body
from fastapi.responses import JSONResponse
import json
import re
from typing import Optional, List, Tuple, Dict, Any

from docx import Document
import pdfplumber

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
from app.services.omr.omr_service import process_omr_exam, generate_omr_template, suggest_omr_crop_quad
from app.db_connect import get_db
from app.db.ocr_tables import OMRTest, OMRAssignment

router = APIRouter()

BASE_OMR_DIR = "uploads/omr"
BASE_OMR_ANSWER_KEY_DIR = "uploads/answer_keys/omr"
BASE_OMR_TEMPLATE_DIR = "uploads/omr_templates"
BASE_OMR_DATA_DIR = "uploads/omr_data"
BASE_OMR_PROFILE_DIR = os.path.join(BASE_OMR_DATA_DIR, "profiles")
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


def _sanitize_ai_uncertainty(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    out: Dict[str, Any] = {}
    enabled = _sanitize_bool_flag(raw.get("enabled"))
    if enabled is not None:
        out["enabled"] = bool(enabled)

    model_path = str(raw.get("model_path") or "").strip()
    if model_path:
        out["model_path"] = model_path

    device = str(raw.get("device") or "").strip()
    if device:
        out["device"] = device

    float_fields = {
        "probe_conf_ratio": (1.00, 3.00),
        "marked_conf_threshold": (0.30, 0.99),
        "empty_conf_threshold": (0.30, 0.99),
    }
    for key, (low, high) in float_fields.items():
        if key not in raw:
            continue
        try:
            val = float(raw.get(key))
        except Exception:
            continue
        out[key] = round(max(low, min(high, val)), 6)

    return out or None


def _sanitize_ai_sid_htr(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    out: Dict[str, Any] = {}
    enabled = _sanitize_bool_flag(raw.get("enabled"))
    if enabled is not None:
        out["enabled"] = bool(enabled)

    model_path = str(raw.get("model_path") or "").strip()
    if model_path:
        out["model_path"] = model_path

    device = str(raw.get("device") or "").strip()
    if device:
        out["device"] = device

    if "min_confidence" in raw:
        try:
            out["min_confidence"] = round(max(0.30, min(0.99, float(raw.get("min_confidence")))), 6)
        except Exception:
            pass

    return out or None


def _sanitize_agentic_rescue(raw: Any) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    out: Dict[str, Any] = {}
    enabled = _sanitize_bool_flag(raw.get("enabled"))
    if enabled is not None:
        out["enabled"] = bool(enabled)

    if "sid_conf_threshold" in raw:
        try:
            out["sid_conf_threshold"] = round(max(0.5, min(2.0, float(raw.get("sid_conf_threshold")))), 6)
        except Exception:
            pass

    return out or None


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

    ocr_engine = str(raw.get("ocr_engine") or "").strip().lower()
    if ocr_engine in {"vietocr_transformer", "internvl", "openai_gpt4o_mini"}:
        out["ocr_engine"] = ocr_engine

    gpu = _sanitize_bool_flag(raw.get("gpu"))
    if gpu is not None:
        out["gpu"] = bool(gpu)

    save_crops = _sanitize_bool_flag(raw.get("save_crops"))
    if save_crops is not None:
        out["save_crops"] = bool(save_crops)

    field_rois_raw = raw.get("field_rois") if isinstance(raw.get("field_rois"), dict) else {}
    field_rois: Dict[str, dict] = {}
    for key in ["truong", "ho_ten", "lop", "mon"]:
        rect = _sanitize_norm_rect(field_rois_raw.get(key))
        if rect is not None:
            field_rois[key] = rect
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
            "ai_uncertainty": None,
            "ai_sid_htr": None,
            "agentic_rescue": None,
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
        "profile_sid_row_offsets": _sanitize_sid_row_offsets(strategy.get("sid_row_offsets")),
        "profile_disable_mcq_rescue": bool(parsed_disable_rescue) if parsed_disable_rescue is not None else False,
        "profile_mcq_decode": _sanitize_mcq_decode(strategy.get("mcq_decode")),
        "profile_threshold_mode": _sanitize_threshold_mode(strategy.get("threshold_mode")),
        "profile_ai_uncertainty": _sanitize_ai_uncertainty(strategy.get("ai_uncertainty")),
        "profile_ai_sid_htr": _sanitize_ai_sid_htr(strategy.get("ai_sid_htr")),
        "profile_agentic_rescue": _sanitize_agentic_rescue(strategy.get("agentic_rescue")),
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


@router.post("/template")
async def create_omr_template(
    uid: int = Form(..., description="ID người dùng"),
    omr_name: Optional[str] = Form(default=None, description="Tên phiếu OMR (mặc định = tên bài thi)"),
    exam_title: str = Form(..., description="Tên bài thi"),
    omr_code: str = Form(..., description="Mã đề gồm 3 chữ số"),
    info_fields: str = Form(default="[]", description="JSON list các trường thông tin"),
    answers: str = Form(default="", description="Chuỗi đáp án cách nhau bởi dấu phẩy, ví dụ 1,2,3"),
    answer_key_file: Optional[UploadFile] = File(default=None, description="File đáp án .doc/.docx/.pdf/.txt"),
    total_questions: int = Form(..., description="Tổng số câu hỏi"),
    options: int = Form(..., description="Số lựa chọn mỗi câu"),
    student_id_digits: int = Form(..., description="Số chữ số Student ID"),
    rows_per_block: int = Form(20),
    num_blocks: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
):
    try:
        uid = _validate_uid(uid)
        omr_name = (omr_name or exam_title or "").strip()
        if not omr_name:
            raise HTTPException(status_code=400, detail="Tên phiếu OMR không được để trống")

        omr_code = _normalize_omr_code(omr_code)
        selected_info_fields = _parse_info_fields(info_fields)

        if total_questions <= 0:
            raise HTTPException(status_code=400, detail="Total Questions phải > 0")
        if options < 2:
            raise HTTPException(status_code=400, detail="Options phải >= 2")
        if student_id_digits <= 0:
            raise HTTPException(status_code=400, detail="Student ID digits phải > 0")

        _check_duplicate_template_identity(
            db=db,
            uid=uid,
            omr_name=omr_name,
            omr_code=omr_code,
            student_id_digits=student_id_digits,
        )

        answer_key, answer_source, parsed_questions = _resolve_shared_answer_key(
            answers=answers,
            answer_key_file=answer_key_file,
            num_choices=options,
            num_questions=total_questions,
        )

        result = await run_in_threadpool(
            generate_omr_template,
            output_folder=BASE_OMR_TEMPLATE_DIR,
            exam_title=exam_title,
            total_questions=parsed_questions,
            options=options,
            student_id_digits=student_id_digits,
            rows_per_block=rows_per_block,
            num_blocks=num_blocks,
            exam_code=omr_code,
            info_fields=selected_info_fields,
        )

        if "error" in result:
            return JSONResponse(status_code=400, content=result)

        template_image = result.get("template_image")
        return JSONResponse(
            content={
                "message": "Tạo phiếu OMR thành công. Nhấn Lưu phiếu để lưu vào cơ sở dữ liệu.",
                "data": result,
                "draft": {
                    "uid": uid,
                    "omr_name": omr_name,
                    "omr_code": omr_code,
                    "omr_quest": parsed_questions,
                    "omr_answer": answer_key,
                    "info_fields": selected_info_fields,
                    "options": options,
                    "rows_per_block": rows_per_block,
                    "student_id_digits": student_id_digits,
                    "answer_source": answer_source,
                    "template_image": template_image,
                },
                "template_url": f"/static/omr_templates/{template_image}" if template_image else None,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Lỗi server", "details": str(e)})


@router.post("/template/save")
async def save_omr_template(
    uid: int = Form(...),
    omr_name: str = Form(...),
    omr_code: str = Form(...),
    omr_quest: int = Form(...),
    omr_answer: str = Form(..., description="JSON list zero-based answers"),
    template_image: Optional[str] = Form(default=None),
    info_fields: str = Form(default="[]"),
    options: Optional[int] = Form(default=None),
    rows_per_block: Optional[int] = Form(default=None),
    student_id_digits: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
):
    try:
        uid = _validate_uid(uid)
        omr_name = (omr_name or "").strip()
        if not omr_name:
            raise HTTPException(status_code=400, detail="Tên bài thi không được để trống")
        omr_code = _normalize_omr_code(omr_code)
        omr_quest = max(1, int(omr_quest))
        sid_digits_checked = max(1, int(student_id_digits)) if student_id_digits is not None else 6

        try:
            parsed_answer = json.loads(omr_answer)
        except Exception:
            raise HTTPException(status_code=400, detail="omr_answer phải là JSON list")
        if not isinstance(parsed_answer, list) or len(parsed_answer) != omr_quest:
            raise HTTPException(status_code=400, detail="Số lượng đáp án không khớp số câu")
        try:
            parsed_answer = [int(x) for x in parsed_answer]
        except Exception:
            raise HTTPException(status_code=400, detail="omr_answer chứa giá trị không hợp lệ")

        _check_duplicate_template_identity(
            db=db,
            uid=uid,
            omr_name=omr_name,
            omr_code=omr_code,
            student_id_digits=sid_digits_checked,
        )

        test_record = OMRTest(
            uuid=uid,
            omr_name=omr_name,
            omr_code=omr_code,
            omr_quest=omr_quest,
            omr_answer=parsed_answer,
        )
        db.add(test_record)
        db.commit()
        db.refresh(test_record)

        info_list = _parse_info_fields(info_fields)
        sidecar_payload = {
            "template_image": os.path.basename(str(template_image or "")) if template_image else None,
            "info_fields": info_list,
            "options": int(options) if options is not None else None,
            "rows_per_block": int(rows_per_block) if rows_per_block is not None else None,
            "student_id_digits": sid_digits_checked,
            "saved_at": datetime.utcnow().isoformat(),
        }
        _write_omr_sidecar(test_record.omrid, sidecar_payload)

        return JSONResponse(
            content={
                "message": "Lưu phiếu OMR thành công",
                "omr_test": {
                    "omrid": test_record.omrid,
                    "omr_name": test_record.omr_name,
                    "omr_code": test_record.omr_code,
                    "omr_quest": test_record.omr_quest,
                },
                "template_url": f"/static/omr_templates/{sidecar_payload['template_image']}" if sidecar_payload.get("template_image") else None,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Lỗi server", "details": str(e)})


@router.get("/form-samples")
async def list_form_samples():
    samples = []
    for sample_file in _list_omr_sample_files():
        code = _safe_profile_code(os.path.splitext(sample_file)[0])
        profile = _resolve_profile(code) or _default_profile(sample_file)
        samples.append(
            {
                "code": code,
                "sample_file": sample_file,
                "sample_url": f"/static/omr_data/{sample_file}",
                "profile": profile,
            }
        )
    return JSONResponse(content={"samples": samples})


@router.get("/form-profiles")
async def list_form_profiles():
    profiles = []
    for sample_file in _list_omr_sample_files():
        code = _safe_profile_code(os.path.splitext(sample_file)[0])
        profiles.append(_resolve_profile(code) or _default_profile(sample_file))
    return JSONResponse(content={"profiles": profiles})


@router.get("/form-profiles/{code}")
async def get_form_profile(code: str):
    profile = _resolve_profile(code)
    if not profile:
        raise HTTPException(status_code=404, detail="Không tìm thấy profile phiếu mẫu")
    return JSONResponse(content={"profile": profile})


@router.post("/form-profiles")
async def save_form_profile(payload: Dict[str, Any] = Body(...)):
    sample_file = os.path.basename(str(payload.get("sample_file") or "").strip())
    if not sample_file:
        raise HTTPException(status_code=400, detail="Thiếu sample_file")
    if sample_file not in _list_omr_sample_files():
        raise HTTPException(status_code=404, detail="Không tìm thấy file phiếu mẫu trong omr_data")

    code = _safe_profile_code(payload.get("code") or os.path.splitext(sample_file)[0])
    if not code:
        raise HTTPException(status_code=400, detail="code profile không hợp lệ")

    existing = _resolve_profile(code) or _default_profile(sample_file)
    strategy_in = payload.get("strategy") if isinstance(payload.get("strategy"), dict) else {}
    existing_strategy = existing.get("strategy") if isinstance(existing.get("strategy"), dict) else {}

    sheet_aspect_raw = strategy_in.get("sheet_aspect_ratio") if "sheet_aspect_ratio" in strategy_in else existing_strategy.get("sheet_aspect_ratio")
    try:
        sheet_aspect_ratio = round(max(1.0, min(2.5, float(sheet_aspect_raw))), 6) if sheet_aspect_raw is not None else None
    except Exception:
        sheet_aspect_ratio = None

    page_size_pt = _sanitize_page_size_pt(
        strategy_in.get("page_size_pt") if "page_size_pt" in strategy_in else existing_strategy.get("page_size_pt")
    )
    corner_markers = _sanitize_corner_markers(
        strategy_in.get("corner_markers") if "corner_markers" in strategy_in else existing_strategy.get("corner_markers")
    )
    scanner_hint = _sanitize_scanner_hint(
        strategy_in.get("scanner_hint") if "scanner_hint" in strategy_in else existing_strategy.get("scanner_hint")
    )

    parsed_disable_rescue = _sanitize_bool_flag(
        strategy_in.get("disable_mcq_rescue") if "disable_mcq_rescue" in strategy_in else existing_strategy.get("disable_mcq_rescue")
    )

    strategy_payload = {
        "crop_quad": _sanitize_quad(strategy_in.get("crop_quad") if "crop_quad" in strategy_in else existing_strategy.get("crop_quad")),
        "sid_roi": _sanitize_norm_rect(strategy_in.get("sid_roi") if "sid_roi" in strategy_in else existing_strategy.get("sid_roi")),
        "mcq_roi": _sanitize_norm_rect(strategy_in.get("mcq_roi") if "mcq_roi" in strategy_in else existing_strategy.get("mcq_roi")),
        "exam_code_roi": _sanitize_norm_rect(strategy_in.get("exam_code_roi") if "exam_code_roi" in strategy_in else existing_strategy.get("exam_code_roi")),
        "handwriting_fields": _sanitize_handwriting_fields(
            strategy_in.get("handwriting_fields") if "handwriting_fields" in strategy_in else existing_strategy.get("handwriting_fields")
        ),
        "sid_row_offsets": _sanitize_sid_row_offsets(strategy_in.get("sid_row_offsets") if "sid_row_offsets" in strategy_in else existing_strategy.get("sid_row_offsets")),
        "mcq_decode": _sanitize_mcq_decode(strategy_in.get("mcq_decode") if "mcq_decode" in strategy_in else existing_strategy.get("mcq_decode")),
        "threshold_mode": _sanitize_threshold_mode(strategy_in.get("threshold_mode") if "threshold_mode" in strategy_in else existing_strategy.get("threshold_mode")),
        "ai_uncertainty": _sanitize_ai_uncertainty(
            strategy_in.get("ai_uncertainty") if "ai_uncertainty" in strategy_in else existing_strategy.get("ai_uncertainty")
        ),
        "ai_sid_htr": _sanitize_ai_sid_htr(
            strategy_in.get("ai_sid_htr") if "ai_sid_htr" in strategy_in else existing_strategy.get("ai_sid_htr")
        ),
        "agentic_rescue": _sanitize_agentic_rescue(
            strategy_in.get("agentic_rescue") if "agentic_rescue" in strategy_in else existing_strategy.get("agentic_rescue")
        ),
        "disable_mcq_rescue": bool(parsed_disable_rescue) if parsed_disable_rescue is not None else False,
    }
    if sheet_aspect_ratio is not None:
        strategy_payload["sheet_aspect_ratio"] = sheet_aspect_ratio
    if page_size_pt is not None:
        strategy_payload["page_size_pt"] = page_size_pt
    if corner_markers is not None:
        strategy_payload["corner_markers"] = corner_markers
    if scanner_hint is not None:
        strategy_payload["scanner_hint"] = scanner_hint

    profile = {
        "code": code,
        "title": str(payload.get("title") or existing.get("title") or os.path.splitext(sample_file)[0]).strip() or os.path.splitext(sample_file)[0],
        "sample_file": sample_file,
        "default_questions": max(1, int(payload.get("default_questions") or existing.get("default_questions") or 40)),
        "total_points": 10,
        "num_choices": max(2, min(6, int(payload.get("num_choices") or existing.get("num_choices") or 4))),
        "rows_per_block": max(1, int(payload.get("rows_per_block") or existing.get("rows_per_block") or 20)),
        "num_blocks": int(payload.get("num_blocks")) if payload.get("num_blocks") not in (None, "", "null") else None,
        "student_id_digits": max(1, int(payload.get("student_id_digits") or existing.get("student_id_digits") or 6)),
        "sid_has_write_row": bool(payload.get("sid_has_write_row") if "sid_has_write_row" in payload else existing.get("sid_has_write_row", True)),
        "strategy": strategy_payload,
        "updated_at": datetime.utcnow().isoformat(),
    }

    with open(_profile_path(code), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    return JSONResponse(content={"message": "Lưu profile thành công", "profile": profile})


def _serialize_assignment(record: OMRAssignment) -> dict:
    form_profile_code = _extract_form_profile_code(record.last_result)
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
        "last_result": record.last_result,
        "form_profile_code": form_profile_code,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


@router.post("/assignments")
async def create_assignment(
    uid: int = Form(...),
    title: str = Form(...),
    created_at_raw: Optional[str] = Form(default=None),
    created_at_label: Optional[str] = Form(default=None),
    question_count: Optional[int] = Form(default=None),
    total_points: Optional[int] = Form(default=None),
    form_profile_code: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
):
    uid = _validate_uid(uid)
    safe_title = str(title or "").strip()
    if not safe_title:
        raise HTTPException(status_code=400, detail="Tên bài thi không được để trống")

    profile = _resolve_profile(form_profile_code)
    if form_profile_code and not profile:
        raise HTTPException(status_code=404, detail="Không tìm thấy profile phiếu mẫu")

    effective_q = int((profile or {}).get("default_questions") or question_count or 40)
    effective_points = 10 if total_points is None else max(1, int(total_points))

    record = OMRAssignment(
        uuid=uid,
        title=safe_title,
        created_at_raw=str(created_at_raw or "").strip() or None,
        created_at_label=str(created_at_label or "").strip() or None,
        question_count=max(1, int(effective_q)),
        total_points=max(1, int(effective_points)),
        graded_count=0,
        answer_sets=[],
        active_code=None,
        last_result=_with_assignment_meta(None, _safe_profile_code(form_profile_code or "")),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return JSONResponse(content={"message": "Tạo bài thi thành công", "assignment": _serialize_assignment(record)})


@router.get("/assignments/{uid}")
async def list_assignments(uid: int, db: Session = Depends(get_db)):
    uid = _validate_uid(uid)
    records = (
        db.query(OMRAssignment)
        .filter(OMRAssignment.uuid == uid)
        .order_by(OMRAssignment.updated_at.desc(), OMRAssignment.aid.desc())
        .all()
    )
    return JSONResponse(content={"assignments": [_serialize_assignment(r) for r in records]})


@router.put("/assignments/{uid}/{aid}")
async def update_assignment(
    uid: int,
    aid: int,
    title: Optional[str] = Form(default=None),
    created_at_raw: Optional[str] = Form(default=None),
    created_at_label: Optional[str] = Form(default=None),
    question_count: Optional[int] = Form(default=None),
    total_points: Optional[int] = Form(default=None),
    graded_count: Optional[int] = Form(default=None),
    answer_sets: Optional[str] = Form(default=None),
    active_code: Optional[str] = Form(default=None),
    last_result: Optional[str] = Form(default=None),
    form_profile_code: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
):
    uid = _validate_uid(uid)
    record = (
        db.query(OMRAssignment)
        .filter(OMRAssignment.uuid == uid, OMRAssignment.aid == int(aid))
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài thi")

    if title is not None:
        safe_title = str(title).strip()
        if not safe_title:
            raise HTTPException(status_code=400, detail="Tên bài thi không được để trống")
        record.title = safe_title
    if created_at_raw is not None:
        record.created_at_raw = str(created_at_raw).strip() or None
    if created_at_label is not None:
        record.created_at_label = str(created_at_label).strip() or None
    if question_count is not None:
        current_profile = _resolve_profile(_extract_form_profile_code(record.last_result))
        if current_profile:
            record.question_count = max(1, int(current_profile.get("default_questions") or record.question_count or 1))
        else:
            record.question_count = max(1, int(question_count))
    if total_points is not None:
        record.total_points = max(1, int(total_points))
    if graded_count is not None:
        record.graded_count = max(0, int(graded_count))
    if active_code is not None:
        record.active_code = str(active_code).strip() or None

    if form_profile_code is not None:
        safe_profile = _safe_profile_code(form_profile_code)
        if safe_profile and not _resolve_profile(safe_profile):
            raise HTTPException(status_code=404, detail="Không tìm thấy profile phiếu mẫu")
        record.last_result = _with_assignment_meta(record.last_result, safe_profile)
        if safe_profile:
            next_profile = _resolve_profile(safe_profile)
            if next_profile:
                record.question_count = max(1, int(next_profile.get("default_questions") or record.question_count or 1))

    if answer_sets is not None:
        try:
            parsed_sets = json.loads(answer_sets)
        except Exception:
            raise HTTPException(status_code=400, detail="answer_sets phải là JSON list")
        if not isinstance(parsed_sets, list):
            raise HTTPException(status_code=400, detail="answer_sets phải là JSON list")
        record.answer_sets = parsed_sets

    if last_result is not None:
        try:
            parsed_result = json.loads(last_result)
        except Exception:
            raise HTTPException(status_code=400, detail="last_result phải là JSON object")
        if not isinstance(parsed_result, dict):
            raise HTTPException(status_code=400, detail="last_result phải là JSON object")
        selected_code = form_profile_code if form_profile_code is not None else _extract_form_profile_code(record.last_result)
        record.last_result = _with_assignment_meta(parsed_result, selected_code)

    db.commit()
    db.refresh(record)
    return JSONResponse(content={"message": "Cập nhật bài thi thành công", "assignment": _serialize_assignment(record)})


@router.delete("/assignments/{uid}/{aid}")
async def delete_assignment(uid: int, aid: int, db: Session = Depends(get_db)):
    uid = _validate_uid(uid)
    record = (
        db.query(OMRAssignment)
        .filter(OMRAssignment.uuid == uid, OMRAssignment.aid == int(aid))
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy bài thi")

    db.delete(record)
    db.commit()
    return JSONResponse(content={"message": "Đã xóa bài thi"})


def _extract_answer_key_text(file_path: str) -> str:
    """Extract plain text from answer key file (.doc/.docx/.pdf/.txt)."""
    lower = file_path.lower()
    if lower.endswith(".doc") or lower.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    if lower.endswith(".pdf"):
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    if lower.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Định dạng file đáp án không hỗ trợ. Dùng: .doc/.docx/.pdf/.txt")


def _normalize_choice_token(token: str, num_choices: int, assume_one_based: bool) -> Optional[int]:
    """Convert answer token (A/B/C or 0/1/2...) to zero-based index."""
    token = token.strip().upper()
    if not token:
        return None

    if re.fullmatch(r"[A-Z]", token):
        idx = ord(token) - ord("A")
        if 0 <= idx < num_choices:
            return idx
        return None

    if re.fullmatch(r"\d+", token):
        num = int(token)
        if assume_one_based:
            if 1 <= num <= num_choices:
                return num - 1
            return None
        if 0 <= num < num_choices:
            return num
        if 1 <= num <= num_choices:
            return num - 1
        return None

    return None


def _parse_answer_key_from_text(
    text: str,
    num_choices: int,
    assume_one_based: Optional[bool] = None,
) -> List[int]:
    """
    Parse answer key from text with formats like:
    - Câu 1: A
    - 1) B
    - 1,A or 1-A
    - A,B,C,D,E
    - 1,2,3,4,5
    """
    if not text or not text.strip():
        raise ValueError("File đáp án trống")

    numbered_matches = re.findall(
        r"(?:C[âa]u\s*)?(\d{1,3})\s*[:\-\.)]?\s*([A-Ea-e]|\d{1,2})",
        text,
        flags=re.IGNORECASE,
    )

    if numbered_matches:
        entries = [(int(q), ans) for q, ans in numbered_matches]
        entries.sort(key=lambda x: x[0])

        numeric_tokens = [ans for _, ans in entries if re.fullmatch(r"\d+", ans.strip())]
        local_assume_one_based = bool(assume_one_based) if assume_one_based is not None else False
        if assume_one_based is None and numeric_tokens:
            nums = [int(x.strip()) for x in numeric_tokens]
            local_assume_one_based = min(nums) >= 1 and max(nums) <= num_choices

        parsed = []
        for _, token in entries:
            idx = _normalize_choice_token(token, num_choices, local_assume_one_based)
            if idx is None:
                raise ValueError(f"Không hiểu đáp án '{token}' trong file")
            parsed.append(idx)
        return parsed

    raw_tokens = [t.strip() for t in re.split(r"[;,\s\n\r\t]+", text) if t.strip()]
    if not raw_tokens:
        raise ValueError("Không tìm thấy đáp án trong file")

    numeric_tokens = [t for t in raw_tokens if re.fullmatch(r"\d+", t)]
    local_assume_one_based = bool(assume_one_based) if assume_one_based is not None else False
    if assume_one_based is None and numeric_tokens:
        nums = [int(x) for x in numeric_tokens]
        local_assume_one_based = min(nums) >= 1 and max(nums) <= num_choices

    parsed = []
    for token in raw_tokens:
        idx = _normalize_choice_token(token, num_choices, local_assume_one_based)
        if idx is not None:
            parsed.append(idx)

    if not parsed:
        raise ValueError("Không parse được đáp án từ file")

    return parsed


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
            answer_text = _extract_answer_key_text(answer_file_path)
            # File dap an: cho phep auto detect 1-5 hoac 0-4.
            answer_key = _parse_answer_key_from_text(answer_text, num_choices, assume_one_based=None)
            answer_source = "file"
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
    elif answers.strip():
        try:
            # Nhap tay: su dung chuan one-based theo UI (A=1..E=5), backend quy doi ve zero-based.
            answer_key = _parse_answer_key_from_text(answers, num_choices, assume_one_based=True)
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

    idx = _normalize_choice_token(text, num_choices, assume_one_based=True)
    if idx is None:
        idx = _normalize_choice_token(text, num_choices, assume_one_based=False)
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


@router.get("/tests/{uid}")
async def list_omr_tests(uid: int, db: Session = Depends(get_db)):
    uid = _validate_uid(uid)
    records = (
        db.query(OMRTest)
        .filter(OMRTest.uuid == uid)
        .order_by(OMRTest.omrid.desc())
        .all()
    )

    tests = []
    for rec in records:
        sidecar = _read_omr_sidecar(rec.omrid)
        template_image = str(sidecar.get("template_image") or "").strip()
        one_based_answers = []
        if isinstance(rec.omr_answer, list):
            for x in rec.omr_answer:
                try:
                    one_based_answers.append(int(x) + 1)
                except Exception:
                    continue
        preview = ",".join(str(x) for x in one_based_answers[:12])
        tests.append(
            {
                "omrid": rec.omrid,
                "omr_name": rec.omr_name,
                "omr_code": rec.omr_code,
                "omr_quest": rec.omr_quest,
                "answer_preview": preview,
                "template_url": f"/static/omr_templates/{template_image}" if template_image else None,
                "created_at": rec.created_at.isoformat() if rec.created_at else None,
            }
        )

    return JSONResponse(content={"omr_tests": tests})


@router.delete("/tests/{uid}/{omrid}")
async def delete_omr_test(uid: int, omrid: int, db: Session = Depends(get_db)):
    uid = _validate_uid(uid)
    record = (
        db.query(OMRTest)
        .filter(OMRTest.uuid == uid, OMRTest.omrid == int(omrid))
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiếu OMR")

    db.delete(record)
    db.commit()
    _delete_omr_sidecar_and_template(omrid)
    return JSONResponse(content={"message": "Đã xóa phiếu OMR"})

@router.post("/grade")
async def grade_exam(
    file: UploadFile = File(...),
    uid: Optional[int] = Form(default=None),
    aid: Optional[int] = Form(default=None),
    form_profile_code: Optional[str] = Form(default=None),
    omr_test_id: Optional[int] = Form(default=None),
    answers: str = Form(default="", description="Chuỗi đáp án cách nhau bởi dấu phẩy, v.d: 1,2,0,1,4"),
    answer_key_file: Optional[UploadFile] = File(default=None, description="File đáp án .doc/.docx/.pdf/.txt"),
    num_questions: int = Form(80),
    num_choices: int = Form(5),
    rows_per_block: int = Form(20),
    num_blocks: Optional[int] = Form(default=None),
    student_id_digits: int = Form(6),
    sid_has_write_row: bool = Form(False),
    crop_x: Optional[float] = Form(default=None),
    crop_y: Optional[float] = Form(default=None),
    crop_w: Optional[float] = Form(default=None),
    crop_h: Optional[float] = Form(default=None),
    crop_tl_x: Optional[float] = Form(default=None),
    crop_tl_y: Optional[float] = Form(default=None),
    crop_tr_x: Optional[float] = Form(default=None),
    crop_tr_y: Optional[float] = Form(default=None),
    crop_br_x: Optional[float] = Form(default=None),
    crop_br_y: Optional[float] = Form(default=None),
    crop_bl_x: Optional[float] = Form(default=None),
    crop_bl_y: Optional[float] = Form(default=None),
    db: Session = Depends(get_db),
):
    try:
        profile = _resolve_profile(form_profile_code)
        if form_profile_code and not profile:
            raise HTTPException(status_code=404, detail="Không tìm thấy profile phiếu mẫu")

        runtime = _build_runtime_config(
            profile,
            num_questions=num_questions,
            num_choices=num_choices,
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

        num_questions = runtime["num_questions"]
        num_choices = runtime["num_choices"]
        rows_per_block = runtime["rows_per_block"]
        num_blocks = runtime["num_blocks"]
        student_id_digits = runtime["student_id_digits"]
        sid_has_write_row = runtime["sid_has_write_row"]
        crop_x = runtime["crop_x"]
        crop_y = runtime["crop_y"]
        crop_w = runtime["crop_w"]
        crop_h = runtime["crop_h"]
        crop_tl_x = runtime["crop_tl_x"]
        crop_tl_y = runtime["crop_tl_y"]
        crop_tr_x = runtime["crop_tr_x"]
        crop_tr_y = runtime["crop_tr_y"]
        crop_br_x = runtime["crop_br_x"]
        crop_br_y = runtime["crop_br_y"]
        crop_bl_x = runtime["crop_bl_x"]
        crop_bl_y = runtime["crop_bl_y"]

        uid_checked = None
        if uid is not None:
            uid_checked = _validate_uid(uid)

        assignment_answer_map: Optional[Dict[str, List[int]]] = None

        # 1. Parse đáp án: ưu tiên assignment, sau đó đề OMR đã lưu,
        # tiếp theo auto nhận diện đề, fallback nhập tay/file.
        if aid is not None:
            uid_checked = _validate_uid(uid)
            record = (
                db.query(OMRAssignment)
                .filter(OMRAssignment.uuid == int(uid_checked), OMRAssignment.aid == int(aid))
                .first()
            )
            if not record:
                raise HTTPException(status_code=404, detail="Không tìm thấy bài thi theo aid")

            assignment_answer_map, active_code = _build_assignment_answer_key_map(record, num_choices)
            if not assignment_answer_map:
                raise HTTPException(status_code=400, detail="Kho đáp án của bài thi đang trống")

            if active_code and active_code in assignment_answer_map:
                answer_key = list(assignment_answer_map[active_code])
            else:
                answer_key = list(next(iter(assignment_answer_map.values())))

            answer_source = "assignment-code-map"
            parsed_num_questions = len(answer_key)
            matched_test = None
            detect_probe = None
        elif omr_test_id is not None:
            uid_checked = _validate_uid(uid)
            answer_key, answer_source, parsed_num_questions = _resolve_answer_key_from_omr_test(
                db=db,
                uid=uid_checked,
                omr_test_id=omr_test_id,
            )
            matched_test = None
            detect_probe = None
        elif (not answers.strip()) and (answer_key_file is None or not answer_key_file.filename):
            uid_checked = _validate_uid(uid)
            # Delay resolving until file is stored, because auto mode needs image decoding first.
            answer_key = None
            answer_source = "auto"
            parsed_num_questions = int(num_questions)
            matched_test = None
            detect_probe = None
        else:
            answer_key, answer_source, parsed_num_questions = _resolve_shared_answer_key(
                answers=answers,
                answer_key_file=answer_key_file,
                num_choices=num_choices,
                num_questions=num_questions,
            )
            matched_test = None
            detect_probe = None

        # 2. Lưu file upload tạm thời
        safe_exam_name = os.path.basename(file.filename)
        file_location = os.path.join(BASE_OMR_DIR, safe_exam_name)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if answer_key is None:
            answer_key, answer_source, parsed_num_questions, matched_test, detect_probe = await _resolve_answer_key_auto_from_sheet(
                db=db,
                uid=uid_checked,
                file_location=file_location,
                num_questions=num_questions,
                num_choices=num_choices,
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

        # 3. Gọi Service xử lý
        result = await run_in_threadpool(
            process_omr_exam,       # Tên hàm
            image_path=file_location,
            output_folder=BASE_OMR_DIR,
            answer_key=answer_key,
            answer_key_by_code=assignment_answer_map,
            questions=num_questions,
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
            profile_sid_roi=runtime["profile_sid_roi"],
            profile_mcq_roi=runtime["profile_mcq_roi"],
            profile_exam_code_roi=runtime["profile_exam_code_roi"],
            profile_sid_row_offsets=runtime["profile_sid_row_offsets"],
            profile_disable_mcq_rescue=runtime["profile_disable_mcq_rescue"],
            profile_mcq_decode=runtime["profile_mcq_decode"],
            profile_threshold_mode=runtime["profile_threshold_mode"],
            profile_ai_uncertainty=runtime["profile_ai_uncertainty"],
            profile_ai_sid_htr=runtime["profile_ai_sid_htr"],
            profile_agentic_rescue=runtime["profile_agentic_rescue"],
            profile_corner_markers=runtime["profile_corner_markers"],
            profile_scanner_hint=runtime["profile_scanner_hint"],
            profile_page_size_pt=runtime["profile_page_size_pt"],
            profile_handwriting_fields=runtime["profile_handwriting_fields"],
        )

        # 4. Kiểm tra lỗi từ service
        if "error" in result:
               print(f"[OMR][GRADE][400] {result.get('error')}")
               return JSONResponse(status_code=400, content=result)

        # 5. Trả về kết quả JSON
        bubble_confidence_json_url = _static_omr_url(result.get("bubble_confidence_json"))
        result["bubble_confidence_json_url"] = bubble_confidence_json_url
        return JSONResponse(content={
            "message": "Chấm điểm thành công",
            "data": result,
            "image_url": _static_omr_url(result.get("result_image")),
            "sid_crop_url": _static_omr_url(result.get("sid_crop_image")),
            "mcq_crop_url": _static_omr_url(result.get("mcq_crop_image")),
            "bubble_confidence_json_url": bubble_confidence_json_url,
            "answer_source": answer_source,
            "parsed_num_questions": parsed_num_questions,
            "matched_omr_test": {
                "omrid": matched_test.omrid,
                "omr_name": matched_test.omr_name,
                "omr_code": matched_test.omr_code,
            } if matched_test is not None else None,
            "auto_detect": {
                "exam_code": detect_probe.get("exam_code") if detect_probe else None,
                "exam_title_detected": detect_probe.get("exam_title_detected") if detect_probe else None,
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Lỗi server", "details": str(e)})


@router.post("/grade-batch")
async def grade_exam_batch(
    files: List[UploadFile] = File(...),
    uid: Optional[int] = Form(default=None),
    aid: Optional[int] = Form(default=None),
    form_profile_code: Optional[str] = Form(default=None),
    omr_test_id: Optional[int] = Form(default=None),
    answers: str = Form(default="", description="Chuỗi đáp án cách nhau bởi dấu phẩy, v.d: 1,2,3,4"),
    answer_key_file: Optional[UploadFile] = File(default=None, description="File đáp án .doc/.docx/.pdf/.txt"),
    num_questions: int = Form(80),
    num_choices: int = Form(5),
    rows_per_block: int = Form(20),
    num_blocks: Optional[int] = Form(default=None),
    student_id_digits: int = Form(6),
    sid_has_write_row: bool = Form(False),
    db: Session = Depends(get_db),
):
    try:
        profile = _resolve_profile(form_profile_code)
        if form_profile_code and not profile:
            raise HTTPException(status_code=404, detail="Không tìm thấy profile phiếu mẫu")

        runtime = _build_runtime_config(
            profile,
            num_questions=num_questions,
            num_choices=num_choices,
            rows_per_block=rows_per_block,
            num_blocks=num_blocks,
            student_id_digits=student_id_digits,
            sid_has_write_row=sid_has_write_row,
        )

        num_questions = runtime["num_questions"]
        num_choices = runtime["num_choices"]
        rows_per_block = runtime["rows_per_block"]
        num_blocks = runtime["num_blocks"]
        student_id_digits = runtime["student_id_digits"]
        sid_has_write_row = runtime["sid_has_write_row"]

        uid_checked = None
        if uid is not None:
            uid_checked = _validate_uid(uid)

        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="Vui lòng tải lên ít nhất 1 ảnh")
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Mỗi lần gửi tối đa 50 ảnh")

        assignment_answer_map: Optional[Dict[str, List[int]]] = None
        auto_mode = False

        if aid is not None:
            uid_checked = _validate_uid(uid)
            record = (
                db.query(OMRAssignment)
                .filter(OMRAssignment.uuid == int(uid_checked), OMRAssignment.aid == int(aid))
                .first()
            )
            if not record:
                raise HTTPException(status_code=404, detail="Không tìm thấy bài thi theo aid")

            assignment_answer_map, active_code = _build_assignment_answer_key_map(record, num_choices)
            if not assignment_answer_map:
                raise HTTPException(status_code=400, detail="Kho đáp án của bài thi đang trống")

            if active_code and active_code in assignment_answer_map:
                answer_key = list(assignment_answer_map[active_code])
            else:
                answer_key = list(next(iter(assignment_answer_map.values())))

            answer_source = "assignment-code-map"
            parsed_num_questions = len(answer_key)
            matched_global_test = None
        elif omr_test_id is not None:
            uid_checked = _validate_uid(uid)
            answer_key, answer_source, parsed_num_questions = _resolve_answer_key_from_omr_test(
                db=db,
                uid=uid_checked,
                omr_test_id=omr_test_id,
            )
            matched_global_test = {
                "omrid": omr_test_id,
            }
        elif (not answers.strip()) and (answer_key_file is None or not answer_key_file.filename):
            uid_checked = _validate_uid(uid)
            auto_mode = True
            answer_key = None
            answer_source = "auto"
            parsed_num_questions = int(num_questions)
            matched_global_test = None
        else:
            answer_key, answer_source, parsed_num_questions = _resolve_shared_answer_key(
                answers=answers,
                answer_key_file=answer_key_file,
                num_choices=num_choices,
                num_questions=num_questions,
            )
            matched_global_test = None

        results = []
        success_count = 0
        failed_count = 0
        zip_image_paths = []

        for upload in files:
            safe_exam_name = os.path.basename(upload.filename or "omr_batch.jpg")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stored_name = f"{timestamp}_{safe_exam_name}"
            file_location = os.path.join(BASE_OMR_DIR, stored_name)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)

            one_answer_key = answer_key
            one_answer_source = answer_source
            one_num_questions = num_questions
            matched_one = None
            detect_probe = None

            if auto_mode:
                try:
                    one_answer_key, one_answer_source, parsed_num_questions, matched_record, detect_probe = await _resolve_answer_key_auto_from_sheet(
                        db=db,
                        uid=uid_checked,
                        file_location=file_location,
                        num_questions=num_questions,
                        num_choices=num_choices,
                        rows_per_block=rows_per_block,
                        num_blocks=num_blocks,
                        student_id_digits=student_id_digits,
                        sid_has_write_row=sid_has_write_row,
                    )
                    matched_one = {
                        "omrid": matched_record.omrid,
                        "omr_name": matched_record.omr_name,
                        "omr_code": matched_record.omr_code,
                    }
                except HTTPException as ex:
                    failed_count += 1
                    results.append(
                        {
                            "file_name": safe_exam_name,
                            "success": False,
                            "error": str(ex.detail),
                            "answer_source": "auto",
                        }
                    )
                    continue

            result = await run_in_threadpool(
                process_omr_exam,
                image_path=file_location,
                output_folder=BASE_OMR_DIR,
                answer_key=one_answer_key,
                answer_key_by_code=assignment_answer_map,
                questions=one_num_questions,
                choices=num_choices,
                rows_per_block=rows_per_block,
                num_blocks=num_blocks,
                student_id_digits=student_id_digits,
                sid_has_write_row=sid_has_write_row,
                crop_tl_x=runtime["crop_tl_x"],
                crop_tl_y=runtime["crop_tl_y"],
                crop_tr_x=runtime["crop_tr_x"],
                crop_tr_y=runtime["crop_tr_y"],
                crop_br_x=runtime["crop_br_x"],
                crop_br_y=runtime["crop_br_y"],
                crop_bl_x=runtime["crop_bl_x"],
                crop_bl_y=runtime["crop_bl_y"],
                profile_sid_roi=runtime["profile_sid_roi"],
                profile_mcq_roi=runtime["profile_mcq_roi"],
                profile_exam_code_roi=runtime["profile_exam_code_roi"],
                profile_sid_row_offsets=runtime["profile_sid_row_offsets"],
                profile_disable_mcq_rescue=runtime["profile_disable_mcq_rescue"],
                profile_mcq_decode=runtime["profile_mcq_decode"],
                profile_threshold_mode=runtime["profile_threshold_mode"],
                profile_ai_uncertainty=runtime["profile_ai_uncertainty"],
                profile_ai_sid_htr=runtime["profile_ai_sid_htr"],
                profile_agentic_rescue=runtime["profile_agentic_rescue"],
                profile_corner_markers=runtime["profile_corner_markers"],
                profile_scanner_hint=runtime["profile_scanner_hint"],
                profile_page_size_pt=runtime["profile_page_size_pt"],
                profile_handwriting_fields=runtime["profile_handwriting_fields"],
            )

            if "error" in result:
                failed_count += 1
                results.append(
                    {
                        "file_name": safe_exam_name,
                        "success": False,
                        "error": result.get("error"),
                        "answer_source": one_answer_source,
                    }
                )
                continue

            success_count += 1
            if result.get("result_image"):
                zip_image_paths.append(os.path.join(BASE_OMR_DIR, result["result_image"]))

            bubble_confidence_json_url = _static_omr_url(result.get("bubble_confidence_json"))
            result["bubble_confidence_json_url"] = bubble_confidence_json_url
            results.append(
                {
                    "file_name": safe_exam_name,
                    "success": True,
                    "data": result,
                    "image_url": _static_omr_url(result.get("result_image")),
                    "sid_crop_url": _static_omr_url(result.get("sid_crop_image")),
                    "mcq_crop_url": _static_omr_url(result.get("mcq_crop_image")),
                    "bubble_confidence_json_url": bubble_confidence_json_url,
                    "answer_source": one_answer_source,
                    "matched_omr_test": matched_one,
                    "auto_detect": {
                        "exam_code": detect_probe.get("exam_code") if detect_probe else None,
                        "exam_title_detected": detect_probe.get("exam_title_detected") if detect_probe else None,
                    },
                }
            )

        zip_url = None
        if len(zip_image_paths) > 0:
            zip_name = f"omr_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.zip"
            zip_path = os.path.join(BASE_OMR_DIR, zip_name)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for abs_path in zip_image_paths:
                    if os.path.exists(abs_path):
                        zf.write(abs_path, arcname=os.path.basename(abs_path))
            zip_url = f"/static/omr/{zip_name}"

        return JSONResponse(
            content={
                "message": "Chấm điểm batch hoàn tất",
                "total_files": len(files),
                "success_count": success_count,
                "failed_count": failed_count,
                "answer_source": answer_source,
                "parsed_num_questions": parsed_num_questions,
                "matched_omr_test": matched_global_test,
                "zip_url": zip_url,
                "results": results,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Lỗi server", "details": str(e)})


@router.post("/suggest-crop")
async def suggest_crop(file: UploadFile = File(...)):
    """Suggest 4-corner crop quad from current OMR CV pipeline."""
    try:
        safe_name = os.path.basename(file.filename or "omr_input.jpg")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_name = f"suggest_{timestamp}_{safe_name}"
        file_location = os.path.join(BASE_OMR_DIR, temp_name)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = await run_in_threadpool(suggest_omr_crop_quad, file_location)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)

        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Lỗi server", "details": str(e)})