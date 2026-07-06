from fastapi import APIRouter

from .shared import *  # noqa: F403 - route modules share the existing helper surface during the split

router = APIRouter()

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

@router.post("/upload-form-sample")
async def upload_form_sample(file: UploadFile = File(...)):
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận PDF, PNG, JPG, JPEG, WebP")
    safe_name = re.sub(r"[^\w.\-]", "-", file.filename or "sample") if file.filename else f"sample{ext}"
    dest = os.path.join(BASE_OMR_DATA_DIR, safe_name)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return JSONResponse(content={"filename": safe_name})

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
    parsed_sid_roi_lock = _sanitize_bool_flag(
        strategy_in.get("sid_roi_lock") if "sid_roi_lock" in strategy_in else existing_strategy.get("sid_roi_lock")
    )

    strategy_payload = {
        "sid_roi": _sanitize_norm_rect(strategy_in.get("sid_roi") if "sid_roi" in strategy_in else existing_strategy.get("sid_roi")),
        "mcq_roi": _sanitize_norm_rect(strategy_in.get("mcq_roi") if "mcq_roi" in strategy_in else existing_strategy.get("mcq_roi")),
        "exam_code_roi": _sanitize_norm_rect(strategy_in.get("exam_code_roi") if "exam_code_roi" in strategy_in else existing_strategy.get("exam_code_roi")),
        "handwriting_fields": _sanitize_handwriting_fields(
            strategy_in.get("handwriting_fields") if "handwriting_fields" in strategy_in else existing_strategy.get("handwriting_fields")
        ),
        "sid_row_offsets": _sanitize_sid_row_offsets(strategy_in.get("sid_row_offsets") if "sid_row_offsets" in strategy_in else existing_strategy.get("sid_row_offsets")),
        "mcq_decode": _sanitize_mcq_decode(strategy_in.get("mcq_decode") if "mcq_decode" in strategy_in else existing_strategy.get("mcq_decode")),
        "threshold_mode": _sanitize_threshold_mode(strategy_in.get("threshold_mode") if "threshold_mode" in strategy_in else existing_strategy.get("threshold_mode")),
        "disable_mcq_rescue": bool(parsed_disable_rescue) if parsed_disable_rescue is not None else False,
        "sid_roi_lock": bool(parsed_sid_roi_lock) if parsed_sid_roi_lock is not None else False,
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
