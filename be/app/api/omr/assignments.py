from fastapi import APIRouter

from .shared import *  # noqa: F403 - route modules share the existing helper surface during the split

router = APIRouter()

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
