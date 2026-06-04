from fastapi import APIRouter

from .shared import *  # noqa: F403 - route modules share the existing helper surface during the split

router = APIRouter()

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
            template_image=os.path.basename(str(template_image or "")) if template_image else None,
            info_fields=_parse_info_fields(info_fields),
            options=int(options) if options is not None else None,
            rows_per_block=int(rows_per_block) if rows_per_block is not None else None,
            student_id_digits=sid_digits_checked,
        )
        db.add(test_record)
        db.commit()
        db.refresh(test_record)

        return JSONResponse(
            content={
                "message": "Lưu phiếu OMR thành công",
                "omr_test": {
                    "omrid": test_record.omrid,
                    "omr_name": test_record.omr_name,
                    "omr_code": test_record.omr_code,
                    "omr_quest": test_record.omr_quest,
                },
                "template_url": f"/static/omr_templates/{test_record.template_image}" if test_record.template_image else None,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Lỗi server", "details": str(e)})

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
        metadata = _omr_template_metadata(rec)
        template_image = str(metadata.get("template_image") or "").strip()
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

    metadata = _omr_template_metadata(record)
    _delete_omr_sidecar_and_template(omrid, metadata)
    db.query(OMRGradeResult).filter(OMRGradeResult.omrid == int(omrid)).update(
        {OMRGradeResult.omrid: None},
        synchronize_session=False,
    )
    db.delete(record)
    db.commit()
    return JSONResponse(content={"message": "Đã xóa phiếu OMR"})
