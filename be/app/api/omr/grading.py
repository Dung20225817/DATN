from fastapi import APIRouter

from .shared import *  # noqa: F403 - route modules share the existing helper surface during the split

router = APIRouter()

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
    disable_mcq_rescue: Optional[bool] = Form(default=None, description="Tắt MCQ Map Search Rescue (dùng cho ablation eval)"),
    density_only_scoring: Optional[bool] = Form(default=None, description="Chỉ dùng density (bỏ darkness) khi tính điểm bong bóng (ablation)"),
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
        if disable_mcq_rescue is not None:
            runtime["profile_disable_mcq_rescue"] = disable_mcq_rescue
        runtime["density_only_scoring"] = bool(density_only_scoring)

        num_questions = runtime["num_questions"]
        num_choices = runtime["num_choices"]
        rows_per_block = runtime["rows_per_block"]
        num_blocks = runtime["num_blocks"]
        student_id_digits = runtime["student_id_digits"]
        sid_has_write_row = runtime["sid_has_write_row"]

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
            profile_sid_roi=runtime["profile_sid_roi"],
            profile_sid_roi_lock=runtime["profile_sid_roi_lock"],
            profile_mcq_roi=runtime["profile_mcq_roi"],
            profile_exam_code_roi=runtime["profile_exam_code_roi"],
            profile_sid_row_offsets=runtime["profile_sid_row_offsets"],
            profile_disable_mcq_rescue=runtime["profile_disable_mcq_rescue"],
            density_only_scoring=runtime["density_only_scoring"],
            profile_mcq_decode=runtime["profile_mcq_decode"],
            profile_threshold_mode=runtime["profile_threshold_mode"],
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
        _inject_handwriting_crop_urls(result)
        bubble_confidence_json_url = _static_omr_url(result.get("bubble_confidence_json"))
        result["bubble_confidence_json_url"] = bubble_confidence_json_url
        persisted_omrid = int(omr_test_id) if omr_test_id is not None else (matched_test.omrid if matched_test is not None else None)
        grade_result_id = _persist_omr_grade_result(
            db,
            uid=uid_checked,
            aid=aid,
            omrid=persisted_omrid,
            source=answer_source,
            file_name=safe_exam_name,
            result=result,
        )
        return JSONResponse(content={
            "message": "Chấm điểm thành công",
            "data": result,
            "grade_result_id": grade_result_id,
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
                profile_sid_roi=runtime["profile_sid_roi"],
                profile_sid_roi_lock=runtime["profile_sid_roi_lock"],
                profile_mcq_roi=runtime["profile_mcq_roi"],
                profile_exam_code_roi=runtime["profile_exam_code_roi"],
                profile_sid_row_offsets=runtime["profile_sid_row_offsets"],
                profile_disable_mcq_rescue=runtime["profile_disable_mcq_rescue"],
                profile_mcq_decode=runtime["profile_mcq_decode"],
                profile_threshold_mode=runtime["profile_threshold_mode"],
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

            _inject_handwriting_crop_urls(result)
            bubble_confidence_json_url = _static_omr_url(result.get("bubble_confidence_json"))
            result["bubble_confidence_json_url"] = bubble_confidence_json_url
            persisted_omrid = None
            if matched_one and matched_one.get("omrid") is not None:
                persisted_omrid = int(matched_one["omrid"])
            elif omr_test_id is not None:
                persisted_omrid = int(omr_test_id)
            grade_result_id = _persist_omr_grade_result(
                db,
                uid=uid_checked,
                aid=aid,
                omrid=persisted_omrid,
                source=one_answer_source,
                file_name=safe_exam_name,
                result=result,
            )
            results.append(
                {
                    "file_name": safe_exam_name,
                    "success": True,
                    "data": result,
                    "grade_result_id": grade_result_id,
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
