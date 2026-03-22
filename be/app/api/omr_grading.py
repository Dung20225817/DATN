# app/api/omr_grading.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import json
import re
from typing import Optional, List, Tuple

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
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

# Import service vừa viết
from app.services.omr.omr_service import process_omr_exam, generate_omr_template, suggest_omr_crop_quad
from app.db_connect import get_db
from app.db.ocr_tables import OMRTest

router = APIRouter()

BASE_OMR_DIR = "uploads/omr"
BASE_OMR_ANSWER_KEY_DIR = "uploads/answer_keys/omr"
BASE_OMR_TEMPLATE_DIR = "uploads/omr_templates"
os.makedirs(BASE_OMR_DIR, exist_ok=True)
os.makedirs(BASE_OMR_ANSWER_KEY_DIR, exist_ok=True)
os.makedirs(BASE_OMR_TEMPLATE_DIR, exist_ok=True)

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
        uid_checked = None
        if uid is not None:
            uid_checked = _validate_uid(uid)

        # 1. Parse đáp án: ưu tiên đề OMR đã lưu, tiếp theo auto nhận diện đề, fallback nhập tay/file.
        if omr_test_id is not None:
            uid_checked = _validate_uid(uid)
            answer_key, answer_source, num_questions = _resolve_answer_key_from_omr_test(
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
            matched_test = None
            detect_probe = None
        else:
            answer_key, answer_source, num_questions = _resolve_shared_answer_key(
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
            answer_key, answer_source, num_questions, matched_test, detect_probe = await _resolve_answer_key_auto_from_sheet(
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
        )

        # 4. Kiểm tra lỗi từ service
        if "error" in result:
               print(f"[OMR][GRADE][400] {result.get('error')}")
               return JSONResponse(status_code=400, content=result)

        # 5. Trả về kết quả JSON
        return JSONResponse(content={
            "message": "Chấm điểm thành công",
            "data": result,
            "image_url": f"/static/omr/{result['result_image']}", # Đường dẫn giả định nếu bạn config static files
            "sid_crop_url": f"/static/omr/{result['sid_crop_image']}" if result.get('sid_crop_image') else None,
            "mcq_crop_url": f"/static/omr/{result['mcq_crop_image']}" if result.get('mcq_crop_image') else None,
            "answer_source": answer_source,
            "parsed_num_questions": num_questions,
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
        uid_checked = None
        if uid is not None:
            uid_checked = _validate_uid(uid)

        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="Vui lòng tải lên ít nhất 1 ảnh")
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Mỗi lần gửi tối đa 50 ảnh")

        auto_mode = False
        if omr_test_id is not None:
            uid_checked = _validate_uid(uid)
            answer_key, answer_source, num_questions = _resolve_answer_key_from_omr_test(
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
            matched_global_test = None
        else:
            answer_key, answer_source, num_questions = _resolve_shared_answer_key(
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
                    one_answer_key, one_answer_source, one_num_questions, matched_record, detect_probe = await _resolve_answer_key_auto_from_sheet(
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
                questions=one_num_questions,
                choices=num_choices,
                rows_per_block=rows_per_block,
                num_blocks=num_blocks,
                student_id_digits=student_id_digits,
                sid_has_write_row=sid_has_write_row,
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
            results.append(
                {
                    "file_name": safe_exam_name,
                    "success": True,
                    "data": result,
                    "image_url": f"/static/omr/{result['result_image']}",
                    "sid_crop_url": f"/static/omr/{result['sid_crop_image']}" if result.get("sid_crop_image") else None,
                    "mcq_crop_url": f"/static/omr/{result['mcq_crop_image']}" if result.get("mcq_crop_image") else None,
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
                "parsed_num_questions": num_questions,
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