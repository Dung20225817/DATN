"""
Handwritten API - 2-step workflow
1) Upload answer file -> preview extracted text
2) Save answer into ocr_test table
3) Grade essays with LLM vision OCR + similarity scoring
"""

from datetime import datetime
import json
import os
import re
import shutil

from docx import Document
import pdfplumber
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from app.db_connect import get_db
from app.db.ocr_tables import OCRTest
from app.services.handwritten_services import (
    parse_answer_text_to_questions,
    process_handwritten_with_llm,
)


class UnicodeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


JSONResponse = UnicodeJSONResponse
router = APIRouter()

BASE_DIR = "uploads/handwritten"
ANSWER_KEY_DIR = "uploads/answer_keys"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(ANSWER_KEY_DIR, exist_ok=True)


def _extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    if ext == ".doc":
        try:
            import textract  # type: ignore

            raw = textract.process(file_path)
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            raise ValueError("File .doc hiện chưa đọc được trên máy chủ này. Vui lòng chuyển sang .docx")

    if ext == ".pdf":
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n\n".join(texts)

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError("Chỉ hỗ trợ file .docx, .pdf hoặc .txt")


@router.post("/upload-answer-key")
async def upload_answer_key(
    uid: int = Form(...),
    answer_key_file: UploadFile = File(...),
):
    """Upload file đáp án và trả về nội dung đã bóc tách để người dùng review."""
    if not answer_key_file.filename:
        raise HTTPException(status_code=400, detail="Thiếu tên file đáp án")

    ext = os.path.splitext(answer_key_file.filename)[1].lower()
    if ext not in {".doc", ".docx", ".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .doc, .docx, .pdf, .txt")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-. ]", "_", answer_key_file.filename)
    saved_name = f"draft_{uid}_{timestamp}_{safe_name}"
    file_path = os.path.join(ANSWER_KEY_DIR, saved_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(answer_key_file.file, buffer)

    try:
        extracted_text = _extract_text_from_file(file_path).replace("\r\n", "\n").replace("\r", "\n").strip()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể đọc file đáp án: {exc}")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="Không trích xuất được nội dung từ file đáp án")

    parsed_questions = parse_answer_text_to_questions(extracted_text)

    return JSONResponse(
        {
            "status": "success",
            "message": "Đã tải đáp án, vui lòng kiểm tra và bấm Lưu đáp án",
            "file_name": answer_key_file.filename,
            "ocr_answer": extracted_text,
            "question_count": len(parsed_questions),
            "question_keys": list(parsed_questions.keys()),
        }
    )


@router.post("/save-answer-key")
async def save_answer_key(
    uid: int = Form(...),
    ocr_name: str = Form(...),
    ocr_answer: str = Form(...),
):
    """Lưu đáp án vào bảng ocr_test."""
    if not ocr_answer.strip():
        raise HTTPException(status_code=400, detail="Nội dung đáp án trống")

    db = next(get_db())
    try:
        record = OCRTest(
            uuid=uid,
            ocr_name=ocr_name.strip(),
            ocr_answer=ocr_answer,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "status": "success",
            "message": "Đã lưu đáp án",
            "ocrid": record.ocrid,
            "ocr_name": record.ocr_name,
        }
    finally:
        db.close()


@router.get("/answer-keys/{uid}")
async def list_answer_keys(uid: int):
    db = next(get_db())
    try:
        rows = (
            db.query(OCRTest)
            .filter(OCRTest.uuid == uid)
            .order_by(OCRTest.created_at.desc())
            .all()
        )

        return {
            "answer_keys": [
                {
                    "ocrid": row.ocrid,
                    "ocr_name": row.ocr_name,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        }
    finally:
        db.close()


@router.get("/answer-key/{ocrid}")
async def get_answer_key(ocrid: int, uid: int):
    db = next(get_db())
    try:
        row = (
            db.query(OCRTest)
            .filter(OCRTest.ocrid == ocrid, OCRTest.uuid == uid)
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Không tìm thấy đáp án")

        return {
            "ocrid": row.ocrid,
            "ocr_name": row.ocr_name,
            "ocr_answer": row.ocr_answer,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
    finally:
        db.close()


@router.get("/answer-key/{ocrid}/download")
async def download_answer_key(ocrid: int, uid: int):
    db = next(get_db())
    try:
        row = (
            db.query(OCRTest)
            .filter(OCRTest.ocrid == ocrid, OCRTest.uuid == uid)
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Khong tim thay dap an")

        base_name = re.sub(r"[^\w\-. ]", "_", (row.ocr_name or "answer_key").strip())
        if not base_name.lower().endswith(".txt"):
            base_name = f"{base_name}.txt"

        return PlainTextResponse(
            content=row.ocr_answer or "",
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{base_name}"'},
        )
    finally:
        db.close()


@router.delete("/answer-key/{ocrid}")
async def delete_answer_key(ocrid: int, uid: int):
    db = next(get_db())
    try:
        row = (
            db.query(OCRTest)
            .filter(OCRTest.ocrid == ocrid, OCRTest.uuid == uid)
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Không tìm thấy đáp án")

        db.delete(row)
        db.commit()
        return {"status": "success", "message": "Đã xóa đáp án"}
    finally:
        db.close()


@router.post("/upload")
async def upload_handwritten_essay(
    uid: int = Form(...),
    essay_images: list[UploadFile] = File(...),
    ocrid: int = Form(...),
    use_answer_correctness: bool = Form(True),
    use_ragas: bool | None = Form(None),
    ocr_model: str = Form("openai_gpt4o_mini"),
):
    """Chấm điểm bài viết tay: OCR theo thứ tự ảnh, ghép nội dung rồi chấm theo từng câu."""
    db = next(get_db())
    try:
        answer_record = (
            db.query(OCRTest)
            .filter(OCRTest.ocrid == ocrid, OCRTest.uuid == uid)
            .first()
        )
        if not answer_record:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy đáp án với ID {ocrid}")
    finally:
        db.close()

    if not essay_images:
        raise HTTPException(status_code=400, detail="Vui lòng tải ít nhất 1 ảnh bài làm")
    if len(essay_images) > 30:
        raise HTTPException(status_code=400, detail="Moi lan cham toi da 30 anh bai lam")

    valid_models = {"openai_gpt4o", "openai_gpt4o_mini"}
    if ocr_model not in valid_models:
        raise HTTPException(status_code=400, detail="ocr_model khong hop le")

    user_dir = os.path.join(BASE_DIR, str(uid))
    os.makedirs(user_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(user_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)

    processing_log = []
    scoring_flag = use_answer_correctness if use_ragas is None else use_ragas
    saved_paths: list[str] = []
    file_names: list[str] = []

    for idx, essay_img in enumerate(essay_images):
        ext = os.path.splitext(essay_img.filename or "")[1].lower() or ".jpg"
        saved_name = f"essay_{idx + 1:03d}{ext}"
        saved_path = os.path.join(session_dir, saved_name)

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(essay_img.file, buffer)
        saved_paths.append(saved_path)
        file_names.append(essay_img.filename or saved_name)

    result = process_handwritten_with_llm(
        image_paths=saved_paths,
        answer_text=answer_record.ocr_answer,
        use_answer_correctness=scoring_flag,
        ocr_model=ocr_model,
    )
    result["file"] = ", ".join(file_names)
    processing_log.extend(result.get("processing_log", []))
    results = [result]

    return JSONResponse(
        {
            "status": "success",
            "message": f"Processed {len(essay_images)} image(s)",
            "results": results,
            "processing_log": processing_log,
            "session_dir": session_dir,
            "answer_key": {
                "ocrid": answer_record.ocrid,
                "ocr_name": answer_record.ocr_name,
            },
        }
    )


@router.get("/sessions/{uid}")
async def list_user_sessions(uid: int):
    user_dir = os.path.join(BASE_DIR, str(uid))
    if not os.path.exists(user_dir):
        return {"sessions": []}

    sessions = []
    for session_name in os.listdir(user_dir):
        session_path = os.path.join(user_dir, session_name)
        if os.path.isdir(session_path):
            images = [f for f in os.listdir(session_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            sessions.append(
                {
                    "session_id": session_name,
                    "image_count": len(images),
                    "path": session_path,
                }
            )

    return {"sessions": sessions}
