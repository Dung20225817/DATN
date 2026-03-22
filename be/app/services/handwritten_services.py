"""
Handwritten Services - Component 4 từ Implementation Plan
Enhanced Question Detection và Processing
"""

import os
import cv2
import numpy as np
import re
import json
import importlib
from difflib import SequenceMatcher
from typing import Dict, List, Optional
from app.services.ocr.line_detector import HybridLineDetector
from app.services.llm.grading_engine import Llama3GradingEngine


class QuestionSplitter:
    """Tách đoạn văn thành các câu hỏi riêng biệt"""
    
    # Patterns hỗ trợ
    QUESTION_PATTERNS = [
        r'^Câu\s+(\d+)\s*:?',      # Câu 1:, Câu 1
        r'^(\d+)\.\s+',             # 1., 2.
        r'^Bài\s+(\d+)\s*:?',       # Bài 1:, Bài 1
        r'^Question\s+(\d+)\s*:?',  # Question 1 (nếu có)
    ]
    
    def detect_questions(self, lines: list) -> list:
        """
        Args:
            lines: [{"text": "Câu 1: ABC", "y": 100}, ...]
        
        Returns: [
            {"id": "Câu 1", "lines": [...]},
            {"id": "Câu 2", "lines": [...]}
        ]
        """
        questions = []
        current_question = None
        
        for line in lines:
            text = line['text'].strip()
            
            # Check xem có phải câu hỏi mới không
            question_match = self._is_question_start(text)
            
            if question_match:
                # Lưu câu hỏi cũ
                if current_question:
                    questions.append(current_question)
                
                # Bắt đầu câu mới
                current_question = {
                    'id': question_match['id'],
                    'lines': [line],
                    'start_y': line['y']
                }
            else:
                # Append vào câu hiện tại
                if current_question:
                    current_question['lines'].append(line)
        
        # Lưu câu cuối
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def _is_question_start(self, text: str) -> dict:
        """Kiểm tra xem dòng có phải đầu câu hỏi không"""
        for pattern in self.QUESTION_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                # Extract question ID
                full_match = match.group(0).strip()
                return {'id': full_match.rstrip(':'), 'pattern': pattern}
        
        return None
    
    def assemble_paragraphs(self, questions: list) -> list:
        """
        Ghép các dòng thành đoạn văn hoàn chỉnh
        
        Returns: [
            {"id": "Câu 1", "content": "Đoạn văn hoàn chỉnh..."},
            ...
        ]
        """
        results = []
        
        for q in questions:
            # Loại bỏ dòng đầu nếu chỉ có "Câu X:"
            lines = q['lines']
            first_line = lines[0]['text'].strip()
            
            # Nếu dòng đầu chỉ có "Câu X:" → bỏ qua
            if self._is_question_start(first_line) and len(first_line) < 20:
                content_lines = [l['text'] for l in lines[1:]]
            else:
                # Nếu "Câu X: Nội dung..." → loại bỏ prefix
                first_cleaned = re.sub(r'^(Câu\s+\d+|Bài\s+\d+|\d+\.)\s*:?\s*', '', first_line)
                content_lines = [first_cleaned] + [l['text'] for l in lines[1:]]
            
            # Ghép thành đoạn văn
            paragraph = ' '.join(content_lines).strip()

            # Giữ lại từng dòng OCR riêng để hiển thị
            all_line_texts = [l['text'] for l in lines]

            results.append({
                'id': q['id'],
                'content': paragraph,
                'ocr_lines': all_line_texts
            })
        
        return results


# Available OCR model names
OCR_MODELS = {
    # ── VietOCR (tiền xử lý truyền thống: tách dòng → predict từng dòng) ──
    "vietocr_transformer":  "VietOCR vgg_transformer (chính xác, chậm hơn)",
    # ── VLM (bỏ qua tiền xử lý, đưa toàn bộ ảnh gốc vào model) ──
    "internvl":  "InternVL2-2B (OpenGVLab, OCR đa ngôn ngữ)",
    "openai_gpt4o_mini": "OpenAI GPT-4o mini (Vision API, xử lý qua cloud)",
}

# Models that receive the full raw image (skip line detection pipeline)
VLM_MODELS = {"internvl", "openai_gpt4o_mini"}


def process_handwritten_essay(
    image_path: str,
    answer_key: dict = None,
    ocr_model: str = "vietocr_transformer",
    use_llama3: bool = True,
    gpu: bool = False
) -> dict:
    """
    Xử lý ảnh essay viết tay - Full pipeline

    Args:
        image_path: Đường dẫn ảnh essay
        answer_key: Dict chứa đáp án mẫu {"Câu 1": "...", ...}
        ocr_model: Tên model OCR (xem OCR_MODELS)
        use_llama3: Dùng Llama3 để cleanup và grading
        gpu: Dùng GPU hay không
    
    Returns: {
        "questions": [{"id": "Câu 1", "content": "...", "score": 8.5}, ...],
        "total_score": 85.0,
        "feedback": "...",
        "processing_log": [...]
    }
    """
    
    log = []

    print("\n" + "="*70)
    print("PROCESSING HANDWRITTEN ESSAY")
    print("="*70)

    _img_stem = os.path.splitext(os.path.basename(image_path))[0]
    _croplist_dir = os.path.join("uploads", "croplist", _img_stem)

    if ocr_model in VLM_MODELS:
        # ── VLM path: bỏ qua toàn bộ tiền xử lý, dùng ảnh gốc trực tiếp ──
        full_img = cv2.imread(image_path)
        if full_img is None:
            return {
                "error": "Cannot read image file",
                "questions": [],
                "total_score": 0,
                "processing_log": log
            }
        line_crops = [full_img]
        log.append(f"[VLM mode] Skipped line detection — using full image for '{ocr_model}'")
        print(f"   [VLM] Full image passed directly to model")
    else:
        # ── Traditional path: deep line detection → crop từng dòng → VietOCR ──
        device = 'cuda' if gpu else 'cpu'
        line_detector = None

        # Ưu tiên 1: CRAFT (craft-text-detector)
        try:
            from app.services.ocr.deep_line_detector import CRAFTLineDetector
            line_detector = CRAFTLineDetector(device=device)
            log.append(f"[LineDetect] CRAFT (device={device})")
        except RuntimeError as _e:
            print(f"   [LineDetect] CRAFT không khả dụng: {_e}")

        # Fallback: HybridLineDetector truyền thống (Hough + Projection)
        if line_detector is None:
            line_detector = HybridLineDetector()
            log.append("[LineDetect] HybridLineDetector (fallback)")

        line_crops = line_detector.detect_lines(image_path)
        log.append(f"Detected {len(line_crops)} lines")

        if len(line_crops) == 0:
            raw = cv2.imread(image_path)
            if raw is None:
                return {
                    "error": "Cannot read image file",
                    "questions": [],
                    "total_score": 0,
                    "processing_log": log
                }
            line_crops = [raw]
            log.append("Fallback: treating full image as 1 text block")

    # Save crop list to uploads/croplist/<image_stem>/ for inspection
    if os.path.exists(_croplist_dir):
        for _f in os.listdir(_croplist_dir):
            if _f.endswith(".jpg"):
                os.remove(os.path.join(_croplist_dir, _f))
    os.makedirs(_croplist_dir, exist_ok=True)
    for _i, _crop in enumerate(line_crops):
        cv2.imwrite(os.path.join(_croplist_dir, f"line_{_i+1:03d}.jpg"), _crop)
    log.append(f"[Debug] Saved {len(line_crops)} crops → {_croplist_dir}")
    print(f"   [Crops] Saved to: {_croplist_dir}")

    # Step 2: OCR — select engine based on ocr_model
    try:
        if ocr_model == "vietocr_transformer":
            from app.services.ocr.reader import OCRReader2
            ocr_engine = OCRReader2(gpu=gpu, model_type='handwritten')
            log.append("✅ OCR model: VietOCR vgg_transformer")
        elif ocr_model == "internvl":
            from app.services.ocr.reader import OCRReaderInternVL
            ocr_engine = OCRReaderInternVL(gpu=gpu)
            log.append("✅ OCR model: InternVL2-2B")
        elif ocr_model == "openai_gpt4o_mini":
            from app.services.ocr.reader import OCRReaderOpenAI4oMini
            ocr_engine = OCRReaderOpenAI4oMini()
            log.append("✅ OCR model: OpenAI GPT-4o mini (Vision API)")
        else:
            from app.services.ocr.reader import OCRReader2
            ocr_engine = OCRReader2(gpu=gpu, model_type='handwritten')
            log.append(f"⚠️ Unknown ocr_model '{ocr_model}', falling back to VietOCR vgg_transformer")
    except RuntimeError as e:
        return {
            "error": str(e),
            "questions": [],
            "total_score": 0,
            "processing_log": log + [f"❌ OCR model init failed: {e}"]
        }

    lines_data = []
    for idx, crop in enumerate(line_crops):
        text = ocr_engine.predict(crop)
        lines_data.append({
            'text': text,
            'y': idx * 50,  # Simplified y-coordinate
            'line_num': idx + 1
        })
    
    log.append(f"✅ OCR completed: {len(lines_data)} lines")
    
    # Step 3: Text Cleanup với Llama3 — 1 request duy nhất cho toàn bộ văn bản
    if use_llama3:
        llm_engine = Llama3GradingEngine()
        log.append("✅ Llama3 engine initialized")

        # Gộp tất cả dòng thành 1 đoạn văn, cleanup 1 lần, sau đó tách lại theo dòng
        full_text = "\n".join(line['text'] for line in lines_data)
        cleaned_full = llm_engine.clean_ocr_text(full_text)
        cleaned_lines = cleaned_full.split("\n")

        for i, line in enumerate(lines_data):
            line['text'] = cleaned_lines[i] if i < len(cleaned_lines) else line['text']

        log.append("✅ Text cleanup completed (1 Llama3 call)")
    
    # Step 4: Question Segmentation
    splitter = QuestionSplitter()
    questions = splitter.detect_questions(lines_data)
    
    # Fallback: nếu không nhận ra câu hỏi nào thì gộp tất cả vào "Câu 1"
    if not questions and lines_data:
        questions = [{'id': 'Câu 1', 'lines': lines_data, 'start_y': 0}]
        log.append("⚠️ No question marker found — treating all text as Câu 1")

    paragraphs = splitter.assemble_paragraphs(questions)
    
    log.append(f"✅ Detected {len(paragraphs)} questions")
    
    # Step 5: Grading (nếu có answer key)
    if answer_key:
        llm_engine = Llama3GradingEngine(force_fallback=not use_llama3)
        
        for para in paragraphs:
            if para['id'] in answer_key:
                grading_result = llm_engine.grade_answer(
                    student_answer=para['content'],
                    correct_answer=answer_key[para['id']],
                    question_id=para['id']
                )
                para.update(grading_result)
                # Đính kèm đáp án dùng để chấm vào kết quả
                para['answer_key_used'] = answer_key[para['id']]
        
        total_score = sum(p.get('score', 0) for p in paragraphs)
        mode = "Llama3" if use_llama3 else "regex fallback"
        log.append(f"✅ Grading completed ({mode}): {total_score:.1f} points")
    else:
        total_score = 0
        log.append("⚠️ No grading (answer key not provided)")
    
    return {
        "questions": paragraphs,
        "total_score": total_score,
        "processing_log": log
    }


QUESTION_HEADER_PATTERN = re.compile(r'^\s*C[âa]u\s*(\d+)\s*:?', re.IGNORECASE)


def parse_answer_text_to_questions(answer_text: str) -> Dict[str, str]:
    """Parse text format: 'Cau 1: ...' -> {'Cau 1': '...'} while keeping line breaks."""
    lines = answer_text.splitlines()
    result: Dict[str, List[str]] = {}
    current_q = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        match = QUESTION_HEADER_PATTERN.match(line.strip())
        if match:
            qid = f"Cau {match.group(1)}"
            current_q = qid
            if qid not in result:
                result[qid] = []

            after_header = QUESTION_HEADER_PATTERN.sub("", line.strip(), count=1).lstrip(": ").rstrip()
            if after_header:
                result[qid].append(after_header)
            continue

        if current_q is not None:
            result[current_q].append(line)

    parsed = {
        qid: "\n".join(block).strip()
        for qid, block in result.items()
        if "\n".join(block).strip()
    }

    if parsed:
        return parsed

    fallback = answer_text.strip()
    return {"Cau 1": fallback} if fallback else {}


def _normalize_question_key(question_id: str) -> str:
    match = re.search(r'(\d+)', question_id or "")
    if not match:
        return "Cau 1"
    return f"Cau {match.group(1)}"


def _simple_similarity(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0

    ref_clean = re.sub(r'\s+', ' ', reference.lower()).strip()
    hyp_clean = re.sub(r'\s+', ' ', hypothesis.lower()).strip()
    seq_ratio = SequenceMatcher(None, ref_clean, hyp_clean).ratio()

    ref_tokens = set(re.findall(r'\w+', ref_clean))
    hyp_tokens = set(re.findall(r'\w+', hyp_clean))
    overlap = len(ref_tokens & hyp_tokens) / max(1, len(ref_tokens | hyp_tokens))
    return round((0.6 * overlap + 0.4 * seq_ratio), 4)


def _answer_correctness(reference: str, hypothesis: str, question_id: str) -> float:
    """Try RAGAS answer_correctness; gracefully fallback to lexical similarity."""
    try:
        datasets_module = importlib.import_module("datasets")
        ragas_module = importlib.import_module("ragas")
        ragas_metrics_module = importlib.import_module("ragas.metrics")

        Dataset = getattr(datasets_module, "Dataset")
        evaluate = getattr(ragas_module, "evaluate")
        answer_correctness = getattr(ragas_metrics_module, "answer_correctness")

        ds = Dataset.from_dict({
            "question": [question_id or "Cau hoi"],
            "answer": [hypothesis],
            "ground_truth": [reference],
        })
        result = evaluate(ds, metrics=[answer_correctness])
        score = float(result["answer_correctness"][0])
        return max(0.0, min(1.0, score))
    except Exception:
        return _simple_similarity(reference, hypothesis)


def _align_student_answers(answer_questions: Dict[str, str], recognized_text: str) -> Dict[str, str]:
    """Align OCR text into per-question blocks using markers, then fallback sequentially."""
    student_questions = parse_answer_text_to_questions(recognized_text)
    normalized_student = {
        _normalize_question_key(k): v
        for k, v in student_questions.items()
        if (v or "").strip()
    }

    # If OCR already has >= 2 question markers, trust parsed mapping.
    if len(normalized_student) >= 2:
        return normalized_student

    answer_keys = [_normalize_question_key(k) for k in answer_questions.keys()]
    question_count = len(answer_keys)
    if question_count <= 1:
        fallback_text = recognized_text.strip()
        return {answer_keys[0] if answer_keys else "Cau 1": fallback_text}

    # Fallback: split by blank-line paragraphs and map in answer order.
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", recognized_text) if b.strip()]
    if len(blocks) >= question_count:
        return {answer_keys[idx]: blocks[idx] for idx in range(question_count)}

    # Final fallback: keep full OCR text for first question and empty for others.
    aligned = {k: "" for k in answer_keys}
    aligned[answer_keys[0]] = recognized_text.strip()
    return aligned


def _build_answer_correctness_reason(reference: str, hypothesis: str, correctness: float) -> str:
    """Generate a concise reason explaining the Answer Correctness score."""
    ref_clean = re.sub(r'\s+', ' ', (reference or "").strip())
    hyp_clean = re.sub(r'\s+', ' ', (hypothesis or "").strip())

    if not hyp_clean:
        return "Bài làm không có nội dung cho câu này nên điểm rất thấp."

    ref_tokens = re.findall(r'\w+', ref_clean.lower())
    hyp_tokens = re.findall(r'\w+', hyp_clean.lower())

    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)

    common = [w for w in ref_tokens if w in hyp_set]
    missing = [w for w in ref_tokens if w not in hyp_set]

    # Keep order and uniqueness for readability.
    common_unique = list(dict.fromkeys(common))[:6]
    missing_unique = list(dict.fromkeys(missing))[:6]

    if correctness >= 0.8:
        level_text = "Nội dung bài làm bám sát đáp án"
    elif correctness >= 0.5:
        level_text = "Bài làm đúng một phần"
    else:
        level_text = "Bài làm lệch đáng kể so với đáp án"

    fragments = [f"{level_text} ({round(correctness * 100, 1)}%)."]
    if common_unique:
        fragments.append(f"Ý trùng khớp: {', '.join(common_unique)}.")
    if missing_unique:
        fragments.append(f"Ý còn thiếu/chưa rõ: {', '.join(missing_unique)}.")

    return " ".join(fragments)


def _cleanup_extracted_text(raw_text: str) -> str:
    """Normalize OCR text while preserving logical line boundaries."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in (raw_text or "").splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty).strip()


def _extract_claims(text: str) -> List[str]:
    """Extract independent claims from an answer block."""
    if not (text or "").strip():
        return []

    chunks = re.split(r"\n+|(?<=[\.!?;:])\s+", text)
    claims: List[str] = []
    seen = set()
    for chunk in chunks:
        item = chunk.strip()
        item = re.sub(r"^[-*•\u2022\d\)\.\s]+", "", item)
        item = re.sub(r"\s+", " ", item).strip(" .;:-")
        if not item:
            continue
        if len(item) < 12 and len(re.findall(r"\w+", item)) < 3:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(item)

    if claims:
        return claims
    return [re.sub(r"\s+", " ", text).strip()]


def _claim_similarity(reference_claim: str, student_claim: str) -> float:
    ref_clean = re.sub(r"\s+", " ", (reference_claim or "").lower()).strip()
    stu_clean = re.sub(r"\s+", " ", (student_claim or "").lower()).strip()
    if not ref_clean and not stu_clean:
        return 1.0
    if not ref_clean or not stu_clean:
        return 0.0

    seq_ratio = SequenceMatcher(None, ref_clean, stu_clean).ratio()
    ref_tokens = set(re.findall(r"\w+", ref_clean))
    stu_tokens = set(re.findall(r"\w+", stu_clean))
    overlap = len(ref_tokens & stu_tokens) / max(1, len(ref_tokens | stu_tokens))
    return round(0.55 * overlap + 0.45 * seq_ratio, 4)


def _analyze_claims(reference_answer: str, student_answer: str) -> dict:
    """Classify student claims into TP/FP/FN by semantic overlap approximation."""
    reference_claims = _extract_claims(reference_answer)
    student_claims = _extract_claims(student_answer)

    if not reference_claims:
        return {
            "tp": [],
            "fp": student_claims,
            "fn": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    used_ref = set()
    tp = []
    fp = []
    threshold = 0.52

    for stu_claim in student_claims:
        best_idx = -1
        best_score = 0.0
        for idx, ref_claim in enumerate(reference_claims):
            if idx in used_ref:
                continue
            score = _claim_similarity(ref_claim, stu_claim)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= threshold:
            used_ref.add(best_idx)
            tp.append(stu_claim)
        else:
            fp.append(stu_claim)

    fn = [ref for idx, ref in enumerate(reference_claims) if idx not in used_ref]

    tp_count = len(tp)
    fp_count = len(fp)
    fn_count = len(fn)
    precision = tp_count / max(1, tp_count + fp_count)
    recall = tp_count / max(1, tp_count + fn_count)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _build_detailed_reason(analysis: dict, score_10: float) -> str:
    tp = analysis.get("tp", [])
    fp = analysis.get("fp", [])
    fn = analysis.get("fn", [])
    precision = analysis.get("precision", 0.0)
    recall = analysis.get("recall", 0.0)

    reason_parts = [
        f"Điểm {score_10:.1f}/10 dựa trên TP={len(tp)}, FP={len(fp)}, FN={len(fn)}; precision={precision:.2f}, recall={recall:.2f}."
    ]
    if tp:
        reason_parts.append(f"Ý đúng nổi bật: {'; '.join(tp[:3])}.")
    if fp:
        reason_parts.append(f"Ý sai/thừa: {'; '.join(fp[:3])}.")
    if fn:
        reason_parts.append(f"Ý thiếu quan trọng: {'; '.join(fn[:3])}.")
    if not fp and not fn and tp:
        reason_parts.append("Bài làm bao phủ gần đầy đủ các ý cốt lõi của đáp án.")
    return " ".join(reason_parts)


_OPENAI_CLIENT = None


def _get_openai_client():
    """Create cached OpenAI client for semantic analysis."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI")
    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _safe_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _f1_from_tpfpfn(tp_count: int, fp_count: int, fn_count: int) -> float:
    precision = tp_count / max(1, tp_count + fp_count)
    recall = tp_count / max(1, tp_count + fn_count)
    if (precision + recall) == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def _llm_semantic_ragas_analysis(
    question_id: str,
    reference_answer: str,
    student_answer: str,
    model_name: str = "gpt-4o-mini",
) -> Optional[dict]:
    """Use LLM to perform semantic TP/FP/FN analysis and return strict JSON shape."""
    try:
        client = _get_openai_client()
    except Exception:
        return None

    system_prompt = (
        "Ban la he thong cham diem OCR theo RAGAS. "
        "Danh gia theo NGU NGHIA, khong yeu cau trung tu vung. "
        "Van phong mo rong hop ly khong tinh FP. "
        "Tra ve JSON hop le duy nhat."
    )

    user_prompt = (
        "Hay thuc hien: OCR text da co san, tach atomic facts, phan loai TP/FP/FN theo ngu nghia, "
        "tinh answer correctness score tren thang 10 va ly do.\n\n"
        f"Question: {question_id}\n"
        f"Ground Truth (dap an chuan):\n{reference_answer}\n\n"
        f"Student Answer:\n{student_answer}\n\n"
        "Yeu cau output JSON duy nhat theo schema:\n"
        "{\n"
        "  \"extracted_text\": \"...\",\n"
        "  \"ragas_analysis\": {\n"
        "    \"true_positives\": [\"...\"],\n"
        "    \"false_positives\": [\"...\"],\n"
        "    \"false_negatives\": [\"...\"]\n"
        "  },\n"
        "  \"answer_correctness_score\": 0.0,\n"
        "  \"answer_correctness_reason\": \"...\"\n"
        "}\n"
        "Luu y: score phu hop voi TP/FP/FN va danh gia semantic."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        ragas_analysis = parsed.get("ragas_analysis", {}) if isinstance(parsed, dict) else {}
        tp = _safe_list(ragas_analysis.get("true_positives"))
        fp = _safe_list(ragas_analysis.get("false_positives"))
        fn = _safe_list(ragas_analysis.get("false_negatives"))

        score_raw = parsed.get("answer_correctness_score", 0.0)
        try:
            score_10 = float(score_raw)
        except Exception:
            score_10 = 0.0
        score_10 = max(0.0, min(10.0, score_10))

        reason = str(parsed.get("answer_correctness_reason", "")).strip()
        extracted_text = str(parsed.get("extracted_text", student_answer or "")).strip()

        return {
            "extracted_text": extracted_text,
            "ragas_analysis": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            },
            "answer_correctness_score": score_10,
            "answer_correctness_reason": reason,
        }
    except Exception:
        return None


def process_handwritten_with_llm(
    image_paths: str | List[str],
    answer_text: str,
    use_answer_correctness: bool = True,
    ocr_model: str = "openai_gpt4o_mini",
) -> dict:
    """Read ordered image set with LLM OCR, then score each question by TP/FP/FN analysis."""
    log = []

    model_map = {
        "openai_gpt4o": "gpt-4o",
        "openai_gpt4o_mini": "gpt-4o-mini",
    }
    selected_model = model_map.get(str(ocr_model).strip().lower(), "gpt-4o-mini")

    try:
        from app.services.ocr.reader import OCRReaderOpenAI4oMini
        ocr_engine = OCRReaderOpenAI4oMini(model_name=selected_model)
        log.append(f"LLM OCR: OpenAI {selected_model}")
    except Exception as exc:
        return {
            "error": f"Khong the khoi tao OCR model {selected_model}: {exc}",
            "questions": [],
            "total_score": 0,
            "total_max_score": 0,
            "recognized_text": "",
            "processing_log": [f"OCR init failed: {exc}"],
        }

    ordered_paths = image_paths if isinstance(image_paths, list) else [image_paths]
    ocr_blocks: List[str] = []
    for idx, path in enumerate(ordered_paths):
        raw_image = cv2.imread(path)
        if raw_image is None:
            log.append(f"Khong the doc anh thu {idx + 1}: {path}")
            continue

        page_text = (ocr_engine.predict(raw_image) or "").strip()
        if page_text:
            ocr_blocks.append(page_text)
        log.append(f"OCR page {idx + 1}: {len(page_text)} chars")

    recognized_text = _cleanup_extracted_text("\n\n".join(ocr_blocks))
    if not recognized_text:
        return {
            "questions": [],
            "total_score": 0,
            "total_max_score": 0,
            "recognized_text": "",
            "processing_log": log + ["Khong trich xuat duoc noi dung tu bo anh"],
        }

    log.append(f"Merged OCR text length: {len(recognized_text)}")

    answer_questions = parse_answer_text_to_questions(answer_text)
    student_questions = _align_student_answers(answer_questions, recognized_text)
    log.append(f"Detected {len(answer_questions)} answer question(s) for grading")

    q_results = []
    total_score = 0.0
    total_max_score = max(1, len(answer_questions)) * 10.0

    for qid, reference_answer in answer_questions.items():
        normalized = _normalize_question_key(qid)
        student_answer = student_questions.get(normalized, "")
        analysis = _analyze_claims(reference_answer, student_answer)
        llm_json = _llm_semantic_ragas_analysis(
            question_id=qid,
            reference_answer=reference_answer,
            student_answer=student_answer,
            model_name=selected_model,
        )

        if llm_json:
            llm_tp = llm_json.get("ragas_analysis", {}).get("true_positives", [])
            llm_fp = llm_json.get("ragas_analysis", {}).get("false_positives", [])
            llm_fn = llm_json.get("ragas_analysis", {}).get("false_negatives", [])

            llm_precision = len(llm_tp) / max(1, len(llm_tp) + len(llm_fp))
            llm_recall = len(llm_tp) / max(1, len(llm_tp) + len(llm_fn))
            llm_f1 = _f1_from_tpfpfn(len(llm_tp), len(llm_fp), len(llm_fn))

            analysis = {
                "tp": llm_tp,
                "fp": llm_fp,
                "fn": llm_fn,
                "precision": round(llm_precision, 4),
                "recall": round(llm_recall, 4),
                "f1": llm_f1,
            }

        if use_answer_correctness:
            # Blend semantic similarity metric with TP/FP/FN F1.
            ragas_score = _answer_correctness(reference_answer, student_answer, question_id=qid)
            correctness = round(0.5 * ragas_score + 0.5 * analysis.get("f1", 0.0), 4)
        else:
            correctness = analysis.get("f1", 0.0)

        score = round(correctness * 10.0, 1)
        total_score += score
        correctness_reason = (
            llm_json.get("answer_correctness_reason", "").strip()
            if llm_json and llm_json.get("answer_correctness_reason")
            else _build_detailed_reason(analysis, score)
        )

        strict_json_output = {
            "extracted_text": (llm_json or {}).get("extracted_text", student_answer),
            "ragas_analysis": {
                "true_positives": analysis.get("tp", []),
                "false_positives": analysis.get("fp", []),
                "false_negatives": analysis.get("fn", []),
            },
            "answer_correctness_score": score,
            "answer_correctness_reason": correctness_reason,
        }

        q_results.append({
            "id": qid.replace("Cau", "Câu"),
            "content": student_answer,
            "score": score,
            "max_score": 10,
            "answer_correctness": correctness,
            "answer_correctness_reason": correctness_reason,
            "feedback": f"Answer Correctness: {round(correctness * 100, 1)}%",
            "answer_key_used": reference_answer,
            "ragas_analysis": {
                "true_positives": analysis.get("tp", []),
                "false_positives": analysis.get("fp", []),
                "false_negatives": analysis.get("fn", []),
                "precision": analysis.get("precision", 0.0),
                "recall": analysis.get("recall", 0.0),
                "f1": analysis.get("f1", 0.0),
            },
            "strict_json_output": strict_json_output,
        })

    log.append(f"Scoring mode: {'AnswerCorrectness' if use_answer_correctness else 'SimpleSimilarityFallback'}")

    return {
        "questions": q_results,
        "total_score": round(total_score, 1),
        "total_max_score": round(total_max_score, 1),
        "recognized_text": recognized_text,
        "processing_log": log,
    }
