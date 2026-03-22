"""
Llama3 Content Grading Engine - Component 3 từ Implementation Plan
Sử dụng Llama3 để chấm điểm dựa trên nội dung
"""

import json
import re

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not installed. Install with: pip install ollama")


class Llama3GradingEngine:
    """Sử dụng Llama3 để chấm điểm dựa trên nội dung"""
    
    def __init__(self, model_name='llama3', force_fallback: bool = False):
        self.model = model_name
        self.fallback_mode = force_fallback or not OLLAMA_AVAILABLE

        if self.fallback_mode:
            print("⚠️ Running in FALLBACK mode (no Llama3). Using regex-based grading.")
        else:
            print(f"✅ Llama3 Grading Engine initialized with model: {model_name}")
    
    def grade_answer(self, student_answer: str, correct_answer, question_id: str) -> dict:
        """
        Chấm điểm 1 câu trả lời.

        correct_answer can be:
          - str  : plain sample answer text (classic mode)
          - dict : rubric entry from extract_criteria_format(), e.g.
                   {'question_text': 'Câu 1', 'total_points': 10,
                    'criteria': {'Nội dung': {'points': 6.0, 'details': ['ý 1', 'ý 2']}, ...}}

        Returns: {
            "score": 8.5,
            "max_score": 10,
            "matched_points": [...],
            "missing_points": [...],
            "feedback": "..."
        }
        """
        # --- Rubric-based grading ---
        if isinstance(correct_answer, dict) and 'criteria' in correct_answer:
            return self._grade_with_rubric(student_answer, correct_answer, question_id)

        # --- Classic string-based grading ---
        if self.fallback_mode:
            return self._fallback_grading(student_answer, correct_answer, question_id)

        return self._llama3_grading(student_answer, correct_answer, question_id)

    def _llama3_grading(self, student_answer: str, correct_answer: str, question_id: str) -> dict:
        """Llama3 grading for classic plain-text answer."""
        prompt = f"""Bạn là giáo viên chấm bài kiểm tra tiếng Việt.

ĐÁP ÁN CHUẨN ({question_id}):
{correct_answer}

CÂU TRẢ LỜI CỦA HỌC SINH:
{student_answer}

NHIỆM VỤ:
1. Phân tích đáp án chuẩn thành các ý chính (bullet points)
2. Kiểm tra câu trả lời của học sinh có đủ các ý chính không
3. Chấm điểm từ 0-10 dựa trên tỷ lệ ý đạt được
4. Đưa ra feedback ngắn gọn

Trả lời theo định dạng JSON sau:
{{
  "main_points": ["Ý 1", "Ý 2", "Ý 3"],
  "matched_points": ["Ý 1", "Ý 2"],
  "missing_points": ["Ý 3"],
  "score": 6.7,
  "max_score": 10,
  "feedback": "Bài làm tốt, đã nắm được 2/3 ý chính. Thiếu ý về..."
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            result = json.loads(response['message']['content'])
            return result
        except Exception as e:
            print(f"⚠️ Llama3 grading error: {e}. Falling back to regex grading.")
            return self._fallback_grading(student_answer, correct_answer, question_id)

    # ------------------------------------------------------------------
    # RUBRIC-BASED GRADING
    # ------------------------------------------------------------------

    def _build_criteria_breakdown(self, criteria: dict, ý_items: list, matched_points: list) -> dict:
        """Build per-criterion breakdown dict from matched/missing point lists."""
        matched_set = set(matched_points)
        breakdown = {}
        for c_name, c_data in criteria.items():
            details = c_data.get('details', [])
            if not details:
                continue
            pts = c_data.get('points', 0)
            c_matched = [item['content'] for item in ý_items
                         if item['criterion'] == c_name and item['content'] in matched_set]
            c_missing = [item['content'] for item in ý_items
                         if item['criterion'] == c_name and item['content'] not in matched_set]
            c_score = sum(item['points'] for item in ý_items
                          if item['criterion'] == c_name and item['content'] in matched_set)
            breakdown[c_name] = {
                'max_points': round(pts, 1),
                'score': round(c_score, 1),
                'matched': c_matched,
                'missing': c_missing,
                'total_ý': len(details)
            }
        return breakdown

    def _grade_with_rubric(self, student_answer: str, rubric: dict, question_id: str) -> dict:
        """
        Grade against a structured rubric (from extract_criteria_format).
        Builds a flat list of all ý across all criteria then scores.
        """
        total_points = rubric.get('total_points', 10)
        criteria = rubric.get('criteria', {})

        # Collect all ý with their per-ý points
        ý_items = []
        for c_name, c_data in criteria.items():
            details = c_data.get('details', [])
            if not details:
                continue
            pts_per_ý = c_data.get('points', 0) / len(details)
            for detail in details:
                ý_items.append({
                    'criterion': c_name,
                    'content': detail,
                    'points': round(pts_per_ý, 3)
                })

        if not ý_items:
            return {
                'main_points': [],
                'matched_points': [],
                'missing_points': [],
                'score': 0.0,
                'max_score': total_points,
                'feedback': 'Không có ý nào được định nghĩa trong đáp án.'
            }

        if self.fallback_mode:
            return self._fallback_grading_with_rubric(student_answer, ý_items, total_points, question_id, criteria)

        # --- Llama3 rubric prompt ---
        point_list_str = '\n'.join(
            f"  [{i+1}] ({item['criterion']}) {item['content']}" for i, item in enumerate(ý_items)
        )
        prompt = f"""Bạn là giáo viên chấm bài essay tiếng Việt.

CÁC Ý CHÍNH CẦN ĐẠT ({question_id}):
{point_list_str}

BÀI LÀM CỦA HỌC SINH:
{student_answer}

NHIỆM VỤ:
1. Kiểm tra bài làm có chứa từng ý không (theo số thứ tự [1]...[{len(ý_items)}])
2. Mỗi ý đạt được = {{"{round(total_points / len(ý_items), 2)}điểm"}}
3. Điểm tổng = số ý đạt × điểm mỗi ý (tối đa {total_points} điểm)

Trả lời JSON:
{{
  "matched_indices": [1, 2],
  "missing_indices": [3],
  "score": 6.7,
  "max_score": {total_points},
  "feedback": "Bài làm đạt ý 1, 2 nhưng thiếu ý 3..."
}}"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'
            )
            raw = json.loads(response['message']['content'])
            matched_idx = set(raw.get('matched_indices', []))
            missing_idx = set(raw.get('missing_indices', []))
            matched_pts = [ý_items[i-1]['content'] for i in sorted(matched_idx) if 1 <= i <= len(ý_items)]
            missing_pts = [ý_items[i-1]['content'] for i in sorted(missing_idx) if 1 <= i <= len(ý_items)]
            return {
                'main_points': [item['content'] for item in ý_items],
                'matched_points': matched_pts,
                'missing_points': missing_pts,
                'score': round(float(raw.get('score', 0)), 1),
                'max_score': total_points,
                'feedback': raw.get('feedback', ''),
                'criteria_breakdown': self._build_criteria_breakdown(criteria, ý_items, matched_pts)
            }
        except Exception as e:
            print(f"⚠️ Llama3 rubric grading error: {e}. Falling back.")
            return self._fallback_grading_with_rubric(student_answer, ý_items, total_points, question_id, criteria)

    def _fallback_grading_with_rubric(self, student_answer: str, ý_items: list,
                                       total_points: float, question_id: str,
                                       criteria: dict = None) -> dict:
        """
        Regex fallback for rubric grading.
        An ý is considered matched if >50% of its significant words
        (len ≥ 3 chars) appear in the student answer.
        """
        student_words = set(re.findall(r'\w{3,}', student_answer.lower()))
        matched = []
        missing = []

        for item in ý_items:
            ý_words = set(re.findall(r'\w{3,}', item['content'].lower()))
            if not ý_words:
                continue
            overlap_ratio = len(ý_words & student_words) / len(ý_words)
            if overlap_ratio >= 0.5:
                matched.append(item['content'])
            else:
                missing.append(item['content'])

        total_ý = len(ý_items)
        score = (len(matched) / total_ý * total_points) if total_ý > 0 else 0.0

        missing_preview = (f" Thiếu: «{missing[0][:40]}»" if missing else "")
        return {
            'main_points': [item['content'] for item in ý_items],
            'matched_points': matched,
            'missing_points': missing,
            'score': round(score, 1),
            'max_score': total_points,
            'feedback': f"Đạt {len(matched)}/{total_ý} ý.{missing_preview}",
            'criteria_breakdown': self._build_criteria_breakdown(criteria or {}, ý_items, matched)
        }
    
    def _fallback_grading(self, student_answer: str, correct_answer: str, question_id: str) -> dict:
        """
        Fallback grading khi không có Llama3
        Dùng keyword matching và length comparison
        """
        # Simple keyword extraction
        correct_words = set(re.findall(r'\w+', correct_answer.lower()))
        student_words = set(re.findall(r'\w+', student_answer.lower()))
        
        # Calculate overlap
        matched_words = correct_words & student_words
        missing_words = correct_words - student_words
        
        # Simple scoring
        if len(correct_words) == 0:
            score = 5.0
        else:
            score = (len(matched_words) / len(correct_words)) * 10
        
        # Length factor
        len_ratio = min(len(student_answer) / max(len(correct_answer), 1), 1.0)
        score = score * (0.7 + 0.3 * len_ratio)  # Penalty if too short
        
        return {
            "main_points": [f"Keyword: {w}" for w in list(correct_words)[:3]],
            "matched_points": [f"Found: {w}" for w in list(matched_words)[:3]],
            "missing_points": [f"Missing: {w}" for w in list(missing_words)[:3]],
            "score": round(score, 1),
            "max_score": 10,
            "feedback": f"Regex-based grading: {len(matched_words)}/{len(correct_words)} keywords matched"
        }
    
    
    def clean_ocr_text(self, raw_text: str) -> str:
        """
        Làm sạch text OCR bằng Llama3
        """
        if self.fallback_mode:
            # Simple regex cleanup
            cleaned = re.sub(r'\s+', ' ', raw_text)  # Remove extra spaces
            cleaned = cleaned.strip()
            return cleaned
        
        prompt = f"""Sửa lỗi từ vựng trong văn bản tiếng Việt sau:

VĂN BẢN GỐC (có lỗi từ vựng):
{raw_text}

YÊU CẦU:
- Sửa lỗi chính tả do viết nhầm
- Thêm dấu câu phù hợp
- Viết hoa đúng quy tắc
- GIỮ NGUYÊN SỐ DÒNG (không gộp hay bỏ dòng)
- Văn bản có thể bị nhầm lẫn giữa các chữ cái hoặc chữ số giống nhau (vd: "c" và "e", "n" và "m", "0" và "o"), hãy sửa lại cho đúng

LƯU Ý: CHỈ TRẢ VỀ VĂN BẢN ĐÃ SỬA, KHÔNG GIẢI THÍCH."""
        
        try:
            response = ollama.chat(
                model=self.model, 
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"⚠️ Text cleanup error: {e}")
            # Fallback to regex cleanup
            cleaned = re.sub(r'\s+', ' ', raw_text)
            return cleaned.strip()
