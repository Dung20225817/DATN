import re
from typing import List, Optional

import pdfplumber
from docx import Document


def extract_answer_key_text(file_path: str) -> str:
    """Extract plain text from answer-key files supported by the OMR workflow."""
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


def normalize_choice_token(token: str, num_choices: int, assume_one_based: bool) -> Optional[int]:
    """Convert an answer token (A/B/C or 0/1/2...) to a zero-based index."""
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


def parse_answer_key_from_text(
    text: str,
    num_choices: int,
    assume_one_based: Optional[bool] = None,
) -> List[int]:
    """
    Parse answer keys from formats such as:
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
            idx = normalize_choice_token(token, num_choices, local_assume_one_based)
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
        idx = normalize_choice_token(token, num_choices, local_assume_one_based)
        if idx is not None:
            parsed.append(idx)

    if not parsed:
        raise ValueError("Không parse được đáp án từ file")

    return parsed
