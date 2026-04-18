from __future__ import annotations

from typing import Dict, List

from .omr_labels import MAP_LABEL, choice_label


LABEL_TO_INDEX = {label.upper(): idx for idx, label in MAP_LABEL.items()}


def _safe_int(raw, default=0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _normalize_answer_key(answer_key, choices: int) -> List[int]:
    normalized: List[int] = []
    for raw in list(answer_key or []):
        val = _safe_int(raw, -1)
        if 0 <= val < int(choices):
            normalized.append(val)
        elif 1 <= val <= int(choices):
            normalized.append(val - 1)
        elif isinstance(raw, str):
            token = str(raw).strip().upper()
            mapped = LABEL_TO_INDEX.get(token, -1)
            if int(mapped) < 0 and len(token) == 1 and ("A" <= token <= "Z"):
                mapped = ord(token) - ord("A")
            if 0 <= int(mapped) < int(choices):
                normalized.append(int(mapped))
            else:
                normalized.append(-1)
        else:
            normalized.append(-1)
    return normalized


def _build_answer_compare(user_answers, answer_key_zero_based, choices: int, graded_questions: int):
    del choices

    compare: List[Dict[str, object]] = []
    wrong_questions: List[int] = []
    uncertain_questions: List[int] = []
    correct_count = 0
    scored_questions = 0

    for idx in range(int(graded_questions)):
        q_num = int(idx + 1)
        selected = int(user_answers[idx]) if idx < len(user_answers) else -1
        correct = int(answer_key_zero_based[idx]) if idx < len(answer_key_zero_based) else -1

        selected_label = choice_label(selected)
        correct_label = choice_label(correct)

        if correct >= 0:
            scored_questions += 1

        if selected < 0 and correct < 0:
            status = "blank-no-key"
            is_correct = False
        elif selected < 0:
            status = "uncertain"
            is_correct = False
            uncertain_questions.append(q_num)
        elif correct < 0:
            status = "no-key"
            is_correct = False
        elif selected == correct:
            status = "correct"
            is_correct = True
            correct_count += 1
        else:
            status = "wrong"
            is_correct = False
            wrong_questions.append(q_num)

        compare.append(
            {
                "question": q_num,
                "selected": int(selected),
                "selected_label": selected_label,
                "correct": int(correct),
                "correct_label": correct_label,
                "status": status,
                "is_correct": bool(is_correct),
            }
        )

    return compare, int(correct_count), wrong_questions, uncertain_questions, int(scored_questions)
