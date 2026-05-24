from __future__ import annotations

import cv2

_COLOR_CORRECT = (0, 220, 0)
_COLOR_WRONG = (0, 0, 255)
_COLOR_MISSING = (0, 200, 255)
_COLOR_SID = (0, 160, 255)
_COLOR_CODE = (255, 200, 0)


def _safe_float(raw, default=0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw, default=-1) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _choice_index(raw, fallback_label=None) -> int:
    value = _safe_int(raw, -1)
    if value >= 0:
        return value

    if isinstance(fallback_label, str):
        label = fallback_label.strip().upper()
        if len(label) == 1 and "A" <= label <= "Z":
            return ord(label) - ord("A")

    return -1


def _row_by_question(mcq_rows):
    rows = {}
    for idx, row in enumerate(list(mcq_rows or [])):
        if not isinstance(row, dict):
            continue
        question = _safe_int(row.get("question"), idx + 1)
        if question > 0:
            rows[question] = row
    return rows


def _circle_for_choice(canvas, row, option_idx: int, color):
    boxes = row.get("cell_boxes") if isinstance(row, dict) else None
    if not isinstance(boxes, list) or option_idx < 0 or option_idx >= len(boxes):
        return

    box = boxes[option_idx]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return

    x1, y1, x2, y2 = [int(v) for v in box]
    cx = int(round(0.5 * (x1 + x2)))
    cy = int(round(0.5 * (y1 + y2)))

    if option_idx == _safe_int(row.get("selected"), -1):
        sel_centroid = row.get("selected_centroid")
        if isinstance(sel_centroid, (list, tuple)) and len(sel_centroid) == 2:
            sx = int(round(_safe_float(sel_centroid[0], float(cx))))
            sy = int(round(_safe_float(sel_centroid[1], float(cy))))
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                cx = int(sx)
                cy = int(sy)

    rad = int(max(4, min(18, 0.4 * min(abs(x2 - x1), abs(y2 - y1)))))
    cv2.circle(canvas, (cx, cy), rad, color, 2)


def _draw_answer_circles(canvas, mcq_rows, answer_compare):
    rows = _row_by_question(mcq_rows)

    for idx, item in enumerate(list(answer_compare or [])):
        if not isinstance(item, dict):
            continue

        question = _safe_int(item.get("question"), idx + 1)
        row = rows.get(question)
        if row is None:
            continue

        selected = _choice_index(item.get("selected"), item.get("selected_label"))
        correct = _choice_index(item.get("correct"), item.get("correct_label"))
        status = str(item.get("status") or "").strip().lower()

        if status == "correct":
            _circle_for_choice(canvas, row, selected if selected >= 0 else correct, _COLOR_CORRECT)
        elif status == "wrong":
            _circle_for_choice(canvas, row, selected, _COLOR_WRONG)
            if correct >= 0 and correct != selected:
                _circle_for_choice(canvas, row, correct, _COLOR_MISSING)
        elif status == "uncertain":
            _circle_for_choice(canvas, row, correct, _COLOR_MISSING)
        elif status == "no-key":
            _circle_for_choice(canvas, row, selected, _COLOR_WRONG)


def _draw_sid_roi(canvas, sid_roi):
    if not isinstance(sid_roi, dict):
        return

    img_h, img_w = canvas.shape[:2]
    x = max(0, min(img_w - 1, _safe_int(sid_roi.get("x"), -1)))
    y = max(0, min(img_h - 1, _safe_int(sid_roi.get("y"), -1)))
    w = max(0, _safe_int(sid_roi.get("w"), 0))
    h = max(0, _safe_int(sid_roi.get("h"), 0))
    x2 = max(0, min(img_w - 1, x + w))
    y2 = max(0, min(img_h - 1, y + h))
    if x2 <= x or y2 <= y:
        return

    cv2.rectangle(canvas, (x, y), (x2, y2), _COLOR_SID, 1)
    cv2.putText(
        canvas,
        "SID ROI",
        (x + 2, max(14, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        _COLOR_SID,
        1,
        cv2.LINE_AA,
    )


def _fill_numeric_selected_cells(canvas, selected_cells, color, alpha: float = 0.34):
    if not isinstance(selected_cells, list):
        return

    img_h, img_w = canvas.shape[:2]
    overlay = canvas.copy()
    used = False

    for cell in selected_cells:
        if not isinstance(cell, dict):
            continue
        digit = _safe_int(cell.get("selected_digit"), -1)
        if digit < 0 or digit > 9:
            continue

        box = cell.get("cell_box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        x1, y1, x2, y2 = [_safe_int(v, -1) for v in box]
        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cx = int(round(0.5 * (x1 + x2)))
        cy = int(round(0.5 * (y1 + y2)))
        rx = int(max(4, 0.36 * abs(x2 - x1)))
        ry = int(max(4, 0.36 * abs(y2 - y1)))
        cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, color, -1)
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, color, 2)
        used = True

    if used:
        cv2.addWeighted(overlay, float(alpha), canvas, 1.0 - float(alpha), 0, dst=canvas)


def _draw_result_overlay(
    img_bgr,
    sid_roi,
    code_roi,
    mcq_roi,
    mcq_rows,
    student_id,
    exam_code,
    score,
    graded_questions,
    handwriting_rois=None,
    answer_compare=None,
    sid_selected_cells=None,
    code_selected_cells=None,
):
    canvas = img_bgr.copy()

    del code_roi, mcq_roi, handwriting_rois
    _draw_sid_roi(canvas, sid_roi)
    _fill_numeric_selected_cells(canvas, sid_selected_cells, _COLOR_SID)
    _fill_numeric_selected_cells(canvas, code_selected_cells, _COLOR_CODE)
    _draw_answer_circles(canvas, mcq_rows, answer_compare)

    text_1 = f"SID: {student_id}"
    text_2 = f"Code: {exam_code}"
    text_3 = f"Score: {score}/{graded_questions}"

    cv2.putText(canvas, text_1, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text_2, (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, text_3, (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)

    return canvas
