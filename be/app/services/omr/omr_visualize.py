from __future__ import annotations

import cv2


def _safe_float(raw, default=0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


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
):
    canvas = img_bgr.copy()

    cv2.rectangle(
        canvas,
        (int(sid_roi["x"]), int(sid_roi["y"])),
        (int(sid_roi["x"] + sid_roi["w"]), int(sid_roi["y"] + sid_roi["h"])),
        (0, 160, 255),
        2,
    )
    cv2.rectangle(
        canvas,
        (int(code_roi["x"]), int(code_roi["y"])),
        (int(code_roi["x"] + code_roi["w"]), int(code_roi["y"] + code_roi["h"])),
        (255, 200, 0),
        2,
    )
    cv2.rectangle(
        canvas,
        (int(mcq_roi["x"]), int(mcq_roi["y"])),
        (int(mcq_roi["x"] + mcq_roi["w"]), int(mcq_roi["y"] + mcq_roi["h"])),
        (0, 255, 0),
        2,
    )

    if isinstance(handwriting_rois, dict):
        for field_name, roi in handwriting_rois.items():
            if not isinstance(roi, dict):
                continue
            x = int(_safe_float(roi.get("x"), -1))
            y = int(_safe_float(roi.get("y"), -1))
            w = int(_safe_float(roi.get("w"), 0))
            h = int(_safe_float(roi.get("h"), 0))
            if w <= 0 or h <= 0:
                continue

            cv2.rectangle(
                canvas,
                (x, y),
                (x + w, y + h),
                (255, 90, 255),
                2,
            )

            label = str(field_name or "").replace("_", " ").upper()
            if label:
                y_label = max(14, y - 6)
                cv2.putText(
                    canvas,
                    label,
                    (x + 2, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 90, 255),
                    1,
                    cv2.LINE_AA,
                )

    for row in list(mcq_rows or []):
        selected = int(row.get("selected", -1))
        if bool(row.get("uncertain", False)):
            continue
        boxes = row.get("cell_boxes")
        if not isinstance(boxes, list) or selected < 0 or selected >= len(boxes):
            continue
        box = boxes[selected]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        cx = int(round(0.5 * (x1 + x2)))
        cy = int(round(0.5 * (y1 + y2)))

        sel_centroid = row.get("selected_centroid")
        if isinstance(sel_centroid, (list, tuple)) and len(sel_centroid) == 2:
            sx = int(round(_safe_float(sel_centroid[0], float(cx))))
            sy = int(round(_safe_float(sel_centroid[1], float(cy))))
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                cx = int(sx)
                cy = int(sy)

        rad = int(max(4, min(18, 0.4 * min(abs(x2 - x1), abs(y2 - y1)))))
        cv2.circle(canvas, (cx, cy), rad, (0, 255, 0), 2)

    text_1 = f"SID: {student_id}"
    text_2 = f"Code: {exam_code}"
    text_3 = f"Score: {score}/{graded_questions}"

    cv2.putText(canvas, text_1, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text_2, (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, text_3, (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)

    return canvas
