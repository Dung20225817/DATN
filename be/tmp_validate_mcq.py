import json, os
from app.services.omr.omr_service import process_omr_exam

img = r"d:\GR2\OCR_CRNN\uploads\omr\camera_1776514660759.jpg"
if not os.path.exists(img):
    img = r"d:\GR2\OCR_CRNN\be\uploads\omr\camera_1776514660759.jpg"

print("img", img, os.path.exists(img))

res = process_omr_exam(
    image_path=img,
    output_folder=r"d:\GR2\OCR_CRNN\be\uploads\omr",
    answer_key=[],
    questions=50,
    choices=4,
    rows_per_block=25,
    student_id_digits=6,
    sid_has_write_row=True,
)

summary = {
    "success": res.get("success"),
    "error": res.get("error"),
    "detected_questions": res.get("detected_questions"),
    "graded_questions": res.get("graded_questions"),
    "uncertain_count": res.get("uncertain_count"),
    "mcq_roi": (res.get("roi_boxes") or {}).get("mcq"),
    "left_x": (res.get("roi_detection") or {}).get("coordinate_mapping_mcq_left_x"),
    "right_x": (res.get("roi_detection") or {}).get("coordinate_mapping_mcq_right_x"),
    "line_h": (res.get("roi_detection") or {}).get("coordinate_mapping_mcq_line_height_px"),
    "black_marker_roi_used": (res.get("roi_detection") or {}).get("coordinate_mapping_mcq_black_marker_roi_used"),
    "black_marker_roi_reason": (res.get("roi_detection") or {}).get("coordinate_mapping_mcq_black_marker_roi_reason"),
    "template_refine_used": (res.get("roi_detection") or {}).get("coordinate_mapping_mcq_template_refine_used"),
    "template_refine_reason": (res.get("roi_detection") or {}).get("coordinate_mapping_mcq_template_refine_reason"),
    "warning_codes": res.get("warning_codes"),
    "bubble_json": res.get("bubble_confidence_json"),
}
print(json.dumps(summary, ensure_ascii=True))
