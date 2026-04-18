# app/services/omr/omr_service.py
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import omr_marker_utils
from .omr_handwriting import (
    HANDWRITING_FIELDS,
    _build_handwriting_rois,
    _parse_handwriting_config,
    _run_handwriting_ocr,
)
from .omr_layout import _build_rois_from_anchors, _parse_roi_cfg, _resolve_coordinate_anchors
from .omr_labels import choice_label
from .omr_mcq import (
    _decode_mcq_with_map,
    _detect_q5_start_drift,
    _infer_mcq_geometry_from_markers,
    _mcq_quality,
    _parse_mcq_decode_config,
    build_mcq_roi_from_black_markers,
    refine_mcq_roi,
)
from .omr_numeric import _decode_numeric_columns, _parse_sid_row_offsets
from .omr_preprocess import (
    _apply_optional_rect_crop,
    _binarize,
    _detect_page_quad,
    _norm_quad_from_points,
    _parse_manual_quad,
    _warp_to_standard_layout,
)
from .omr_scoring import _build_answer_compare, _normalize_answer_key
from .omr_visualize import _draw_result_overlay

WIDTH_IMG = 1000
HEIGHT_IMG = 1400
A4_WARP_W = 2480
A4_WARP_H = 3508

def _safe_float(raw, default=0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw, default=0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _safe_bool(raw, default=False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_exam_code_key(raw) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        if len(digits) <= 3:
            return digits.zfill(3)
        return digits

    return text.upper()


def _parse_numeric_value(value: str, digits: int) -> Optional[List[int]]:
    text = str(value or "").strip()
    if len(text) != int(max(1, digits)):
        return None
    if "?" in text or (not text.isdigit()):
        return None
    return [int(ch) for ch in text]


def _numeric_shift_delta(primary_value: str, alt_value: str, digits: int) -> Optional[int]:
    primary_digits = _parse_numeric_value(primary_value, digits)
    alt_digits = _parse_numeric_value(alt_value, digits)
    if primary_digits is None or alt_digits is None:
        return None

    deltas = {(int(alt) - int(base)) % 10 for base, alt in zip(primary_digits, alt_digits)}
    if len(deltas) != 1:
        return None
    return int(next(iter(deltas)))


def _build_choice_labels(num_choices: int) -> List[str]:
    labels: List[str] = []
    for i in range(max(1, int(num_choices))):
        if i < 26:
            labels.append(chr(ord("A") + i))
        else:
            labels.append(str(i + 1))
    return labels


def _sanitize_filename_part(text: str, fallback: str = "omr") -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip())
    safe = safe.strip("_")
    return safe[:60] if safe else fallback


def generate_omr_template(
    output_folder: str,
    exam_title: str,
    total_questions: int,
    options: int,
    student_id_digits: int,
    rows_per_block: int = 20,
    num_blocks=None,
    exam_code: str = "000",
    info_fields=None,
):
    """Generate blank OMR template image used by /api/omr/template."""
    width_img = 1000
    height_img = 1400

    total_questions = max(1, int(total_questions))
    options = max(2, int(options))
    student_id_digits = max(1, int(student_id_digits))
    rows_per_block = max(1, int(rows_per_block))
    exam_code = str(exam_code or "000").strip()
    if not re.fullmatch(r"\d{3}", exam_code):
        exam_code = "000"
    if not isinstance(info_fields, list):
        info_fields = []
    info_fields = [str(x).strip() for x in info_fields if str(x).strip()]

    if num_blocks is None:
        block_count = max(1, int(math.ceil(total_questions / float(rows_per_block))))
    else:
        block_count = max(1, int(num_blocks))
    if block_count * rows_per_block < total_questions:
        block_count = int(math.ceil(total_questions / float(rows_per_block)))

    os.makedirs(output_folder, exist_ok=True)

    img = np.full((height_img, width_img, 3), 255, dtype=np.uint8)
    black = (0, 0, 0)

    # Four corner markers for alignment.
    marker_w, marker_h = 34, 22
    cv2.rectangle(img, (24, 22), (24 + marker_w, 22 + marker_h), black, -1)
    cv2.rectangle(img, (width_img - 24 - marker_w, 22), (width_img - 24, 22 + marker_h), black, -1)
    cv2.rectangle(img, (24, height_img - 22 - marker_w), (24 + marker_w, height_img - 22), black, -1)
    cv2.rectangle(
        img,
        (width_img - 24 - marker_w, height_img - 22 - marker_w),
        (width_img - 24, height_img - 22),
        black,
        -1,
    )

    title = (exam_title or "OMR Exam").strip()
    cv2.putText(img, title[:80], (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, black, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"Total Questions: {total_questions} | Options: {options}",
        (70, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        black,
        2,
        cv2.LINE_AA,
    )

    # Student ID panel.
    sid_x = int(width_img * 0.085)
    sid_y = int(height_img * 0.165)
    sid_w = int(width_img * 0.255)
    sid_h = int(height_img * 0.225)
    cv2.rectangle(img, (sid_x, sid_y), (sid_x + sid_w, sid_y + sid_h), black, 3)
    cv2.putText(img, "Student ID", (sid_x + 8, sid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, black, 2, cv2.LINE_AA)

    write_row_h = max(24, int(sid_h * 0.16))
    sid_col_edges = np.linspace(sid_x, sid_x + sid_w, student_id_digits + 1, dtype=int)
    for c in range(student_id_digits):
        sx1, sx2 = int(sid_col_edges[c]), int(sid_col_edges[c + 1])
        pad = max(2, int((sx2 - sx1) * 0.18))
        cv2.rectangle(
            img,
            (sx1 + pad, sid_y + 5),
            (sx2 - pad, sid_y + write_row_h - 5),
            (30, 30, 30),
            1,
        )

    bubble_top = sid_y + write_row_h
    sid_row_edges = np.linspace(bubble_top, sid_y + sid_h, 10 + 1, dtype=int)
    for y in sid_row_edges[1:-1]:
        cv2.line(img, (sid_x, int(y)), (sid_x + sid_w, int(y)), (70, 70, 70), 1)
    for x in sid_col_edges[1:-1]:
        cv2.line(img, (int(x), bubble_top), (int(x), sid_y + sid_h), (70, 70, 70), 1)
    cv2.line(img, (sid_x, bubble_top), (sid_x + sid_w, bubble_top), (70, 70, 70), 1)

    for r in range(10):
        y1, y2 = int(sid_row_edges[r]), int(sid_row_edges[r + 1])
        cy = y1 + (y2 - y1) // 2
        cv2.putText(img, str(r), (sid_x - 16, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1, cv2.LINE_AA)
        for c in range(student_id_digits):
            x1, x2 = int(sid_col_edges[c]), int(sid_col_edges[c + 1])
            cx = x1 + (x2 - x1) // 2
            radius = max(5, min(9, int((y2 - y1) * 0.32)))
            cv2.circle(img, (cx, cy), radius, (45, 45, 45), 1)

    # Exam code panel.
    code_x = sid_x + sid_w + int(width_img * 0.045)
    code_y = sid_y
    code_w = int(width_img * 0.12)
    code_h = sid_h
    cv2.rectangle(img, (code_x, code_y), (code_x + code_w, code_y + code_h), black, 3)
    cv2.putText(img, "Ma de", (code_x + 6, code_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, black, 2, cv2.LINE_AA)

    code_write_h = max(24, int(code_h * 0.16))
    code_col_edges = np.linspace(code_x, code_x + code_w, 3 + 1, dtype=int)
    for c in range(3):
        cx1, cx2 = int(code_col_edges[c]), int(code_col_edges[c + 1])
        pad = max(2, int((cx2 - cx1) * 0.18))
        cv2.rectangle(
            img,
            (cx1 + pad, code_y + 5),
            (cx2 - pad, code_y + code_write_h - 5),
            (30, 30, 30),
            1,
        )

    code_bubble_top = code_y + code_write_h
    code_row_edges = np.linspace(code_bubble_top, code_y + code_h, 10 + 1, dtype=int)
    for y in code_row_edges[1:-1]:
        cv2.line(img, (code_x, int(y)), (code_x + code_w, int(y)), (70, 70, 70), 1)
    for x in code_col_edges[1:-1]:
        cv2.line(img, (int(x), code_bubble_top), (int(x), code_y + code_h), (70, 70, 70), 1)
    cv2.line(img, (code_x, code_bubble_top), (code_x + code_w, code_bubble_top), (70, 70, 70), 1)

    for r in range(10):
        y1, y2 = int(code_row_edges[r]), int(code_row_edges[r + 1])
        cy = y1 + (y2 - y1) // 2
        cv2.putText(img, str(r), (code_x - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1, cv2.LINE_AA)
        for c in range(3):
            x1, x2 = int(code_col_edges[c]), int(code_col_edges[c + 1])
            cx = x1 + (x2 - x1) // 2
            radius = max(5, min(9, int((y2 - y1) * 0.32)))
            cv2.circle(img, (cx, cy), radius, (45, 45, 45), 1)

    if info_fields:
        info_x = code_x + code_w + int(width_img * 0.045)
        info_y = sid_y
        info_w = min(int(width_img * 0.36), width_img - info_x - 30)
        info_h = sid_h
        if info_w > 120:
            cv2.rectangle(img, (info_x, info_y), (info_x + info_w, info_y + info_h), black, 3)
            rows = len(info_fields)
            row_h = max(24, int(info_h / max(1, rows)))
            for idx, field_name in enumerate(info_fields):
                y1 = info_y + idx * row_h
                y2 = info_y + info_h if idx == rows - 1 else info_y + (idx + 1) * row_h
                if idx > 0:
                    cv2.line(img, (info_x, y1), (info_x + info_w, y1), (90, 90, 90), 1)
                cv2.putText(
                    img,
                    f"{field_name}:",
                    (info_x + 8, y1 + max(18, min(28, row_h - 6))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (35, 35, 35),
                    1,
                    cv2.LINE_AA,
                )
                line_y = min(y2 - 8, y1 + max(20, row_h - 8))
                cv2.line(img, (info_x + 110, line_y), (info_x + info_w - 8, line_y), (130, 130, 130), 1)

    # MCQ frame.
    mcq_x = int(width_img * 0.075)
    mcq_y = int(height_img * 0.430)
    mcq_w = int(width_img * 0.855)
    mcq_h = int(height_img * 0.470)
    cv2.rectangle(img, (mcq_x, mcq_y), (mcq_x + mcq_w, mcq_y + mcq_h), black, 3)

    block_gap = int(mcq_w * 0.02) if block_count > 1 else 0
    total_gap = block_gap * max(0, block_count - 1)
    usable_w = max(1, mcq_w - total_gap)
    block_w = max(1, int(usable_w / block_count))
    choice_labels = _build_choice_labels(options)

    q_counter = 1
    for block_idx in range(block_count):
        block_start_q = block_idx * rows_per_block
        rows_in_block = min(rows_per_block, total_questions - block_start_q)
        if rows_in_block <= 0:
            break

        bx = block_idx * (block_w + block_gap)
        bx2 = mcq_w if block_idx == block_count - 1 else min(bx + block_w, mcq_w)
        abs_x1 = mcq_x + bx
        abs_x2 = mcq_x + bx2

        cv2.rectangle(img, (abs_x1, mcq_y), (abs_x2, mcq_y + mcq_h), black, 2)

        b_w = max(1, bx2 - bx)
        bubble_left = int(b_w * 0.16)
        bubble_right = int(b_w * 0.95)
        bubble_x1 = abs_x1 + bubble_left
        bubble_x2 = abs_x1 + bubble_right
        cv2.line(img, (bubble_x1, mcq_y), (bubble_x1, mcq_y + mcq_h), (70, 70, 70), 1)

        row_edges = np.linspace(mcq_y, mcq_y + mcq_h, rows_in_block + 1, dtype=int)
        col_edges = np.linspace(bubble_x1, bubble_x2, options + 1, dtype=int)

        for r in range(rows_in_block):
            ry1, ry2 = int(row_edges[r]), int(row_edges[r + 1])
            cy = ry1 + (ry2 - ry1) // 2
            cv2.line(img, (abs_x1, ry1), (abs_x2, ry1), (90, 90, 90), 1)
            cv2.putText(
                img,
                str(q_counter),
                (abs_x1 + 5, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )
            for c in range(options):
                cx1, cx2 = int(col_edges[c]), int(col_edges[c + 1])
                cx = cx1 + (cx2 - cx1) // 2
                radius = max(6, min(10, int((ry2 - ry1) * 0.25)))
                cv2.circle(img, (cx, cy), radius, (45, 45, 45), 1)
                if r == 0:
                    cv2.putText(
                        img,
                        choice_labels[c],
                        (cx - 7, mcq_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (35, 35, 35),
                        1,
                        cv2.LINE_AA,
                    )
            q_counter += 1

        cv2.line(img, (abs_x1, mcq_y + mcq_h), (abs_x2, mcq_y + mcq_h), (90, 90, 90), 1)

    cv2.putText(
        img,
        "Fill bubbles completely with dark pen. One choice per question.",
        (70, height_img - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = _sanitize_filename_part(title, fallback="exam")
    out_name = f"omr_template_{safe_title}_{stamp}.png"
    out_path = os.path.join(output_folder, out_name)
    cv2.imwrite(out_path, img)

    return {
        "success": True,
        "template_image": out_name,
        "mcq_layout": {
            "questions": total_questions,
            "choices": options,
            "rows_per_block": rows_per_block,
            "num_blocks": block_count,
        },
        "student_id_digits": student_id_digits,
    }


def suggest_omr_crop_quad(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Khong the doc file anh"}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    quad, strategy = _detect_page_quad(gray)
    if quad is None:
        quad = np.array(
            [
                [0.08 * w, 0.08 * h],
                [0.92 * w, 0.08 * h],
                [0.92 * w, 0.92 * h],
                [0.08 * w, 0.92 * h],
            ],
            dtype=np.float32,
        )
        strategy = "default"

    markers = omr_marker_utils._extract_black_square_markers_from_gray(
        gray,
        min_area_ratio=0.00002,
        max_area_ratio=0.015,
        max_markers=240,
    )

    return {
        "success": True,
        "strategy": str(strategy),
        "quad": _norm_quad_from_points(quad, w, h),
        "marker_count": int(len(markers)),
        "image": {"width": int(w), "height": int(h)},
    }


def process_omr_exam(
    image_path: str,
    output_folder: str,
    answer_key: list,
    answer_key_by_code: Optional[Dict[str, List[int]]] = None,
    questions=80,
    choices=5,
    rows_per_block=20,
    num_blocks=None,
    student_id_digits=6,
    sid_has_write_row=False,
    crop_x=None,
    crop_y=None,
    crop_w=None,
    crop_h=None,
    crop_tl_x=None,
    crop_tl_y=None,
    crop_tr_x=None,
    crop_tr_y=None,
    crop_br_x=None,
    crop_br_y=None,
    crop_bl_x=None,
    crop_bl_y=None,
    profile_sid_roi=None,
    profile_mcq_roi=None,
    profile_exam_code_roi=None,
    profile_sid_row_offsets=None,
    profile_disable_mcq_rescue=False,
    profile_mcq_decode=None,
    profile_threshold_mode=None,
    profile_ai_uncertainty=None,
    profile_ai_sid_htr=None,
    profile_agentic_rescue=None,
    profile_corner_markers=None,
    profile_scanner_hint=None,
    profile_page_size_pt=None,
    profile_handwriting_fields=None,
    _internal_retry=False,
):
    del profile_ai_uncertainty
    del profile_ai_sid_htr
    del profile_agentic_rescue
    del profile_corner_markers
    del profile_scanner_hint
    del _internal_retry

    try:
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            return {"error": "Khong the doc file anh"}

        os.makedirs(output_folder, exist_ok=True)

        questions = max(1, _safe_int(questions, 80))
        choices = max(2, _safe_int(choices, 5))
        rows_per_block = max(1, _safe_int(rows_per_block, 20))
        student_id_digits = max(1, _safe_int(student_id_digits, 6))
        sid_has_write_row = bool(sid_has_write_row)

        if num_blocks is None:
            block_count = max(1, int(math.ceil(float(questions) / float(rows_per_block))))
        else:
            block_count = max(1, _safe_int(num_blocks, 1))
        inferred_block_count = 0

        warnings: List[str] = []
        warning_codes: List[str] = []

        img_input, rect_crop_used = _apply_optional_rect_crop(img_raw, crop_x, crop_y, crop_w, crop_h)
        if rect_crop_used:
            warnings.append("Da ap dung crop rectangle truoc global warp.")

        manual_quad = _parse_manual_quad(
            crop_tl_x,
            crop_tl_y,
            crop_tr_x,
            crop_tr_y,
            crop_br_x,
            crop_br_y,
            crop_bl_x,
            crop_bl_y,
        )

        img_std, warp_strategy, global_warp_used, warp_info = _warp_to_standard_layout(
            img_input,
            width_img=WIDTH_IMG,
            height_img=HEIGHT_IMG,
            a4_warp_w=A4_WARP_W,
            a4_warp_h=A4_WARP_H,
            manual_quad_norm=manual_quad,
        )
        if manual_quad is not None and warp_strategy != "manual-quad":
            warnings.append("Crop 4 diem thu cong khong hop le, fallback sang detector tu dong.")
            warning_codes.append("MANUAL_QUAD_INVALID")

        if not global_warp_used:
            warnings.append("Khong tim thay 4 goc marker ro rang, da fallback ve resize thong thuong.")
            warning_codes.append("COORD_GLOBAL_WARP_FALLBACK")

        gray = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
        gray_norm, binary_inv, threshold_meta = _binarize(gray, profile_threshold_mode)

        sid_roi_cfg = _parse_roi_cfg(profile_sid_roi, WIDTH_IMG, HEIGHT_IMG)
        code_roi_cfg = _parse_roi_cfg(profile_exam_code_roi, WIDTH_IMG, HEIGHT_IMG)
        mcq_roi_cfg = _parse_roi_cfg(profile_mcq_roi, WIDTH_IMG, HEIGHT_IMG)

        markers = omr_marker_utils._extract_black_square_markers_from_gray(
            gray_norm,
            min_area_ratio=0.00002,
            max_area_ratio=0.012,
            max_markers=320,
        )
        mcq_geometry = _infer_mcq_geometry_from_markers(markers, WIDTH_IMG, HEIGHT_IMG)

        inferred_block_count = _safe_int(mcq_geometry.get("block_count"), 0)
        geom_bands = mcq_geometry.get("block_bands")
        if inferred_block_count <= 1 and isinstance(geom_bands, (list, tuple)):
            inferred_block_count = max(0, len(list(geom_bands)))
        if num_blocks is None and inferred_block_count >= 2:
            if block_count <= 1 or questions <= rows_per_block or abs(inferred_block_count - block_count) <= 1:
                block_count = int(inferred_block_count)
                warning_codes.append("MCQ_BLOCKS_INFERRED")
                warnings.append(f"Tu dong suy luan so khoi MCQ = {block_count} tu marker fiducial.")

        anchors, anchor_fallback_used = _resolve_coordinate_anchors(
            WIDTH_IMG,
            HEIGHT_IMG,
            markers,
            rows_per_block,
            sid_roi_cfg,
            code_roi_cfg,
            mcq_roi_cfg,
        )

        if anchor_fallback_used:
            warnings.append("Thieu marker neo cuc bo, da fallback ve anchor theo ty le template.")
            warning_codes.append("COORD_ANCHOR_FALLBACK")

        roi_boxes, line_h_anchor = _build_rois_from_anchors(
            anchors,
            WIDTH_IMG,
            HEIGHT_IMG,
            rows_per_block,
            sid_roi_cfg,
            code_roi_cfg,
            mcq_roi_cfg,
        )

        sid_roi = roi_boxes["sid"]
        code_roi = roi_boxes["code"]
        mcq_roi = roi_boxes["mcq"]

        mcq_block_bands: List[Tuple[float, float]] = []
        if isinstance(geom_bands, (list, tuple)):
            for item in list(geom_bands):
                if not isinstance(item, dict):
                    continue
                x1 = _safe_float(item.get("x_min"), -1.0)
                x2 = _safe_float(item.get("x_max"), -1.0)
                if x2 <= x1:
                    continue
                mcq_block_bands.append((float(x1), float(x2)))
        if mcq_block_bands and len(mcq_block_bands) >= block_count:
            mcq_block_bands = sorted(mcq_block_bands, key=lambda band: float(band[0]))
        else:
            mcq_block_bands = []

        geom_top_center = _safe_float(mcq_geometry.get("top_center_y"), -1.0)
        geom_line_h = _safe_float(mcq_geometry.get("line_h"), -1.0)
        mcq_marker_roi_meta: Dict[str, object] = {"used": False, "reason": "skipped"}
        if mcq_roi_cfg is None:
            marker_roi, mcq_marker_roi_meta = build_mcq_roi_from_black_markers(
                mcq_roi=mcq_roi,
                mcq_geometry=mcq_geometry,
                img_w=WIDTH_IMG,
                img_h=HEIGHT_IMG,
                top_padding_px=5,
                side_padding_px=10,
                bottom_padding_px=15,
            )
            if bool(mcq_marker_roi_meta.get("used", False)):
                mcq_roi = dict(marker_roi)
                warnings.append("Da dat ROI MCQ theo cac o vuong den cua khung MCQ.")
                warning_codes.append("MCQ_BLACK_MARKER_ROI_APPLIED")

        mcq_refine_meta: Dict[str, object] = {"used": False, "reason": "skipped"}
        mcq_anchor_height_meta: Dict[str, object] = {"used": False, "reason": "derived-from-mcq-refine"}
        mcq_crop_clean = img_std[
            mcq_roi["y"] : mcq_roi["y"] + mcq_roi["h"],
            mcq_roi["x"] : mcq_roi["x"] + mcq_roi["w"],
        ]
        if mcq_roi_cfg is None:
            mcq_roi, mcq_crop_clean, mcq_refine_meta = refine_mcq_roi(
                source_img=img_std,
                gray_img=gray_norm,
                mcq_roi=mcq_roi,
                top_padding_px=5,
                side_padding_px=10,
                bottom_padding_px=15,
            )

            if bool(mcq_refine_meta.get("used", False)):
                warnings.append("Da refine ROI MCQ bang template matching anchor de loai bo vung chu huong dan.")
                warning_codes.append("MCQ_TEMPLATE_REFINE_APPLIED")

                refined_top = _safe_float(mcq_refine_meta.get("y_anchor_top"), -1.0)
                refined_bottom = _safe_float(mcq_refine_meta.get("y_anchor_bottom"), -1.0)
                refined_line_h = _safe_float(mcq_refine_meta.get("line_h"), -1.0)
                if refined_top > 0.0 and refined_bottom > refined_top:
                    mcq_geometry["top_center_y"] = float(refined_top)
                    mcq_geometry["bottom_center_y"] = float(refined_bottom)
                if refined_line_h > 0.0:
                    mcq_geometry["line_h"] = float(refined_line_h)
                anchor_distance = float(refined_bottom - refined_top)
                mcq_anchor_height_meta = {
                    "used": bool(anchor_distance > 0.0),
                    "reason": "derived-from-refine-mcq-roi",
                    "y_top_anchor": round(float(refined_top), 4),
                    "y_bottom_anchor": round(float(refined_bottom), 4),
                    "anchor_distance": round(float(anchor_distance), 4),
                    "line_h": round(float(refined_line_h), 4) if refined_line_h > 0.0 else None,
                    "match_count": _safe_int(mcq_refine_meta.get("matches"), 0),
                    "top_markers": list(mcq_refine_meta.get("top_row_centers") or []),
                    "bottom_markers": list(mcq_refine_meta.get("bottom_row_centers") or []),
                    "refined_roi": dict(mcq_roi),
                }
            else:
                mcq_anchor_height_meta = {
                    "used": False,
                    "reason": f"mcq-refine-not-used:{mcq_refine_meta.get('reason')}",
                }

        if mcq_roi_cfg is None:
            left_expand_px = max(6, min(20, int(round(0.012 * float(mcq_roi["w"])))))
            if left_expand_px > 0:
                new_x = max(0, int(mcq_roi["x"]) - int(left_expand_px))
                gained = int(mcq_roi["x"]) - int(new_x)
                if gained > 0:
                    mcq_roi["x"] = int(new_x)
                    mcq_roi["w"] = int(min(WIDTH_IMG - int(new_x), int(mcq_roi["w"]) + int(gained)))
                    mcq_crop_clean = img_std[
                        mcq_roi["y"] : mcq_roi["y"] + mcq_roi["h"],
                        mcq_roi["x"] : mcq_roi["x"] + mcq_roi["w"],
                    ]

        handwriting_cfg = _parse_handwriting_config(profile_handwriting_fields)
        handwriting_rois = _build_handwriting_rois(
            anchors,
            sid_roi,
            mcq_roi,
            WIDTH_IMG,
            HEIGHT_IMG,
            handwriting_cfg.get("field_rois"),
        )

        sid_offsets = _parse_sid_row_offsets(profile_sid_row_offsets, student_id_digits)

        sid_result_with_write = _decode_numeric_columns(
            gray_norm,
            binary_inv,
            sid_roi,
            digits=student_id_digits,
            has_write_row=True,
            row_offsets=sid_offsets,
        )
        sid_result_without_write = _decode_numeric_columns(
            gray_norm,
            binary_inv,
            sid_roi,
            digits=student_id_digits,
            has_write_row=False,
            row_offsets=sid_offsets,
        )

        if bool(sid_has_write_row):
            sid_result = sid_result_with_write
            sid_result_alt = sid_result_without_write
            sid_decode_mode = "with-write-row"
            alt_mode = "without-write-row"
        else:
            sid_result = sid_result_without_write
            sid_result_alt = sid_result_with_write
            sid_decode_mode = "without-write-row"
            alt_mode = "with-write-row"

        primary_value = str(sid_result.get("value") or "")
        alt_value = str(sid_result_alt.get("value") or "")
        primary_ok = sid_result.get("status") == "ok" and "?" not in primary_value
        alt_ok = sid_result_alt.get("status") == "ok" and "?" not in alt_value

        primary_conf = float(sid_result.get("confidence") or 0.0)
        alt_conf = float(sid_result_alt.get("confidence") or 0.0)
        shift_delta = _numeric_shift_delta(primary_value, alt_value, student_id_digits) if (primary_ok and alt_ok) else None

        use_alt = False
        if bool(sid_has_write_row):
            primary_all_zero = primary_value == ("0" * int(max(1, student_id_digits)))
            if alt_ok and not primary_ok:
                use_alt = True
            elif alt_ok and primary_ok and (alt_conf >= (primary_conf + 0.08)):
                use_alt = True
            elif alt_ok and primary_ok and primary_value.startswith("99") and not alt_value.startswith("99"):
                use_alt = True
            elif alt_ok and primary_ok and shift_delta in (1, 9) and alt_conf >= (primary_conf - 0.15):
                use_alt = True
            elif alt_ok and primary_ok and primary_all_zero and primary_value != alt_value and shift_delta in (1, 9) and alt_conf >= 1.15:
                use_alt = True
        else:
            if alt_ok and not primary_ok:
                use_alt = True

        if use_alt:
            sid_result = sid_result_alt
            sid_decode_mode = f"auto-{alt_mode}"
            warnings.append("SID co dau hieu lech hang theo cot, da tu dong doi che do decode SID.")
            warning_codes.append("SID_AUTO_SWITCH_ROW_MODE")

        code_result = _decode_numeric_columns(
            gray_norm,
            binary_inv,
            code_roi,
            digits=3,
            has_write_row=False,
            row_offsets=None,
        )

        mcq_cfg = _parse_mcq_decode_config(profile_mcq_decode)
        geom_top_center = _safe_float(mcq_geometry.get("top_center_y"), -1.0)
        geom_line_h = _safe_float(mcq_geometry.get("line_h"), -1.0)

        anchor_top_y = 0.5 * (anchors["mcq_left_top"][1] + anchors["mcq_right_top"][1])
        anchor_bottom_y = 0.5 * (anchors["mcq_left_bottom"][1] + anchors["mcq_right_bottom"][1])
        anchor_left_x = 0.5 * (anchors["mcq_left_top"][0] + anchors["mcq_left_bottom"][0])
        anchor_right_x = 0.5 * (anchors["mcq_right_top"][0] + anchors["mcq_right_bottom"][0])

        anchor_line_h = (anchor_bottom_y - anchor_top_y) / max(1.0, float(max(2, rows_per_block) - 1))
        roi_line_h = max(8.0, float(mcq_roi["h"]) / max(2.0, float(rows_per_block)))
        geom_line_h_ok = 8.0 <= geom_line_h <= 60.0
        geom_line_h_consistent = geom_line_h_ok and ((0.65 * roi_line_h) <= geom_line_h <= (1.45 * roi_line_h))
        anchor_line_h_consistent = (0.70 * roi_line_h) <= anchor_line_h <= (1.35 * roi_line_h)
        map_line_h_consistent = (0.70 * roi_line_h) <= line_h_anchor <= (1.35 * roi_line_h)

        if geom_line_h_consistent:
            line_h = float(geom_line_h)
        elif anchor_line_h_consistent:
            line_h = float(anchor_line_h)
        elif map_line_h_consistent:
            line_h = float(line_h_anchor)
        else:
            line_h = float(roi_line_h)
        line_h = max(8.0, min(44.0, float(line_h)))

        top_center_y = float(geom_top_center) if geom_top_center > 0.0 else float(anchor_top_y)
        fid_top_y = _safe_float(mcq_geometry.get("fid_top_y"), -1.0)
        if fid_top_y > 0.0 and top_center_y > (fid_top_y + 1.8 * line_h):
            top_center_y = float(fid_top_y + line_h)

        top_center_upper_guard = float(mcq_roi["y"] + max(2.2 * line_h, 0.45 * float(mcq_roi["h"])))
        if not ((mcq_roi["y"] - 0.5 * line_h) <= top_center_y <= top_center_upper_guard):
            top_center_y = float(mcq_roi["y"] + 0.5 * line_h)

        geom_left_x = _safe_float(mcq_geometry.get("left_x"), -1.0)
        geom_right_x = _safe_float(mcq_geometry.get("right_x"), -1.0)
        if mcq_block_bands:
            anchor_left_x = float(mcq_block_bands[0][0])
            anchor_right_x = float(mcq_block_bands[min(len(mcq_block_bands), block_count) - 1][1])
        elif geom_right_x - geom_left_x >= float(mcq_roi["w"]) * 0.45:
            anchor_left_x = float(geom_left_x)
            anchor_right_x = float(geom_right_x)

        anchor_left_x = max(float(mcq_roi["x"]), float(anchor_left_x))
        anchor_right_x = min(float(mcq_roi["x"] + mcq_roi["w"]), float(anchor_right_x))
        span_x = float(anchor_right_x - anchor_left_x)
        if span_x < (0.55 * float(mcq_roi["w"])) or span_x > (1.02 * float(mcq_roi["w"])):
            anchor_left_x = float(mcq_roi["x"] + 0.06 * mcq_roi["w"])
            anchor_right_x = float(mcq_roi["x"] + 0.94 * mcq_roi["w"])

        mcq_result = _decode_mcq_with_map(
            gray_norm,
            binary_inv,
            mcq_roi,
            questions=questions,
            choices=choices,
            rows_per_block=rows_per_block,
            block_count=block_count,
            left_x=anchor_left_x,
            right_x=anchor_right_x,
            top_center_y=top_center_y,
            line_h=line_h,
            decode_cfg=mcq_cfg,
            top_shift_px=0.0,
            block_bands=mcq_block_bands,
        )

        mcq_map_search_meta: Dict[str, object] = {
            "used": False,
            "reason": "skipped",
            "initial_uncertain": int(len(list(mcq_result.get("uncertain_questions") or []))),
        }

        # If marker-anchored ROI is still highly uncertain, search nearby map parameters.
        # This keeps ownership in MCQ module while avoiding hardcoded service-side ROI rewrites.
        initial_uncertain = int(len(list(mcq_result.get("uncertain_questions") or [])))
        search_uncertain_gate = max(12, int(round(0.60 * float(max(1, questions)))))
        if (not bool(profile_disable_mcq_rescue)) and initial_uncertain >= search_uncertain_gate:
            mcq_map_search_meta = {
                "used": False,
                "reason": "no-better-candidate",
                "initial_uncertain": int(initial_uncertain),
                "gate": int(search_uncertain_gate),
            }

            baseline_quality = float(_mcq_quality(mcq_result))
            baseline_double = int(len(list(mcq_result.get("double_mark_questions") or [])))
            baseline_rank = (int(initial_uncertain), int(baseline_double), float(-baseline_quality))
            baseline_line_h = float(line_h)
            baseline_top_center_y = float(top_center_y)

            best_result = mcq_result
            best_rank = baseline_rank
            best_line_h = float(line_h)
            best_top_shift_px = 0.0
            best_block_bands = list(mcq_block_bands)

            line_scales = [1.00, 0.92, 0.88, 1.08, 0.84, 1.16]
            shift_multipliers = [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]

            candidate_band_sets: List[List[Tuple[float, float]]] = [list(mcq_block_bands)] if mcq_block_bands else [[]]

            tested = 0
            for cand_bands in candidate_band_sets:
                for scale in line_scales:
                    cand_line_h = max(6.0, min(44.0, float(line_h) * float(scale)))
                    for shift_mul in shift_multipliers:
                        cand_shift_px = float(shift_mul) * float(cand_line_h)
                        cand_result = _decode_mcq_with_map(
                            gray_norm,
                            binary_inv,
                            mcq_roi,
                            questions=questions,
                            choices=choices,
                            rows_per_block=rows_per_block,
                            block_count=block_count,
                            left_x=anchor_left_x,
                            right_x=anchor_right_x,
                            top_center_y=top_center_y,
                            line_h=cand_line_h,
                            decode_cfg=mcq_cfg,
                            top_shift_px=cand_shift_px,
                            block_bands=cand_bands,
                        )
                        tested += 1

                        cand_uncertain = int(len(list(cand_result.get("uncertain_questions") or [])))
                        cand_double = int(len(list(cand_result.get("double_mark_questions") or [])))
                        cand_quality = float(_mcq_quality(cand_result))
                        cand_rank = (int(cand_uncertain), int(cand_double), float(-cand_quality))

                        if cand_rank < best_rank:
                            best_rank = cand_rank
                            best_result = cand_result
                            best_line_h = float(cand_line_h)
                            best_top_shift_px = float(cand_shift_px)
                            best_block_bands = list(cand_bands)

            improved = False
            if best_rank < baseline_rank:
                uncertain_gain = int(baseline_rank[0] - best_rank[0])
                if uncertain_gain >= 2:
                    improved = True
                elif uncertain_gain >= 1 and best_rank[1] <= baseline_rank[1]:
                    improved = True
                elif uncertain_gain == 0 and best_rank[1] < baseline_rank[1]:
                    improved = True

            if improved:
                mcq_result = best_result
                line_h = float(best_line_h)
                top_center_y = float(top_center_y + best_top_shift_px)
                top_guard = float(max(3.0, 0.35 * float(line_h)))
                top_center_min = float(mcq_roi["y"] + top_guard)
                top_center_max = float(mcq_roi["y"] + mcq_roi["h"] - top_guard)
                if top_center_min > top_center_max:
                    top_center_min = float(mcq_roi["y"])
                    top_center_max = float(mcq_roi["y"] + mcq_roi["h"])
                top_center_y = max(top_center_min, min(top_center_max, float(top_center_y)))
                mcq_block_bands = list(best_block_bands)

                mcq_map_search_meta = {
                    "used": True,
                    "reason": "improved",
                    "tested": int(tested),
                    "initial_uncertain": int(baseline_rank[0]),
                    "final_uncertain": int(best_rank[0]),
                    "initial_double_mark": int(baseline_rank[1]),
                    "final_double_mark": int(best_rank[1]),
                    "line_h_before": round(float(baseline_line_h), 4),
                    "line_h_after": round(float(best_line_h), 4),
                    "top_center_y_before": round(float(baseline_top_center_y), 4),
                    "top_center_y_after": round(float(top_center_y), 4),
                    "top_shift_px": round(float(best_top_shift_px), 4),
                    "block_bands_used": bool(best_block_bands),
                }
                warning_codes.append("MCQ_MAP_SEARCH_RESCUE")
                warnings.append(
                    "MCQ map rescue da dieu chinh line-height/top-shift (va block bands neu can) de giam uncertain."
                )
            else:
                mcq_map_search_meta = {
                    "used": False,
                    "reason": "no-better-candidate",
                    "tested": int(tested),
                    "initial_uncertain": int(baseline_rank[0]),
                    "final_uncertain": int(best_rank[0]),
                    "initial_double_mark": int(baseline_rank[1]),
                    "final_double_mark": int(best_rank[1]),
                }
        elif bool(profile_disable_mcq_rescue):
            mcq_map_search_meta = {
                "used": False,
                "reason": "disabled-by-profile",
                "initial_uncertain": int(initial_uncertain),
            }
        else:
            mcq_map_search_meta = {
                "used": False,
                "reason": "below-uncertain-gate",
                "initial_uncertain": int(initial_uncertain),
                "gate": int(search_uncertain_gate),
            }

        drift_suspected = _detect_q5_start_drift(mcq_result, mcq_cfg["min_mark_score"])
        auto_expand = False
        auto_expand_lines = 0

        if drift_suspected:
            warning_codes.append("MCQ_COORD_DRIFT")
            if not bool(profile_disable_mcq_rescue):
                expand_lines = 4
                expand_px = int(round(float(expand_lines) * float(line_h)))
                retry_roi = dict(mcq_roi)
                if expand_px > 0:
                    y_new = max(0, int(retry_roi["y"]) - int(expand_px))
                    gained = int(retry_roi["y"]) - y_new
                    retry_roi["y"] = int(y_new)
                    retry_roi["h"] = int(min(HEIGHT_IMG - retry_roi["y"], int(retry_roi["h"]) + gained))

                retry_top_center = float(top_center_y - float(expand_lines) * float(line_h))
                if retry_top_center < (retry_roi["y"] + 0.25 * line_h):
                    retry_top_center = float(retry_roi["y"] + 0.5 * line_h)

                retry = _decode_mcq_with_map(
                    gray_norm,
                    binary_inv,
                    retry_roi,
                    questions=questions,
                    choices=choices,
                    rows_per_block=rows_per_block,
                    block_count=block_count,
                    left_x=anchor_left_x,
                    right_x=anchor_right_x,
                    top_center_y=retry_top_center,
                    line_h=line_h,
                    decode_cfg=mcq_cfg,
                    top_shift_px=0.0,
                    block_bands=mcq_block_bands,
                )

                if _mcq_quality(retry) >= _mcq_quality(mcq_result):
                    mcq_result = retry
                    mcq_roi = retry_roi
                    top_center_y = float(retry_top_center)
                    auto_expand = True
                    auto_expand_lines = int(expand_lines)
                    warning_codes.append("MCQ_AUTO_EXPAND_UP")
                    warnings.append("Phat hien roi MCQ co dau hieu bat dau tu cau 5, da tu dong mo rong ROI len 4 dong.")
                else:
                    warnings.append("Phat hien drift roi MCQ nhung thu nghiem mo rong len khong tot hon decode hien tai.")
            else:
                warnings.append("Phat hien drift roi MCQ nhung profile da khoa rescue tu dong.")

        user_answers = list(mcq_result.get("user_answers") or [])
        if len(user_answers) < questions:
            user_answers.extend([-1] * (questions - len(user_answers)))

        answer_confidences = list(mcq_result.get("answer_confidences") or [])
        if len(answer_confidences) < questions:
            answer_confidences.extend([0.0] * (questions - len(answer_confidences)))

        answer_map = [choice_label(int(val)) for val in user_answers]

        detected_exam_code_raw = str(code_result.get("value") or "").strip()
        detected_exam_code_key = _normalize_exam_code_key(detected_exam_code_raw)

        manual_answer_key = list(answer_key or [])
        selected_answer_key = list(manual_answer_key)
        selected_answer_code: Optional[str] = None
        answer_key_source = "manual-or-file"

        normalized_code_map: Dict[str, List[int]] = {}
        display_code_map: Dict[str, str] = {}
        if isinstance(answer_key_by_code, dict):
            for raw_code, raw_values in answer_key_by_code.items():
                norm_code = _normalize_exam_code_key(raw_code)
                if not norm_code:
                    continue
                values = [int(v) for v in list(raw_values or [])]
                if not values:
                    continue
                normalized_code_map[norm_code] = values
                display_code_map[norm_code] = str(raw_code)

        if normalized_code_map:
            answer_key_source = "assignment-code-map"
            if detected_exam_code_key and detected_exam_code_key in normalized_code_map:
                selected_answer_key = list(normalized_code_map[detected_exam_code_key])
                selected_answer_code = str(display_code_map.get(detected_exam_code_key) or detected_exam_code_raw)
            else:
                selected_answer_key = []
                if detected_exam_code_raw:
                    warnings.append(f"Khong tim thay dap an cho CODE '{detected_exam_code_raw}' trong kho dap an.")
                    warning_codes.append("ANSWER_CODE_NOT_FOUND")

        answer_key_zero = _normalize_answer_key(selected_answer_key, choices)
        compare_questions = min(len(answer_key_zero), len(user_answers), int(questions))

        answer_compare, correct_count, wrong_questions, uncertain_compare, scored_questions = _build_answer_compare(
            user_answers,
            answer_key_zero,
            choices,
            compare_questions,
        )
        graded_questions = int(min(int(questions), int(scored_questions)))

        detection_uncertain_questions = sorted(set(int(q) for q in list(mcq_result.get("uncertain_questions") or [])))
        uncertain_questions = sorted(set(int(q) for q in uncertain_compare))
        double_mark_questions = sorted(set(int(q) for q in list(mcq_result.get("double_mark_questions") or [])))
        if double_mark_questions:
            warning_codes.append("MCQ_DOUBLE_MARK")
            warnings.append(f"Phat hien da to nhieu lua chon o cac cau: {', '.join(str(q) for q in double_mark_questions[:20])}")
        uncertain_count = len([q for q in uncertain_questions if 1 <= q <= questions])

        score = float(correct_count)
        detected_questions = int(questions)
        ungraded_questions = max(0, detected_questions - graded_questions)

        run_tag = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        result_name = f"omr_result_{run_tag}.jpg"
        sid_crop_name = f"omr_sid_{run_tag}.jpg"
        mcq_crop_name = f"omr_mcq_{run_tag}.jpg"
        bubble_name = f"bubble_confidence_{run_tag}.json"

        handwriting_payload = _run_handwriting_ocr(
            img_std,
            handwriting_rois,
            output_folder=output_folder,
            run_tag=run_tag,
            handwriting_cfg=handwriting_cfg,
        )
        warnings.extend(list(handwriting_payload.get("warnings") or []))
        warning_codes.extend(list(handwriting_payload.get("warning_codes") or []))

        sid_crop = img_std[
            sid_roi["y"] : sid_roi["y"] + sid_roi["h"],
            sid_roi["x"] : sid_roi["x"] + sid_roi["w"],
        ]
        if mcq_crop_clean is not None and getattr(mcq_crop_clean, "size", 0) > 0:
            mcq_crop = mcq_crop_clean
        else:
            mcq_crop = img_std[
                mcq_roi["y"] : mcq_roi["y"] + mcq_roi["h"],
                mcq_roi["x"] : mcq_roi["x"] + mcq_roi["w"],
            ]

        img_result = _draw_result_overlay(
            img_std,
            sid_roi,
            code_roi,
            mcq_roi,
            mcq_result.get("rows") or [],
            sid_result.get("value") or "",
            code_result.get("value") or "",
            score,
            graded_questions,
        )

        result_path = os.path.join(output_folder, result_name)
        sid_crop_path = os.path.join(output_folder, sid_crop_name)
        mcq_crop_path = os.path.join(output_folder, mcq_crop_name)
        bubble_path = os.path.join(output_folder, bubble_name)

        cv2.imwrite(result_path, img_result)
        cv2.imwrite(sid_crop_path, sid_crop)
        cv2.imwrite(mcq_crop_path, mcq_crop)

        bubble_payload = {
            "pipeline_version": "omr-coordinate-map-v1",
            "image": os.path.basename(image_path),
            "warp_strategy": str(warp_strategy),
            "roi_boxes": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "coordinate_mapping": {
                "global_warp": bool(global_warp_used),
                "global_warp_target": {"width": int(A4_WARP_W), "height": int(A4_WARP_H)},
                "global_warp_info": warp_info,
                "anchor_fallback_used": bool(anchor_fallback_used),
                "mcq_marker_roi": mcq_marker_roi_meta,
                "mcq_refine": mcq_refine_meta,
                "mcq_anchor_height": mcq_anchor_height_meta,
                "inferred_block_count": int(inferred_block_count),
                "line_height_px": round(float(line_h), 4),
                "drift_suspected": bool(drift_suspected),
                "auto_expand_upward": bool(auto_expand),
                "auto_expand_lines": int(auto_expand_lines),
                "mcq_map_search": mcq_map_search_meta,
                "fid_top_y": round(float(_safe_float(mcq_geometry.get("fid_top_y"), 0.0)), 4),
                "fid_bottom_y": round(float(_safe_float(mcq_geometry.get("fid_bottom_y"), 0.0)), 4),
                "anchors": {
                    key: {"x": round(float(val[0]), 4), "y": round(float(val[1]), 4)}
                    for key, val in anchors.items()
                },
            },
            "handwriting": {
                "enabled": bool(handwriting_payload.get("enabled", False)),
                "ocr_engine": str(handwriting_payload.get("ocr_engine") or "vietocr_transformer"),
                "field_rois": handwriting_rois,
                "values": handwriting_payload.get("values") or {},
                "fields": handwriting_payload.get("fields") or {},
            },
            "rows": mcq_result.get("rows") or [],
        }

        try:
            with open(bubble_path, "w", encoding="utf-8") as fp:
                json.dump(bubble_payload, fp, ensure_ascii=False, indent=2)
        except Exception:
            bubble_name = None
            warnings.append("Khong the ghi telemetry bubble confidence JSON.")
            warning_codes.append("BUBBLE_JSON_WRITE_FAILED")

        warp_layout_score = 0.0
        if global_warp_used:
            warp_layout_score += 0.50
        warp_layout_score += min(0.40, len(markers) / 500.0)
        if not anchor_fallback_used:
            warp_layout_score += 0.10

        roi_strategy = "coordinate-mapping-anchor-grid"
        if sid_roi_cfg is not None or code_roi_cfg is not None or mcq_roi_cfg is not None:
            roi_strategy += " + profile-roi"
        if anchor_fallback_used:
            roi_strategy += " + percent-fallback"

        correct_answers = []
        for i in range(detected_questions):
            if i < len(selected_answer_key):
                correct_answers.append(_safe_int(selected_answer_key[i], -1))
            else:
                correct_answers.append(-1)

        correct_answers_graded = [
            int(item.get("correct", -1))
            for item in answer_compare
            if _safe_int(item.get("correct", -1), -1) >= 0
        ]

        warnings = _dedupe_keep_order(warnings)
        warning_codes = _dedupe_keep_order(warning_codes)

        return {
            "success": True,
            "pipeline_version": "omr-coordinate-map-v1",
            "score": score,
            "student_id": str(sid_result.get("value") or ""),
            "student_id_status": str(sid_result.get("status") or "uncertain"),
            "student_id_confidence": float(sid_result.get("confidence") or 0.0),
            "exam_code": str(code_result.get("value") or ""),
            "exam_code_status": str(code_result.get("status") or "uncertain"),
            "exam_code_confidence": float(code_result.get("confidence") or 0.0),
            "matched_answer_code": selected_answer_code,
            "answer_key_source": answer_key_source,
            "exam_title_detected": None,
            "user_answers": [int(v) for v in user_answers],
            "answer_map": answer_map,
            "answer_confidences": [float(v) for v in answer_confidences],
            "answer_compare": answer_compare,
            "correct_answers": correct_answers,
            "correct_answers_graded": correct_answers_graded,
            "graded_questions": int(graded_questions),
            "answer_key_questions": int(len(answer_key_zero)),
            "detected_questions": int(detected_questions),
            "ungraded_questions": int(ungraded_questions),
            "ungraded_count": int(ungraded_questions),
            "uncertain_questions": uncertain_questions,
            "detection_uncertain_questions": detection_uncertain_questions,
            "uncertain_count": int(uncertain_count),
            "wrong_questions": wrong_questions,
            "double_mark_questions": double_mark_questions,
            "roi_boxes": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "roi_boxes_refined": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "roi_boxes_json": {
                "student_id": sid_roi,
                "exam_code": code_roi,
                "mcq": mcq_roi,
                "handwriting": handwriting_rois,
            },
            "handwriting_fields": handwriting_payload.get("values") or {},
            "handwriting": {
                "enabled": bool(handwriting_payload.get("enabled", False)),
                "ocr_engine": str(handwriting_payload.get("ocr_engine") or "vietocr_transformer"),
                "gpu": bool(handwriting_payload.get("gpu", False)),
                "save_crops": bool(handwriting_payload.get("save_crops", True)),
                "field_rois": handwriting_rois,
                "fields": handwriting_payload.get("fields") or {},
                "crop_images": handwriting_payload.get("crop_images") or {},
                "preprocessed_crop_images": handwriting_payload.get("preprocessed_crop_images") or {},
            },
            "roi_detection": {
                "strategy": roi_strategy,
                "profile_page_size_pt": profile_page_size_pt if isinstance(profile_page_size_pt, dict) else None,
                "marker_count": int(len(markers)),
                "sid_decode_mode": sid_decode_mode,
                "handwriting_enabled": bool(handwriting_payload.get("enabled", False)),
                "handwriting_ocr_engine": str(handwriting_payload.get("ocr_engine") or "vietocr_transformer"),
                "coordinate_mapping_global_warp": bool(global_warp_used),
                "coordinate_mapping_anchor_fallback": bool(anchor_fallback_used),
                "coordinate_mapping_mcq_map_used": True,
                "coordinate_mapping_mcq_blocks_inferred": int(inferred_block_count),
                "coordinate_mapping_mcq_black_marker_roi_used": bool(mcq_marker_roi_meta.get("used", False)),
                "coordinate_mapping_mcq_black_marker_roi_reason": str(mcq_marker_roi_meta.get("reason") or ""),
                "coordinate_mapping_mcq_template_refine_used": bool(mcq_refine_meta.get("used", False)),
                "coordinate_mapping_mcq_template_refine_reason": str(mcq_refine_meta.get("reason") or ""),
                "coordinate_mapping_mcq_anchor_height_used": bool(mcq_anchor_height_meta.get("used", False)),
                "coordinate_mapping_mcq_anchor_height_reason": str(mcq_anchor_height_meta.get("reason") or ""),
                "coordinate_mapping_mcq_line_height_px": round(float(line_h), 4),
                "coordinate_mapping_mcq_drift_suspected": bool(drift_suspected),
                "coordinate_mapping_mcq_auto_expand_upward": bool(auto_expand),
                "coordinate_mapping_mcq_auto_expand_lines": int(auto_expand_lines),
                "coordinate_mapping_mcq_map_search_used": bool(mcq_map_search_meta.get("used", False)),
                "coordinate_mapping_mcq_map_search_reason": str(mcq_map_search_meta.get("reason") or ""),
                "coordinate_mapping_mcq_q1_y": round(float(top_center_y), 4),
                "coordinate_mapping_mcq_left_x": round(float(anchor_left_x), 4),
                "coordinate_mapping_mcq_right_x": round(float(anchor_right_x), 4),
                "coordinate_mapping_mcq_double_mark_count": int(len(double_mark_questions)),
            },
            "sid_layout": {"digits": int(student_id_digits), "rows": 10},
            "mcq_layout": {
                "questions": int(questions),
                "choices": int(choices),
                "rows_per_block": int(rows_per_block),
                "num_blocks": int(block_count),
                "block_gap_px": 0,
                "block_width_px": int(round(float(mcq_roi["w"]) / max(1.0, float(block_count)))),
            },
            "threshold_info": {
                "method": f"{threshold_meta['mode']} threshold",
                "otsu_value": float(threshold_meta["otsu_value"]),
            },
            "warp_strategy": str(warp_strategy),
            "warp_layout_score": round(float(warp_layout_score), 5),
            "warnings": warnings,
            "warning_codes": warning_codes,
            "bubble_confidence_json": bubble_name,
            "sid_crop_image": sid_crop_name,
            "mcq_crop_image": mcq_crop_name,
            "result_image": result_name,
        }

    except Exception as exc:
        return {"error": f"Loi xu ly OMR: {exc}"}
