import tempfile
import unittest
from pathlib import Path

from app.api.omr.shared import _build_runtime_config, _resolve_profile
from app.services.omr.omr_layout import _build_rois_from_anchors
from app.services.omr.omr_service import (
    _derive_mcq_center_bands_from_span,
    _reject_long_form_low_mcq_top,
    _validate_mcq_block_bands,
    process_omr_exam,
)


class Omr120FlowTests(unittest.TestCase):
    @staticmethod
    def _run_120(image_path: Path, output_dir: Path):
        profile = _resolve_profile("120pdf")
        runtime = _build_runtime_config(
            profile,
            num_questions=120,
            num_choices=4,
            rows_per_block=30,
            num_blocks=4,
            student_id_digits=6,
            sid_has_write_row=True,
        )
        answer_key = [1, 0, 2, 3, 2, 1, 1, 2, 2, 2] + [0] * 110
        return process_omr_exam(
            image_path=str(image_path),
            output_folder=str(output_dir),
            answer_key=answer_key,
            answer_key_by_code=None,
            questions=runtime["num_questions"],
            choices=runtime["num_choices"],
            rows_per_block=runtime["rows_per_block"],
            num_blocks=runtime["num_blocks"],
            student_id_digits=runtime["student_id_digits"],
            sid_has_write_row=runtime["sid_has_write_row"],
            profile_sid_roi=runtime["profile_sid_roi"],
            profile_sid_roi_lock=runtime["profile_sid_roi_lock"],
            profile_mcq_roi=runtime["profile_mcq_roi"],
            profile_exam_code_roi=runtime["profile_exam_code_roi"],
            profile_sid_row_offsets=runtime["profile_sid_row_offsets"],
            profile_disable_mcq_rescue=runtime["profile_disable_mcq_rescue"],
            profile_mcq_decode=runtime["profile_mcq_decode"],
            profile_threshold_mode=runtime["profile_threshold_mode"],
            profile_page_size_pt=runtime["profile_page_size_pt"],
            profile_handwriting_fields=runtime["profile_handwriting_fields"],
        )

    @staticmethod
    def _run_40(image_path: Path, output_dir: Path):
        profile = _resolve_profile("40-cau-pdf")
        runtime = _build_runtime_config(
            profile,
            num_questions=40,
            num_choices=4,
            rows_per_block=17,
            num_blocks=3,
            student_id_digits=6,
            sid_has_write_row=True,
        )
        answer_key = [0, 1, 2, 3, 2] + [0] * 35
        return process_omr_exam(
            image_path=str(image_path),
            output_folder=str(output_dir),
            answer_key=answer_key,
            answer_key_by_code=None,
            questions=runtime["num_questions"],
            choices=runtime["num_choices"],
            rows_per_block=runtime["rows_per_block"],
            num_blocks=runtime["num_blocks"],
            student_id_digits=runtime["student_id_digits"],
            sid_has_write_row=runtime["sid_has_write_row"],
            profile_sid_roi=runtime["profile_sid_roi"],
            profile_sid_roi_lock=runtime["profile_sid_roi_lock"],
            profile_mcq_roi=runtime["profile_mcq_roi"],
            profile_exam_code_roi=runtime["profile_exam_code_roi"],
            profile_sid_row_offsets=runtime["profile_sid_row_offsets"],
            profile_disable_mcq_rescue=runtime["profile_disable_mcq_rescue"],
            profile_mcq_decode=runtime["profile_mcq_decode"],
            profile_threshold_mode=runtime["profile_threshold_mode"],
            profile_page_size_pt=runtime["profile_page_size_pt"],
            profile_handwriting_fields=runtime["profile_handwriting_fields"],
        )

    def test_rejects_low_long_form_mcq_top_fid(self):
        self.assertTrue(
            _reject_long_form_low_mcq_top(
                {"used": True, "y_top_anchor": 685.6, "line_h": 37.3},
                1400,
            )
        )
        self.assertFalse(
            _reject_long_form_low_mcq_top(
                {"used": True, "y_top_anchor": 422.3, "line_h": 28.4},
                1400,
            )
        )

    def test_auto_sid_roi_is_inset_inside_sid_anchors(self):
        anchors = {
            "sid_top": (560.0, 48.0),
            "sid_bottom": (560.0, 390.0),
            "code_top": (800.0, 60.0),
            "code_bottom": (800.0, 390.0),
            "mcq_left_top": (230.0, 430.0),
            "mcq_left_bottom": (230.0, 1280.0),
            "mcq_right_top": (900.0, 430.0),
            "mcq_right_bottom": (900.0, 1280.0),
        }

        rois, _ = _build_rois_from_anchors(
            anchors,
            img_w=1000,
            img_h=1400,
            rows_per_block=30,
            sid_roi_cfg=None,
            code_roi_cfg=None,
            mcq_roi_cfg=None,
            block_count_hint=4,
        )
        sid_roi = rois["sid"]
        old_auto_left = int(round(anchors["sid_top"][0] + 1000 * 0.007))
        old_auto_right = old_auto_left + int(round(1000 * 0.188))

        self.assertGreater(sid_roi["y"], anchors["sid_top"][1])
        self.assertLess(sid_roi["y"] + sid_roi["h"], anchors["sid_bottom"][1])
        self.assertGreater(sid_roi["x"], old_auto_left)
        self.assertLess(sid_roi["x"] + sid_roi["w"], old_auto_right)

    def test_120_profile_uses_auto_sid_roi_and_decodes_known_sheet(self):
        image_path = Path(__file__).with_name("20260421_100238_118524_test2.jpg")
        if not image_path.exists():
            self.skipTest(f"Missing fixture: {image_path}")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = self._run_120(image_path, output_dir)
            bubble_payload = None
            bubble_name = result.get("bubble_confidence_json")
            if bubble_name:
                import json

                bubble_payload = json.loads((output_dir / str(bubble_name)).read_text(encoding="utf-8"))

        self.assertTrue(result.get("success"), result.get("error"))
        self.assertEqual(result.get("student_id"), "233443")
        self.assertEqual(result.get("exam_code"), "111")
        self.assertEqual(result.get("user_answers", [])[:10], [1, 0, 2, 3, 2, 1, 1, 2, 2, 2])
        self.assertLess(int((result.get("roi_boxes") or {}).get("mcq", {}).get("y", 9999)), 500)
        mcq_roi = (result.get("roi_boxes") or {}).get("mcq", {})
        self.assertGreaterEqual(int(mcq_roi.get("x", 0)) + int(mcq_roi.get("w", 0)), 900)
        self.assertLessEqual(int(mcq_roi.get("x", 0)) + int(mcq_roi.get("w", 9999)), 930)
        self.assertIn("SID_PROFILE_ROI_IGNORED_LONG_FORM", result.get("warning_codes") or [])
        self.assertIn("SID_AUTO_ROI_INSET_APPLIED", result.get("warning_codes") or [])
        self.assertIn("MCQ_LONG_FORM_BUBBLE_BANDS_DETECTED", result.get("warning_codes") or [])
        self.assertIn("MCQ_LONG_FORM_ROI_EXPANDED_FOR_BANDS", result.get("warning_codes") or [])
        roi_detection = result.get("roi_detection") or {}
        self.assertEqual(roi_detection.get("coordinate_mapping_mcq_decode_x_source"), "detected-bubbles")
        self.assertTrue(roi_detection.get("coordinate_mapping_mcq_long_form_block_bands_used"))
        self.assertTrue(roi_detection.get("coordinate_mapping_mcq_answer_band_detection_used"))
        self.assertTrue(roi_detection.get("coordinate_mapping_mcq_roi_expanded_for_bands"))
        self.assertTrue(roi_detection.get("coordinate_mapping_mcq_long_form_row_offsets_used"))
        self.assertEqual(roi_detection.get("coordinate_mapping_mcq_row_offsets_source"), "detected-bubbles")
        self.assertNotIn("MCQ_BLOCK_BANDS_DISABLED_LONG_FORM", result.get("warning_codes") or [])
        self.assertNotIn("MCQ_BLOCK_BANDS_REJECTED", result.get("warning_codes") or [])
        self.assertIsInstance(bubble_payload, dict)
        sid_roi = (result.get("roi_boxes") or {}).get("student_id") or {}
        anchors = ((bubble_payload.get("coordinate_mapping") or {}).get("anchors") or {})
        sid_top_y = float((anchors.get("sid_top") or {}).get("y"))
        sid_bottom_y = float((anchors.get("sid_bottom") or {}).get("y"))
        self.assertGreater(float(sid_roi.get("y")), sid_top_y)
        self.assertLess(float(sid_roi.get("y")) + float(sid_roi.get("h")), sid_bottom_y)
        rows = bubble_payload.get("rows") or []
        self.assertGreaterEqual(len(rows), 91)
        q91_centers = [round((box[0] + box[2]) / 2.0) for box in rows[90].get("cell_boxes") or []]
        self.assertEqual(q91_centers, [796, 826, 856, 886])
        q11_y = round(sum((box[1] + box[3]) / 2.0 for box in rows[10].get("cell_boxes") or []) / 4.0)
        q21_y = round(sum((box[1] + box[3]) / 2.0 for box in rows[20].get("cell_boxes") or []) / 4.0)
        self.assertGreaterEqual(q11_y, 760)
        self.assertLessEqual(q11_y, 776)
        self.assertGreaterEqual(q21_y, 1072)
        self.assertLessEqual(q21_y, 1090)
        row_meta = ((bubble_payload.get("coordinate_mapping") or {}).get("row_offsets_meta") or {})
        self.assertEqual(row_meta.get("source"), "detected-bubbles")
        self.assertGreaterEqual(len(row_meta.get("row_offsets_px") or []), 30)

    def test_120_rejects_cascaded_low_mcq_marker_roi_and_no_name_fallback(self):
        image_path = Path(__file__).with_name("20260418_121820_651438_camera_1776489482496.jpg")
        if not image_path.exists():
            self.skipTest(f"Missing fixture: {image_path}")

        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_120(image_path, Path(tmp))

        self.assertTrue(result.get("success"), result.get("error"))
        self.assertIn("MCQ_LONG_FORM_TOP_FID_REJECTED", result.get("warning_codes") or [])
        self.assertEqual(((result.get("roi_boxes") or {}).get("handwriting") or {}), {})
        self.assertEqual(((result.get("handwriting") or {}).get("fields") or {}).get("ho_ten", {}).get("status"), "missing-roi")

    def test_40_profile_revalidates_block_bands_after_roi_and_decodes_known_sheet(self):
        image_path = Path(__file__).with_name("20260418_121820_651438_camera_1776489482496.jpg")
        if not image_path.exists():
            self.skipTest(f"Missing fixture: {image_path}")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = self._run_40(image_path, output_dir)
            import json

            bubble_payload = json.loads((output_dir / str(result.get("bubble_confidence_json"))).read_text(encoding="utf-8"))

        self.assertTrue(result.get("success"), result.get("error"))
        self.assertEqual(result.get("student_id"), "111111")
        self.assertEqual(result.get("exam_code"), "111")
        self.assertEqual(result.get("user_answers", [])[:5], [0, 1, 2, 3, 2])
        self.assertIn("SID_AUTO_ROI_INSET_APPLIED", result.get("warning_codes") or [])
        sid_roi = (result.get("roi_boxes") or {}).get("student_id") or {}
        self.assertGreaterEqual(int(sid_roi.get("x", 0)), 576)
        self.assertLessEqual(int(sid_roi.get("w", 9999)), 184)
        self.assertLessEqual(int(sid_roi.get("x", 0)) + int(sid_roi.get("w", 9999)), 758)

        roi_detection = result.get("roi_detection") or {}
        self.assertFalse(roi_detection.get("coordinate_mapping_mcq_long_form_block_bands_used"))
        self.assertFalse(roi_detection.get("coordinate_mapping_mcq_long_form_row_offsets_used"))
        self.assertEqual(roi_detection.get("coordinate_mapping_mcq_decode_x_source"), "geometry")
        self.assertTrue(roi_detection.get("coordinate_mapping_mcq_block_bands_revalidated_after_roi"))
        self.assertEqual(roi_detection.get("coordinate_mapping_mcq_map_search_reason"), "locked-by-short-form-block-bands")
        self.assertNotIn("MCQ_MAP_SEARCH_RESCUE", result.get("warning_codes") or [])

        coord = bubble_payload.get("coordinate_mapping") or {}
        self.assertEqual((coord.get("block_bands_final_meta") or {}).get("source"), "geometry")
        self.assertEqual((coord.get("block_bands_final_meta") or {}).get("reason"), "ok")
        self.assertEqual(len(coord.get("block_bands") or []), 3)

    def test_long_form_block_bands_validate_and_reject_outlier(self):
        roi = {"x": 212, "y": 427, "w": 647, "h": 973}
        bands = _derive_mcq_center_bands_from_span(240.0, 839.0, roi, 4, 4)
        validated, meta = _validate_mcq_block_bands(bands, roi, 4, 4, True, source="unit")
        self.assertTrue(meta.get("used"))
        self.assertEqual(len(validated), 4)

        outlier = list(validated)
        outlier[-1] = (930.0, 1030.0)
        rejected, rejected_meta = _validate_mcq_block_bands(outlier, roi, 4, 4, True, source="unit")
        self.assertEqual(rejected, [])
        self.assertEqual(rejected_meta.get("reason"), "outside-mcq-roi")


if __name__ == "__main__":
    unittest.main()
