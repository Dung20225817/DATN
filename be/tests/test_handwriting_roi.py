import unittest

from app.api.omr.shared import _sanitize_handwriting_fields
from app.services.omr.omr_handwriting import _build_handwriting_rois, _parse_handwriting_config


class HandwritingRoiTests(unittest.TestCase):
    def test_backend_merges_legacy_name_rois_to_single_ho_ten(self):
        cfg = _parse_handwriting_config(
            {
                "enabled": True,
                "field_rois": {
                    "ho_ten_1": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05},
                    "ho_ten_2": {"x": 0.12, "y": 0.26, "w": 0.32, "h": 0.05},
                },
            }
        )

        self.assertEqual(set(cfg["field_rois"].keys()), {"ho_ten"})
        self.assertEqual(cfg["field_rois"]["ho_ten"], {"x": 100, "y": 280, "w": 340, "h": 154})

        rois = _build_handwriting_rois(None, None, None, 1000, 1400, cfg["field_rois"])
        self.assertEqual(set(rois.keys()), {"ho_ten"})
        self.assertEqual(rois["ho_ten"], {"x": 100, "y": 280, "w": 340, "h": 154})

    def test_backend_prefers_direct_ho_ten_over_legacy_rois(self):
        cfg = _parse_handwriting_config(
            {
                "field_rois": {
                    "ho_ten": {"x": 0.2, "y": 0.3, "w": 0.4, "h": 0.08},
                    "ho_ten_1": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05},
                    "ho_ten_2": {"x": 0.12, "y": 0.26, "w": 0.32, "h": 0.05},
                },
            }
        )

        self.assertEqual(cfg["field_rois"], {"ho_ten": {"x": 200, "y": 420, "w": 400, "h": 112}})

    def test_profile_sanitizer_outputs_only_ho_ten_for_legacy_rois(self):
        sanitized = _sanitize_handwriting_fields(
            {
                "enabled": True,
                "save_crops": True,
                "field_rois": {
                    "truong": {"x": 0.01, "y": 0.01, "w": 0.1, "h": 0.1},
                    "ho_ten_1": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05},
                    "ho_ten_2": {"x": 0.12, "y": 0.26, "w": 0.32, "h": 0.05},
                },
            }
        )

        self.assertEqual(set(sanitized["field_rois"].keys()), {"ho_ten"})
        self.assertEqual(
            sanitized["field_rois"]["ho_ten"],
            {"x": 0.1, "y": 0.2, "w": 0.34, "h": 0.11},
        )

    def test_backend_infers_short_form_name_roi_when_profile_missing(self):
        rois = _build_handwriting_rois(
            anchors={},
            sid_roi={"x": 568, "y": 63, "w": 200, "h": 316},
            mcq_roi={"x": 195, "y": 691, "w": 597, "h": 698},
            img_w=1000,
            img_h=1400,
            cfg_field_rois={},
        )

        self.assertEqual(set(rois.keys()), {"ho_ten"})
        self.assertEqual(rois["ho_ten"], {"x": 283, "y": 120, "w": 277, "h": 88})

    def test_backend_does_not_infer_short_form_name_roi_for_long_forms(self):
        rois = _build_handwriting_rois(
            anchors={},
            sid_roi={"x": 520, "y": 50, "w": 220, "h": 300},
            mcq_roi={"x": 160, "y": 690, "w": 780, "h": 700},
            img_w=1000,
            img_h=1400,
            cfg_field_rois={},
            long_form_mode=True,
        )

        self.assertEqual(rois, {})


if __name__ == "__main__":
    unittest.main()
