import unittest

import numpy as np

from app.services.omr.omr_numeric import _decode_numeric_columns
from app.services.omr.omr_visualize import _draw_result_overlay


class OmrVisualizeTests(unittest.TestCase):
    def assert_pixel_close(self, actual, expected, tolerance=2):
        self.assertTrue(
            np.all(np.abs(actual.astype(np.int16) - np.asarray(expected, dtype=np.int16)) <= tolerance),
            msg=f"{actual.tolist()} != {list(expected)}",
        )

    def test_result_overlay_uses_answer_compare_colors(self):
        image = np.full((260, 260, 3), 255, dtype=np.uint8)
        rows = []
        for q_num in range(1, 6):
            y1 = 70 + (q_num * 28)
            rows.append(
                {
                    "question": q_num,
                    "selected": -1,
                    "cell_boxes": [
                        [40, y1, 60, y1 + 20],
                        [70, y1, 90, y1 + 20],
                        [100, y1, 120, y1 + 20],
                        [130, y1, 150, y1 + 20],
                    ],
                }
            )

        rows[0]["selected"] = 0
        rows[1]["selected"] = 1
        rows[3]["selected"] = 0

        canvas = _draw_result_overlay(
            image,
            sid_roi={"x": 5, "y": 5, "w": 20, "h": 20},
            code_roi={"x": 30, "y": 5, "w": 20, "h": 20},
            mcq_roi={"x": 10, "y": 90, "w": 180, "h": 160},
            mcq_rows=rows,
            student_id="123456",
            exam_code="111",
            score=1,
            graded_questions=5,
            handwriting_rois=None,
            answer_compare=[
                {"question": 1, "selected": 0, "correct": 0, "status": "correct"},
                {"question": 2, "selected": 1, "correct": 2, "status": "wrong"},
                {"question": 3, "selected": -1, "correct": 3, "status": "uncertain"},
                {"question": 4, "selected": 0, "correct": -1, "status": "no-key"},
                {"question": 5, "selected": -1, "correct": -1, "status": "blank-no-key"},
            ],
            sid_selected_cells=[
                {"digit_index": 0, "selected_digit": 1, "cell_box": [12, 120, 28, 136], "score": 0.9},
            ],
            code_selected_cells=[
                {"digit_index": 0, "selected_digit": 2, "cell_box": [42, 120, 58, 136], "score": 0.9},
            ],
        )

        self.assertTrue(np.array_equal(canvas[5, 5], np.array([0, 160, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[5, 30], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[90, 10], np.array([255, 255, 255], dtype=np.uint8)))
        self.assert_pixel_close(canvas[128, 20], [168, 223, 255])
        self.assert_pixel_close(canvas[128, 50], [255, 236, 168])
        self.assertTrue(np.array_equal(canvas[121, 13], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[121, 43], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[108, 58], np.array([0, 220, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[136, 88], np.array([0, 0, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[136, 118], np.array([0, 200, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[164, 148], np.array([0, 200, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[192, 58], np.array([0, 0, 255], dtype=np.uint8)))
        self.assertTrue(np.array_equal(canvas[220, 58], np.array([255, 255, 255], dtype=np.uint8)))

    def test_numeric_decoder_returns_selected_cell_boxes(self):
        gray = np.full((140, 80), 255, dtype=np.uint8)
        binary = np.zeros((140, 80), dtype=np.uint8)
        roi = {"x": 10, "y": 20, "w": 30, "h": 100}

        selected_digits = [1, 2, 3]
        col_edges = np.linspace(roi["x"], roi["x"] + roi["w"], 4, dtype=np.float32)
        row_edges = np.linspace(roi["y"], roi["y"] + roi["h"], 11, dtype=np.float32)
        for col_idx, digit in enumerate(selected_digits):
            x1 = int(round(float(col_edges[col_idx])))
            x2 = int(round(float(col_edges[col_idx + 1])))
            y1 = int(round(float(row_edges[digit])))
            y2 = int(round(float(row_edges[digit + 1])))
            gray[y1:y2, x1:x2] = 0
            binary[y1:y2, x1:x2] = 255

        result = _decode_numeric_columns(gray, binary, roi, digits=3)

        self.assertEqual(result["value"], "123")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["scores"]), 10)
        self.assertEqual(len(result["selected_cells"]), 3)
        self.assertEqual(
            [(cell["digit_index"], cell["selected_digit"]) for cell in result["selected_cells"]],
            [(0, 1), (1, 2), (2, 3)],
        )
        self.assertEqual(result["selected_cells"][0]["cell_box"], [10, 30, 20, 40])


if __name__ == "__main__":
    unittest.main()
