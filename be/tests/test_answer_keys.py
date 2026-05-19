import unittest

from app.services.omr.answer_keys import normalize_choice_token, parse_answer_key_from_text


class AnswerKeyTests(unittest.TestCase):
    def test_normalize_choice_token(self):
        self.assertEqual(normalize_choice_token("A", 4, assume_one_based=False), 0)
        self.assertEqual(normalize_choice_token("4", 4, assume_one_based=True), 3)
        self.assertIsNone(normalize_choice_token("Z", 4, assume_one_based=False))

    def test_parse_numbered_and_csv_formats(self):
        self.assertEqual(parse_answer_key_from_text("Câu 1: A\nCâu 2: C", 4), [0, 2])
        self.assertEqual(parse_answer_key_from_text("1,2,3,4", 4), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
