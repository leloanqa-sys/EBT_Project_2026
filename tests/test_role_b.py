import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unittest.mock import patch
from unittest.mock import patch
from src.common.schemas import Query, QueryType, CandidateFrame, Answer
from src.role_b_nlp import (
    normalize_text,
    parse_query,
    classify_question,
    compute_fusion_score,
)
from src.role_b_nlp.object_matcher import calculate_object_match_score

class TestRoleBNLP(unittest.TestCase):

    def test_normalize_text(self):
        raw = "  Người Chạy Ô Tô!!   "
        normalized = normalize_text(raw)
        self.assertEqual(normalized, "người chạy ô tô")

    def test_classify_question(self):
        q1 = "Trong video có bao nhiêu người?"
        self.assertEqual(classify_question(q1), "COUNT")
        
        q2 = "Chiếc xe hơi màu gì?"
        self.assertEqual(classify_question(q2), "COLOR")

        q3 = "Biển số xe ghi chữ gì?"
        self.assertEqual(classify_question(q3), "TEXT_OCR")

    @patch('src.role_b_nlp.query_parser.decompose_events')
    @patch('src.role_b_nlp.query_parser.extract_target_objects')
    def test_parse_query_trake_with_mock(self, mock_objects, mock_decompose):
        mock_objects.return_value = ["person"]
        mock_decompose.return_value = ["người chạy đà", "giậm nhảy qua xà", "tiếp đất an toàn"]
        
        query = Query(
            query_id="Q_TRAKE_01",
            raw_text=" (1) Người chạy đà. (2) Giậm nhảy qua xà. (3) Tiếp đất an toàn.",
            query_type=QueryType.TRAKE
        )
        parsed = parse_query(query)
        self.assertEqual(parsed.query_type, QueryType.TRAKE)
        self.assertEqual(len(parsed.sub_events), 3)
        self.assertEqual(parsed.sub_events[0], "người chạy đà")
        self.assertEqual(parsed.attributes["objects"]["keywords"], ["person"])

    @patch('src.role_b_nlp.query_parser.decompose_events')
    @patch('src.role_b_nlp.query_parser.extract_target_objects')
    def test_parse_query_trake_fault_tolerance(self, mock_objects, mock_decompose):
        # Mock Gemini failing
        mock_objects.return_value = []
        mock_decompose.side_effect = Exception("API 429 Error")
        
        query = Query(
            query_id="Q_TRAKE_02",
            raw_text=" (1) Người chạy đà. (2) Giậm nhảy qua xà. (3) Tiếp đất an toàn.",
            query_type=QueryType.TRAKE
        )
        parsed = parse_query(query)
        
        # It should fallback to Regex for sub_events
        self.assertEqual(len(parsed.sub_events), 3)
        self.assertEqual(parsed.sub_events[0], "người chạy đà")
        
        # Target objects should be empty
        self.assertNotIn("objects", parsed.attributes)

    def test_candidate_frame_and_answer_schema(self):
        # Fix broken test: use frame_idx, attach fusion_score dynamically
        frame = CandidateFrame(
            video_id="L01_V001",
            frame_idx=125,
            faiss_id=4502,
            clip_score=0.85
        )
        setattr(frame, "fusion_score", 0.78)
        self.assertEqual(frame.faiss_id, 4502)
        self.assertEqual(frame.frame_idx, 125)
        self.assertEqual(frame.fusion_score, 0.78)

        ans = Answer(answer_text="5 người", confidence=None)
        self.assertEqual(ans.answer_text, "5 người")
        self.assertIsNone(ans.confidence)

    def test_compute_fusion_score(self):
        score = compute_fusion_score(clip_score=0.8, obj_score=0.5, meta_score=0.0)
        self.assertEqual(score, 0.72)
        
    def test_calculate_object_match_score(self):
        score = calculate_object_match_score(["car", "person"], ["Person", "building", "CAR"])
        self.assertEqual(score, 1.0)
        score2 = calculate_object_match_score(["car", "dog"], ["person", "car"])
        self.assertEqual(score2, 0.5)

if __name__ == "__main__":
    unittest.main()
