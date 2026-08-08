import unittest
from src.common.schemas import Query, QueryType, CandidateFrame, Answer
from src.role_b_nlp import (
    normalize_text,
    parse_query,
    classify_question,
    extract_object_keywords,
    calculate_object_match_score,
    compute_fusion_score,
)

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

    def test_parse_query_trake(self):
        query = Query(
            query_id="Q_TRAKE_01",
            raw_text=" (1) Người chạy đà. (2) Giậm nhảy qua xà. (3) Tiếp đất an toàn.",
            query_type=QueryType.TRAKE
        )
        parsed = parse_query(query)
        self.assertEqual(parsed.query_type, QueryType.TRAKE)
        self.assertEqual(len(parsed.sub_events), 3)
        self.assertEqual(parsed.sub_events[0], "người chạy đà")

    def test_candidate_frame_and_answer_schema(self):
        frame = CandidateFrame(
            video_id="L01_V001",
            frame_id=125,
            faiss_id=4502,
            clip_score=0.85,
            fusion_score=0.78
        )
        self.assertEqual(frame.faiss_id, 4502)
        self.assertEqual(frame.fusion_score, 0.78)

        ans = Answer(answer_text="5 người", confidence=None)
        self.assertEqual(ans.answer_text, "5 người")
        self.assertIsNone(ans.confidence)

    def test_compute_fusion_score(self):
        score = compute_fusion_score(clip_score=0.8, obj_score=0.5, meta_score=0.0)
        self.assertEqual(score, 0.66)

if __name__ == "__main__":
    unittest.main()
