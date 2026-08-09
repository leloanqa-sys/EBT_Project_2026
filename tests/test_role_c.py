import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from dataclasses import replace
from src.common.schemas import CandidateFrame, SubmissionItem, QueryType


# Ensure project root is in sys.path regardless of execution method
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.role_c_logic.ranking import (
    cluster_by_event,
    _representative,
    rank_5budget,
    _get_score
)
from src.role_c_logic.output_formatter import build_submission, format_submission
from src.role_c_logic.pipeline_kis import run_kis

class TestRoleCLogic(unittest.TestCase):

    def test_representative_immutability(self):
        """Verify _representative uses dataclasses.replace and does NOT mutate original object."""
        c1 = CandidateFrame(faiss_id=1, video_id="V1", frame_idx=10, clip_score=0.5)
        c2 = CandidateFrame(faiss_id=2, video_id="V1", frame_idx=20, clip_score=0.9)
        c3 = CandidateFrame(faiss_id=3, video_id="V1", frame_idx=30, clip_score=0.7)
        
        cluster = [c1, c2, c3]
        rep = _representative(cluster)
        
        # Best score in cluster is c2 (0.9), median frame_idx is c2 (20)
        self.assertEqual(rep.faiss_id, 2)
        self.assertEqual(rep.frame_idx, 20)
        self.assertEqual(rep.clip_score, 0.9)
        
        # Verify original c2 was NOT mutated if median had differed
        c4 = CandidateFrame(faiss_id=4, video_id="V1", frame_idx=100, clip_score=0.95)
        cluster2 = [c1, c2, c3, c4]  # median is index len//2 = 2 -> c3.frame_idx = 30
        rep2 = _representative(cluster2)
        
        self.assertEqual(rep2.faiss_id, 4)
        self.assertEqual(rep2.frame_idx, 30)  # rep2 gets median frame_idx
        self.assertEqual(c4.frame_idx, 100)   # original c4 retains frame_idx=100 (unmutated!)

    def test_cluster_by_event_gap(self):
        """Verify cluster_by_event groups frames within gap_threshold from cluster[0]."""
        c1 = CandidateFrame(faiss_id=1, video_id="V1", frame_idx=10, clip_score=0.8)
        c2 = CandidateFrame(faiss_id=2, video_id="V1", frame_idx=20, clip_score=0.6)  # 20 - 10 = 10 <= 15 -> same cluster
        c3 = CandidateFrame(faiss_id=3, video_id="V1", frame_idx=25, clip_score=0.7)  # 25 - 10 = 15 <= 15 -> same cluster
        c4 = CandidateFrame(faiss_id=4, video_id="V1", frame_idx=30, clip_score=0.9)  # 30 - 10 = 20 > 15  -> new cluster
        
        candidates = [c1, c2, c3, c4]
        representatives = cluster_by_event(candidates, gap_threshold=15)
        
        self.assertEqual(len(representatives), 2)
        # Cluster 1: [c1, c2, c3] -> max score is c1 (0.8), median frame_idx is c2 (20)
        self.assertEqual(representatives[0].frame_idx, 20)
        # Cluster 2: [c4] -> c4 (30)
        self.assertEqual(representatives[1].frame_idx, 30)

    def test_rank_5budget_unique_videos(self):
        """Verify ranks 2-5 select candidates from 4 unique video_ids."""
        candidates = [
            CandidateFrame(faiss_id=1, video_id="V1", frame_idx=10, clip_score=0.95), # Top 1
            CandidateFrame(faiss_id=2, video_id="V1", frame_idx=50, clip_score=0.90), # V1 (same as top 1)
            CandidateFrame(faiss_id=3, video_id="V2", frame_idx=15, clip_score=0.85), # V2
            CandidateFrame(faiss_id=4, video_id="V2", frame_idx=40, clip_score=0.80), # V2 (same as V2)
            CandidateFrame(faiss_id=5, video_id="V3", frame_idx=10, clip_score=0.75), # V3
            CandidateFrame(faiss_id=6, video_id="V4", frame_idx=12, clip_score=0.70), # V4
            CandidateFrame(faiss_id=7, video_id="V5", frame_idx=18, clip_score=0.65), # V5
        ]
        
        ranked_items = rank_5budget(candidates, strategy="diversify")
        
        self.assertEqual(len(ranked_items), 7)
        self.assertEqual(ranked_items[0].video_id, "V1") # Rank 1
        
        # Ranks 2-5 must be from V2, V3, V4, V5 (4 distinct video_ids)
        rank_2_5_videos = [item.video_id for item in ranked_items[1:5]]
        self.assertEqual(set(rank_2_5_videos), {"V2", "V3", "V4", "V5"})
        
        # Verify frame_id mapping
        self.assertEqual(ranked_items[0].frame_id, 10)

    def test_format_submission_and_kis(self):
        """Smoke test for output formatter and KIS pipeline."""
        sample_query = "người nói chuyện"
        out_csv = run_kis(sample_query, query_id="test_unit_001", top_k_raw=50)
        
        self.assertTrue(os.path.exists(out_csv))
        with open(out_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # 1 header line + up to 50 rows
        self.assertTrue(len(lines) > 1)
        self.assertTrue("rank,video_id,frame_id" in lines[0])

if __name__ == "__main__":
    unittest.main()
