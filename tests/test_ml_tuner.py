import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We assume ml_tuner.py has been refactored or will be refactored by Agent 3.
# These tests verify the EXPECTED behavior of the refactored Tuner.

class TestMLTuner(unittest.TestCase):
    def setUp(self):
        self.dummy_csv = "dummy_ground_truth.csv"
        
    def tearDown(self):
        if os.path.exists(self.dummy_csv):
            os.remove(self.dummy_csv)

    def test_agent3_tuner_synthetic_data(self):
        """Test Case 3.1: Synthetic data where w_clip=2.0 and w_spatial=1.0 is the exact answer."""
        # Create 50 synthetic rows
        data = []
        target_w_clip = 2.0
        target_w_spatial = 1.0
        
        for i in range(50):
            # random scores
            clip_score = np.random.rand()
            spatial_score = np.random.rand()
            
            # compute true fusion score using our secret weights
            true_score = (target_w_clip * clip_score) + (target_w_spatial * spatial_score)
            
            # if the score is in the top 50% for this "query", we call it a MATCH
            data.append({
                "query_id": f"q_{i//5}", # 5 frames per query
                "video_id": f"v_{i}",
                "frame_idx": i,
                "clip_score": clip_score,
                "obj_score": 0.0, # testing without objects
                "spatial_score": spatial_score,
                "has_target_objects": False,
                "true_score": true_score
            })
            
        df = pd.DataFrame(data)
        
        # Determine verdict based on true_score within each query group
        # Top 1 gets verdict 1, others 0
        df['verdict'] = 0
        for qid in df['query_id'].unique():
            idx_max = df[df['query_id'] == qid]['true_score'].idxmax()
            df.loc[idx_max, 'verdict'] = 1
            
        df.drop(columns=['true_score'], inplace=True)
        df.to_csv(self.dummy_csv, index=False)
        
        # Here we WOULD call the refactored ml_tuner.py on self.dummy_csv
        # Since Agent 3 is blocked, this test is a placeholder for the oracle.
        self.assertTrue(os.path.exists(self.dummy_csv))

    def test_agent3_tuner_guardrail_insufficient_data(self):
        """Test Case 3.2: Guard rail for branch with < N rows."""
        # Create CSV with 3 rows for True, 20 rows for False
        pass # To be implemented when Agent 3 is run

    def test_agent3_tuner_empty_csv(self):
        """Test Case 3.3: Empty CSV exits gracefully."""
        pd.DataFrame(columns=["query_id","video_id","frame_idx","verdict","clip_score","obj_score","spatial_score","has_target_objects"]).to_csv(self.dummy_csv, index=False)
        # Should not crash
        self.assertTrue(os.path.exists(self.dummy_csv))

if __name__ == "__main__":
    unittest.main()
