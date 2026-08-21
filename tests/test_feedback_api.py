import os
import sys
import unittest
import json
import csv
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app
from api.routes.feedback_routes import ALL_GT_FILE

class TestFeedbackAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Ensure we have a clean test file
        self.test_gt_file = ALL_GT_FILE
        if self.test_gt_file.exists():
            # Backup original if needed, but since it's eval data, we might not want to nuke it.
            # To be safe and test cleanly, let's just count rows
            with open(self.test_gt_file, "r", encoding="utf-8") as f:
                self.initial_rows = len(f.readlines())
        else:
            self.initial_rows = 0

    def test_agent2_feedback_payload_valid(self):
        """Test Case 2.1: Valid 8-field payload appends correctly to CSV."""
        payload = {
            "query_id": "mock_query_999",
            "video_id": "V1",
            "frame_idx": 100,
            "verdict": 1,
            "siglip_score": 0.87,
            "obj_score": 0.65,
            "spatial_score": 0.0,
            "has_target_objects": True
        }
        
        response = self.client.post("/api/v1/feedback", json=payload)
        self.assertEqual(response.status_code, 200)
        
        # Verify CSV
        self.assertTrue(self.test_gt_file.exists())
        with open(self.test_gt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        last_line = lines[-1].strip()
        cols = last_line.split(",")
        self.assertEqual(cols[0], "mock_query_999")
        self.assertEqual(cols[4], "0.8700")

    def test_agent2_feedback_payload_missing(self):
        """Test Case 2.2: Missing field should 422 and NOT write to CSV."""
        if self.test_gt_file.exists():
            with open(self.test_gt_file, "r", encoding="utf-8") as f:
                rows_before = len(f.readlines())
        else:
            rows_before = 0
            
        payload_missing = {
            # Missing query_id (required)
            "video_id": "V1",
            "frame_idx": 100,
            "verdict": 1,
            "siglip_score": 0.87,
        }
        
        response = self.client.post("/api/v1/feedback", json=payload_missing)
        self.assertEqual(response.status_code, 422)
        
        if self.test_gt_file.exists():
            with open(self.test_gt_file, "r", encoding="utf-8") as f:
                rows_after = len(f.readlines())
            self.assertEqual(rows_before, rows_after)

if __name__ == "__main__":
    unittest.main()
