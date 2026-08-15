"""
Feedback Routes — /api/v1/feedback
========================================
Handles Ground Truth data collection from the Tester UI.
"""
import os
import csv
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
GT_FILE = EVAL_DIR / "ground_truth.csv"

router = APIRouter(tags=["feedback"])

class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Unique ID of the query")
    video_id: str = Field(..., description="Video ID")
    frame_idx: int = Field(..., description="Frame index")
    verdict: int = Field(..., description="1 for MATCH, 0 for MISMATCH")
    clip_score: float = Field(..., description="CLIP similarity score at time of search")
    obj_score: float = Field(..., description="Object-match score at time of search")
    spatial_score: float = Field(..., description="Spatial constraint score at time of search")
    has_target_objects: bool = Field(..., description="Branch flag matching fusion_score.py's weight-selection condition")

@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Saves Tester's verdict (Match/Mismatch) to the Ground Truth CSV file.
    This data will be used by ml_tuner.py to optimize weights.
    """
    try:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = GT_FILE.exists()
        
        with open(GT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not file_exists:
                writer.writerow(["query_id", "video_id", "frame_idx", "verdict", "clip_score", "obj_score", "spatial_score", "has_target_objects"])
            
            writer.writerow([req.query_id, req.video_id, req.frame_idx, req.verdict, req.clip_score, req.obj_score, req.spatial_score, req.has_target_objects])
            
        return {"status": "success", "message": "Feedback saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@router.get("/feedback/stats")
async def get_feedback_stats():
    """Returns basic stats about the collected ground truth data."""
    if not GT_FILE.exists():
        return {"total_samples": 0, "matches": 0, "mismatches": 0}
        
    try:
        total = 0
        matches = 0
        with open(GT_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if str(row.get("verdict", "0")) == "1":
                    matches += 1
        return {
            "total_samples": total,
            "matches": matches,
            "mismatches": total - matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
