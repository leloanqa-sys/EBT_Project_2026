"""
Feedback Routes — /api/v1/feedback
========================================
Handles Ground Truth data collection from the Web UI.
Saves verdicts to outputs/verdicts/ for ML Tuner.
"""
import os
import csv
from pathlib import Path
from typing import Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VERDICTS_DIR = PROJECT_ROOT / "outputs" / "verdicts"
ALL_GT_FILE = VERDICTS_DIR / "ground_truth_all.csv"

router = APIRouter(tags=["feedback"])

class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Unique ID of the query")
    video_id: str = Field(..., description="Video ID")
    frame_idx: int = Field(..., description="Frame index")
    verdict: Union[str, int] = Field(..., description="Verdict: 'MATCH' (1), 'UNCERTAIN' (2), 'MISMATCH' (0)")
    clip_score: float = Field(default=0.0, description="CLIP score")
    obj_score: float = Field(default=0.0, description="Object score")
    spatial_score: float = Field(default=0.0, description="Spatial score")
    fusion_score: float = Field(default=0.0, description="Fusion score")

@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Saves Tester's verdict (MATCH / UNCERTAIN / MISMATCH) to:
      1. outputs/verdicts/human_verdict_{query_id}.csv (per query)
      2. outputs/verdicts/ground_truth_all.csv (aggregated)
    """
    try:
        VERDICTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Normalize verdict string
        v_str = str(req.verdict).upper()
        if v_str in ("1", "MATCH"):
            verdict_clean = "MATCH"
        elif v_str in ("2", "UNCERTAIN"):
            verdict_clean = "UNCERTAIN"
        else:
            verdict_clean = "MISMATCH"
            
        # 1. Save to query-specific verdict file
        q_file = VERDICTS_DIR / f"human_verdict_{req.query_id}.csv"
        q_exists = q_file.exists()
        with open(q_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not q_exists:
                writer.writerow(["query_id", "video_id", "frame_id", "verdict", "clip_score", "obj_score", "spatial_score", "fusion_score"])
            writer.writerow([req.query_id, req.video_id, req.frame_idx, verdict_clean, f"{req.clip_score:.4f}", f"{req.obj_score:.2f}", f"{req.spatial_score:.2f}", f"{req.fusion_score:.4f}"])
            
        # 2. Append to all ground truth file
        all_exists = ALL_GT_FILE.exists()
        with open(ALL_GT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not all_exists:
                writer.writerow(["query_id", "video_id", "frame_id", "verdict", "clip_score", "obj_score", "spatial_score", "fusion_score"])
            writer.writerow([req.query_id, req.video_id, req.frame_idx, verdict_clean, f"{req.clip_score:.4f}", f"{req.obj_score:.2f}", f"{req.spatial_score:.2f}", f"{req.fusion_score:.4f}"])
            
        return {"status": "success", "message": f"Verdict '{verdict_clean}' saved successfully", "verdict": verdict_clean}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@router.get("/feedback/stats")
async def get_feedback_stats():
    """Returns stats about all collected ground truth verdicts."""
    if not ALL_GT_FILE.exists():
        return {"total_samples": 0, "matches": 0, "uncertains": 0, "mismatches": 0}
        
    try:
        total = 0
        matches = 0
        uncertains = 0
        mismatches = 0
        with open(ALL_GT_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                v = str(row.get("verdict", "")).upper()
                if v == "MATCH":
                    matches += 1
                elif v == "UNCERTAIN":
                    uncertains += 1
                else:
                    mismatches += 1
        return {
            "total_samples": total,
            "matches": matches,
            "uncertains": uncertains,
            "mismatches": mismatches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

