import time
import uuid
import sys
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.role_c_logic.pipeline_trake import TRAKEPipeline

router = APIRouter(tags=["search"])

class TRAKERequest(BaseModel):
    query: str = Field(..., description="Toàn bộ truy vấn")
    events: List[str] = Field(..., description="Danh sách các sự kiện con theo thứ tự thời gian")
    top_k: int = Field(default=5, description="Số lượng chuỗi video kết quả")

class TRAKEResponse(BaseModel):
    query_id: str
    latency_ms: float
    total_sequences: int
    sequences: List[dict]

@router.post("/search/trake", response_model=TRAKEResponse)
async def search_trake(req: TRAKERequest):
    try:
        pipeline = TRAKEPipeline(detect_threshold=0.3)
        query_id = f"trake_{uuid.uuid4().hex[:8]}"
        
        result = pipeline.run(
            query_id=query_id,
            sub_events=req.events,
            top_k=req.top_k,
            max_time_gap_seconds=30.0
        )
        
        seqs_out = []
        for rank, seq_obj in enumerate(result.get("sequences", [])):
            frames_out = []
            for frame_idx, c_obj in enumerate(seq_obj["frames"]):
                pts = c_obj.pts_time
                pts_str = f"{int(pts // 60):02d}:{int(pts % 60):02d}"
                frames_out.append({
                    "event_idx": frame_idx,
                    "frame_idx": c_obj.frame_idx,
                    "timestamp": pts_str,
                    "frame_url": f"/api/v1/image/{c_obj.video_id}/{c_obj.frame_idx}",
                    "clip_score": round(c_obj.clip_score, 4)
                })
                
            seqs_out.append({
                "rank": rank + 1,
                "video_id": seq_obj["video_id"],
                "avg_score": round(seq_obj["avg_score"], 4),
                "frames": frames_out
            })
            
        return {
            "query_id": query_id,
            "latency_ms": result.get("latency_ms", 0.0),
            "total_sequences": len(seqs_out),
            "sequences": seqs_out
        }
        
    except Exception as e:
        print(f"[TRAKE API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
