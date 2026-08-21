import time
import uuid
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.role_c_logic.pipeline_qa import QAPipeline

router = APIRouter(tags=["search"])

class QARequest(BaseModel):
    query: str = Field(..., description="Mô tả sự kiện")
    question: str = Field(..., description="Câu hỏi chi tiết về sự kiện")
    top_k: int = Field(default=5, description="Số lượng kết quả lấy làm bằng chứng")

class QAResponse(BaseModel):
    query_id: str
    answer: str
    latency_ms: float
    evidence: List[dict]

# Reuse the get_video_metadata from kis_routes or define a simpler one
from api.routes.kis_routes import get_video_metadata

@router.post("/search/qa", response_model=QAResponse)
async def search_qa(req: QARequest):
    from fastapi.concurrency import run_in_threadpool
    from api.main import _get_searcher
    try:
        searcher_instance = _get_searcher()
        pipeline = QAPipeline(detect_threshold=0.3, searcher=searcher_instance)
        query_id = f"qa_{uuid.uuid4().hex[:8]}"
        
        result = await run_in_threadpool(
            pipeline.run,
            query_id=query_id,
            event_description=req.query,
            question=req.question,
            top_k=req.top_k
        )
        
        evidence_list = []
        for rank, c_obj in enumerate(result.get("evidence_candidates", [])):
            pts = c_obj.pts_time
            pts_str = f"{int(pts // 60):02d}:{int(pts % 60):02d}"
            
            evidence_list.append({
                "rank": rank + 1,
                "video_id": c_obj.video_id,
                "frame_idx": c_obj.frame_idx,
                "timestamp": pts_str,
                "frame_url": f"/api/v1/image/{c_obj.video_id}/{c_obj.frame_idx}",
                "siglip_score": round(c_obj.siglip_score, 4)
            })
            
        return {
            "query_id": query_id,
            "answer": result.get("answer", ""),
            "latency_ms": result.get("latency_ms", 0.0),
            "evidence": evidence_list
        }
        
    except Exception as e:
        print(f"[QA API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
