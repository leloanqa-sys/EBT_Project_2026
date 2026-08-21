"""
KIS Search Routes — /api/v1/search/kis
========================================
Handles Known Item Search queries:
  1. Receives raw Vietnamese text query
  2. Runs through NLP → Retrieval → Fusion → Ranking pipeline
  3. Returns JSON with ranked results + explainability data
"""
import time
import uuid
import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ── Resolve project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from tools.review_tool import resolve_keyframe_b64
import base64
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter(tags=["search"])


# ── Request/Response Models ──

class SearchRequest(BaseModel):
    """Search request payload."""
    query: str = Field(..., min_length=1, description="Raw Vietnamese text query")
    query_type: str = Field(default="KIS", description="Query type: KIS, QA, TRAKE")
    top_k: int = Field(default=100, ge=1, le=500, description="Number of results to return")
    question: Optional[str] = Field(default=None, description="Question text for QA type")


class ResultItem(BaseModel):
    rank: int
    video_id: str
    frame_idx: int
    siglip_score: float
    obj_score: float
    fusion_score: float
    pts_time: float = 0.0
    timestamp: str = "00:00"
    frame_url: str
    detected_labels: List[str] = []
    watch_url: Optional[str] = None
    video_title: Optional[str] = None
    vqa_answer: Optional[str] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

class ParsedInfo(BaseModel):
    normalized_text: str = ""
    extracted_objects: List[str] = []
    sub_events: List[str] = []


class SearchResponse(BaseModel):
    query_id: str
    query: str
    query_type: str
    total_results: int
    search_time_ms: float
    cache_hit: bool = False
    parsed_info: ParsedInfo
    results: List[ResultItem]


# ── Metadata cache helper for Tester Verification ──
from functools import lru_cache
import json

@lru_cache(maxsize=1000)
def get_video_metadata(video_id: str) -> dict:
    meta_path = PROJECT_ROOT / "data" / "raw" / "media-info" / f"{video_id}.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


from fastapi.responses import FileResponse, Response
import urllib.parse
from pydantic import BaseModel, Field

# ── Dynamic Image Resolver Endpoint ──
@router.get("/image/{video_id}/{frame_idx}")
def get_frame_image(video_id: str, frame_idx: int):
    """
    Resolve frame image dynamically from .zip or .mp4 files using tools logic.
    """
    try:
        b64_str, keyframe_n, expected_fname = resolve_keyframe_b64(
            video_id,
            frame_idx,
            keyframes_root=str(PROJECT_ROOT / "data" / "raw" / "keyframes"),
            allow_remote=False
        )
        
        if b64_str:
            # b64_str is like "data:image/jpeg;base64,....."
            img_data = base64.b64decode(b64_str.split(",")[1])
            mime_type = b64_str.split(";")[0].split(":")[1]
            return Response(
                content=img_data,
                media_type=mime_type,
                headers={"Cache-Control": "public, max-age=86400"}
            )
            
    except Exception as e:
        print(f"[API] Image resolver error: {e}")
        pass
        
    raise HTTPException(status_code=404, detail="Frame image not found")



def _run_real_search(query: str, query_type: str, top_k: int, question: Optional[str] = None) -> dict:
    """Run the actual search pipeline using MVPPipeline."""
    from src.pipeline import MVPPipeline
    import time
    import uuid
    from api.main import _get_searcher

    start_time = time.perf_counter()

    # We only use searcher check to ensure fail-fast, but MVPPipeline will load it anyway
    searcher = _get_searcher()
    if searcher is None:
        raise HTTPException(status_code=503, detail="VectorSearcher not loaded")

    pipeline = MVPPipeline(detect_threshold=0.3, top_k_retrieve=500, searcher=searcher)
    query_id = f"api_{uuid.uuid4().hex[:8]}"
    
    # Run pipeline
    result = pipeline.run(query_id, query, query_type, question=question)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Read inferred query_type from IR Graph
    inferred_type = query_type
    if result.operator_trace and "ir_graph" in result.operator_trace:
        inferred_type = result.operator_trace["ir_graph"].get("query_type", query_type)

    # Build response format
    results = []
    
    # Extract target objects from trace/ir
    target_objs = []
    if result.operator_trace and "ir_graph" in result.operator_trace:
        ir = result.operator_trace["ir_graph"]
        target_objs = [e.get("label", "") for e in ir.get("entities", [])]
        
    for c_obj in result.candidates:
        if len(results) >= top_k:
            break
            
        rank = len(results) + 1
        pts = c_obj.pts_time
        pts_str = f"{int(pts // 60):02d}:{int(pts % 60):02d}"

        meta = get_video_metadata(c_obj.video_id)
        watch_base = meta.get("watch_url", "")
        v_title = meta.get("title", c_obj.video_id)
        sec_int = int(pts)
        watch_url = f"{watch_base}&t={sec_int}s" if watch_base else None

        # Fetch labels from shared MetadataCache (reuse from executor, not new instance per request)
        detected_labels = []
        try:
            db_meta = pipeline._executor.meta_cache.get_metadata(c_obj.video_id, c_obj.frame_idx)
            if db_meta:
                detected_labels = list(dict.fromkeys([d["class_entity"] for d in db_meta]))[:8]
        except Exception:
            pass

        # Bug #2 Fix: DO NOT recalculate obj_score here.
        # The pipeline executor already computed obj_score using the full synonyms_map + taxonomy_loader.
        # Re-calculating with a simple string match would give inconsistent/wrong values.
        obj_score = getattr(c_obj, 'obj_score', 0.0)
        # Use the REAL fusion_score from the pipeline (includes Spatial & VQA logic)
        fusion_score = c_obj.fusion_score
        has_target_objects = len(target_objs) > 0

        # Simulate TRAKE segments based on KIS frame
        start_frame = None
        end_frame = None
        if inferred_type == "TRAKE":
            start_frame = max(0, c_obj.frame_idx - 75)
            end_frame = c_obj.frame_idx + 75

        results.append({
            "rank": rank,
            "video_id": c_obj.video_id,
            "frame_idx": c_obj.frame_idx,
            "siglip_score": round(c_obj.siglip_score, 4),
            "obj_score": round(obj_score, 4),
            "spatial_score": round(getattr(c_obj, 'spatial_score', 0.0), 4),
            "fusion_score": round(fusion_score, 4),
            "has_target_objects": has_target_objects,
            "pts_time": round(pts, 2),
            "timestamp": pts_str,
            "frame_url": f"/api/v1/image/{c_obj.video_id}/{c_obj.frame_idx}",
            "watch_url": watch_url,
            "video_title": v_title,
            "detected_labels": detected_labels,
            "vqa_answer": getattr(c_obj, 'vqa_answer', None),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    # Prepare extracted info for UI based on trace (we didn't pass the raw IR graph back in MVP result yet, so mock it for UI)
    extracted_objects = []
    if result.operator_trace and "operator_traces" in result.operator_trace:
         for op in result.operator_trace["operator_traces"]:
             if op["operator_name"] == "DETECT":
                 extracted_objects.append(op.get("details", {}).get("class", ""))
    
    return {
        "query_id": query_id,
        "query": query,
        "query_type": inferred_type,
        "total_results": len(results),
        "search_time_ms": round(elapsed_ms, 1),
        "cache_hit": False,
        "parsed_info": {
            "normalized_text": query.lower(),
            "extracted_objects": [obj for obj in extracted_objects if obj],
            "sub_events": [],
        },
        "results": results,
    }


# ── Endpoints ──

@router.post("/search/kis", response_model=SearchResponse)
async def search_kis(req: SearchRequest):
    """
    KIS Search Endpoint — Known Item Search.
    Accepts a Vietnamese text query and returns ranked video frames
    with explainability scores (CLIP, Object Detection, Fusion).
    """
    from fastapi.concurrency import run_in_threadpool
    try:
        # If QA, the query itself is the question
        question_text = req.question
        if req.query_type in ("QA", "TRAKE") and not question_text:
            question_text = req.query
            
        result = await run_in_threadpool(
            _run_real_search,
            query=req.query,
            query_type=req.query_type,
            top_k=req.top_k,
            question=question_text,
        )
    except Exception as e:
        print(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

    return result


@router.get("/search/status")
async def search_status():
    """Check if the search pipeline is ready."""
    from api.main import _get_searcher
    searcher = _get_searcher()
    return {
        "pipeline_ready": searcher is not None,
        "mode": "LIVE" if searcher else "DEMO",
    }
