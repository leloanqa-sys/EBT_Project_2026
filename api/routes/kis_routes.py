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
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

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
        except Exception as e:
            import logging
            logging.error(f"[get_video_metadata] Error reading {meta_path}: {e}")
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
            allow_remote=True
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
        import logging, traceback
        logging.error(f"[API] Image resolver error for {video_id} {frame_idx}: {e}")
        logging.error(traceback.format_exc())
        
    raise HTTPException(status_code=404, detail="Frame image not found")



def _run_real_search(query: str, query_type: str, top_k: int, question: Optional[str] = None) -> dict:
    """Run the actual search pipeline using NLP Compiler, Hybrid Search and Lazy OCR."""
    import time
    import uuid
    from src.database.db_manager import DatabaseManager
    from src.database.legacy_detection_store import LegacyDetectionStore
    from src.retrieval.hybrid_searcher import HybridSearcher
    from src.retrieval.vector_index import FAISSIndex
    from src.common.schemas import ScoringPlan
    from src.nlp.gemini_parser import parse_query_to_ir
    
    start_time = time.perf_counter()
    query_id = f"api_{uuid.uuid4().hex[:8]}"
    
    # 1. Compile Query with NLP
    ir = parse_query_to_ir(query_id=query_id, raw_text=query, query_type=query_type)
    
    # 2. Initialize Data Layer & Hybrid Searcher
    from api.main import get_db, get_faiss, get_detection_store
    db = get_db()
    faiss_index = get_faiss()
    detection_store = get_detection_store()
    retriever = HybridSearcher(db=db, faiss_index=faiss_index, detection_store=detection_store)
    
    # Extract scoring weights from environment or use defaults
    visual_top_k = int(os.environ.get("SCORING_VISUAL_TOP_K", 2000))
    w_visual = float(os.environ.get("SCORING_W_VISUAL", 1.0))
    w_ocr = float(os.environ.get("SCORING_W_OCR", 0.5))
    w_metadata = float(os.environ.get("SCORING_W_METADATA", 0.2))
    w_object = float(os.environ.get("SCORING_W_OBJECT", 0.5))

    scoring_plan = ScoringPlan(
        visual_top_k=visual_top_k, 
        w_visual=w_visual,
        w_ocr=w_ocr,
        w_metadata=w_metadata,
        w_object=w_object
    )
    
    # Apply dynamic routing based on Query Class (Exp D)
    from src.retrieval.hybrid_searcher import route_scoring_plan
    scoring_plan = route_scoring_plan(ir.query_class, scoring_plan)
    
    # 3. Retrieve Candidates (Hybrid Search + Lazy OCR on-the-fly)
    raw_results = retriever.search(ir, scoring_plan=scoring_plan, top_k_final=top_k, enable_lazy_ocr=True)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not raw_results:
        return {"query_id": query_id, "query": query, "query_type": query_type, "total_results": 0, "search_time_ms": elapsed_ms, "parsed_info": {}, "results": []}
        
    # --- QA OCR-First Engine + VLM Temporal Predictor ---
    # QA: Use OCR-first pipeline (SigLIP-guided frames → vote → VLM verify top-1)
    # Temporal: VLMAuditor predicts [start, end] bounds for top candidate
    qa_answer_global = None
    try:
        from src.retrieval.vlm_auditor import VLMAuditor
        auditor = VLMAuditor(retriever.db)
        top_cand = raw_results[0]

        if query_type == "QA" and question:
            try:
                from src.retrieval.qa_ocr_engine import answer_question_ocr_first
                qa_answer_global = answer_question_ocr_first(
                    question=question,
                    ranked_candidates=raw_results,  # already sorted by score DESC
                    top_n_ocr=8,
                )
                # Stash on candidate for backward compat
                top_cand.candidate.vqa_answer = qa_answer_global
                print(f"[kis_routes] QA OCR-First answer: '{qa_answer_global}'")
            except Exception as qa_e:
                print(f"[kis_routes] QA OCR Engine failed, fallback to VLM: {qa_e}")
                qa_answer_global = auditor.answer_question(question, top_cand)
                top_cand.candidate.vqa_answer = qa_answer_global

        vlm_start, vlm_end = auditor.predict_temporal_bounds(ir.raw_text, top_cand)

        # Apply the VLM bounds to the top candidate
        top_cand.candidate.pts_time = (vlm_start + vlm_end) / 2.0
    except Exception as e:
        print(f"[kis_routes] VLM Temporal Predictor failed: {e}")
        vlm_start, vlm_end = None, None

    results = []
    
    for idx, cand in enumerate(raw_results[:top_k]):
        if len(results) >= top_k:
            break
            
        rank = idx + 1
        
        # Determine actual sequence bounds from DP window or fallback to peak
        if cand.temporal_evidence and cand.temporal_evidence.window:
            start_pts = cand.temporal_evidence.window.start_time
            end_pts = cand.temporal_evidence.window.end_time
            # Ensure minimum 5-second window for UI aesthetics if the sequence was too fast
            if end_pts - start_pts < 5.0:
                mid = (start_pts + end_pts) / 2.0
                start_pts = max(0.0, mid - 2.5)
                end_pts = mid + 2.5
        else:
            start_pts = max(0.0, cand.candidate.pts_time - 2.5)
            end_pts = cand.candidate.pts_time + 2.5
            
        pts = (start_pts + end_pts) / 2.0
        pts_str = f"{int(pts // 60):02d}:{int(pts % 60):02d}"

        meta = db.get_video_metadata(cand.candidate.video_id) or {}
        watch_base = meta.get("source", "")
        v_title = meta.get("title", cand.candidate.video_id)
        sec_int = int(pts)
        watch_url = f"{watch_base}&t={sec_int}s" if watch_base else None

        fps = cand.candidate.fps if hasattr(cand.candidate, 'fps') and cand.candidate.fps else 30.0
        
        results.append({
            "rank": rank,
            "video_id": cand.candidate.video_id,
            "frame_idx": cand.candidate.frame_idx,
            "siglip_score": round(cand.score.visual, 4),
            "obj_score": round(cand.score.object, 4),
            "spatial_score": round(cand.score.metadata, 4),
            "fusion_score": round(cand.score.final, 4),
            "has_target_objects": len(ir.object_targets) > 0 and cand.score.object > 0,
            "pts_time": round(pts, 2),
            "timestamp": pts_str,
            "frame_url": f"/api/v1/image/{cand.candidate.video_id}/{cand.candidate.frame_idx}",
            "watch_url": watch_url,
            "video_title": v_title,
            "detected_labels": ir.object_targets,
            "vqa_answer": getattr(cand.candidate, "vqa_answer", None),
            "start_frame": max(0, int(start_pts * fps)) if query_type == "TRAKE" else None,
            "end_frame": int(end_pts * fps) if query_type == "TRAKE" else None,
        })
    
    # Pass parsed NLP info down to UI
    parsed_info = {
        "normalized_text": ir.raw_text,
        "extracted_objects": [ir.dense_caption_en],
        "sub_events": [t.term for t in ir.text_targets],
    }
    
    return {
        "query_id": query_id,
        "query": query,
        "query_type": query_type,
        "total_results": len(results),
        "search_time_ms": round(elapsed_ms, 1),
        "cache_hit": False,
        "qa_answer": qa_answer_global,  # Top-level QA answer (QA mode only)
        "parsed_info": parsed_info,
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
