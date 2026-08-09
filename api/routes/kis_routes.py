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
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter(tags=["search"])


# ── Request/Response Models ──

class SearchRequest(BaseModel):
    """Search request payload."""
    query: str = Field(..., min_length=1, description="Raw Vietnamese text query")
    query_type: str = Field(default="KIS", description="Query type: KIS, QA, TRAKE")
    top_k: int = Field(default=20, ge=1, le=100, description="Number of results to return")
    question: Optional[str] = Field(default=None, description="Question text for QA type")


class ResultItem(BaseModel):
    rank: int
    video_id: str
    frame_idx: int
    clip_score: float
    obj_score: float
    fusion_score: float
    pts_time: float = 0.0
    timestamp: str = "00:00"
    frame_url: str
    detected_labels: List[str] = []


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


# ── Demo data for when VectorSearcher is not available ──

DEMO_RESULTS = [
    {"rank": 1, "video_id": "L25_V086", "frame_idx": 14550, "clip_score": 0.2469, "obj_score": 0.50, "fusion_score": 0.2728, "detected_labels": ["person", "clothing", "building"]},
    {"rank": 2, "video_id": "L23_V025", "frame_idx": 11812, "clip_score": 0.2423, "obj_score": 0.33, "fusion_score": 0.2362, "detected_labels": ["person", "microphone"]},
    {"rank": 3, "video_id": "L25_V004", "frame_idx": 25799, "clip_score": 0.2312, "obj_score": 0.50, "fusion_score": 0.2619, "detected_labels": ["person", "clothing", "tree"]},
    {"rank": 4, "video_id": "L25_V072", "frame_idx": 21600, "clip_score": 0.2306, "obj_score": 0.00, "fusion_score": 0.1614, "detected_labels": ["person"]},
    {"rank": 5, "video_id": "L25_V054", "frame_idx": 62250, "clip_score": 0.2200, "obj_score": 0.33, "fusion_score": 0.2206, "detected_labels": ["person", "car"]},
    {"rank": 6, "video_id": "L22_V026", "frame_idx": 22862, "clip_score": 0.2171, "obj_score": 0.00, "fusion_score": 0.1520, "detected_labels": ["person", "building"]},
    {"rank": 7, "video_id": "L25_V038", "frame_idx": 33450, "clip_score": 0.2191, "obj_score": 0.25, "fusion_score": 0.2034, "detected_labels": ["person", "tree", "bench"]},
    {"rank": 8, "video_id": "L29_V005", "frame_idx": 12753, "clip_score": 0.2165, "obj_score": 0.00, "fusion_score": 0.1516, "detected_labels": ["person"]},
    {"rank": 9, "video_id": "L25_V045", "frame_idx": 3000, "clip_score": 0.2170, "obj_score": 0.50, "fusion_score": 0.2519, "detected_labels": ["person", "clothing"]},
    {"rank": 10, "video_id": "L24_V011", "frame_idx": 15872, "clip_score": 0.2135, "obj_score": 0.33, "fusion_score": 0.2161, "detected_labels": ["person", "table"]},
]


def _build_frame_url(video_id: str, frame_idx: int) -> str:
    """Build the URL path for a keyframe image."""
    return f"/keyframes/{video_id}/{frame_idx:04d}.jpg"


def _run_demo_search(query: str, query_type: str, top_k: int) -> dict:
    """Return demo results when the real pipeline is unavailable."""
    results = []
    for item in DEMO_RESULTS[:top_k]:
        results.append({
            **item,
            "frame_url": _build_frame_url(item["video_id"], item["frame_idx"]),
        })

    return {
        "query_id": f"demo_{uuid.uuid4().hex[:8]}",
        "query": query,
        "query_type": query_type,
        "total_results": len(results),
        "search_time_ms": 42.5,
        "cache_hit": False,
        "parsed_info": {
            "normalized_text": query.lower(),
            "extracted_objects": ["person", "clothing"],
            "sub_events": [],
        },
        "results": results,
    }


def _run_real_search(query: str, query_type: str, top_k: int, question: Optional[str] = None) -> dict:
    """Run the actual search pipeline."""
    from src.common.schemas import Query, QueryType, CandidateFrame
    from src.role_b_nlp.query_parser import parse_query
    from src.role_b_nlp.object_matcher import fuse_candidates, parse_json_to_detected_objects
    from src.role_c_logic.ranking import cluster_by_event, rank_5budget

    # Import main module to get searcher singleton
    from api.main import _get_searcher

    start_time = time.perf_counter()

    searcher = _get_searcher()
    if searcher is None:
        raise HTTPException(status_code=503, detail="VectorSearcher not loaded")

    # Map query_type string to enum
    qt_map = {"KIS": QueryType.KIS, "QA": QueryType.QA, "TRAKE": QueryType.TRAKE}
    qt = qt_map.get(query_type.upper(), QueryType.KIS)

    # Step 1: Retrieval — get raw candidates from FAISS
    candidates = searcher.search_by_text(query, top_k=300)

    # Step 2: NLP Parse
    q_obj = Query(
        query_id=f"api_{uuid.uuid4().hex[:8]}",
        raw_text=query,
        query_type=qt,
        question_text=question,
    )
    parsed_q = parse_query(q_obj)

    # Step 3: Fusion
    candidates = fuse_candidates(candidates, parsed_q)

    # Step 4: Clustering + Ranking
    clustered = cluster_by_event(candidates, gap_threshold=15)
    ranked_items = rank_5budget(clustered, strategy="diversify")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Build candidate lookup map for fast retrieval
    cand_map = {(c.video_id, c.frame_idx): c for c in candidates}

    # Build response
    results = []
    target_objs = parsed_q.attributes.get("objects", {}).get("keywords", [])

    for item in ranked_items[:top_k]:
        c_obj = cand_map.get((item.video_id, item.frame_id))
        clip_sc = round(c_obj.clip_score, 4) if c_obj else 0.0
        pts = c_obj.pts_time if c_obj else 0.0
        pts_str = f"{int(pts // 60):02d}:{int(pts % 60):02d}"

        detected = parse_json_to_detected_objects(item.video_id, item.frame_id)
        detected_labels = list(dict.fromkeys([obj.label for obj in detected]))  # deduplicate labels

        if target_objs and detected_labels:
            from src.role_b_nlp.object_matcher import calculate_object_match_score
            obj_sc = round(calculate_object_match_score(target_objs, detected_labels), 4)
        else:
            obj_sc = 0.0

        results.append({
            "rank": item.rank,
            "video_id": item.video_id,
            "frame_idx": item.frame_id,
            "clip_score": clip_sc,
            "obj_score": obj_sc,
            "fusion_score": round(item.confidence_score, 4),
            "pts_time": round(pts, 2),
            "timestamp": pts_str,
            "frame_url": _build_frame_url(item.video_id, item.frame_id),
            "detected_labels": detected_labels[:8],  # top 8 labels for UI
        })

    return {
        "query_id": q_obj.query_id,
        "query": query,
        "query_type": query_type,
        "total_results": len(results),
        "search_time_ms": round(elapsed_ms, 1),
        "cache_hit": False,
        "parsed_info": {
            "normalized_text": parsed_q.normalized_text,
            "extracted_objects": parsed_q.attributes.get("objects", {}).get("keywords", []),
            "sub_events": parsed_q.sub_events,
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
    try:
        result = _run_real_search(
            query=req.query,
            query_type=req.query_type,
            top_k=req.top_k,
            question=req.question,
        )
    except Exception as e:
        # Fallback to demo mode if pipeline is not available
        print(f"[API] Pipeline error, falling back to DEMO mode: {e}")
        result = _run_demo_search(
            query=req.query,
            query_type=req.query_type,
            top_k=req.top_k,
        )

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
