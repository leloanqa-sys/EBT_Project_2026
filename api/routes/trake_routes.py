"""
TRAKE Search Routes — /api/v1/search/trake
==========================================
Handles Temporal Relation and Knowledge Event (TRAKE) queries.
Uses HybridSearcher with DP temporal alignment + TRAKEFormatter for output.

AIC TRAKE CSV format: video_id, e1_frame_idx, e2_frame_idx, ..., eN_frame_idx
Optional QA answer: text answer appended when question is provided.
"""
import time
import uuid
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter(tags=["search"])


class TRAKERequest(BaseModel):
    query: str = Field(..., description="Toàn bộ mô tả chuỗi sự kiện (Tiếng Việt)")
    question: Optional[str] = Field(default=None, description="Câu hỏi QA kèm theo (nếu có)")
    top_k: int = Field(default=5, ge=1, le=20, description="Số video kết quả trả về")


class TRAKEResponse(BaseModel):
    query_id: str
    latency_ms: float
    total_sequences: int
    n_events: int
    sequences: List[dict]
    qa_answer: Optional[str] = None


def _run_trake_search(query: str, question: Optional[str], top_k: int) -> dict:
    """Run the TRAKE pipeline using HybridSearcher + TRAKEFormatter."""
    from src.database.db_manager import DatabaseManager
    from src.database.legacy_detection_store import LegacyDetectionStore
    from src.retrieval.hybrid_searcher import HybridSearcher
    from src.retrieval.vector_index import FAISSIndex
    from src.retrieval.trake_formatter import format_trake_results
    from src.common.schemas import ScoringPlan
    from src.nlp.gemini_parser import parse_query_to_ir
    from src.retrieval.hybrid_searcher import route_scoring_plan

    start_time = time.perf_counter()
    query_id = f"trake_{uuid.uuid4().hex[:8]}"

    # 1. Parse query → IR (extracts temporal_sequence automatically)
    ir = parse_query_to_ir(query_id=query_id, raw_text=query, query_type="TRAKE")
    n_events = len(ir.temporal_sequence) if ir.temporal_sequence else 1
    logger.info(f"[TRAKE] Events parsed: {n_events} → {ir.temporal_sequence}")

    # 2. Init retrieval stack
    from api.main import get_db, get_faiss, get_detection_store
    db = get_db()
    faiss_index = get_faiss()
    detection_store = get_detection_store()
    retriever = HybridSearcher(db=db, faiss_index=faiss_index, detection_store=detection_store)

    # 3. Scoring plan — TRAKE benefits from higher visual_top_k for multi-step DP
    visual_top_k = int(os.environ.get("SCORING_VISUAL_TOP_K", 2000))
    scoring_plan = ScoringPlan(
        visual_top_k=visual_top_k,
        w_visual=float(os.environ.get("SCORING_W_VISUAL", 1.0)),
        w_ocr=float(os.environ.get("SCORING_W_OCR", 0.3)),
        w_metadata=float(os.environ.get("SCORING_W_METADATA", 0.2)),
        w_object=0.0,
    )
    scoring_plan = route_scoring_plan(ir.query_class, scoring_plan)

    # 4. Retrieve with DP temporal alignment (Step 7 in HybridSearcher)
    raw_results = retriever.search(
        ir,
        scoring_plan=scoring_plan,
        top_k_final=top_k * 20,  # Retrieve more to have good per-video coverage
        enable_lazy_ocr=True,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not raw_results:
        return {
            "query_id": query_id,
            "latency_ms": round(elapsed_ms, 1),
            "total_sequences": 0,
            "n_events": n_events,
            "sequences": [],
            "qa_answer": None,
        }

    # 5. Format into TRAKE sequences
    sequences = format_trake_results(
        ranked_candidates=raw_results,
        ir=ir,
        top_k_videos=top_k,
        question=question,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # 6. Build API response
    qa_answer = sequences[0].qa_answer if sequences and sequences[0].qa_answer else None
    seqs_out = [seq.to_dict() for seq in sequences]

    return {
        "query_id": query_id,
        "latency_ms": round(elapsed_ms, 1),
        "total_sequences": len(seqs_out),
        "n_events": n_events,
        "sequences": seqs_out,
        "qa_answer": qa_answer,
    }


@router.post("/search/trake", response_model=TRAKEResponse)
async def search_trake(req: TRAKERequest):
    """
    TRAKE Search Endpoint — Temporal Relation and Knowledge Events.
    Accepts a Vietnamese description of a temporal event sequence.
    Returns ordered frame sequences: one frame per event per candidate video.
    """
    from fastapi.concurrency import run_in_threadpool
    try:
        result = await run_in_threadpool(
            _run_trake_search,
            query=req.query,
            question=req.question,
            top_k=req.top_k,
        )
        return result
    except Exception as e:
        logger.error(f"[TRAKE API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
