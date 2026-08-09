from typing import List, Optional
from src.common.schemas import CandidateFrame, QueryType, Query
from src.role_a_retrieval.searcher import VectorSearcher
from src.role_b_nlp.query_parser import parse_query
from src.role_b_nlp.object_matcher import fuse_candidates
from src.role_c_logic.ranking import cluster_by_event, rank_5budget
from src.role_c_logic.output_formatter import build_submission, format_submission

_SEARCHER_INSTANCE: Optional[VectorSearcher] = None

def get_searcher() -> VectorSearcher:
    """
    Singleton pattern for VectorSearcher engine to ensure FAISS index
    and hot-path mapping arrays are loaded into RAM only once.
    """
    global _SEARCHER_INSTANCE
    if _SEARCHER_INSTANCE is None:
        _SEARCHER_INSTANCE = VectorSearcher()
    return _SEARCHER_INSTANCE

def run_kis(
    raw_query: str,
    query_id: str,
    top_k_raw: int = 300,
    gap_threshold: int = 15,
    strategy: str = "diversify"
) -> str:
    """
    End-to-End Pipeline for Textual KIS (Known Item Search):
    1. Retrieval: Fetch top_k_raw candidates from Role A FAISS engine.
    2. Parse: Parse raw_query into ParsedQuery using Role B NLP Engine.
    3. Fusion: Apply Role B score fusion with Object Detection Re-ranking.
    4. Temporal Clustering: Group candidates by event window (gap_threshold) per video.
    5. 5-Budget Ranking: Prioritize Rank 1, diversify video_ids at Ranks 2-5.
    6. Formatting: Export submission CSV file to outputs/ directory.
    """
    searcher = get_searcher()
    candidates = searcher.search_by_text(raw_query, top_k=top_k_raw)
    
    # Parse query using Role B
    q_obj = Query(query_id=query_id, query_type=QueryType.KIS, raw_text=raw_query)
    parsed_q = parse_query(q_obj)
    
    # Fusion
    candidates = fuse_candidates(candidates, parsed_q)
    
    clustered = cluster_by_event(candidates, gap_threshold=gap_threshold)
    ranked_items = rank_5budget(clustered, strategy=strategy)
    
    submission = build_submission(ranked_items, query_id=query_id, query_type=QueryType.KIS)
    out_path = format_submission(submission)
    return out_path



if __name__ == "__main__":
    # Smoke test for KIS Pipeline MVP
    sample_query = "một diễn giả mặc áo đỏ phát biểu ngoài trời"
    sample_id = "kis_smoke_001"
    print(f"[Smoke Test] Running KIS Pipeline for query: '{sample_query}'...")
    output_csv = run_kis(sample_query, query_id=sample_id)
    print(f"[Smoke Test] SUCCESS! Submission CSV generated at: {output_csv}")
