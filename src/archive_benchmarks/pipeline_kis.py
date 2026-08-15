from typing import List, Optional
from src.common.schemas import VisualIRGraph, CandidateFrame, QueryType, Query
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

from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor

def run_kis(
    raw_query: str,
    query_id: str,
    top_k_raw: int = 300,
    gap_threshold: int = 15,
    strategy: str = "diversify"
) -> str:
    """
    M2 Pipeline:
    1. Parse: NLP Engine -> VisualIRGraph
    2. Plan: VisualIRGraph -> ExecutionPlan (DAG of capabilities)
    3. Execute: DAG -> Filtered Candidates + Execution Trace
    4. Rank & Format.
    """
    searcher = get_searcher()
    
    # 1. Parse Query
    q_obj = Query(query_id=query_id, query_type=QueryType.KIS, raw_text=raw_query)
    parsed_ir = parse_query(q_obj)
    
    # 2. Plan
    plan = create_deterministic_plan(parsed_ir)
    
    # 3. Execute
    executor = DeterministicExecutor(searcher)
    candidates = executor.execute_plan(plan, parsed_ir, top_k_raw=top_k_raw)
    
    # M0 Object Matcher Fusion fallback (if needed)
    # candidates = fuse_candidates(candidates, parsed_ir) # Disabled since Executor handles logic now
    
    # 4. Temporal Clustering & Ranking
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
