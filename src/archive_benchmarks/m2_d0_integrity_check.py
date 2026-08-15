import json
import time
from typing import List, Dict, Any
from src.common.schemas import Query, QueryType, CandidateFrame, VisualIRGraph
from src.role_a_retrieval.searcher import VectorSearcher
from src.role_b_nlp.query_parser import parse_query
from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor
from src.role_c_logic.capability_registry import CapabilityStatus

def calculate_metrics_synthetic(candidates: List[str], gt_frames: List[str]):
    top_500 = set(candidates[:500])
    gt_set = set(gt_frames)
    found_gt = gt_set.intersection(top_500)
    recall_500 = len(found_gt) / len(gt_set) if gt_set else 0.0
    return recall_500

def test_1_known_good_query():
    print("\n--- TEST 1: Known-Good Query Check ---")
    searcher = VectorSearcher()
    # A generic query that should yield something. If index is empty, it will be caught here.
    query_text = "người" # very general, should return something
    cands = searcher.search_by_text(query_text, top_k=10)
    
    if not cands:
        print("[FAIL] FAISS returned 0 candidates. FAISS Index might be empty or mock.")
    else:
        print(f"[PASS] FAISS returned {len(cands)} candidates for '{query_text}'.")
        print("Top 3 IDs:", [f"{c.video_id}_{c.frame_idx}" for c in cands[:3]])

def test_2_id_contract_check():
    print("\n--- TEST 2: ID Contract Check ---")
    searcher = VectorSearcher()
    cands = searcher.search_by_text("test", top_k=1)
    if not cands:
        print("[FAIL] Cannot run Test 2: no candidates returned from FAISS.")
        return
        
    cand = cands[0]
    cand_id_format = f"{cand.video_id}_{cand.frame_idx}"
    print(f"Candidate ID format from Retriever: '{cand_id_format}'")
    print(f"Expected format in Ground Truth (from PO): 'video_001_frame_0023' or similar.")
    print("If these formats do NOT match EXACTLY, intersection will fail (Recall=0).")
    print("[WARN] Need Manual Review of format alignment between GT and Retriever.")

def test_3_synthetic_evaluator():
    print("\n--- TEST 3: Synthetic Evaluator Check ---")
    gt = ["A", "B", "C"]
    retrieved = ["X", "A", "Y", "B", "Z"]
    recall = calculate_metrics_synthetic(retrieved, gt)
    
    expected = 2/3
    if abs(recall - expected) < 1e-5:
        print(f"[PASS] Synthetic Recall Evaluator is correct: {recall*100:.1f}%")
    else:
        print(f"[FAIL] Evaluator returned {recall}, expected {expected}")

def test_4_actual_candidate_trace():
    print("\n--- TEST 4: Actual Candidate Trace Flow ---")
    searcher = VectorSearcher()
    executor = DeterministicExecutor(searcher)
    
    ir = VisualIRGraph(query_id="test_001", raw_text="Một người", query_type=QueryType.KIS)
    plan = create_deterministic_plan(ir)
    
    print("Executing plan...")
    cands = executor.execute_plan(plan, ir, top_k_raw=10)
    
    if not cands:
        print("[FAIL] Executor returned 0 candidates. Dummy operators might be clearing the list.")
    else:
        print(f"[PASS] Executor successfully returned {len(cands)} candidates.")
        print("Final candidate trace sample:", [f"{c.video_id}_{c.frame_idx}" for c in cands[:3]])
        print("Please check outputs/traces/test_001.json for operator-level trace.")

if __name__ == "__main__":
    try:
        test_1_known_good_query()
        test_2_id_contract_check()
        test_3_synthetic_evaluator()
        test_4_actual_candidate_trace()
    except Exception as e:
        print(f"Error during integrity check: {e}")
