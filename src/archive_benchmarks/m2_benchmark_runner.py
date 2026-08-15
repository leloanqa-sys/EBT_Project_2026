import os
import json
import time
from typing import List, Dict, Any
from src.common.schemas import Query, QueryType
from src.role_a_retrieval.searcher import VectorSearcher
from src.role_b_nlp.query_parser import parse_query
from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor
from src.role_c_logic.capability_registry import CapabilityStatus

def load_ground_truth(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground Truth file not found at {file_path}. Please provide human-annotated GT.")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_metrics(candidates: List[Any], gt_frames: List[str]):
    """
    GT match logic: assume candidates have video_id and frame_idx.
    GT frames format: 'video_id_frame_idx'
    """
    cand_ids = [f"{c.video_id}_{c.frame_idx}" for c in candidates]
    
    # Recall@500
    top_500 = set(cand_ids[:500])
    gt_set = set(gt_frames)
    found_gt = gt_set.intersection(top_500)
    recall_500 = len(found_gt) / len(gt_set) if gt_set else 0.0
    
    # Precision@5
    top_5 = cand_ids[:5]
    correct_in_top_5 = sum(1 for cid in top_5 if cid in gt_set)
    p_5 = correct_in_top_5 / 5.0
    
    return recall_500, p_5, len(found_gt)

def run_benchmark():
    gt_file = "data/m2_20_queries_gt.json"
    print(f"Loading Ground Truth from {gt_file}...")
    try:
        queries = load_ground_truth(gt_file)
    except FileNotFoundError as e:
        print(e)
        return
        
    searcher = VectorSearcher()
    executor = DeterministicExecutor(searcher)
    
    results = []
    
    print("Running M2 Decision Gate Benchmark...")
    for idx, q_data in enumerate(queries):
        qid = q_data["query_id"]
        text = q_data["text"]
        q_type = q_data.get("type", "Simple")
        gt_frames = q_data["gt_frames"]
        
        print(f"\n[{idx+1}/{len(queries)}] [{q_type}] {text}")
        
        # 1. Cấu hình A: Base V0 (FAISS only)
        start_a = time.time()
        cands_a = searcher.search_by_text(text, top_k=500)
        latency_a = (time.time() - start_a) * 1000
        recall_a, p5_a, gt_count_a = calculate_metrics(cands_a, gt_frames)
        
        # 2. Compile IR (dùng chung cho B và C)
        q_obj = Query(query_id=qid, query_type=QueryType.QA, raw_text=text)
        ir = parse_query(q_obj)
        plan_full = create_deterministic_plan(ir)
        
        # 3. Cấu hình B: M2 READY ONLY
        plan_ready = create_deterministic_plan(ir)
        for step in plan_ready.steps:
            if step.status == CapabilityStatus.EXPERIMENTAL:
                step.status = CapabilityStatus.UNSUPPORTED # Force skip
                
        start_b = time.time()
        cands_b = executor.execute_plan(plan_ready, ir, top_k_raw=500)
        latency_b = (time.time() - start_b) * 1000
        recall_b, p5_b, gt_count_b = calculate_metrics(cands_b, gt_frames)
        
        # 4. Cấu hình C: M2 FULL (READY + EXPERIMENTAL)
        start_c = time.time()
        cands_c = executor.execute_plan(plan_full, ir, top_k_raw=500)
        latency_c = (time.time() - start_c) * 1000
        recall_c, p5_c, gt_count_c = calculate_metrics(cands_c, gt_frames)
        
        res = {
            "query_id": qid,
            "type": q_type,
            "gt_total": len(gt_frames),
            "A_latency": latency_a,
            "A_Recall500": recall_a,
            "A_P5": p5_a,
            "A_GT_count": gt_count_a,
            
            "B_latency": latency_b,
            "B_Recall500": recall_b,
            "B_P5": p5_b,
            "B_GT_count": gt_count_b,
            "B_RecallRetention": (gt_count_b / gt_count_a) if gt_count_a > 0 else 0,
            
            "C_latency": latency_c,
            "C_Recall500": recall_c,
            "C_P5": p5_c,
            "C_GT_count": gt_count_c,
            "C_RecallRetention": (gt_count_c / gt_count_a) if gt_count_a > 0 else 0,
        }
        results.append(res)
        print(f"  [Base V0]  Recall@500: {recall_a:.2f} | P@5: {p5_a:.2f} | Latency: {latency_a:.1f}ms")
        print(f"  [M2 READY] Recall@500: {recall_b:.2f} (Ret: {res['B_RecallRetention']*100:.1f}%) | P@5: {p5_b:.2f} | Latency: {latency_b:.1f}ms")
        print(f"  [M2 FULL]  Recall@500: {recall_c:.2f} (Ret: {res['C_RecallRetention']*100:.1f}%) | P@5: {p5_c:.2f} | Latency: {latency_c:.1f}ms")

    # Tính tổng kết
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("outputs/m2_benchmark_report.csv", index=False)
    print("\n--- FINAL BENCHMARK SUMMARY ---")
    for t in df['type'].unique():
        sub = df[df['type'] == t]
        print(f"Category: {t} (N={len(sub)})")
        print(f"  Base V0 P@5: {sub['A_P5'].mean():.2f}")
        print(f"  M2 READY P@5: {sub['B_P5'].mean():.2f}")
        print(f"  M2 FULL P@5: {sub['C_P5'].mean():.2f}")
        print(f"  M2 READY Recall Retention: {sub['B_RecallRetention'].mean()*100:.1f}%")
        print(f"  Latency P95 (READY): {sub['B_latency'].quantile(0.95):.1f}ms")

if __name__ == "__main__":
    run_benchmark()
