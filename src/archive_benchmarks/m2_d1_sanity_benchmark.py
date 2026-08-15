import json
from src.role_a_retrieval.searcher import VectorSearcher

def run_m2_d1_sanity_benchmark(gt_path: str = "data/m2_10_human_gt.json"):
    print("--- M2-D.1: Human-GT Sanity Benchmark (Base V0) ---")
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print(f"[FAIL] Could not find {gt_path}")
        return
        
    searcher = VectorSearcher()
    
    total_queries = len(gt_data)
    hits_at_500 = 0
    
    for item in gt_data:
        qid = item["query_id"]
        qtext = item["query"]
        
        # Determine the target video_id
        target_vid = item["gt"]["video_id"]
        
        print(f"\nQuery [{qid}]: {qtext}")
        print(f"Target GT Video: {target_vid}")
        
        # Base V0 Retrieval (Text -> FAISS)
        cands = searcher.search_by_text(qtext, top_k=500)
        
        # Evaluate Dataset Presence & ID Mapping & Retrieval hit
        retrieved_vids = set(c.video_id for c in cands)
        
        if target_vid in retrieved_vids:
            print(f"[PASS] Video {target_vid} found in Top-500!")
            hits_at_500 += 1
        else:
            print(f"[FAIL] Video {target_vid} NOT FOUND in Top-500.")
            
    recall = hits_at_500 / total_queries if total_queries > 0 else 0
    print(f"\n--- Benchmark Summary ---")
    print(f"Base Recall@500 on {total_queries} Human-GT queries: {recall * 100:.1f}%")
    
    if recall > 0:
        print("[VERDICT: PROCEED] Base V0 demonstrates retrieval capability. M2-D is clear to proceed.")
    else:
        print("[VERDICT: STOP] Base V0 Recall is 0. Bottleneck lies in data/index/retrieval alignment.")

if __name__ == "__main__":
    run_m2_d1_sanity_benchmark()
