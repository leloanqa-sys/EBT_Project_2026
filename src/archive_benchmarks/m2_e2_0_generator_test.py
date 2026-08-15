import os
import json
import time
import shutil
from src.role_a_retrieval.searcher import VectorSearcher
from src.common.schemas import VisualIRGraph, Entity, Attribute

# Re-use the MOCK_IR_GRAPHS
MOCK_IR_GRAPHS = {
    "KIS_01": VisualIRGraph(query_id="KIS_01", raw_text="người đàn ông áo đỏ", query_type="KIS", entities=[Entity(id="e1", label="Person")]),
    "KIS_02": VisualIRGraph(query_id="KIS_02", raw_text="chiếc xe màu trắng", query_type="KIS", entities=[Entity(id="e1", label="Car")]),
    "KIS_03": VisualIRGraph(query_id="KIS_03", raw_text="người phụ nữ cầm ô", query_type="KIS", entities=[Entity(id="e1", label="Person"), Entity(id="e2", label="Umbrella")]),
    "KIS_04": VisualIRGraph(query_id="KIS_04", raw_text="con chó ngoài trời", query_type="KIS", entities=[Entity(id="e1", label="Dog")]),
    "QA_01": VisualIRGraph(query_id="QA_01", raw_text="người phát biểu", query_type="QA", entities=[Entity(id="e1", label="Person")]),
    "QA_02": VisualIRGraph(query_id="QA_02", raw_text="xe di chuyển", query_type="QA", entities=[Entity(id="e1", label="Car")]),
    "QA_03": VisualIRGraph(query_id="QA_03", raw_text="người ngồi bàn", query_type="QA", entities=[Entity(id="e1", label="Person"), Entity(id="e2", label="Table")]),
    "TRAKE_01": VisualIRGraph(query_id="TRAKE_01", raw_text="người rót nước", query_type="TRAKE", entities=[Entity(id="e1", label="Person"), Entity(id="e2", label="Cup")]),
    "TRAKE_02": VisualIRGraph(query_id="TRAKE_02", raw_text="vận động viên", query_type="TRAKE", entities=[Entity(id="e1", label="Person")]),
    "TRAKE_03": VisualIRGraph(query_id="TRAKE_03", raw_text="sự kiện", query_type="TRAKE", entities=[])
}

def clear_cache(cache_dir="data/processed/cache/query_cache"):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

def run_generator_test():
    with open("data/m2_10_human_gt.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    searcher = VectorSearcher()
    
    modes = {
        "G0": "Base V0 (raw)",
        "G1": "M2 (Entity-only)",
        "G2": "M2 (Raw-text)"
    }
    
    print("| Query | Mode | Cache | Recall@500 | Latency (ms) |")
    print("|---|---|---|---|---|")
    
    overall_recalls = {"G0": 0, "G1": 0, "G2": 0}
    
    for mode in ["G0", "G1", "G2"]:
        # Clear cache before testing each mode to guarantee COLD run first
        clear_cache()
        hits = 0
        
        for item in gt_data:
            qid = item["query_id"]
            target_vid = item["gt"]["video_id"]
            
            # Determine query text based on mode
            if mode == "G0":
                query_text = item["query"] # Original raw user query
            else:
                ir_graph = MOCK_IR_GRAPHS.get(qid, VisualIRGraph(query_id=qid, raw_text=item["query"], query_type="KIS"))
                if mode == "G1":
                    entity_labels = [e.label for e in ir_graph.entities]
                    query_text = " ".join(entity_labels) if entity_labels else ir_graph.raw_text
                else: # G2
                    query_text = ir_graph.raw_text
            
            # --- COLD RUN ---
            start_t = time.time()
            cands_cold = searcher.search_by_text(query_text, top_k=500)
            latency_cold = (time.time() - start_t) * 1000
            
            retrieved_vids_cold = set(c.video_id for c in cands_cold)
            recall_cold = 100 if target_vid in retrieved_vids_cold else 0
            
            print(f"| {qid} | {mode} | miss | {recall_cold} | {latency_cold:.1f} |")
            
            # --- WARM RUN ---
            start_t = time.time()
            cands_warm = searcher.search_by_text(query_text, top_k=500)
            latency_warm = (time.time() - start_t) * 1000
            
            retrieved_vids_warm = set(c.video_id for c in cands_warm)
            recall_warm = 100 if target_vid in retrieved_vids_warm else 0
            
            print(f"| {qid} | {mode} | hit | {recall_warm} | {latency_warm:.1f} |")
            
            if recall_cold > 0:
                hits += 1
                
        overall_recalls[mode] = (hits / len(gt_data)) * 100
        print("|---|---|---|---|---|")

    print("\n### Summary of Recall@500")
    for mode, recall in overall_recalls.items():
        print(f"- **{mode}**: {recall:.1f}%")

if __name__ == "__main__":
    run_generator_test()
