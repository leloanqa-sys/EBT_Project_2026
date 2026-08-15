import json
import time
import os
import numpy as np
from src.role_a_retrieval.searcher import VectorSearcher
from src.common.schemas import VisualIRGraph, Entity, Attribute, Relation
from src.role_c_logic.capability_registry import registry, CapabilityStatus
from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor

# Offline Mocks
MOCK_IR_GRAPHS = {
    "KIS_01": VisualIRGraph(query_id="KIS_01", raw_text="người đàn ông áo đỏ", query_type="KIS", entities=[Entity(id="e1", label="Person")], attributes=[Attribute(entity_id="e1", name="color", value="red")]),
    "KIS_02": VisualIRGraph(query_id="KIS_02", raw_text="chiếc xe màu trắng", query_type="KIS", entities=[Entity(id="e1", label="Car")], attributes=[Attribute(entity_id="e1", name="color", value="white")]),
    "KIS_03": VisualIRGraph(query_id="KIS_03", raw_text="người phụ nữ cầm ô", query_type="KIS", entities=[Entity(id="e1", label="Person"), Entity(id="e2", label="Umbrella")]),
    "KIS_04": VisualIRGraph(query_id="KIS_04", raw_text="con chó ngoài trời", query_type="KIS", entities=[Entity(id="e1", label="Dog")]),
    "QA_01": VisualIRGraph(query_id="QA_01", raw_text="người phát biểu", query_type="QA", entities=[Entity(id="e1", label="Person")]),
    "QA_02": VisualIRGraph(query_id="QA_02", raw_text="xe di chuyển", query_type="QA", entities=[Entity(id="e1", label="Car")]),
    "QA_03": VisualIRGraph(query_id="QA_03", raw_text="người ngồi bàn", query_type="QA", entities=[Entity(id="e1", label="Person"), Entity(id="e2", label="Table")]),
    "TRAKE_01": VisualIRGraph(query_id="TRAKE_01", raw_text="người rót nước", query_type="TRAKE", entities=[Entity(id="e1", label="Person"), Entity(id="e2", label="Cup")]),
    "TRAKE_02": VisualIRGraph(query_id="TRAKE_02", raw_text="vận động viên", query_type="TRAKE", entities=[Entity(id="e1", label="Person")]),
    "TRAKE_03": VisualIRGraph(query_id="TRAKE_03", raw_text="sự kiện", query_type="TRAKE", entities=[])
}

# Add a specific spatial query for testing O2 properly since the 10 GT don't explicitly have left_of
MOCK_IR_GRAPHS["SPATIAL_MOCK"] = VisualIRGraph(
    query_id="SPATIAL_MOCK", raw_text="con chó bên trái cái cây", query_type="KIS", 
    entities=[Entity(id="e1", label="Dog"), Entity(id="e2", label="Tree")],
    relations=[Relation(source_id="e1", target_id="e2", relation_type="left_of")]
)

def run_e1_isolation():
    with open("data/m2_10_human_gt.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    searcher = VectorSearcher()
    executor = DeterministicExecutor(searcher)
    
    # Warm up cache
    print("Warming up Cache...")
    for item in gt_data:
        searcher.search_by_text(item["query"], top_k=500)
    
    configs = {
        "G0": [],
        "O1_DETECT": ["DETECT"],
        "O2_SPATIAL": ["SPATIAL_LEFT_OF"],
        "O3_ATTRIBUTE": ["FILTER_ATTRIBUTE"]
    }
    
    print("\n| Config | Recall@500 | Retention | Reduction | P@5 | Latency P50 (ms) | Latency P95 (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    
    results = {}
    
    # Run G0 first to get baseline GT and Candidates
    baseline_gts_retrieved = {}
    baseline_cands_count = {}
    
    for conf_name, allowed_ops in configs.items():
        # Setup registry
        for op in registry.operators.values():
            if op.name == "CLIP_RETRIEVE" or op.name in allowed_ops:
                op.status = CapabilityStatus.READY
            else:
                op.status = CapabilityStatus.UNSUPPORTED
                
        hits = 0
        total_queries = len(gt_data)
        
        latencies = []
        retention_scores = []
        reduction_scores = []
        p_at_5_hits = 0
        
        for item in gt_data:
            qid = item["query_id"]
            target_vid = item["gt"]["video_id"]
            ir_graph = MOCK_IR_GRAPHS.get(qid, VisualIRGraph(query_id=qid, raw_text=item["query"], query_type="KIS"))
            
            # Special logic: to test O2 properly, we inject the SPATIAL_MOCK if O2 is active and it's KIS_04 (dog)
            if conf_name == "O2_SPATIAL" and qid == "KIS_04":
                ir_graph = MOCK_IR_GRAPHS["SPATIAL_MOCK"]
                ir_graph.query_id = qid # Keep original ID for tracing
                
            start_t = time.time()
            plan = create_deterministic_plan(ir_graph)
            cands = executor.execute_plan(plan, ir_graph, top_k_raw=500)
            latency = (time.time() - start_t) * 1000
            latencies.append(latency)
            
            retrieved_vids = [c.video_id for c in cands]
            retrieved_vids_set = set(retrieved_vids)
            
            hit = 1 if target_vid in retrieved_vids_set else 0
            hits += hit
            
            if target_vid in retrieved_vids[:5]:
                p_at_5_hits += 1
                
            # Retention & Reduction
            if conf_name == "G0":
                baseline_gts_retrieved[qid] = hit
                baseline_cands_count[qid] = len(cands)
            else:
                base_hit = baseline_gts_retrieved[qid]
                if base_hit == 1:
                    retention = 1.0 if hit == 1 else 0.0
                    retention_scores.append(retention)
                
                base_cands = baseline_cands_count[qid]
                if base_cands > 0:
                    reduction = 1.0 - (len(cands) / base_cands)
                    reduction_scores.append(reduction)
                    
        recall = (hits / total_queries) * 100
        p_at_5 = (p_at_5_hits / total_queries) * 100
        
        avg_retention = (sum(retention_scores) / len(retention_scores)) * 100 if retention_scores else (100.0 if conf_name == "G0" else 0.0)
        avg_reduction = (sum(reduction_scores) / len(reduction_scores)) * 100 if reduction_scores else 0.0
        
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        results[conf_name] = {
            "Recall": recall, "Retention": avg_retention, "Reduction": avg_reduction,
            "P@5": p_at_5, "P50": p50, "P95": p95
        }
        
        print(f"| {conf_name} | {recall:.1f}% | {avg_retention:.1f}% | {avg_reduction:.1f}% | {p_at_5:.1f}% | {p50:.1f} | {p95:.1f} |")

if __name__ == "__main__":
    run_e1_isolation()
