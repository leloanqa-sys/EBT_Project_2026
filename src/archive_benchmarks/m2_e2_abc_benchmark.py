import json
import time
from src.role_a_retrieval.searcher import VectorSearcher
from src.common.schemas import VisualIRGraph, Entity, Attribute
from src.role_c_logic.capability_registry import registry, CapabilityStatus
from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor

# Offline Mocks to save API limit
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

def run_benchmark():
    with open("data/m2_10_human_gt.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    searcher = VectorSearcher()
    executor = DeterministicExecutor(searcher)
    
    results = {"A": 0, "B": 0, "C": 0}
    total_queries = len(gt_data)
    
    print("--- M2-E.2: A/B/C Benchmark (10 GT Mock) ---\n")
    
    for run_mode in ["A", "B", "C"]:
        # Configure Registry for the Run
        if run_mode == "A":
            registry.operators["DETECT"].status = CapabilityStatus.UNSUPPORTED
            registry.operators["FILTER_ATTRIBUTE"].status = CapabilityStatus.UNSUPPORTED
        elif run_mode == "B":
            registry.operators["DETECT"].status = CapabilityStatus.READY
            registry.operators["FILTER_ATTRIBUTE"].status = CapabilityStatus.EXPERIMENTAL
        elif run_mode == "C":
            registry.operators["DETECT"].status = CapabilityStatus.READY
            registry.operators["FILTER_ATTRIBUTE"].status = CapabilityStatus.READY
            
        hits = 0
        latencies = []
        
        for item in gt_data:
            qid = item["query_id"]
            target_vid = item["gt"]["video_id"]
            ir_graph = MOCK_IR_GRAPHS.get(qid, VisualIRGraph(query_id=qid, raw_text=item["query"], query_type=item["type"]))
            
            start_t = time.time()
            
            if run_mode == "A":
                cands = searcher.search_by_text(ir_graph.raw_text, top_k=500)
            else:
                plan = create_deterministic_plan(ir_graph)
                if run_mode == "B":
                    for s in plan.steps:
                        if registry.operators.get(s.operator_name) and registry.operators[s.operator_name].status == CapabilityStatus.EXPERIMENTAL:
                            s.status = CapabilityStatus.UNSUPPORTED
                
                cands = executor.execute_plan(plan, ir_graph, top_k_raw=500)
                
            elapsed = (time.time() - start_t) * 1000
            latencies.append(elapsed)
            
            retrieved_vids = set(c.video_id for c in cands)
            if target_vid in retrieved_vids:
                hits += 1
                
        recall = hits / total_queries * 100
        avg_latency = sum(latencies) / len(latencies)
        results[run_mode] = {"Recall": recall, "Avg_Latency_ms": avg_latency}
        print(f"Run {run_mode} | Recall@500: {recall:.1f}% | Avg Latency: {avg_latency:.1f} ms")
        
    print("\n--- Benchmark Complete ---")

if __name__ == "__main__":
    run_benchmark()
