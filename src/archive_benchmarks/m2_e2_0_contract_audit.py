import json
import os
from src.common.schemas import VisualIRGraph, Entity, Attribute, Relation
from src.role_c_logic.capability_registry import registry, CapabilityStatus
from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor
from src.role_a_retrieval.searcher import VectorSearcher

def audit_query(searcher, executor, label: str, raw_text: str, ir_graph: VisualIRGraph):
    print(f"\n{'='*50}")
    print(f"QUERY {label}: {raw_text}")
    print(f"{'='*50}")
    
    plan = create_deterministic_plan(ir_graph)
    print("\n[PLAN CONTRACT]")
    for s in plan.steps:
        print(f"  - [{s.status.value}] {s.operator_name} (target: {s.target})")
        
    print("\n[EXECUTION TRACE]")
    
    # Execute
    cands = executor.execute_plan(plan, ir_graph, top_k_raw=500)
    
    # Read trace from JSON
    trace_path = f"outputs/traces/{ir_graph.query_id}.json"
    if os.path.exists(trace_path):
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
            
        for op_trace in trace["operator_traces"]:
            print(f"  -> {op_trace['operator_name']:20s} | In: {op_trace['candidates_in']:4d} | Out: {op_trace['candidates_out']:4d} | Latency: {op_trace['latency_ms']:6.1f}ms | Status: {op_trace['status']}")
            
        print(f"\nFinal Candidates: {trace['final_candidates']}")
        print(f"Total Latency: {trace['total_latency_ms']:.1f}ms")
    else:
        print(f"NO TRACE FOUND AT {trace_path}!")
        
    # Validation logic based on PO's rule
    clip_ran = any(op['operator_name'] == 'CLIP_RETRIEVE' and op['status'] == 'READY' for op in trace.get("operator_traces", [])) if os.path.exists(trace_path) else False
    if not clip_ran:
        print("\n[VERDICT = PLAN INVALID] (CLIP_RETRIEVE was not executed or generated 0 candidates)")

def run_audit():
    print("Initializing FAISS searcher...")
    searcher = VectorSearcher()
    executor = DeterministicExecutor(searcher)
    
    # Ensure capabilities are READY
    registry.operators["DETECT"].status = CapabilityStatus.READY
    registry.operators["FILTER_ATTRIBUTE"].status = CapabilityStatus.READY
    registry.operators["SPATIAL_LEFT_OF"].status = CapabilityStatus.READY
    registry.operators["CLIP_RETRIEVE"].status = CapabilityStatus.READY
    
    # Query A: Simple
    ir_a = VisualIRGraph(
        query_id="QA",
        raw_text="một người đàn ông",
        query_type="KIS",
        entities=[Entity(id="e1", label="Person")]
    )
    audit_query(searcher, executor, "A: Simple", ir_a.raw_text, ir_a)
    
    # Query B: Spatial
    ir_b = VisualIRGraph(
        query_id="QB",
        raw_text="một con chó bên trái cái cây",
        query_type="KIS",
        entities=[Entity(id="e1", label="Dog"), Entity(id="e2", label="Tree")],
        relations=[Relation(source_id="e1", target_id="e2", relation_type="left_of")]
    )
    audit_query(searcher, executor, "B: Spatial", ir_b.raw_text, ir_b)
    
    # Query C: Attribute
    ir_c = VisualIRGraph(
        query_id="QC",
        raw_text="một chiếc xe màu đỏ",
        query_type="KIS",
        entities=[Entity(id="e1", label="Car")],
        attributes=[Attribute(entity_id="e1", name="color", value="red")]
    )
    audit_query(searcher, executor, "C: Attribute", ir_c.raw_text, ir_c)
    
if __name__ == "__main__":
    run_audit()
