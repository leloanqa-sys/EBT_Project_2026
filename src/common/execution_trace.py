import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class OperatorTrace:
    operator_name: str
    target: str
    latency_ms: float
    candidates_in: int
    candidates_out: int
    status: str

@dataclass
class ExecutionTraceLog:
    query_id: str
    query_text: str
    ir_graph: Dict[str, Any]
    plan_steps: List[Dict[str, Any]]
    
    initial_candidates: int = 0
    final_candidates: int = 0
    
    operator_traces: List[OperatorTrace] = field(default_factory=list)
    total_latency_ms: float = 0.0
    
    recall_before: Optional[float] = None
    recall_after: Optional[float] = None
    precision_at_5: Optional[float] = None

class TraceLogger:
    """
    Logger cho M2-C.
    Ghi nhận chi tiết luồng thực thi: IR -> Plan -> Operator candidate reduction -> Latency.
    """
    def __init__(self, log_dir: str = "outputs/traces"):
        self.log_dir = log_dir
        self.traces: List[ExecutionTraceLog] = []
        os.makedirs(self.log_dir, exist_ok=True)
        
    def save_trace(self, trace: ExecutionTraceLog):
        self.traces.append(trace)
        file_path = os.path.join(self.log_dir, f"{trace.query_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(trace), f, ensure_ascii=False, indent=2)
            
    def print_trace_summary(self, trace: ExecutionTraceLog):
        print(f"\n=== TRACE SUMMARY: {trace.query_id} ===")
        print(f"Query: {trace.query_text}")
        print(f"Candidates: {trace.initial_candidates} -> {trace.final_candidates}")
        print(f"Total Latency: {trace.total_latency_ms:.1f}ms")
        print("Operator Steps:")
        for op in trace.operator_traces:
            print(f"  - {op.operator_name}({op.target}) [{op.status}]: {op.candidates_in} -> {op.candidates_out} ({op.latency_ms:.1f}ms)")
        print("===================================\n")
