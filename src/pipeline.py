"""
MVP Pipeline - End-to-End Visual Information Retrieval
=====================================================
Connects all components: M1 (NLP) -> M2 (FAISS Retrieval) -> Operators (SQLite) -> Submission

Design decisions (self-proposed based on evidence):
1. DETECT score threshold = 0.3 (MVP):
   - 584 classes, top-100 fixed output, scores from 0.875 down to 0.002
   - score >= 0.3 indicates the detector is "fairly confident" the object exists
   - score < 0.3 in top-100 -> UNKNOWN (keep candidate) - preservative
   - class not in top-100 -> UNKNOWN (keep candidate) - top-100 is not closed-world
   
2. SPATIAL: Existential semantics (∃ pair satisfies), UNKNOWN=Keep

3. Generator: raw_text -> CLIP (G2 configuration, frozen after G0/G1/G2 audit)

4. Top-K: retrieve 500, return top-100 for submission
"""
import os
import time
import json
import sqlite3
import csv
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from src.common.schemas import (
    CandidateFrame, VisualIRGraph, SubmissionItem, SubmissionOutput, QueryType
)
from src.role_a_retrieval.searcher import VectorSearcher
from src.role_b_nlp.gemini_nlp_engine import compile_to_visual_ir
from src.role_c_logic.deterministic_planner import create_deterministic_plan
from src.role_c_logic.executor import DeterministicExecutor

DETECT_SCORE_THRESHOLD = 0.3  # MVP default — pending official PO approval
TOP_K_RETRIEVE = 500
TOP_K_SUBMIT = 100


@dataclass
class PipelineResult:
    query_id: str
    query_text: str
    query_type: str
    candidates: List[CandidateFrame]
    latency_ms: float
    detect_threshold_used: float
    operator_trace: Dict[str, Any]


class MVPPipeline:
    """
    Full end-to-end pipeline connecting M1 + M2.
    Singleton-style: load FAISS index once, reuse across queries.
    """

    def __init__(self,
                 detect_threshold: float = DETECT_SCORE_THRESHOLD,
                 top_k_retrieve: int = TOP_K_RETRIEVE,
                 searcher: Optional[VectorSearcher] = None):
        self.detect_threshold = detect_threshold
        self.top_k_retrieve = top_k_retrieve
        self._searcher: Optional[VectorSearcher] = searcher
        self._executor: Optional[DeterministicExecutor] = None

    def _load(self):
        if self._searcher is None:
            print("  [MVP] Loading FAISS index + metadata cache...")
            self._searcher = VectorSearcher()
        
        if self._executor is None:
            self._executor = DeterministicExecutor(self._searcher)
            # Set score threshold on executor
            self._executor.detect_score_threshold = self.detect_threshold
            print(f"  [MVP] Ready. DETECT threshold = {self.detect_threshold}")

    def run(self, query_id: str, raw_text: str, query_type: str = "KIS") -> PipelineResult:
        """
        Full pipeline for one query.
        Step 1: M1 - Parse NLP -> VisualIR (with Gemini API, cached)
        Step 2: M2 - CLIP retrieve top-K candidates
        Step 3: Operators - DETECT / SPATIAL (SQLite, UNKNOWN=Keep)
        Step 4: Return ranked candidates
        """
        self._load()
        t_start = time.time()

        # === STEP 1: M1 - NLP Parsing ===
        ir_graph = compile_to_visual_ir(query_id, raw_text, query_type)

        # === STEP 2+3: M2 - Plan + Execute ===
        plan = create_deterministic_plan(ir_graph)
        candidates = self._executor.execute_plan(plan, ir_graph, top_k_raw=self.top_k_retrieve)
        
        # === STEP 4: Soft Scoring & Ranking ===
        from src.role_c_logic.ranking import compute_fusion_scores
        compute_fusion_scores(candidates)
        candidates.sort(key=lambda c: c.fusion_score, reverse=True)
        
        # === STEP 5: VQA Re-ranking (Role D) ===
        # [DISABLED] The local machine is running out of memory (OS Error 1455 Paging file too small) 
        # when trying to load Qwen2-VL-2B. We will bypass this for the MVP testing phase.
        '''
        top_20 = candidates[:20]
        if top_20:
            try:
                import sys
                import os
                import base64
                PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
                from tools.review_tool import resolve_keyframe_b64
                from src.role_c_logic.vqa_model import get_vqa_engine
                
                vqa_engine = get_vqa_engine()
                for c in top_20:
                    b64_str, _, _ = resolve_keyframe_b64(c.video_id, c.frame_idx, keyframes_root=os.path.join(PROJECT_ROOT, "data", "raw", "keyframes"))
                    if b64_str:
                        image_bytes = base64.b64decode(b64_str.split(",")[1])
                        # Check if image matches the query
                        if vqa_engine.verify_image_match(image_bytes, raw_text):
                            c.fusion_score += 10.0 # Huge bonus for VQA match
                        else:
                            c.fusion_score -= 2.0  # Penalty for mismatch
            except Exception as e:
                print(f"[Pipeline] VQA Re-ranking failed or skipped: {e}")
                
        # Re-sort after VQA bonuses
        candidates.sort(key=lambda c: c.fusion_score, reverse=True)
        '''

        latency = (time.time() - t_start) * 1000

        # Load trace for reporting
        trace_path = f"outputs/traces/{query_id}.json"
        op_trace = {}
        if os.path.exists(trace_path):
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    op_trace = json.load(f)
            except Exception:
                pass

        return PipelineResult(
            query_id=query_id,
            query_text=raw_text,
            query_type=query_type,
            candidates=candidates,
            latency_ms=latency,
            detect_threshold_used=self.detect_threshold,
            operator_trace=op_trace,
        )

    def run_batch(self, queries: List[Dict]) -> List[PipelineResult]:
        """
        Run multiple queries. queries = [{"id":..., "text":..., "type":...}, ...]
        """
        results = []
        for i, q in enumerate(queries):
            qid = q.get("id", f"Q{i+1:03d}")
            text = q.get("text", "")
            qtype = q.get("type", "KIS")
            print(f"  [{i+1}/{len(queries)}] {qid}: {text[:60]}...")
            result = self.run(qid, text, qtype)
            print(f"    -> {len(result.candidates)} candidates | {result.latency_ms:.0f}ms")
            results.append(result)
        return results


def to_submission(result: PipelineResult, top_k: int = TOP_K_SUBMIT) -> SubmissionOutput:
    """Convert PipelineResult to SubmissionOutput for CSV export."""
    items = []
    for rank, c in enumerate(result.candidates[:top_k], start=1):
        items.append(SubmissionItem(
            rank=rank,
            video_id=c.video_id,
            frame_id=c.frame_idx,
            confidence_score=c.clip_score,
        ))
    return SubmissionOutput(
        query_id=result.query_id,
        query_type=QueryType(result.query_type) if result.query_type in ("KIS", "QA", "TRAKE") else QueryType.KIS,
        items=items,
    )


def export_csv(submissions: List[SubmissionOutput], output_path: str = "outputs/submission_mvp.csv"):
    """Export submissions to AIC-compatible CSV format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "rank", "video_id", "frame_id", "clip_score"])
        for sub in submissions:
            for item in sub.items:
                writer.writerow([sub.query_id, item.rank, item.video_id, item.frame_id, f"{item.confidence_score:.6f}"])
    print(f"  [MVP] Submission saved to {output_path}")
    return output_path


def print_trace_summary(result: PipelineResult):
    """Pretty-print operator trace for a result."""
    trace = result.operator_trace
    if not trace:
        print("  [No trace available]")
        return
    print(f"  Candidates: {trace.get('initial_candidates', '?')} -> {trace.get('final_candidates', '?')}")
    for op in trace.get("operator_traces", []):
        bar = "[OK]" if op["status"] in ("READY", "UNTESTABLE") else "[SKIP]"
        print(f"  {bar} {op['operator_name']:20s} {op['candidates_in']:4d} -> {op['candidates_out']:4d} ({op['latency_ms']:.1f}ms) [{op['status']}]")
