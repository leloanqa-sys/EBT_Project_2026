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

3. Generator: raw_text -> SigLIP2 (768-dim, google/siglip2-base-patch16-224)

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

    def run(self, query_id: str, raw_text: str, query_type: str = "KIS", question: Optional[str] = None) -> PipelineResult:
        """
        Full pipeline for one query.
        Step 1: M1 - Parse NLP -> VisualIR (with Gemini API, cached)
        Step 2: M2 - SigLIP2 retrieve top-K candidates via FAISS
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
        compute_fusion_scores(candidates, ir_graph=ir_graph)
        
        # Apply NMS after fusion_score is computed to penalize near-duplicates properly
        candidates = self._executor._apply_nms(candidates)
        
        candidates.sort(key=lambda c: c.fusion_score, reverse=True)
        
        # === STEP 5: VQA Re-ranking & Active Relevance Feedback (Dual-Layer) ===
        try:
            from src.role_c_logic.executor import should_escalate_to_vlm
            from src.role_c_logic.vlm_client import GeminiVisionClient
            from src.role_c_logic.prompt_generator import ir_graph_to_prompt
            from src.role_c_logic.ranking import diversify_candidates
            
            # QUOTA OPTIMIZATION:
            # Lọc đa dạng video TRƯỚC KHI gửi cho VLM.
            # Tránh lãng phí 10 lượt gọi API chỉ cho 1 video duy nhất.
            pre_vlm_pool = diversify_candidates(candidates, top_k=20, distinct_top_n=10)
            top_vlm_pool = pre_vlm_pool[:10]
            escalated_candidates = [c for c in top_vlm_pool if should_escalate_to_vlm(c, ir_graph)]
            
            if escalated_candidates:
                vlm_client = GeminiVisionClient()
                vlm_prompt = ir_graph_to_prompt(ir_graph, question=question)
                prompt_version = "v2_qa" if question else "v1"
                
                print(f"  [Pipeline] Escalating {len(escalated_candidates)} candidates to Gemini VLM (Pass 1 - Strict)...")
                results = vlm_client.verify_candidates_batch(escalated_candidates, vlm_prompt, ir_graph.raw_text, prompt_version)
                
                num_matches = 0
                for c, res in zip(escalated_candidates, results):
                    is_match = res.get("match", False)
                    vqa_answer = res.get("answer", None)
                    c.vqa_answer = vqa_answer
                    
                    if is_match:
                        num_matches += 1
                        c.fusion_score += 5.0 # RANK_1_BONUS
                        ans_str = f" | Ans: {vqa_answer}" if vqa_answer else ""
                        print(f"    -> VLM Match for {c.video_id}:{c.frame_idx} (Bonus +5.0){ans_str}")
                    else:
                        print(f"    -> VLM Mismatch for {c.video_id}:{c.frame_idx}")
                        
                # Active Relevance Feedback: If 100% mismatch, fallback to relaxed conceptual association layer
                if num_matches == 0 and getattr(ir_graph, 'relaxed_query_en', None):
                    relaxed_query = ir_graph.relaxed_query_en.strip()
                    strict_query = getattr(ir_graph, 'clip_query_en', '') or ir_graph.raw_text
                    if relaxed_query and relaxed_query.lower() != strict_query.lower():
                        print(f"  [Pipeline] [Active Feedback] 100% VLM Mismatch detected on strict pass! Activating Relaxed Conceptual Layer: '{relaxed_query}'")
                        
                        # Retrieve new candidates with relaxed query
                        relaxed_candidates = self._searcher.search_by_text(relaxed_query, top_k=self.top_k_retrieve)
                        
                        # Filter out already checked frames
                        checked_keys = {f"{c.video_id}_{c.frame_idx}" for c in escalated_candidates}
                        filtered_relaxed = [c for c in relaxed_candidates if f"{c.video_id}_{c.frame_idx}" not in checked_keys]
                        
                        for c in filtered_relaxed:
                            c._temp_detect_scores = []
                            c._temp_spatial_scores = []
                            c.is_ambiguous = False
                            
                        from src.role_c_logic.ranking import compute_fusion_scores
                        compute_fusion_scores(filtered_relaxed, ir_graph=ir_graph)
                        filtered_relaxed = self._executor._apply_nms(filtered_relaxed)
                        filtered_relaxed.sort(key=lambda c: c.fusion_score, reverse=True)
                        
                        # Escalate top 10 from relaxed layer to VLM
                        relaxed_vlm_pool = filtered_relaxed[:10]
                        relaxed_escalated = [c for c in relaxed_vlm_pool if should_escalate_to_vlm(c, ir_graph)]
                        
                        if relaxed_escalated:
                            print(f"  [Pipeline] Escalating {len(relaxed_escalated)} relaxed candidates to Gemini VLM (Pass 2 - Relaxed)...")
                            relaxed_results = vlm_client.verify_candidates_batch(relaxed_escalated, vlm_prompt, ir_graph.raw_text, f"{prompt_version}_relaxed")
                            for c, res in zip(relaxed_escalated, relaxed_results):
                                is_match = res.get("match", False)
                                vqa_answer = res.get("answer", None)
                                c.vqa_answer = vqa_answer
                                if is_match:
                                    c.fusion_score += 4.5 # Slightly lower bonus than strict match but high enough to take Rank 1
                                    ans_str = f" | Ans: {vqa_answer}" if vqa_answer else ""
                                    print(f"    -> VLM Match (Relaxed) for {c.video_id}:{c.frame_idx} (Bonus +4.5){ans_str}")
                                else:
                                    print(f"    -> VLM Mismatch (Relaxed) for {c.video_id}:{c.frame_idx}")
                        
                        # Merge candidates
                        candidates.extend(filtered_relaxed)
                        
        except Exception as e:
            print(f"  [Pipeline] Gemini VLM Re-ranking / Active Feedback failed: {e}")
            
        # === STEP 6: Final Ranking (AIC Confidence Portfolio Strategy) ===
        # Optimizes for mean(R@1, R@5, R@20, R@50, R@100)
        from src.role_c_logic.ranking import rank_confidence_portfolio
        candidates = rank_confidence_portfolio(candidates, top_k=self.top_k_retrieve)

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
            confidence_score=getattr(c, 'fusion_score', c.siglip_score),
            answer=c.vqa_answer,
            vqa_answer=c.vqa_answer
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
        
        # Check if it's QA based on query_type
        is_qa = bool(submissions and submissions[0].query_type == QueryType.QA)
        if is_qa:
            writer.writerow(["query_id", "rank", "video_id", "frame_id", "answer", "confidence_score"])
        else:
            writer.writerow(["query_id", "rank", "video_id", "frame_id", "confidence_score"])
            
        for sub in submissions:
            for item in sub.items:
                if is_qa:
                    writer.writerow([sub.query_id, item.rank, item.video_id, item.frame_id, item.answer or item.vqa_answer or "", f"{item.confidence_score:.6f}"])
                else:
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
