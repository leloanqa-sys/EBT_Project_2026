import os
import json
import time
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from src.common.schemas import CandidateFrame, VisualIRGraph
from src.role_a_retrieval.searcher import VectorSearcher
from src.role_c_logic.deterministic_planner import ExecutionPlan, PlanStep
from src.role_c_logic.capability_registry import CapabilityStatus
from src.common.execution_trace import ExecutionTraceLog, OperatorTrace, TraceLogger

DB_PATH = "data/processed/metadata.db"

class MetadataCache:
    """
    E1.2: SQLite-backed metadata cache.
    Single frame lookup: ~1.7ms (vs 7304ms with raw file I/O).
    In-memory cache layer on top of DB to avoid repeat queries for same frame.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.mem_cache: Dict[Tuple[str, int], Optional[List[Dict]]] = {}

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get_metadata(self, video_id: str, frame_idx: int) -> Optional[List[Dict]]:
        """Returns list of detection dicts for the frame, or None if not found."""
        cache_key = (video_id, frame_idx)
        if cache_key in self.mem_cache:
            return self.mem_cache[cache_key]

        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT class_entity, score, ymin, xmin, ymax, xmax FROM detections WHERE video_id=? AND frame_n=?",
                (video_id, frame_idx)
            ).fetchall()
            if not rows:
                self.mem_cache[cache_key] = None
                return None
            result = [{"class_entity": r[0], "score": r[1], "ymin": r[2], "xmin": r[3], "ymax": r[4], "xmax": r[5]} for r in rows]
            self.mem_cache[cache_key] = result
            return result
        except Exception:
            self.mem_cache[cache_key] = None
            return None

def extract_boxes(detections: List[Dict], target_class: str) -> List[Tuple[float, float, float, float]]:
    """Extract bounding boxes from list of detection dicts for a given class."""
    if not detections:
        return []
    boxes = []
    target_lower = target_class.lower()
    for d in detections:
        if d["class_entity"] == target_lower:
            try:
                boxes.append((d["ymin"], d["xmin"], d["ymax"], d["xmax"]))
            except (KeyError, TypeError):
                pass
    return boxes

class DeterministicExecutor:
    def __init__(self, searcher: VectorSearcher):
        self.searcher = searcher
        self.logger = TraceLogger()
        self.meta_cache = MetadataCache()
        # MVP score threshold for DETECT.
        # score >= threshold in top-100 -> SATISFIED (Keep)
        # class in top-100 but score < threshold -> UNKNOWN (Keep, borderline)
        # class not in top-100 -> UNKNOWN (Keep, top-100 is not closed-world)
        # Only override to 0.0 to disable filtering entirely.
        self.detect_score_threshold: float = 0.3

    def _op_detect(self, candidates: List[CandidateFrame], args: Dict[str, Any]) -> List[CandidateFrame]:
        target_class = args.get("class", "").lower()
        if not target_class:
            return candidates

        for c in candidates:
            meta = self.meta_cache.get_metadata(c.video_id, c.frame_idx)
            if meta is None:
                # No DB record -> keep obj_score as 0.0 (neutral)
                continue

            # Find the highest score for target class in this frame's top-100
            best_score = max((d["score"] for d in meta if d["class_entity"] == target_class), default=0.0)
            
            # Soft Scoring: Add to candidate's obj_score
            c.obj_score += best_score

        return candidates

    def _op_spatial(self, candidates: List[CandidateFrame], args: Dict[str, Any], relation: str) -> List[CandidateFrame]:
        source_label = args.get("source_label", "")
        target_label = args.get("target_label", "")
        
        if not source_label or not target_label:
            return candidates
            
        for c in candidates:
            meta = self.meta_cache.get_metadata(c.video_id, c.frame_idx)
            if not meta:
                continue
                
            source_boxes = extract_boxes(meta, source_label)
            target_boxes = extract_boxes(meta, target_label)
            
            if not source_boxes or not target_boxes:
                continue
                
            # Soft Scoring: Check if ANY source/target pair satisfies relation
            satisfied = False
            for s_box in source_boxes:
                s_center_y = (s_box[0] + s_box[2]) / 2.0
                s_center_x = (s_box[1] + s_box[3]) / 2.0
                for t_box in target_boxes:
                    t_center_y = (t_box[0] + t_box[2]) / 2.0
                    t_center_x = (t_box[1] + t_box[3]) / 2.0
                    
                    if relation == "left_of" and s_center_x < t_center_x:
                        satisfied = True
                    elif relation == "right_of" and s_center_x > t_center_x:
                        satisfied = True
                    elif relation == "above" and s_center_y < t_center_y:
                        satisfied = True
                    elif relation == "below" and s_center_y > t_center_y:
                        satisfied = True
                        
                    if satisfied:
                        break
                if satisfied:
                    break
                    
            if satisfied:
                c.spatial_score += 1.0
                
        return candidates

    def _apply_nms(self, candidates: List[CandidateFrame], min_seconds: float = 5.0) -> List[CandidateFrame]:
        """Filters out frames that are too close in time for the same video using soft penalty."""
        # Sort candidates temporarily by fusion_score to ensure highest scores are kept without penalty
        candidates.sort(key=lambda c: getattr(c, 'fusion_score', c.clip_score), reverse=True)
        
        added_videos = {} # video_id -> list of (pts_time, fusion_score)
        for c in candidates:
            pts = c.pts_time
            prev_entries = added_videos.get(c.video_id, [])
            
            # If there's a frame within min_seconds, apply a penalty
            if any(abs(pts - t) < min_seconds for t, _ in prev_entries):
                c.fusion_score -= 1.0 # Soft Penalty for NMS
                
            added_videos.setdefault(c.video_id, []).append((pts, c.fusion_score))
            
        return candidates

    def execute_plan(self, plan: ExecutionPlan, ir_graph: VisualIRGraph, top_k_raw: int = 300) -> List[CandidateFrame]:
        trace = ExecutionTraceLog(
            query_id=plan.query_id,
            query_text=ir_graph.raw_text,
            ir_graph=ir_graph.model_dump(),
            plan_steps=[{"op": step.operator_name, "target": step.target, "status": step.status.value} for step in plan.steps]
        )
        
        start_total = time.time()
        candidates: List[CandidateFrame] = []
        
        for step in plan.steps:
            op_start = time.time()
            candidates_in = len(candidates)
            
            if step.status in (CapabilityStatus.UNSUPPORTED, CapabilityStatus.DEFER):
                trace.operator_traces.append(OperatorTrace(
                    operator_name=step.operator_name,
                    target=step.target,
                    latency_ms=0.0,
                    candidates_in=candidates_in,
                    candidates_out=candidates_in,
                    status="SKIPPED (UNSUPPORTED)" if step.status == CapabilityStatus.UNSUPPORTED else "SKIPPED (DEFERRED_TO_VLM)"
                ))
                continue
                
            if step.operator_name == "CLIP_RETRIEVE":
                # Ensure raw_text is used as per PO's generator audit
                candidates = self.searcher.search_by_text(ir_graph.raw_text, top_k=step.args.get("k", top_k_raw))
                trace.initial_candidates = len(candidates)
                status_str = "READY"
            elif step.operator_name == "DETECT":
                candidates = self._op_detect(candidates, step.args)
                status_str = "READY"
            elif step.operator_name.startswith("SPATIAL_"):
                # Resolve labels from IR
                source_id = step.args.get("source")
                target_id = step.args.get("target")
                source_label = next((e.label for e in ir_graph.entities if e.id == source_id), "")
                target_label = next((e.label for e in ir_graph.entities if e.id == target_id), "")
                step.args["source_label"] = source_label
                step.args["target_label"] = target_label
                
                relation_map = {
                    "SPATIAL_LEFT_OF": "left_of",
                    "SPATIAL_RIGHT_OF": "right_of",
                    "SPATIAL_ABOVE": "above",
                    "SPATIAL_BELOW": "below"
                }
                
                if step.operator_name in relation_map:
                    relation = relation_map[step.operator_name]
                    candidates = self._op_spatial(candidates, step.args, relation)
                    status_str = "READY"
                else:
                    status_str = "UNKNOWN/UNSUPPORTED SPATIAL"
            elif step.operator_name == "FILTER_ATTRIBUTE":
                # UNTESTABLE -> Returns candidates unchanged
                status_str = "UNTESTABLE"
            else:
                # Fallback for UNKNOWN
                status_str = "UNKNOWN (Kept)"
                
            candidates_out = len(candidates)
            op_latency = (time.time() - op_start) * 1000
            
            trace.operator_traces.append(OperatorTrace(
                operator_name=step.operator_name,
                target=step.target,
                latency_ms=op_latency,
                candidates_in=candidates_in,
                candidates_out=candidates_out,
                status=status_str
            ))
            
        # Apply NMS (Non-Maximum Suppression) to remove near-duplicate frames from the same video
        candidates = self._apply_nms(candidates)
            
        trace.final_candidates = len(candidates)
        trace.total_latency_ms = (time.time() - start_total) * 1000
        self.logger.save_trace(trace)
        
        return candidates
