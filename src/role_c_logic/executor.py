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
from src.role_c_logic.synonyms_map import CLASS_SYNONYMS_MAP
from src.role_c_logic.taxonomy_loader import TAXONOMY_CLASSES_SET
from src.common.thresholds import (
    SPATIAL_MIN_DIST,
    AREA_DIFF_AMBIGUOUS_THRESHOLD,
    AMBIGUOUS_PLACEHOLDER_SCORE
)

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
    target_lower = target_class.lower().strip()
    
    if target_lower in CLASS_SYNONYMS_MAP:
        aliases = set(CLASS_SYNONYMS_MAP[target_lower])
    elif target_lower in TAXONOMY_CLASSES_SET:
        aliases = {target_lower}
    else:
        aliases = {target_lower}
        
    for d in detections:
        if d["class_entity"].lower().strip() in aliases:
            try:
                boxes.append((d["ymin"], d["xmin"], d["ymax"], d["xmax"]))
            except (KeyError, TypeError):
                pass
    return boxes

def safe_mean(scores: List[float]) -> float:
    """
    Rào lỗi kiến trúc (Guardrail): CẤM CỘNG DỒN (+=) điểm số tùy tiện.
    Hàm này tổng hợp điểm của nhiều thực thể trên 1 Candidate.
    Dùng trung bình (mean) để đảm bảo score không bao giờ vượt quá 1.0.
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

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

    def _log_unmapped_target(self, target_class: str):
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "unmapped_targets.csv")
        
        if not os.path.exists(log_path):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("timestamp,target_class\n")
                
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.time()},{target_class}\n")

    def _op_detect(self, candidates: List[CandidateFrame], args: Dict[str, Any]) -> List[CandidateFrame]:
        target_class = args.get("class", "").lower().strip()
        if not target_class:
            return candidates

        # Tầng 2: Nếu có alias map tay, dùng nó (ưu tiên vì đã curate kỹ)
        if target_class in CLASS_SYNONYMS_MAP:
            aliases = set(CLASS_SYNONYMS_MAP[target_class])
        # Tầng 1: Nếu target trùng thẳng tên 1 class thật trong taxonomy
        elif target_class in TAXONOMY_CLASSES_SET:
            aliases = {target_class}
        # Miss thật sự: fallback + log
        else:
            aliases = {target_class}
            self._log_unmapped_target(target_class)

        for c in candidates:
            meta = self.meta_cache.get_metadata(c.video_id, c.frame_idx)
            best_score = 0.0
            if meta is not None:
                # Find the highest score for any valid alias in this frame
                best_score = max((d["score"] for d in meta if d["class_entity"].lower().strip() in aliases), default=0.0)
            
            # Khởi tạo mảng lưu điểm tạm thời nếu chưa có
            if not hasattr(c, '_temp_detect_scores'):
                c._temp_detect_scores = []
            c._temp_detect_scores.append(best_score)
            
            # Guardrail: Gán đè (=) qua hàm aggregate, CẤM +=
            c.obj_score = safe_mean(c._temp_detect_scores)

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
                
            # Khởi tạo mảng điểm nếu chưa có
            if not hasattr(c, '_temp_spatial_scores'):
                c._temp_spatial_scores = []
                
            best_relation_score = 0.0
            
            for s_box in source_boxes:
                s_ymin, s_xmin, s_ymax, s_xmax = s_box
                s_center_y = (s_ymin + s_ymax) / 2.0
                s_center_x = (s_xmin + s_xmax) / 2.0
                s_area = (s_ymax - s_ymin) * (s_xmax - s_xmin)
                
                for t_box in target_boxes:
                    t_ymin, t_xmin, t_ymax, t_xmax = t_box
                    t_center_y = (t_ymin + t_ymax) / 2.0
                    t_center_x = (t_xmin + t_xmax) / 2.0
                    t_area = (t_ymax - t_ymin) * (t_xmax - t_xmin)
                    
                    satisfied = False
                    
                    # 2D Geometry
                    if relation in ("left_of", "right_of"):
                        # X-distance threshold
                        dist_x = abs(s_center_x - t_center_x)
                        # Y-Axis Alignment: Đồng giao trục Y
                        y_overlap = min(s_ymax, t_ymax) > max(s_ymin, t_ymin)
                        
                        if dist_x >= SPATIAL_MIN_DIST and y_overlap:
                            if relation == "left_of" and s_center_x < t_center_x:
                                satisfied = True
                            elif relation == "right_of" and s_center_x > t_center_x:
                                satisfied = True
                                
                    elif relation in ("above", "below"):
                        dist_y = abs(s_center_y - t_center_y)
                        x_overlap = min(s_xmax, t_xmax) > max(s_xmin, t_xmin)
                        
                        if dist_y >= SPATIAL_MIN_DIST and x_overlap:
                            if relation == "above" and s_center_y < t_center_y:
                                satisfied = True
                            elif relation == "below" and s_center_y > t_center_y:
                                satisfied = True
                                
                    # 3D Depth Heuristics
                    elif relation in ("front", "in_front_of", "behind"):
                        max_area = max(s_area, t_area)
                        if max_area > 0:
                            area_diff_ratio = abs(s_area - t_area) / max_area
                            
                            ymax_says_front = s_ymax > t_ymax
                            area_says_front = s_area > t_area
                            signals_conflict = (ymax_says_front != area_says_front)
                            
                            if signals_conflict or area_diff_ratio < AREA_DIFF_AMBIGUOUS_THRESHOLD:
                                c.is_ambiguous = True
                                # Điểm hòa vốn cho cases cần hỏi VLM (không set satisfied để tránh bị ghi đè thành 1.0)
                                best_relation_score = max(best_relation_score, AMBIGUOUS_PLACEHOLDER_SCORE)
                            else:
                                if relation in ("front", "in_front_of") and ymax_says_front and area_says_front:
                                    satisfied = True
                                elif relation == "behind" and not ymax_says_front and not area_says_front:
                                    satisfied = True
                                    
                    if satisfied:
                        best_relation_score = 1.0
                        break
                if best_relation_score == 1.0:
                    break
                    
            c._temp_spatial_scores.append(best_relation_score)
            c.spatial_score = safe_mean(c._temp_spatial_scores)
            
        return candidates

    def _apply_nms(self, candidates: List[CandidateFrame], min_seconds: float = 4.0, max_per_video: int = 5) -> List[CandidateFrame]:
        """
        Hard NMS: Ensures that each rank corresponds to a completely distinct [s, e] segment.
        Drops any candidate that is within min_seconds of a higher-scoring candidate in the same video.
        Also applies a hard cap on maximum distinct segments per video.
        """
        candidates.sort(key=lambda c: getattr(c, 'fusion_score', c.siglip_score), reverse=True)
        result = []
        added_videos: Dict[str, List[float]] = {}
        
        for c in candidates:
            vid = c.video_id
            pts = c.pts_time
            prev_entries = added_videos.get(vid, [])
            
            # 1. Hard cap per video (diversity constraint)
            if len(prev_entries) >= max_per_video:
                continue
                
            # 2. Hard NMS temporal window check
            is_overlap = False
            for t in prev_entries:
                if abs(pts - t) < min_seconds:
                    is_overlap = True
                    break
                    
            if not is_overlap:
                result.append(c)
                added_videos.setdefault(vid, []).append(pts)
                
        return result

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
                # Merge raw, translated, and relaxed views so a long Vietnamese
                # query can recover frames missed by one tokenizer formulation.
                query_candidates: Dict[Tuple[str, int], Tuple[CandidateFrame, float]] = {}
                search_queries = [
                    (ir_graph.raw_text, 1.0),
                    (getattr(ir_graph, "clip_query_en", ""), 1.0),
                    (getattr(ir_graph, "relaxed_query_en", ""), 0.5),
                ]
                seen_queries = set()
                for search_query, query_weight in search_queries:
                    search_query = search_query.strip() if search_query else ""
                    if not search_query or search_query in seen_queries:
                        continue
                    seen_queries.add(search_query)
                    retrieved = self.searcher.search_by_text(search_query, top_k=step.args.get("k", top_k_raw))
                    for rank, candidate in enumerate(retrieved, start=1):
                        key = (candidate.video_id, candidate.frame_idx)
                        reciprocal_rank = query_weight / (60.0 + rank)
                        previous = query_candidates.get(key)
                        if previous is None:
                            query_candidates[key] = (candidate, reciprocal_rank)
                        else:
                            previous[0].siglip_score = max(previous[0].siglip_score, candidate.siglip_score)
                            query_candidates[key] = (previous[0], previous[1] + reciprocal_rank)
                candidates = [item[0] for item in sorted(query_candidates.values(), key=lambda item: item[1], reverse=True)[:top_k_raw]]
                for c in candidates:
                    c._temp_detect_scores = []
                    c._temp_spatial_scores = []
                    c.is_ambiguous = False
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
                    "SPATIAL_BELOW": "below",
                    "SPATIAL_BEHIND": "behind",
                    "SPATIAL_FRONT": "front"
                }
                
                if step.operator_name in relation_map:
                    relation = relation_map[step.operator_name]
                    candidates = self._op_spatial(candidates, step.args, relation)
                    status_str = "READY"
                else:
                    status_str = "UNKNOWN/UNSUPPORTED SPATIAL"
            elif step.operator_name == "FILTER_ATTRIBUTE":
                # Unfreeze logic: Đánh dấu candidates là ambiguous để VLM verify thuộc tính chi tiết (màu sắc/text)
                attr_name = step.args.get("name", "").lower()
                attr_val = step.args.get("value", "").lower()
                # Chỉ lọc những thuộc tính màu sắc hoặc văn bản
                if attr_name in ("color", "shirt_color", "pants_color", "text", "ocr", "written"):
                    for c in candidates:
                        c.is_ambiguous = True
                status_str = "EXPERIMENTAL"
            elif step.operator_name == "EVENT_ACTION":
                # Unfreeze logic: Ép buộc escalate lên VLM để verify hành động (Action/Event)
                for c in candidates:
                    c.is_ambiguous = True
                status_str = "EXPERIMENTAL"
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
            
        trace.final_candidates = len(candidates)
        trace.total_latency_ms = (time.time() - start_total) * 1000
        self.logger.save_trace(trace)
        self.logger.print_trace_summary(trace) # Added to show which pipeline sequence worked/broke
        
        return candidates

def should_escalate_to_vlm(c: CandidateFrame, ir_graph: VisualIRGraph = None) -> bool:
    """
    Agent 3: Escalation conditions.
    """
    if getattr(c, 'is_ambiguous', False):
        return True
        
    if c.siglip_score < 0.5: # Low retrieval confidence
        return True
        
    if ir_graph:
        if getattr(ir_graph, 'scoring_plan', None) and ir_graph.scoring_plan.vlm_required:
            return True
        if len(ir_graph.events) > 0:
            return True
        for a in ir_graph.attributes:
            if a.name.lower() in ['text', 'ocr', 'written']:
                return True
                
    return False
