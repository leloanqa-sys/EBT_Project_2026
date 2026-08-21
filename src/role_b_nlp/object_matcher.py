from typing import List, Dict, Any, Optional
import os
import json
import glob
from dataclasses import dataclass
from src.common.schemas import CandidateFrame, VisualIRGraph
from src.role_b_nlp.fusion_score import compute_fusion_score

@dataclass
class DetectedObject:
    label: str
    confidence: float

_MAP_KEYFRAMES_CACHE: Dict[str, Dict[int, int]] = {}

def get_n_from_map_csv(video_id: str, frame_idx: int, map_root: str = "data/raw/map-keyframes") -> Optional[int]:
    """Look up keyframe sequence index 'n' from map-keyframes CSV for a given frame_idx."""
    if video_id not in _MAP_KEYFRAMES_CACHE:
        csv_path = os.path.join(map_root, f"{video_id}.csv")
        mapping = {}
        if os.path.exists(csv_path):
            try:
                import csv
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            mapping[int(row["frame_idx"])] = int(row["n"])
                        except (ValueError, KeyError):
                            pass
            except Exception:
                pass
        _MAP_KEYFRAMES_CACHE[video_id] = mapping

    return _MAP_KEYFRAMES_CACHE[video_id].get(frame_idx)


def parse_json_to_detected_objects(video_id: str, frame_idx: int, objects_root: str = "data/raw/objects") -> List[DetectedObject]:
    """
    Đọc JSON Faster R-CNN và chuyển thành danh sách DetectedObject.
    """
    vid_dir = os.path.join(objects_root, video_id)
    if not os.path.exists(vid_dir):
        return []

    # 1. Try mapping frame_idx -> n -> {n:03d}.json
    json_path = None
    n = get_n_from_map_csv(video_id, frame_idx)
    if n is not None:
        candidate = os.path.join(vid_dir, f"{n:03d}.json")
        if os.path.exists(candidate):
            json_path = candidate

    # 2. Fallbacks
    if not json_path:
        for p in [os.path.join(vid_dir, f"{frame_idx:03d}.json"), os.path.join(vid_dir, f"{frame_idx}.json")]:
            if os.path.exists(p):
                json_path = p
                break

    if not json_path or not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entities = data.get("detection_class_entities", [])
        scores = data.get("detection_scores", [])
        
        objects = []
        for i in range(len(entities)):
            raw_conf = scores[i] if i < len(scores) else 0.0
            try:
                conf = float(raw_conf)
            except (ValueError, TypeError):
                conf = 0.0

            if conf >= 0.1:  # Threshold basic
                objects.append(DetectedObject(label=str(entities[i]), confidence=conf))
        return objects
    except Exception as e:
        return []

def calculate_object_match_score(target_objects: List[str], detected_classes: List[str]) -> float:
    """
    Computes overlap ratio score between target physical objects in query and detected classes in frame.
    Score ranges from 0.0 to 1.0.
    """
    if not target_objects or not detected_classes:
        return 0.0
        
    target_set = set(obj.lower() for obj in target_objects)
    detected_set = set(obj.lower() for obj in detected_classes)
    
    matched = target_set.intersection(detected_set)
    return len(matched) / len(target_set)

def fuse_candidates(candidates: List[CandidateFrame], parsed_query: Any, objects_root: str = "data/raw/objects") -> List[CandidateFrame]:
    """
    Thực hiện contract A-B-C: Tính điểm fusion score cho danh sách candidates.
    """
    target_objs = parsed_query.attributes.get("objects", {}).get("keywords", [])
    
    for cand in candidates:
        if target_objs:
            detected = parse_json_to_detected_objects(cand.video_id, cand.frame_idx, objects_root)
            detected_classes = [obj.label for obj in detected]
            obj_score = calculate_object_match_score(target_objs, detected_classes)
        else:
            obj_score = 0.0
            
        cand.fusion_score = compute_fusion_score(
            clip_score=cand.clip_score,
            obj_score=obj_score,
            has_target_objects=bool(target_objs)
        )
        
    return candidates
