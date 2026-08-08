def compute_fusion_score(
    clip_score: float,
    obj_score: float = 0.0,
    meta_score: float = 0.0,
    w_clip: float = 0.70,
    w_obj: float = 0.20,
    w_meta: float = 0.10
) -> float:
    """
    Role B Score Fusion Module:
    Combines CLIP similarity score, object detection match score, and metadata match score into a single raw score.
    
    Formula: raw_score = w_clip * clip_score + w_obj * obj_score + w_meta * meta_score
    """
    raw_score = (w_clip * clip_score) + (w_obj * obj_score) + (w_meta * meta_score)
    return round(raw_score, 4)
