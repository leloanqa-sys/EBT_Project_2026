def compute_fusion_score(
    clip_score: float,
    obj_score: float = 0.0,
    meta_score: float = 0.0,
    has_target_objects: bool = False,
    w_clip: float = 0.25,
    w_obj: float = 0.65,
    w_meta: float = 0.10
) -> float:
    """
    Role B Score Fusion Module:
    Policy: 'NLP & Object Detection is Primary, CLIP is Fallback'.
    
    If target_objects are present:
        w_obj = 0.65 (Primary)
        w_clip = 0.25 (Secondary)
        w_meta = 0.10
    Else (No target objects):
        w_clip = 0.90 (Fallback)
        w_obj = 0.0
        w_meta = 0.10
    """
    if not has_target_objects:
        wc, wo, wm = 0.90, 0.0, 0.10
    else:
        wc, wo, wm = w_clip, w_obj, w_meta

    raw_score = (wc * clip_score) + (wo * obj_score) + (wm * meta_score)
    return round(raw_score, 4)
