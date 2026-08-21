def compute_fusion_score(siglip_score: float, obj_score: float = 0.0, meta_score: float = 0.0, has_target_objects: bool = True) -> float:
    # Formula: 0.65 * siglip_score + 0.40 * obj_score + 0.10 * meta_score
    return round(0.65 * siglip_score + 0.40 * obj_score + 0.10 * meta_score, 4)
