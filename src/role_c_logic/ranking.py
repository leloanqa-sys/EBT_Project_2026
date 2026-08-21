from typing import List
from dataclasses import replace
from itertools import groupby

from src.common.schemas import CandidateFrame, SubmissionItem, SubmissionOutput, QueryType

import os
import json

# Ngưỡng từ khóa bối cảnh (Scene Keywords)
SCENE_KEYWORDS = {"beach", "sea", "ocean", "forest", "mountain", "room", "classroom", "stadium", "street", "park", "kitchen", "office", "garden", "road"}

def compute_fusion_scores(candidates: List[CandidateFrame], ir_graph = None):
    """
    Applies Soft Scoring to compute the final fusion_score for each candidate.
    Weights are adapted dynamically:
    1. Read and validate Gemini ScoringPlan if available.
    2. Fallback to dynamic heuristics if Gemini weights are missing/invalid.
    3. Fallback to tuner output or defaults.
    """
    # 1. Khởi tạo bộ trọng số mặc định (Default Weights)
    w_clip = 1.0
    w_obj = 0.5
    w_spatial = 0.5

    # 2. Thử load cấu hình tối ưu từ ML Tuner tuning_results.json nếu có
    tuner_config_p = "outputs/tuning_results.json"
    if os.path.exists(tuner_config_p):
        try:
            with open(tuner_config_p, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Check category specific weights first
                cat_weights = config.get("category_weights", {})
                global_weights = config.get("global_weights", config.get("optimal_weights", {}))
                
                context_key = getattr(getattr(ir_graph, 'scoring_plan', None), 'context_type', 'general_scene')
                chosen_w = cat_weights.get(context_key, global_weights)
                
                w_clip = chosen_w.get("w_siglip", chosen_w.get("w_clip", w_clip))
                w_obj = chosen_w.get("w_obj", w_obj)
                w_spatial = chosen_w.get("w_spatial", w_spatial)
        except Exception:
            pass

    # 3. Sử dụng Gemini ScoringPlan (nếu có và hợp lệ)
    applied_gemini_plan = False
    if ir_graph and hasattr(ir_graph, 'scoring_plan') and ir_graph.scoring_plan:
        plan = ir_graph.scoring_plan
        plan_w_siglip = getattr(plan, 'w_siglip', getattr(plan, 'w_clip', 1.0))
        # Validate weights bounds [0.0, 2.0] to prevent hallucinations/overflow
        if 0.0 <= plan_w_siglip <= 2.0 and 0.0 <= plan.w_obj <= 2.0 and 0.0 <= plan.w_spatial <= 2.0:
            w_clip = plan_w_siglip
            w_obj = plan.w_obj
            w_spatial = plan.w_spatial
            applied_gemini_plan = True
            print(f"  [Scoring] Using Gemini ScoringPlan: context={plan.context_type} | weights=[w_siglip={w_clip}, w_obj={w_obj}, w_spatial={w_spatial}] | rationale: {plan.rationale}")

    # 4. Fallback sang Dynamic Heuristic nếu không áp dụng được Gemini plan
    if not applied_gemini_plan and ir_graph:
        raw_txt = ir_graph.raw_text.lower()
        has_color = any(a.name.lower() in ("color", "shirt_color", "pants_color") for a in ir_graph.attributes)
        has_scene = any(kw in raw_txt for kw in SCENE_KEYWORDS)

        # A. Ngữ cảnh Màu sắc (Color-heavy Context)
        if has_color:
            w_clip = 1.50
            w_obj = 0.20
            w_spatial = 0.30
        # B. Ngữ cảnh Bối cảnh (Scene-heavy Context)
        elif has_scene:
            w_clip = 1.80
            w_obj = 0.05
            w_spatial = 0.10
        # C. Ngữ cảnh Không gian (Spatial-heavy Context)
        elif ir_graph.relations:
            w_clip = 0.80
            w_obj = 0.60
            w_spatial = 1.20

    # 4. Tính điểm Fusion (Có chuẩn hóa)
    if candidates:
        max_clip = max(c.siglip_score for c in candidates)
        min_clip = min(c.siglip_score for c in candidates)
        clip_range = max_clip - min_clip
        if clip_range == 0:
            clip_range = 1.0  # Chống lỗi chia 0
    else:
        max_clip, min_clip, clip_range = 0, 0, 1.0

    w_obj = 0.0
    w_spatial = 0.0
    for c in candidates:
        norm_clip = (c.siglip_score - min_clip) / clip_range
        c.fusion_score = (w_clip * norm_clip) + (w_obj * c.obj_score) + (w_spatial * c.spatial_score)


def _get_score(candidate: CandidateFrame) -> float:
    """Helper to safely retrieve score (fusion_score or siglip_score)."""
    return getattr(candidate, 'fusion_score', candidate.siglip_score)

def _representative(cluster: List[CandidateFrame]) -> CandidateFrame:
    """
    Selects representative candidate for an event cluster.
    - Picks candidate with highest fusion/clip score.
    - Uses median frame_idx to minimize boundary error.
    - Uses dataclasses.replace to create a clean copy without mutating original object.
    """
    best = max(cluster, key=_get_score)
    mid_frame_idx = cluster[len(cluster) // 2].frame_idx
    return replace(best, frame_idx=mid_frame_idx)

def cluster_by_event(candidates: List[CandidateFrame], gap_threshold: int = 15) -> List[CandidateFrame]:
    """
    Groups candidate frames into temporal event clusters per video.
    Clustering condition: (frame_idx - cluster[0].frame_idx <= gap_threshold).
    """
    if not candidates:
        return []

    # Sort candidates by (video_id, frame_idx) for grouping
    sorted_c = sorted(candidates, key=lambda c: (c.video_id, c.frame_idx))
    clustered_representatives: List[CandidateFrame] = []

    for video_id, group_iter in groupby(sorted_c, key=lambda c: c.video_id):
        group = list(group_iter)
        cluster = [group[0]]
        for c in group[1:]:
            if c.frame_idx - cluster[0].frame_idx <= gap_threshold:
                cluster.append(c)
            else:
                clustered_representatives.append(_representative(cluster))
                cluster = [c]
        clustered_representatives.append(_representative(cluster))

    return clustered_representatives

def rank_confidence_portfolio(
    candidates: List[CandidateFrame],
    top_k: int = 100
) -> List[CandidateFrame]:
    """
    AIC 2026 Confidence Portfolio Strategy:
    Optimizes for Final Score = mean(R@1, R@5, R@20, R@50, R@100).
    
    Logic:
    - If top candidates have strong confidence (e.g., VLM confirmed >= 5.0 or high fusion):
      * Rank 1: Peak frame of best video.
      * Rank 2-5: Temporal neighbors of Top 1 video (carpet bomb window to guarantee R@5 hit).
      * Rank 6-15: Top frame + neighbors of 2nd and 3rd candidate videos.
      * Rank 16-100: Diversified candidates across other videos to maximize R@50/R@100 recall.
    - If low confidence / general search:
      * Ranks 1-5: Top candidates from distinct videos.
      * Ranks 6-100: Score-sorted pool.
    """
    if not candidates:
        return []

    # Group candidates by video_id
    video_groups = {}
    for c in candidates:
        video_groups.setdefault(c.video_id, []).append(c)

    for vid in video_groups:
        video_groups[vid].sort(key=_get_score, reverse=True)

    sorted_vids = sorted(video_groups.keys(), key=lambda v: _get_score(video_groups[v][0]), reverse=True)
    if not sorted_vids:
        return candidates[:top_k]

    best_vid = sorted_vids[0]
    best_cand = video_groups[best_vid][0]
    best_score = _get_score(best_cand)

    is_high_confidence = (best_score >= 5.0)  # VLM Match bonus applied

    result: List[CandidateFrame] = []
    placed_ids = set()

    def _add_cand(c: CandidateFrame):
        cid = (c.video_id, c.frame_idx)
        if cid not in placed_ids:
            placed_ids.add(cid)
            result.append(c)

    if is_high_confidence:
        # High confidence: Carpet bomb top video neighbors into Ranks 1-5
        _add_cand(best_cand)
        
        # Add up to 4 neighbors from top video for Ranks 2-5
        top_vid_cands = video_groups[best_vid]
        for c in top_vid_cands[1:]:
            if len(result) >= 5:
                break
            _add_cand(c)

        # If top video didn't have 5 frames in pool, generate close temporal frames
        fps = best_cand.fps if (best_cand.fps and 10.0 <= best_cand.fps <= 60.0) else 25.0
        offsets = [-10, 10, -25, 25, -50, 50]
        for off in offsets:
            if len(result) >= 5:
                break
            syn_frame = max(0, best_cand.frame_idx + off)
            syn_c = replace(
                best_cand,
                frame_idx=syn_frame,
                pts_time=syn_frame / fps,
                fusion_score=best_cand.fusion_score - 0.01 * (abs(off) / fps)
            )
            _add_cand(syn_c)

        # Ranks 6-20: Top 1 frame and close neighbors of 2nd, 3rd, 4th confident videos
        for vid in sorted_vids[1:5]:
            if len(result) >= 20:
                break
            v_cands = video_groups[vid]
            for c in v_cands[:3]:  # Top 3 frames of this video
                if len(result) >= 20:
                    break
                _add_cand(c)

        # Ranks 21-100: Diversified across remaining videos
        for vid in sorted_vids[5:]:
            if len(result) >= top_k:
                break
            if video_groups[vid]:
                _add_cand(video_groups[vid][0])

        # Backfill remaining slots
        for vid in sorted_vids:
            for c in video_groups[vid]:
                if len(result) >= top_k:
                    break
                _add_cand(c)

    else:
        # Exploratory mode: Top 5 distinct videos
        for vid in sorted_vids[:5]:
            if video_groups[vid]:
                _add_cand(video_groups[vid][0])

        # Ranks 6-20: Secondary frames of top 5 videos + next distinct videos
        for vid in sorted_vids[:5]:
            for c in video_groups[vid][1:3]:
                if len(result) >= 20:
                    break
                _add_cand(c)

        for vid in sorted_vids[5:]:
            if len(result) >= top_k:
                break
            _add_cand(video_groups[vid][0])

        # Backfill remaining slots
        for vid in sorted_vids:
            for c in video_groups[vid]:
                if len(result) >= top_k:
                    break
                _add_cand(c)

    return result[:top_k]


def diversify_candidates(
    candidates: List[CandidateFrame],
    top_k: int = 100,
    distinct_top_n: int = 5
) -> List[CandidateFrame]:
    """
    Guarantees that the Top-N positions contain up to distinct_top_n UNIQUE video_ids.
    Positions beyond Top-N are filled by remaining candidates sorted by fusion_score.
    """
    sorted_c = sorted(candidates, key=_get_score, reverse=True)
    if not sorted_c:
        return []

    result: List[CandidateFrame] = []
    used_video_ids = set()

    # 1. Rank 1: Top candidate overall
    top1 = sorted_c[0]
    result.append(top1)
    used_video_ids.add(top1.video_id)

    remaining = sorted_c[1:]

    # 2. Ranks 2 to distinct_top_n: Select candidates from distinct videos
    distinct_pool: List[CandidateFrame] = []
    for c in remaining:
        if c.video_id not in used_video_ids:
            distinct_pool.append(c)
            used_video_ids.add(c.video_id)
        if len(result) + len(distinct_pool) >= distinct_top_n:
            break

    result.extend(distinct_pool)

    # 3. Backfill up to distinct_top_n if fewer unique videos were available
    placed_ids = {id(c) for c in result}
    for c in remaining:
        if len(result) >= distinct_top_n:
            break
        if id(c) not in placed_ids:
            result.append(c)
            placed_ids.add(id(c))

    # 4. Fill ranks (distinct_top_n + 1) to top_k with remaining candidates by score
    placed_ids = {id(c) for c in result}
    for c in remaining:
        if len(result) >= top_k:
            break
        if id(c) not in placed_ids:
            result.append(c)
            placed_ids.add(id(c))

    return result

def rank_5budget(
    candidates: List[CandidateFrame],
    strategy: str = "diversify"
) -> List[SubmissionItem]:
    """
    Role C 5-Budget Ranking Strategy:
    - Rank 1: Top candidate overall (safest).
    - Ranks 2-5: Best candidates from up to 4 DISTINCT video_ids (if strategy == 'diversify').
    - Ranks 6-100: Remaining candidates sorted by score descending.
    
    Maps candidate.frame_idx -> SubmissionItem.frame_id cleanly.
    """
    if strategy == "diversify":
        result = diversify_candidates(candidates, top_k=100, distinct_top_n=5)
    else:
        sorted_c = sorted(candidates, key=_get_score, reverse=True)
        result = sorted_c[:100]

    submission_items = [
        SubmissionItem(
            rank=idx + 1,
            video_id=c.video_id,
            frame_id=c.frame_idx,  # Explicit mapping from CandidateFrame.frame_idx -> SubmissionItem.frame_id
            confidence_score=_get_score(c)
        )
        for idx, c in enumerate(result[:100])
    ]

    return submission_items

def rank_candidates_5budget(
    query_id: str,
    candidates: List[CandidateFrame],
    query_type: QueryType = QueryType.KIS,
    answer: str = None
) -> SubmissionOutput:
    """Wrapper function returning SubmissionOutput for backward compatibility."""
    items = rank_5budget(candidates, strategy="diversify")
    if answer:
        for item in items:
            item.answer = answer

    return SubmissionOutput(
        query_id=query_id,
        query_type=query_type,
        items=items
    )

