from typing import List
from dataclasses import replace
from itertools import groupby

from src.common.schemas import CandidateFrame, SubmissionItem, SubmissionOutput, QueryType

def compute_fusion_scores(candidates: List[CandidateFrame], w_clip: float = 1.0, w_obj: float = 0.5, w_spatial: float = 0.5):
    """
    Applies Soft Scoring to compute the final fusion_score for each candidate.
    Weights can be tuned via ML Tuner.
    """
    for c in candidates:
        c.fusion_score = (w_clip * c.clip_score) + (w_obj * c.obj_score) + (w_spatial * c.spatial_score)

def _get_score(candidate: CandidateFrame) -> float:
    """Helper to safely retrieve score (fusion_score or clip_score)."""
    return getattr(candidate, 'fusion_score', candidate.clip_score)

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
    sorted_c = sorted(candidates, key=_get_score, reverse=True)
    if not sorted_c:
        return []

    result: List[CandidateFrame] = []
    used_video_ids = set()

    # Rank 1: Top candidate
    top1 = sorted_c[0]
    result.append(top1)
    used_video_ids.add(top1.video_id)

    remaining = sorted_c[1:]

    if strategy == "diversify":
        # Ranks 2-5: Select candidates from 4 unique video_ids
        distinct_video_pool: List[CandidateFrame] = []
        for c in remaining:
            if c.video_id not in used_video_ids:
                distinct_video_pool.append(c)
                used_video_ids.add(c.video_id)
            if len(distinct_video_pool) >= 4:
                break

        result.extend(distinct_video_pool)

        # If distinct video pool has fewer than 4 candidates, backfill from remaining
        placed_set = set(id(c) for c in result)
        for c in remaining:
            if len(result) >= 5:
                break
            if id(c) not in placed_set:
                result.append(c)
                placed_set.add(id(c))

        # Ranks 6-100: Fill rest of slots
        placed_set = set(id(c) for c in result)
        for c in remaining:
            if len(result) >= 100:
                break
            if id(c) not in placed_set:
                result.append(c)
                placed_set.add(id(c))
    else:
        result.extend(remaining[:99])

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

