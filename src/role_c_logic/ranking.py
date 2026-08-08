from typing import List
from src.common.schemas import CandidateFrame, SubmissionItem, SubmissionOutput, QueryType

def rank_candidates_5budget(
    query_id: str,
    candidates: List[CandidateFrame],
    query_type: QueryType = QueryType.KIS,
    answer: str = None
) -> SubmissionOutput:
    """
    Role C 5-Budget Ranking Strategy:
    Sử dụng cand.fusion_score để sắp xếp và tối ưu R-Score.
    """
    items: List[SubmissionItem] = []
    
    # Sort candidates by fusion_score descending
    sorted_candidates = sorted(candidates, key=lambda c: c.fusion_score, reverse=True)
    
    seen_videos = set()
    placed_items = []
    
    # Rank 1: Safest top candidate
    if sorted_candidates:
        top_cand = sorted_candidates[0]
        placed_items.append(top_cand)
        seen_videos.add(top_cand.video_id)
        
    # Ranks 2-5: Diversify video hypotheses
    for cand in sorted_candidates[1:]:
        if len(placed_items) >= 5:
            break
        if cand.video_id not in seen_videos:
            placed_items.append(cand)
            seen_videos.add(cand.video_id)
            
    # Fill remaining slots up to 100
    for cand in sorted_candidates:
        if len(placed_items) >= 100:
            break
        if cand not in placed_items:
            placed_items.append(cand)

    for rank_idx, cand in enumerate(placed_items[:100], start=1):
        items.append(SubmissionItem(
            rank=rank_idx,
            video_id=cand.video_id,
            frame_id=cand.frame_id,
            answer=answer,
            confidence_score=cand.fusion_score
        ))

    return SubmissionOutput(
        query_id=query_id,
        query_type=query_type,
        items=items
    )
