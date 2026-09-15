"""
Evaluation Metrics for Video Retrieval & Temporal Localization
==============================================================
Standardized metrics for AIC 2026:
- R@1, R@5, R@10, R@20, R@50, R@100, R@500
- Video Recall@K (correct video in top K)
- Temporal Recall@K (correct video + inside [s, e) in top K)
- Temporal Distance d(t, [s, e))
"""

from typing import List, Dict, Any, Tuple
from src.common.schemas import CandidateFrame


def is_inside_interval(t: float, start: float, end: float, semantics: str = "[s,e)") -> bool:
    """
    Standard ground truth temporal interval verification.
    Default semantics is half-open interval [s, e).
    """
    if semantics == "[s,e)":
        return start <= t < end
    elif semantics == "[s,e]":
        return start <= t <= end
    elif semantics == "(s,e)":
        return start < t < end
    else:
        return start <= t <= end


def temporal_distance(t: float, start: float, end: float) -> float:
    """
    Computes distance from time t to interval [s, e).
    d(t, [s, e)) =
        s - t  if t < s
        0.0    if s <= t < e
        t - e  if t >= e
    """
    if t < start:
        return start - t
    elif t < end:
        return 0.0
    else:
        return t - end


def evaluate_candidate_ranking(
    ranked_candidates: List[CandidateFrame],
    ground_truth_list: List[Dict[str, Any]],
    cutoffs: List[int] = [1, 5, 10, 20, 50, 100, 500]
) -> Dict[str, Any]:
    """
    Evaluates a ranked candidate list against a list of ground truth items.
    Returns:
        video_recall: Dict[str, bool] (e.g. "R@1": True, "R@5": True...)
        temporal_recall: Dict[str, bool]
        min_temporal_distance: float
        best_candidate_rank: Optional[int]
    """
    if not ground_truth_list or not ranked_candidates:
        return {
            "video_recall": {f"R@{k}": 0.0 for k in cutoffs},
            "temporal_recall": {f"R@{k}": 0.0 for k in cutoffs},
            "min_temporal_distance": float("inf"),
            "best_video_rank": None,
            "best_temporal_rank": None
        }

    # Find earliest match for video and temporal
    best_video_rank = None
    best_temporal_rank = None
    min_dist = float("inf")

    for rank_idx, cand in enumerate(ranked_candidates):
        rank = rank_idx + 1
        cand_vid = cand.video_id
        cand_pts = cand.pts_time

        for gt in ground_truth_list:
            gt_vid = gt["video_id"]
            s = float(gt["start_time"])
            e = float(gt["end_time"])
            sem = gt.get("interval_semantics", "[s,e)")

            if cand_vid == gt_vid:
                if best_video_rank is None:
                    best_video_rank = rank

                dist = temporal_distance(cand_pts, s, e)
                if dist < min_dist:
                    min_dist = dist

                if is_inside_interval(cand_pts, s, e, sem):
                    if best_temporal_rank is None:
                        best_temporal_rank = rank

    video_recall = {}
    temporal_recall = {}

    for k in cutoffs:
        video_recall[f"R@{k}"] = 1.0 if (best_video_rank is not None and best_video_rank <= k) else 0.0
        temporal_recall[f"R@{k}"] = 1.0 if (best_temporal_rank is not None and best_temporal_rank <= k) else 0.0

    return {
        "video_recall": video_recall,
        "temporal_recall": temporal_recall,
        "min_temporal_distance": min_dist if min_dist != float("inf") else -1.0,
        "best_video_rank": best_video_rank,
        "best_temporal_rank": best_temporal_rank
    }
