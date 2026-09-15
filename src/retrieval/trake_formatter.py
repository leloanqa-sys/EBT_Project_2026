"""
TRAKE Formatter — Event Sequence Result Builder
================================================
Converts HybridSearcher RankedCandidates (with DP temporal windows)
into AIC TRAKE submission format.

AIC TRAKE CSV format:
    video_id, e1_frame_idx, e2_frame_idx, ..., eN_frame_idx

Each eK_frame_idx is the frame best representing event K within
the temporal window found by the DP alignment in HybridSearcher Step 7.

Pipeline:
    RankedCandidates (sorted by final_score)
        → Group by video_id
        → For each video: assign best frame per event from temporal window
        → Score: mean of per-event siglip scores
        → Sort videos by overall score
        → Return List[TRAKESequence]

Optional QA on TRAKE:
    If question is provided, run QAOCREngine on the top-1 video's best frames
    to generate a text answer alongside the frame indices.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from src.common.schemas import RankedCandidate, TemporalWindow
from src.common.schemas import QueryIR

logger = logging.getLogger(__name__)


@dataclass
class TRAKEEventFrame:
    """Best frame for a single event step."""
    event_idx: int          # 0-based index into temporal_sequence
    event_description: str  # The event string (English from NLP)
    video_id: str
    frame_idx: int
    pts_time: float
    siglip_score: float
    frame_url: str = ""


@dataclass
class TRAKESequence:
    """
    Full sequence result for one video.
    Matches AIC format: video_id + one frame_idx per event.
    """
    rank: int
    video_id: str
    event_frames: List[TRAKEEventFrame] = field(default_factory=list)
    avg_score: float = 0.0
    qa_answer: Optional[str] = None   # Text answer if question provided

    @property
    def frame_indices(self) -> List[int]:
        """Ordered list of frame_idx, one per event."""
        return [ef.frame_idx for ef in self.event_frames]

    def to_csv_row(self) -> str:
        """AIC TRAKE CSV row: video_id,e1_frame,e2_frame,..."""
        frames_str = ",".join(str(f) for f in self.frame_indices)
        return f"{self.video_id},{frames_str}"

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "avg_score": round(self.avg_score, 4),
            "frames": [
                {
                    "event_idx": ef.event_idx,
                    "event_description": ef.event_description,
                    "frame_idx": ef.frame_idx,
                    "pts_time": round(ef.pts_time, 2),
                    "siglip_score": round(ef.siglip_score, 4),
                    "frame_url": ef.frame_url,
                }
                for ef in self.event_frames
            ],
            "qa_answer": self.qa_answer,
        }


def _assign_frames_to_events(
    video_frames: List[RankedCandidate],
    n_events: int,
    event_descriptions: List[str],
) -> List[Optional[TRAKEEventFrame]]:
    """
    Given all candidates from one video (sorted by pts_time),
    assign the best frame to each event step using greedy forward scan.

    Strategy:
    - Events must be in chronological order (E1 before E2 before ...)
    - For each event K, find the frame with highest siglip_score for that event
      that occurs AFTER the frame chosen for event K-1 (temporal ordering constraint)
    - Uses score.visual per event index embedded in the DP result

    Since HybridSearcher already ran multi-step DP (visual_maps per step),
    we reconstruct event assignment from the temporal window data.
    """
    if not video_frames or n_events == 0:
        return []

    # Sort by pts_time for temporal ordering
    frames_by_time = sorted(video_frames, key=lambda r: r.candidate.pts_time)

    result: List[Optional[TRAKEEventFrame]] = [None] * n_events
    last_pts = -1.0

    for ev_idx in range(n_events):
        best_score = -1.0
        best_rc = None

        for rc in frames_by_time:
            pts = rc.candidate.pts_time
            if pts <= last_pts:
                continue  # Must be strictly after previous event

            # Use final score as proxy (incorporates DP alignment)
            sc = rc.score.final
            if sc > best_score:
                best_score = sc
                best_rc = rc

        if best_rc is not None:
            result[ev_idx] = TRAKEEventFrame(
                event_idx=ev_idx,
                event_description=event_descriptions[ev_idx] if ev_idx < len(event_descriptions) else f"Event {ev_idx+1}",
                video_id=best_rc.candidate.video_id,
                frame_idx=best_rc.candidate.frame_idx,
                pts_time=best_rc.candidate.pts_time,
                siglip_score=best_rc.score.visual,
                frame_url=f"/api/v1/image/{best_rc.candidate.video_id}/{best_rc.candidate.frame_idx}",
            )
            last_pts = best_rc.candidate.pts_time
            logger.debug(
                f"  [TRAKE] Event {ev_idx} → {best_rc.candidate.video_id}/"
                f"{best_rc.candidate.frame_idx} @ {pts:.1f}s (score={sc:.4f})"
            )
        else:
            # Could not find a chronologically ordered frame — use earliest remaining
            remaining = [r for r in frames_by_time if r.candidate.pts_time > last_pts]
            if remaining:
                rc = remaining[0]
                result[ev_idx] = TRAKEEventFrame(
                    event_idx=ev_idx,
                    event_description=event_descriptions[ev_idx] if ev_idx < len(event_descriptions) else f"Event {ev_idx+1}",
                    video_id=rc.candidate.video_id,
                    frame_idx=rc.candidate.frame_idx,
                    pts_time=rc.candidate.pts_time,
                    siglip_score=rc.score.visual,
                    frame_url=f"/api/v1/image/{rc.candidate.video_id}/{rc.candidate.frame_idx}",
                )
                last_pts = rc.candidate.pts_time

    return result


def format_trake_results(
    ranked_candidates: List[RankedCandidate],
    ir: QueryIR,
    top_k_videos: int = 5,
    question: Optional[str] = None,
) -> List[TRAKESequence]:
    """
    Main entry point: convert HybridSearcher output to TRAKE sequences.

    Args:
        ranked_candidates: All ranked candidates from HybridSearcher
        ir: Parsed QueryIR (contains temporal_sequence with event descriptions)
        top_k_videos: Max number of distinct videos to return
        question: Optional text question for QA-on-TRAKE (generates qa_answer)

    Returns:
        List of TRAKESequence sorted by avg_score DESC
    """
    events = ir.temporal_sequence or []
    n_events = len(events)

    if n_events == 0:
        logger.warning("[TRAKEFormatter] No temporal_sequence in IR — treating as single event")
        events = [ir.dense_caption_en or ir.raw_text]
        n_events = 1

    # ── Group candidates by video_id ──────────────────────────────────────────
    video_groups: Dict[str, List[RankedCandidate]] = {}
    for rc in ranked_candidates:
        vid = rc.candidate.video_id
        video_groups.setdefault(vid, []).append(rc)

    logger.info(f"[TRAKEFormatter] {len(video_groups)} distinct videos, {n_events} events")

    # ── Score each video by best sequential coverage ──────────────────────────
    video_scores: List[Tuple[str, float, List[Optional[TRAKEEventFrame]]]] = []

    for vid, frames in video_groups.items():
        event_frames = _assign_frames_to_events(frames, n_events, events)

        # Score = mean of assigned frame scores (penalize missing events)
        valid_frames = [ef for ef in event_frames if ef is not None]
        if not valid_frames:
            continue

        scores = [ef.siglip_score for ef in valid_frames]
        coverage = len(valid_frames) / max(1, n_events)  # 0-1: how many events covered
        avg_sc = (sum(scores) / len(scores)) * coverage   # Penalize incomplete coverage

        video_scores.append((vid, avg_sc, event_frames))

    # Sort by score descending
    video_scores.sort(key=lambda x: x[1], reverse=True)

    # ── Build TRAKESequence objects ───────────────────────────────────────────
    sequences: List[TRAKESequence] = []

    for rank_idx, (vid, avg_sc, event_frames) in enumerate(video_scores[:top_k_videos]):
        valid_frames = [ef for ef in event_frames if ef is not None]

        seq = TRAKESequence(
            rank=rank_idx + 1,
            video_id=vid,
            event_frames=valid_frames,
            avg_score=avg_sc,
        )

        # ── Optional: QA answer for top-1 video ──────────────────────────────
        if question and rank_idx == 0 and valid_frames:
            try:
                from src.retrieval.qa_ocr_engine import answer_question_ocr_first

                # Use the event frames as candidates for OCR (best quality frames)
                # Re-create a minimal "candidates" list from event frames
                qa_candidates = [
                    rc for rc in ranked_candidates
                    if rc.candidate.video_id == vid
                ]
                # Sort by final score desc (highest quality first → OCR priority)
                qa_candidates.sort(key=lambda r: r.score.final, reverse=True)

                answer = answer_question_ocr_first(
                    question=question,
                    ranked_candidates=qa_candidates,
                    top_n_ocr=min(8, len(qa_candidates)),
                )
                seq.qa_answer = answer
                logger.info(f"[TRAKEFormatter] QA answer for {vid}: '{answer}'")
            except Exception as e:
                logger.error(f"[TRAKEFormatter] QA answering failed: {e}")
                seq.qa_answer = None

        sequences.append(seq)

    logger.info(f"[TRAKEFormatter] Returning {len(sequences)} TRAKE sequences")
    return sequences
