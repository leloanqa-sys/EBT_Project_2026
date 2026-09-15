"""
Text Retriever — FTS5 BM25 Probabilistic Search with Temporal Locality
=======================================================================
Searches text_segments (OCR, ASR, TITLE, DESCRIPTION) using SQLite FTS5.

ARCHITECTURE FIX (v2):
  Previous: FTS5 → {video_id: score} → all frames of that video get text_score
  Problem:  Video with "Lausanne" at t=1s and unrelated content at t=30s
            → ALL frames of that video received the score (false positive)

  Fixed:    FTS5 → segment with [start_time, end_time] → only frames whose
            pts_time falls within [start_time - Δ, end_time + Δ] get the score.

  Sources without timestamp (TITLE/DESCRIPTION) → video-level with PENALTY_FACTOR.
  Sources with timestamp (OCR, ASR) → frame-level with temporal window check.

Design:
  - Input: List[TextTarget] from QueryIR
  - Output: {frame_id: text_score} dict (frame-level, not video-level)
  - Scoring: bm25_raw × term_weight × source_confidence, sigmoid normalized
  - Temporal window: Δt = 5 seconds around [start_time, end_time]
"""

import math
from typing import List, Dict, Optional

from src.common.schemas import TextTarget
from src.database.db_manager import DatabaseManager

# Sources WITH reliable timestamps → frame-level scoring
TEMPORAL_SOURCES = {"OCR", "ASR"}
# Sources WITHOUT timestamps → video-level (penalized)
METADATA_SOURCES = {"TITLE", "DESCRIPTION"}

SOURCE_CONFIDENCE = {
    "ASR": 0.8,   # Temporal alignment ✓, STT errors possible
    "OCR": 0.9,   # High fidelity if text visible on screen
    "TITLE": 1.0, # Authoritative, but no temporal locality → penalized
    "DESCRIPTION": 0.7,  # Authoritative, no temporal locality → penalized
}

# Penalty for sources without timestamps (applied to video-level scores)
VIDEO_LEVEL_PENALTY = 0.4  # TITLE/DESCRIPTION score × 0.4

# Temporal expansion window (seconds) around [start_time, end_time]
TEMPORAL_DELTA_SEC = 5.0


class TextRetriever:
    """FTS5 BM25 probabilistic text retrieval with temporal locality."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._frame_cache: Optional[Dict] = None  # {(video_id, frame_idx): (frame_id, pts_time)}

    def _get_frame_lookup(self) -> Dict:
        """Lazy-load frame lookup table: {video_id: [(frame_id, frame_idx, pts_time)]}"""
        if self._frame_cache is None:
            rows = self.db.execute_query(
                "SELECT frame_id, video_id, frame_idx, pts_time FROM frames ORDER BY video_id, pts_time"
            )
            cache = {}
            for r in rows:
                vid = r["video_id"]
                if vid not in cache:
                    cache[vid] = []
                cache[vid].append((r["frame_id"], r["frame_idx"], r["pts_time"]))
            self._frame_cache = cache
        return self._frame_cache

    def _frames_in_window(self, video_id: str, start_time: Optional[float], end_time: Optional[float]) -> List[int]:
        """Return frame_ids whose pts_time falls within [start-Δ, end+Δ]."""
        lookup = self._get_frame_lookup()
        frames = lookup.get(video_id, [])
        if not frames:
            return []

        if start_time is None or end_time is None:
            # No temporal info → return all frames of this video (video-level)
            return [f[0] for f in frames]

        lo = start_time - TEMPORAL_DELTA_SEC
        hi = end_time + TEMPORAL_DELTA_SEC
        return [fid for fid, fidx, pts in frames if lo <= pts <= hi]

    def retrieve(
        self,
        text_targets: List[TextTarget],
        top_k_fts: int = 200,
    ) -> Dict[int, float]:
        """
        Search FTS5 for each text_target term.
        Returns: {frame_id: combined_text_score}

        Frame-level scoring:
          - OCR/ASR segments: score goes only to frames within [start-Δ, end+Δ]
          - TITLE/DESCRIPTION: score × VIDEO_LEVEL_PENALTY goes to ALL frames of video

        Aggregation per frame_id: max(scores across all term hits)
        Final: sigmoid normalize across all scored frames.
        """
        active_terms = [t for t in text_targets if t.weight >= 0.2]
        if not active_terms:
            return {}

        # {frame_id: accumulated_score}
        frame_scores: Dict[int, float] = {}

        for target in active_terms:
            results = self.db.search_fts(target.term, top_k=top_k_fts)
            if not results:
                continue

            for row in results:
                video_id = row["video_id"]
                source = row["source"]
                bm25_raw = abs(float(row["bm25_score"]))
                src_conf = SOURCE_CONFIDENCE.get(source, 0.6)
                seg_conf = float(row.get("confidence", 1.0))
                start_t = row.get("start_time")
                end_t = row.get("end_time")

                base_score = bm25_raw * target.weight * src_conf * seg_conf

                if source in TEMPORAL_SOURCES and start_t is not None:
                    # Frame-level: only frames in temporal window
                    candidate_fids = self._frames_in_window(video_id, start_t, end_t)
                    for fid in candidate_fids:
                        frame_scores[fid] = frame_scores.get(fid, 0.0) + base_score
                else:
                    # Video-level (penalized): all frames of this video
                    all_fids = self._frames_in_window(video_id, None, None)
                    penalized = base_score * VIDEO_LEVEL_PENALTY
                    for fid in all_fids:
                        frame_scores[fid] = frame_scores.get(fid, 0.0) + penalized

        if not frame_scores:
            return {}

        # Sigmoid normalization
        max_score = max(frame_scores.values())
        if max_score > 0:
            frame_scores = {
                fid: 1.0 / (1.0 + math.exp(-(score / max_score) * 6 - 3))
                for fid, score in frame_scores.items()
            }

        return frame_scores
