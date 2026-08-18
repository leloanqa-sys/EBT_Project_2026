import time
from typing import List, Dict, Optional
from src.pipeline import MVPPipeline

class TRAKEPipeline:
    def __init__(self, detect_threshold: float = 0.3, searcher=None):
        # We reuse MVPPipeline as the base retriever for individual sub-events.
        # searcher: inject singleton VectorSearcher from API layer to avoid reloading FAISS on every request.
        self.kis_pipeline = MVPPipeline(detect_threshold=detect_threshold, top_k_retrieve=300, searcher=searcher)

    def run(self, query_id: str, main_query: str, sub_events: List[str], top_k: int = 5, max_time_gap_seconds: float = 30.0) -> Dict:
        """
        Runs the TRAKE Pipeline (Temporal Retrieval and Alignment of Key Events).
        Uses Dynamic Programming to find the best sequence of frames.
        """
        start_time = time.time()
        
        if not sub_events:
            return {"sequences": [], "latency_ms": 0}
            
        N = len(sub_events)
        
        # 1. Retrieve candidates for EACH sub-event
        # event_candidates[i] is a list of CandidateFrame for sub_event[i]
        event_candidates = []
        for i, text in enumerate(sub_events):
            full_context_text = f"{main_query}. Phân cảnh: {text}"
            res = self.kis_pipeline.run(f"{query_id}_e{i}", full_context_text, query_type="TRAKE_PART")
            # Limit to top 300 per event to increase intersection chances
            event_candidates.append(res.candidates[:300])
            
        # 2. Group by Video ID
        # Only consider videos that appear in ALL sub-events' candidate lists (or at least many of them)
        # For strict TRAKE, it must appear in all.
        video_sets = []
        for cands in event_candidates:
            vset = set(c.video_id for c in cands)
            video_sets.append(vset)
            
        # Intersection of videos
        common_videos = set.intersection(*video_sets) if video_sets else set()
        
        if not common_videos:
            return {"sequences": [], "latency_ms": (time.time() - start_time) * 1000}
            
        # 3. Dynamic Programming for Alignment
        # For each video, find the optimal strictly increasing sequence of timestamps
        best_sequences = []
        
        for vid in common_videos:
            # Extract frames for this video per event, sorted by pts_time
            V_cands = []
            for cands in event_candidates:
                frames = [c for c in cands if c.video_id == vid]
                frames.sort(key=lambda x: x.pts_time)
                V_cands.append(frames)
                
            # DP State: dp[i][j] = max score for prefix of length i ending at frame j of event i
            # where V_cands[i][j]
            # Since V_cands[i] can have multiple frames, we track max score and backpointer
            dp = [[] for _ in range(N)]
            backpointers = [[] for _ in range(N)]
            
            # Helper to get fusion_score safely (fallback to clip_score if not computed)
            def _get_fscore(cf):
                return getattr(cf, 'fusion_score', cf.clip_score)
                
            # Init dp for event 0
            for j, f0 in enumerate(V_cands[0]):
                dp[0].append(_get_fscore(f0))
                backpointers[0].append(-1)
                
            # Fill DP
            valid_sequence_exists = True
            for i in range(1, N):
                for j, fi in enumerate(V_cands[i]):
                    max_prev_score = -1.0
                    best_prev_idx = -1
                    
                    fi_score = _get_fscore(fi)
                    for k, fk in enumerate(V_cands[i-1]):
                        # Constraint: Time must be increasing, allowing same-frame events
                        time_diff = fi.pts_time - fk.pts_time
                        if 0 <= time_diff <= max_time_gap_seconds:
                            score = dp[i-1][k] + fi_score
                            # Optional: apply penalty for large time gaps
                            penalty = 0.05 * (time_diff / max_time_gap_seconds)
                            score -= penalty
                            
                            if score > max_prev_score:
                                max_prev_score = score
                                best_prev_idx = k
                                
                    dp[i].append(max_prev_score)
                    backpointers[i].append(best_prev_idx)
                    
                if all(s == -1.0 for s in dp[i]):
                    valid_sequence_exists = False
                    break
                    
            if valid_sequence_exists:
                # Find best end frame
                best_end_idx = -1
                max_total_score = -1.0
                for j, score in enumerate(dp[N-1]):
                    if score > max_total_score:
                        max_total_score = score
                        best_end_idx = j
                        
                if best_end_idx != -1:
                    # Backtrack to build the sequence
                    seq = []
                    curr_idx = best_end_idx
                    for i in range(N-1, -1, -1):
                        seq.append(V_cands[i][curr_idx])
                        curr_idx = backpointers[i][curr_idx]
                    seq.reverse()
                    
                    avg_score = max_total_score / N
                    best_sequences.append({
                        "video_id": vid,
                        "avg_score": avg_score,
                        "frames": seq
                    })
                    
        # 4. Sort and return Top-K sequences
        best_sequences.sort(key=lambda x: x["avg_score"], reverse=True)
        return {
            "sequences": best_sequences[:top_k],
            "latency_ms": (time.time() - start_time) * 1000
        }
