"""
Hybrid Searcher â€” Phase 4: Multi-target SigLIP + Conditional w_ocr
====================================================================
Orchestrates multi-modal retrieval from a single QueryIR input.

Phase 4 changes vs Phase 3:
  1. Multi-target SigLIP (Exp C):
     Instead of encoding the concatenated visual_targets string into a single vector,
     each visual_target is encoded separately and frame scores are averaged
     using weighted mean: score(frame) = Î£ sim(frame, t_i) * w_i / Î£ w_i
     This avoids embedding dilution for multi-concept queries.
     Activated when len(ir.visual_targets) > 1.
     Falls back to single-vector (Exp A/B behavior) for single-concept queries.

  2. Separate w_ocr from w_metadata (Exp D routing support):
     w_metadata = TITLE + DESCRIPTION weight (video-level, penalized in TextRetriever)
     w_ocr      = OCR + ASR weight (frame-level, temporally localized)
     This split allows Conditional Routing (ScoringRouter) to adjust w_ocr
     based on query_class without touching w_metadata.

  3. Full logging (no silent failures):
     Every score contribution is logged at DEBUG level.
     Candidate pool size, multi-target vectors count, FTS5 hits â€” all visible.

Data Flow:
  QueryIR
    â”œâ”€â”€ visual_targets  â”€â”€â–º Multi-target SigLIP (FAISS)   â†’ {frame_id: siglip_score}
    â”œâ”€â”€ text_targets    â”€â”€â–º TextRetriever (FTS5)          â†’ {frame_id: text_score}
    â””â”€â”€ object_targets  â”€â”€â–º (w_object locked=0, skipped)

  Union Pool (frame_ids from both modalities)
    â”‚
    â–¼
  Single bulk DB hit â†’ frame metadata (video_id, pts_time, fps)
    â”‚
    â–¼
  Score Assembly:
    visual_score  = multi-target weighted SigLIP score
    ocr_score     = text_retriever score for OCR/ASR-sourced frames
    meta_score    = text_retriever score for TITLE/DESC-sourced frames
    object_score  = 0.0 (locked)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    final_score   = w_visual Ã— visual + w_ocr Ã— ocr + w_metadata Ã— meta
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  Returns List[RankedCandidate] sorted by final_score DESC

I/O Budget (unchanged from Phase 3):
  - FAISS: 0 disk I/O (mmap or in-memory index), N encodes for N visual_targets
  - FTS5: 1 disk hit per text term
  - Frame metadata: 1 bulk SELECT IN (?)
  Total: 3-4 disk hits regardless of pool size.
"""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.common.schemas import QueryIR
from src.retrieval.vector_index import FAISSIndex
from src.retrieval.text_retriever import TextRetriever
from src.database.db_manager import DatabaseManager
from src.database.legacy_detection_store import LegacyDetectionStore
from src.common.schemas import (
    CandidateFrame, VisualEvidence, MetadataEvidence,
    TemporalEvidence, CandidateScore, RankedCandidate, ScoringPlan
)
from src.common.config import SAFE_CANDIDATE_BASELINE

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Multi-modal retrieval engine (Phase 4).
    Input:  QueryIR + ScoringPlan
    Output: List[RankedCandidate]
    """

    def __init__(
        self,
        db: DatabaseManager,
        faiss_index: FAISSIndex,
        detection_store: Optional[LegacyDetectionStore] = None,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
    ):
        self.db = db
        self.faiss_index = faiss_index
        self.detection_store = detection_store
        self.text_retriever = TextRetriever(db)
        self._encoder = None
        self._model_name = siglip_model_name

    # â”€â”€ Encoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _load_encoder(self):
        """Lazy-load SigLIP encoder (once per process)."""
        if self._encoder is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModel

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        logger.info(f"[HybridSearcher] Loading SigLIP: {self._model_name} on {device}")
        processor = AutoProcessor.from_pretrained(self._model_name, local_files_only=True)
        model = AutoModel.from_pretrained(self._model_name, local_files_only=True).to(device).eval()
        self._encoder = (processor, model, device)

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized 768-dim float32 vector."""
        self._load_encoder()
        processor, model, device = self._encoder
        import torch
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        ).to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            feat = (
                out.pooler_output
                if hasattr(out, "pooler_output") and out.pooler_output is not None
                else out
            )
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            return feat.cpu().numpy().astype(np.float32)

    # â”€â”€ Multi-target SigLIP (Exp C) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _multi_target_visual_search(
        self, ir: QueryIR, top_k: int
    ) -> List[Dict[int, float]]:
        """
        Encode visual_targets. 
        For KIS: Combine into a single string to leverage SigLIP's compositional reasoning.
        For TRAKE: Multi-target MAX aggregation to preserve independent sequence events.
        """
        if ir.query_type != "TRAKE" or not ir.temporal_sequence:
            # SINGLE VECTOR PATH (Compositional Reasoning for KIS)
            # This is critical so SigLIP understands "woman wearing pink" rather than 
            # finding "woman" and "pink" in different parts of the image and averaging.
            concept = ir.get_siglip_query()
            if not concept:
                concept = ir.raw_text[:64]
                logger.debug(f"[HybridSearcher] No visual_targets → single fallback: '{concept}'")
            else:
                logger.debug(f"[HybridSearcher] KIS Single Compositional Query: '{concept}'")
                
            vec = self._encode_text(concept)
            scores, _, frame_ids = self.faiss_index.search_vectors(vec, top_k=top_k)
            return [{int(fid): float(sc) for sc, fid in zip(scores, frame_ids) if fid > 0}]

        # ---------------------------------------------------------
        # TRAKE PATH: Sequence Alignment
        # ---------------------------------------------------------
        multi_targets = [(step, 1.0) for step in ir.temporal_sequence]
        logger.info(f"[HybridSearcher] Sequential TRAKE SigLIP: {len(multi_targets)} events")
        
        step_maps = []
        for concept, weight in multi_targets:
            logger.debug(f"  concept='{concept}'")
            vec = self._encode_text(concept)
            scores, _, frame_ids = self.faiss_index.search_vectors(vec, top_k=top_k)
            step_map = {}
            for sc, fid in zip(scores, frame_ids):
                if fid > 0:
                    step_map[int(fid)] = float(sc)
            step_maps.append(step_map)
            
        return step_maps
    def search(
        self,
        ir: QueryIR,
        scoring_plan: Optional[ScoringPlan] = None,
        top_k_final: int = 100,
        enable_lazy_ocr: bool = True,
    ) -> List[RankedCandidate]:
        if scoring_plan is None:
            scoring_plan = SAFE_CANDIDATE_BASELINE

        # ── Step 1: Multi-target Visual Retrieval (SigLIP) ─────────────────────
        visual_maps = self._multi_target_visual_search(ir, top_k=scoring_plan.visual_top_k)
        
        # Merge maps to get all visual candidates
        visual_frame_ids = set()
        for vmap in visual_maps:
            visual_frame_ids.update(vmap.keys())

        # ── Step 2: Text Retrieval via FTS5 (Chạy SAU Lazy OCR) ────────
        text_frame_scores: Dict[int, float] = {}
        if ir.text_targets and (scoring_plan.w_metadata > 0 or scoring_plan.w_ocr > 0):
            text_frame_scores = self.text_retriever.retrieve(ir.text_targets, top_k_fts=200)
            logger.info(f"[HybridSearcher] Step2 text_frame_scores: {len(text_frame_scores):,} frames")
        else:
            logger.debug("[HybridSearcher] Step2 skipped (no text_targets or weights=0)")

        # ── Step 3: Union Pool ─────────────────────────────────────────
        # Include frames from FTS5 that FAISS missed (text-only candidates).
        text_only_fids = [fid for fid in text_frame_scores if fid not in visual_frame_ids]
        union_frame_ids = list(visual_frame_ids) + text_only_fids
        logger.info(
            f"[HybridSearcher] Step3 union pool: {len(union_frame_ids):,} frames "
            f"(visual={len(visual_frame_ids):,} + text_only={len(text_only_fids):,})"
        )

        # â”€â”€ Step 4: Bulk Metadata Fetch (1 disk hit) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        frames_dict = self.db.get_frames_bulk(union_frame_ids)
        logger.debug(f"[HybridSearcher] Step4 resolved {len(frames_dict):,} frame metadata rows")

        # ── Step 5: Object Evidence ─────────────────────────
        detected_map: Dict[str, dict] = {}
        if self.detection_store and (scoring_plan.w_object > 0 or getattr(ir, 'object_constraints', [])):
            logger.debug("[HybridSearcher] Fetching bbox signals for scoring/constraints")
            pairs = [
                (frames_dict[fid]["video_id"], frames_dict[fid]["frame_idx"])
                for fid in union_frame_ids if fid in frames_dict
            ]
            detected_map = self.detection_store.get_coarse_batch_objects(
                pairs, min_score=scoring_plan.object_min_score
            )

        # â”€â”€ Step 6: Score Assembly & Ranking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ranked = []

        for frame_id in union_frame_ids:
            finfo = frames_dict.get(frame_id)
            if not finfo:
                logger.debug(f"[HybridSearcher] frame_id={frame_id} not in frames_dict, skipping")
                continue

            vid = finfo["video_id"]
            # Base score for single vectors, handled properly in Step 7 DP
            siglip_score = visual_maps[0].get(frame_id, 0.0) if visual_maps else 0.0
            text_score = text_frame_scores.get(frame_id, 0.0)

            # Object score (locked=0 in Phase 4)
            obj_score = 0.0
            if ir.object_targets and self.detection_store and scoring_plan.w_object > 0:
                key = f"{vid}:{finfo['frame_idx']}"
                detected = detected_map.get(key, {})
                matched = sum(
                    1 for tgt in ir.object_targets
                    # Handle new dict format in detected_map if we modified legacy_detection_store
                    if any(tgt.lower() == lbl.lower() for lbl in detected.keys())
                )
                obj_score = min(matched / max(len(ir.object_targets), 1), 1.0)

            # Phase 4: text_score from TextRetriever already blends OCR + METADATA.
            # Prevent double-counting by using the maximum of the two weights.
            effective_text_weight = max(scoring_plan.w_metadata, scoring_plan.w_ocr)
            final_score = (
                scoring_plan.w_visual * siglip_score
                + effective_text_weight * text_score
                + scoring_plan.w_object * obj_score
            )

            # Build FAISS id for provenance
            faiss_id_val = frame_id

            candidate = CandidateFrame(
                faiss_id=faiss_id_val,
                video_id=vid,
                frame_idx=finfo["frame_idx"],
                pts_time=finfo["pts_time"],
                fps=finfo["fps"],
            )
            ev = VisualEvidence(siglip_score=siglip_score, obj_score=obj_score)
            meta_ev = MetadataEvidence(score=text_score)
            score = CandidateScore(
                visual=siglip_score,
                object=obj_score,
                metadata=text_score,  # total text signal
                temporal=0.0,
                final=final_score,
            )
            ranked.append(RankedCandidate(
                candidate=candidate,
                visual_evidence=ev,
                metadata_evidence=meta_ev,
                temporal_evidence=TemporalEvidence(candidate_time=finfo["pts_time"]),
                score=score,
                rank=-1,
            ))

        # ── Step 7: Event-Centric Soft Sequence DP & NMS ─────────
        video_groups: Dict[str, List[RankedCandidate]] = {}
        for r in ranked:
            video_groups.setdefault(r.candidate.video_id, []).append(r)
            
        event_candidates = []
        WINDOW_SIZE = 15.0 # 15 seconds window
        n_steps = len(visual_maps)
        
        from src.common.schemas import TemporalWindow
        
        for vid, frames in video_groups.items():
            frames.sort(key=lambda x: x.candidate.pts_time)
            
            # DP for Longest Increasing Subsequence of actions
            for r in frames:
                window_frames = [
                    f for f in frames 
                    if abs(f.candidate.pts_time - r.candidate.pts_time) <= (WINDOW_SIZE / 2.0)
                ]
                
                dp = [0.0] * n_steps
                dp_start = [r.candidate.pts_time] * n_steps
                dp_end = [r.candidate.pts_time] * n_steps
                
                for wf in window_frames:
                    fid = wf.candidate.faiss_id
                    new_dp = list(dp)
                    new_start = list(dp_start)
                    new_end = list(dp_end)
                    
                    for k in range(n_steps):
                        sc = visual_maps[k].get(fid, 0.0)
                        if sc > 0:
                            # Start fresh at k
                            if sc > new_dp[k]:
                                new_dp[k] = sc
                                new_start[k] = wf.candidate.pts_time
                                new_end[k] = wf.candidate.pts_time
                            
                            # Extend from j < k
                            for j in range(k):
                                if dp[j] > 0 and dp[j] + sc > new_dp[k]:
                                    new_dp[k] = dp[j] + sc
                                    new_start[k] = dp_start[j]
                                    new_end[k] = wf.candidate.pts_time
                    dp = new_dp
                    dp_start = new_start
                    dp_end = new_end
                
                max_dp = max(dp) if dp else 0.0
                best_k = dp.index(max_dp) if max_dp > 0 else 0
                
                # Soft assignment: if it skips steps, it divides by total steps
                seq_visual = max_dp / max(1, n_steps)
                
                max_text = max((f.score.metadata for f in window_frames), default=0.0)
                max_obj = max((f.score.object for f in window_frames), default=0.0)
                
                event_score = (
                    scoring_plan.w_visual * seq_visual + 
                    effective_text_weight * max_text + 
                    scoring_plan.w_object * max_obj
                )
                
                # [EXPERIMENTAL] Tier-2 Counting Gate
                multiplier = 1.0
                if getattr(ir, 'object_constraints', []) and detected_map:
                    for constraint in ir.object_constraints:
                        req_class = constraint.class_name.lower()
                        req_count = constraint.min_count
                        
                        # Find max count for this class in the window
                        max_found = 0
                        for wf in window_frames:
                            key = f"{vid}:{wf.candidate.frame_idx}"
                            det = detected_map.get(key, {})
                            # det is {class_entity: {'max_score': float, 'count': int}}
                            for lbl, val in det.items():
                                if lbl.lower() == req_class:
                                    if isinstance(val, dict):
                                        max_found = max(max_found, val.get('count', 0))
                                    else:
                                        max_found = max(max_found, 1) # Fallback
                        
                        if max_found >= req_count:
                            multiplier *= 1.2 # Bonus for hitting count
                        elif max_found > 0 and max_found < max(1, req_count // 2):
                            multiplier *= 0.5 # Penalty for severe miss
                        elif max_found == 0:
                            multiplier *= 0.2 # Extreme penalty if totally absent
                        # Else (e.g. occluded, slightly less), do nothing (1.0)
                        
                event_score *= multiplier
                
                r.score.final = max(r.score.final, event_score)
                r.score.visual = seq_visual
                if max_dp > 0:
                    r.temporal_evidence.window = TemporalWindow(start_time=dp_start[best_k], end_time=dp_end[best_k])
                    
            # 1D Non-Maximum Suppression (NMS)
            frames.sort(key=lambda x: x.score.final, reverse=True)
            active_events = []
            for r in frames:
                is_redundant = any(
                    abs(r.candidate.pts_time - ev.candidate.pts_time) <= WINDOW_SIZE
                    for ev in active_events
                )
                if not is_redundant:
                    active_events.append(r)
            event_candidates.extend(active_events)

        # Sort events by final score descending before VLM
        event_candidates.sort(key=lambda r: r.score.final, reverse=True)
        top_candidates = event_candidates[:top_k_final]

        # ── Step 8: VLM Reranker (Top 10 Candidates) ─────────
        # Uses Gemini 1.5 Flash Lite to rerank the top candidates visually, bypassing the "blind" FAISS scores.
        # Since images are not local (remotezip), VLMAuditor fetches them on-the-fly.
        if top_candidates:
            try:
                from src.retrieval.vlm_auditor import VLMAuditor
                auditor = VLMAuditor(self.db)
                # We rerank Top 10 to ensure we capture the best without breaking the bank
                top_candidates = auditor.rerank_top_k(ir.raw_text, top_candidates, top_k=10)
            except Exception as e:
                logger.error(f"[HybridSearcher] VLM Reranker failed, skipping: {e}")

        for i, r in enumerate(top_candidates):
            r.rank = i + 1

        logger.info(
            f"[HybridSearcher] Done | candidates returned={len(top_candidates)} "
            f"(pool={len(ranked):,})"
        )
        return top_candidates







# =========================================================================
# SCORING ROUTER (Merged from scoring_router.py)
# =========================================================================

_ROUTING_TABLE: dict = {
    "Q_ENTITY":    {"w_ocr": 0.4, "w_object": 0.0},
    "Q_METADATA":  {"w_ocr": 0.4, "w_object": 0.0},
    "Q_VISUAL":    {"w_ocr": 0.0, "w_object": 0.0},
    "Q_COMPOSITE": {"w_ocr": 0.2, "w_object": 0.0},  # baseline
}

# Fallback for unknown query_class values
_DEFAULT_ROUTE = {"w_ocr": 0.2, "w_object": 0.0}


def route_scoring_plan(query_class: str, base_plan: ScoringPlan) -> ScoringPlan:
    """
    Return a new ScoringPlan with w_ocr and w_object adjusted by query_class.
    All other weights (w_visual, w_metadata, ...) are inherited unchanged.

    This function is the ONLY place where conditional routing logic lives.
    It must remain frozen once benchmark starts.

    Args:
        query_class: From QueryIR.query_class — one of Q_VISUAL/Q_ENTITY/Q_METADATA/Q_COMPOSITE
        base_plan:   The Safe Candidate Baseline plan to route from

    Returns:
        A new ScoringPlan with routed weights (base_plan is not mutated).
    """
    route = _ROUTING_TABLE.get(query_class, _DEFAULT_ROUTE)

    # Build routed plan — copy all fields, override only w_ocr and w_object
    routed = base_plan.model_copy(update={
        "w_ocr": route["w_ocr"],
        "w_object": route.get("w_object", 0.1)
    })

    logger.debug(
        f"[ScoringRouter] query_class={query_class} → "
        f"w_ocr: {base_plan.w_ocr:.2f} → {routed.w_ocr:.2f} | "
        f"w_visual={routed.w_visual:.2f} w_metadata={routed.w_metadata:.2f} "
        f"w_object={routed.w_object:.2f}"
    )

    if query_class not in _ROUTING_TABLE:
        logger.warning(
            f"[ScoringRouter] Unknown query_class='{query_class}' → using default route "
            f"(w_ocr={_DEFAULT_ROUTE['w_ocr']})"
        )

    return routed
