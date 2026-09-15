"""
Script 04: End-to-End Baseline Evaluation Harness
=================================================
Runs a complete baseline retrieval test over the new Data Layer:
1. Ingests test Query & Ground Truth into aic2026.db
2. Retrieves Top 500 candidates via SigLIP (FAISS + in-memory mapping + SQLite)
3. Evaluates Video Recall@K, Temporal Recall@K, Temporal Distance
4. Demonstrates Object Evidence extraction via LegacyDetectionStore adapter
5. Logs full Provenance to retrieval_runs & candidate_evidence tables.
"""

import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager
from src.database.legacy_detection_store import LegacyDetectionStore
from src.retrieval.visual_retriever import VisualRetriever
from src.evaluation.metrics import evaluate_candidate_ranking
from src.evaluation.run_tracker import RunTracker
from src.common.schemas import (
    CandidateFrame, VisualEvidence, MetadataEvidence,
    TemporalEvidence, CandidateScore, RankedCandidate, ScoringPlan
)


def main():
    print("=" * 70)
    print("  PHASE 1 - SCRIPT 04: END-TO-END BASELINE EVALUATION")
    print("=" * 70)

    db = DatabaseManager()
    detection_store = LegacyDetectionStore()
    retriever = VisualRetriever(db=db)
    tracker = RunTracker(db=db)

    # 1. Register a test query & ground truth into aic2026.db
    query_id = "test_q01_gas_station"
    query_text = "Có thể thấy trong cảnh quay có 4 tài xế xe ôm công nghệ trong trạm xăng, trong đó 3 người đứng đợi còn 1 người lái xe từ trái sang phải khung hình. Trước đó là cảnh một người đậy nắp bình xăng xe máy của họ. Có thông tin về giá dầu mazut được hiển thị trong khung hình."
    query_en = "There are 4 technology motorbike taxi drivers at the gas station, 3 of them are waiting and 1 is driving from left to right. Before that, someone was closing the gas cap of their motorbike. There is information about the price of mazut oil displayed."

    # Insert test query
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO queries (query_id, batch_id, query_type, raw_text)
            VALUES (?, ?, ?, ?)
            """,
            (query_id, "batch1", "KIS", query_text)
        )
        # Insert known GT for this clip: L22_V001 in temporal range [700.0, 720.0)
        conn.execute(
            """
            INSERT OR REPLACE INTO ground_truth (gt_id, query_id, video_id, start_time, end_time, interval_semantics)
            VALUES (1, ?, 'L22_V001', 700.0, 720.0, '[s,e)')
            """,
            (query_id,)
        )

    print(f"\n1. Registered Query: {query_id}")
    print(f"   Text (VI): {query_text[:65]}...")
    print(f"   Text (EN): {query_en[:65]}...")
    print(f"   Ground Truth: L22_V001 @ [700.0s, 720.0s)")

    # 2. Retrieve Top 500 via Visual Retriever (SigLIP)
    print("\n2. Executing Visual Retrieval (SigLIP2 Top-500)...")
    results = retriever.retrieve(query_en, top_k=500)
    print(f"   Retrieved {len(results)} candidate frames.")

    # 3. Create Ranked Candidates
    ranked_candidates = []
    for rank, (cand, visual_ev) in enumerate(results, start=1):
        score = CandidateScore(
            visual=visual_ev.siglip_score,
            object=0.0,
            spatial=0.0,
            metadata=0.0,
            temporal=0.0,
            final=visual_ev.siglip_score
        )
        rc = RankedCandidate(
            candidate=cand,
            visual_evidence=visual_ev,
            metadata_evidence=MetadataEvidence(),
            temporal_evidence=TemporalEvidence(candidate_time=cand.pts_time),
            score=score,
            rank=rank
        )
        ranked_candidates.append(rc)

    # 4. Evaluate against Ground Truth
    gt_list = db.get_ground_truth(query_id)
    eval_metrics = evaluate_candidate_ranking(
        [rc.candidate for rc in ranked_candidates],
        gt_list,
        cutoffs=[1, 5, 10, 20, 50, 100, 500]
    )

    print("\n" + "=" * 70)
    print("  EVALUATION METRICS REPORT (SigLIP Baseline)")
    print("=" * 70)
    print(f"  Best Video Match Rank:    #{eval_metrics['best_video_rank']}")
    print(f"  Best Temporal Match Rank: #{eval_metrics['best_temporal_rank']}")
    print(f"  Min Temporal Distance:    {eval_metrics['min_temporal_distance']:.2f}s")
    print("\n  Recall Cutoffs:")
    for k in [1, 5, 10, 20, 50, 100, 500]:
        v_rec = eval_metrics['video_recall'][f'R@{k}']
        t_rec = eval_metrics['temporal_recall'][f'R@{k}']
        print(f"    - R@{k:3d}  | Video Recall: {v_rec * 100:5.1f}% | Temporal Recall: {t_rec * 100:5.1f}%")

    # 5. Object Detection Evidence Extraction Test (Top 3 Candidates)
    print("\n" + "=" * 70)
    print("  OBJECT EVIDENCE ADAPTER (LegacyDetectionStore Test)")
    print("=" * 70)
    pairs = [(rc.candidate.video_id, rc.candidate.frame_idx) for rc in ranked_candidates[:3]]
    detected_map = detection_store.get_coarse_batch_objects(pairs, min_score=0.3)

    for i, rc in enumerate(ranked_candidates[:3], start=1):
        cand = rc.candidate
        key = f"{cand.video_id}:{cand.frame_idx}"
        objs = detected_map.get(key, {})
        top_3_objs = sorted(objs.items(), key=lambda x: x[1], reverse=True)[:3]
        obj_str = ", ".join([f"{k}: {v:.2f}" for k, v in top_3_objs]) if top_3_objs else "None"
        print(f"  Rank #{i:02d} | {cand.video_id:8s} | Frame: {cand.frame_idx:5d} | PTS: {cand.pts_time:6.1f}s | SigLIP: {rc.score.visual:.4f} | Detected: {obj_str}")

    # 6. Log Provenance to SQLite
    scoring_plan = ScoringPlan(w_visual=1.0, visual_top_k=500, candidate_top_k=100)
    run_id = tracker.create_run(query_id, pipeline_version="v2.0_baseline", scoring_plan=scoring_plan)
    tracker.log_results_and_evidence(run_id, query_id, ranked_candidates)
    print(f"\n[RunTracker] Logged full provenance for run_id: {run_id} ({len(ranked_candidates)} candidates) to SQLite.")


if __name__ == "__main__":
    main()
