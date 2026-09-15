"""
Retrieval Run Tracker & Provenance Logger
=========================================
Logs each query execution run, candidate ranks (visual vs final), and evidence
breakdowns into SQLite (aic2026.db) for quantitative MLOps ablation analysis.
"""

import json
import uuid
from typing import List, Optional
from src.database.db_manager import DatabaseManager
from src.common.schemas import RankedCandidate, ScoringPlan


class RunTracker:
    """Provenance tracker logging retrieval runs & candidate evidence."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_run(self, query_id: str, pipeline_version: str = "v2.0", scoring_plan: Optional[ScoringPlan] = None) -> str:
        """Initialize a new retrieval run record."""
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        plan_json = scoring_plan.model_dump_json() if scoring_plan else "{}"
        
        insert_sql = """
            INSERT INTO retrieval_runs (run_id, query_id, pipeline_version, scoring_plan_json)
            VALUES (?, ?, ?, ?)
        """
        with self.db.transaction() as conn:
            conn.execute(insert_sql, (run_id, query_id, pipeline_version, plan_json))
        return run_id

    def log_results_and_evidence(self, run_id: str, query_id: str, ranked_candidates: List[RankedCandidate]):
        """Bulk logs final ranks and component evidence breakdown."""
        if not ranked_candidates:
            return

        result_rows = []
        evidence_rows = []

        # Get frame_id to canonical ID mapping
        lookup_rows = self.db.execute_query("SELECT frame_id, video_id, frame_idx FROM frames")
        vf_map = {(r["video_id"], r["frame_idx"]): r["frame_id"] for r in lookup_rows}

        for idx, rc in enumerate(ranked_candidates):
            final_rank = idx + 1
            cand = rc.candidate
            frame_id = vf_map.get((cand.video_id, cand.frame_idx), -1)
            if frame_id == -1:
                continue

            visual_rank = rc.rank if rc.rank > 0 else final_rank
            visual_score = rc.visual_evidence.siglip_score
            final_score = rc.score.final

            result_rows.append((
                run_id, query_id, frame_id, cand.video_id,
                visual_rank, visual_score, final_rank, final_score
            ))

            evidence_rows.append((
                run_id, query_id, frame_id,
                rc.visual_evidence.siglip_score,
                rc.visual_evidence.obj_score,
                rc.visual_evidence.spatial_score,
                rc.metadata_evidence.score,
                rc.temporal_evidence.score,
                rc.score.final,
                rc.vqa_answer
            ))

        # Bulk insert into SQLite
        insert_results_sql = """
            INSERT OR REPLACE INTO retrieval_results (
                run_id, query_id, frame_id, video_id,
                visual_rank, visual_score, final_rank, final_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.bulk_insert(insert_results_sql, result_rows)

        insert_evidence_sql = """
            INSERT OR REPLACE INTO candidate_evidence (
                run_id, query_id, frame_id,
                siglip_score, object_score, spatial_score,
                metadata_score, temporal_score, fusion_score, vqa_answer
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.bulk_insert(insert_evidence_sql, evidence_rows)
