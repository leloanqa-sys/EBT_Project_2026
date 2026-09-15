"""
Legacy Detection Store Adapter
==============================
Provides decoupled access to object detection evidence from legacy metadata.db (2.2GB).
Supports two-tier retrieval:
- Tier 1: Coarse object verification (frame_n, class_entity, max_score, count)
- Tier 2: Fine bounding box extraction (ymin, xmin, ymax, xmax) for spatial reasoning.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional

DEFAULT_METADATA_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "metadata.db"
)


class LegacyDetectionStore:
    """Read-only adapter for 20.2M detections in legacy metadata.db."""

    def __init__(self, db_path: str = DEFAULT_METADATA_DB_PATH):
        self.db_path = os.path.abspath(db_path)
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Legacy metadata.db not found at {self.db_path}")

    def _get_ro_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def get_coarse_frame_objects(self, video_id: str, frame_n: int, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Tier 1: Coarse object query.
        Returns unique object classes present in the frame with their max confidence score.
        """
        query = """
            SELECT class_entity, MAX(score) as max_score, COUNT(*) as object_count
            FROM detections
            WHERE video_id = ? AND frame_n = ? AND score >= ?
            GROUP BY class_entity
            ORDER BY max_score DESC
        """
        conn = self._get_ro_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, (video_id, frame_n, min_score))
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_fine_bounding_boxes(self, video_id: str, frame_n: int, class_entity: Optional[str] = None, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Tier 2: Fine bbox query for spatial reasoning.
        """
        if class_entity:
            query = """
                SELECT class_entity, score, ymin, xmin, ymax, xmax
                FROM detections
                WHERE video_id = ? AND frame_n = ? AND class_entity = ? AND score >= ?
                ORDER BY score DESC
            """
            params = (video_id, frame_n, class_entity, min_score)
        else:
            query = """
                SELECT class_entity, score, ymin, xmin, ymax, xmax
                FROM detections
                WHERE video_id = ? AND frame_n = ? AND score >= ?
                ORDER BY score DESC
            """
            params = (video_id, frame_n, min_score)

        conn = self._get_ro_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_coarse_batch_objects(self, candidate_pairs: List[tuple], min_score: float = 0.3) -> Dict[str, Dict[str, float]]:
        """
        Batch query for multiple (video_id, frame_idx) pairs.
        Returns: { "video_id:frame_idx": { "class_entity": max_score, ... } }
        """
        if not candidate_pairs:
            return {}

        conn = self._get_ro_connection()
        results = {}
        try:
            cur = conn.cursor()
            # Query in chunks of 100 to avoid SQLite variable limits
            chunk_size = 100
            for i in range(0, len(candidate_pairs), chunk_size):
                chunk = candidate_pairs[i:i + chunk_size]
                conditions = " OR ".join("(video_id = ? AND frame_n = ?)" for _ in chunk)
                flat_params = []
                for vid, fn in chunk:
                    flat_params.extend([vid, fn])
                flat_params.append(min_score)

                query = f"""
                    SELECT video_id, frame_n, class_entity, MAX(score) as max_score, COUNT(*) as object_count
                    FROM detections
                    WHERE ({conditions}) AND score >= ?
                    GROUP BY video_id, frame_n, class_entity
                """
                cur.execute(query, tuple(flat_params))
                for r in cur.fetchall():
                    key = f"{r['video_id']}:{r['frame_n']}"
                    if key not in results:
                        results[key] = {}
                    results[key][r['class_entity']] = {'max_score': r['max_score'], 'count': r['object_count']}
            return results
        finally:
            conn.close()
