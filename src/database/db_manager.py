"""
Database Manager for SQLite (aic2026.db)
========================================
Handles SQLite connection lifecycle, PRAGMA optimizations (WAL, busy_timeout),
transactions, and batch operations. No connection pooling (single local DB).
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from src.database.schema import SCHEMA_SQL, MIGRATION_STMTS

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "database", "aic2026.db"
)


class DatabaseManager:
    """Singleton/Instance SQLite Manager with WAL & transaction support."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_pragmas_and_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_pragmas_and_schema(self):
        with self.transaction() as conn:
            conn.executescript(SCHEMA_SQL)
        # Run backward-safe migrations (idempotent — errors silently ignored)
        for stmt in MIGRATION_STMTS:
            try:
                with self.transaction() as conn:
                    conn.execute(stmt)
            except Exception as e:
                err_msg = str(e).lower()
                if "duplicate column name" in err_msg or "already exists" in err_msg:
                    pass  # Column/index already exists → OK
                else:
                    import logging
                    logging.warning(f"Migration statement failed: {stmt}. Error: {e}")

    @contextmanager
    def transaction(self):
        """Context manager for atomic transaction block with auto-commit/rollback."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Read-only query execution."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
        finally:
            conn.close()

    def execute_single(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """Read-only query returning single row."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            return cur.fetchone()
        finally:
            conn.close()

    def bulk_insert(self, query: str, data: List[Tuple]) -> int:
        """Batch insert using executemany inside a transaction."""
        if not data:
            return 0
        with self.transaction() as conn:
            cur = conn.cursor()
            cur.executemany(query, data)
            return cur.rowcount

    # -------------------------------------------------------------------------
    # High-level Entity Helpers
    # -------------------------------------------------------------------------

    def get_frame_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        row = self.execute_single(
            "SELECT * FROM frames WHERE frame_id = ?", (frame_id,)
        )
        return dict(row) if row else None

    def get_frames_bulk(self, frame_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch multiple frames by ID in a single disk hit."""
        if not frame_ids:
            return {}
        placeholders = ",".join("?" for _ in frame_ids)
        rows = self.execute_query(
            f"SELECT * FROM frames WHERE frame_id IN ({placeholders})",
            tuple(frame_ids)
        )
        return {r["frame_id"]: dict(r) for r in rows}

    def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        row = self.execute_single(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        )
        return dict(row) if row else None

    def get_ground_truth(self, query_id: str) -> List[Dict[str, Any]]:
        rows = self.execute_query(
            "SELECT * FROM ground_truth WHERE query_id = ?", (query_id,)
        )
        return [dict(r) for r in rows]

    def search_fts(self, query_terms: str, top_k: int = 200) -> List[Dict[str, Any]]:
        """
        Probabilistic BM25 search via SQLite FTS5.
        Returns top_k matches sorted by BM25 score (lower = better in SQLite BM25).
        NOTE: SQLite FTS5 BM25 is negative-scored (more negative = better match).
        query_terms: space-separated keywords. FTS5 uses prefix matching & stemming (porter).
        """
        if not query_terms or not query_terms.strip():
            return []
        
        sql = """
            SELECT
                ts.video_id,
                ts.segment_id,
                ts.source,
                ts.content,
                ts.start_time,
                ts.end_time,
                ts.confidence,
                bm25(fts_text_index) AS bm25_score
            FROM fts_text_index
            JOIN text_segments ts ON fts_text_index.rowid = ts.segment_id
            WHERE fts_text_index MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
        """
        try:
            rows = self.execute_query(sql, (query_terms, top_k))
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"  [FTS5] Search error (query='{query_terms}'): {e}")
            return []

    def insert_text_segments(self, segments: List[Dict[str, Any]]) -> int:
        """
        Bulk insert text segments and rebuild FTS5 index incrementally.
        Each segment: {video_id, source, content, start_time, end_time, confidence}
        """
        if not segments:
            return 0
        
        rows = [(
            s["video_id"], s["source"], s["content"],
            s.get("start_time"), s.get("end_time"), s.get("confidence", 1.0)
        ) for s in segments]
        
        insert_sql = """
            INSERT INTO text_segments (video_id, source, content, start_time, end_time, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with self.transaction() as conn:
            cur = conn.cursor()
            cur.executemany(insert_sql, rows)
            conn.commit()

            # Rebuild FTS5 index from content table
            cur.execute("INSERT INTO fts_text_index(fts_text_index) VALUES('rebuild')")
        return len(rows)
