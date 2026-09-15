"""
Database Schema for AIC 2026 (aic2026.db)
=========================================
Source of Truth for Videos, Frames, Categories, Ground Truth, and Provenance Logging.
"""

SCHEMA_SQL = """
-- =========================================================================
-- GROUP 1: DATASET TAXONOMY & ASSETS
-- =========================================================================

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    filename TEXT,
    duration REAL DEFAULT 0.0,
    fps REAL DEFAULT 0.0,
    width INTEGER DEFAULT 0,
    height INTEGER DEFAULT 0,
    category TEXT,
    subcategory TEXT,
    title TEXT,
    description TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    frame_idx INTEGER NOT NULL,
    pts_time REAL NOT NULL,
    fps REAL NOT NULL DEFAULT 30.0,
    keyframe_n INTEGER,
    UNIQUE(video_id, frame_idx)
);

CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id);
CREATE INDEX IF NOT EXISTS idx_frames_video_idx ON frames(video_id, frame_idx);
CREATE INDEX IF NOT EXISTS idx_frames_pts ON frames(video_id, pts_time);

-- =========================================================================
-- GROUP 2: CATEGORIES & METADATA
-- =========================================================================

CREATE TABLE IF NOT EXISTS categories (
    category_id TEXT PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS video_categories (
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    category_id TEXT NOT NULL REFERENCES categories(category_id),
    PRIMARY KEY(video_id, category_id)
);

-- =========================================================================
-- GROUP 3: QUERIES, IR & GROUND TRUTH
-- =========================================================================

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    query_type TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    question_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_ir (
    query_id TEXT PRIMARY KEY REFERENCES queries(query_id),
    ir_version TEXT NOT NULL DEFAULT '2.0',
    ir_json TEXT NOT NULL,
    parser_model TEXT,
    parser_version TEXT,
    parser_confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ground_truth (
    gt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL REFERENCES queries(query_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    interval_semantics TEXT NOT NULL DEFAULT '[s,e)',
    frame_start INTEGER,
    frame_end INTEGER,
    answer_text TEXT,
    source TEXT DEFAULT 'official'
);

CREATE INDEX IF NOT EXISTS idx_gt_query ON ground_truth(query_id);
CREATE INDEX IF NOT EXISTS idx_gt_video ON ground_truth(video_id);

-- =========================================================================
-- GROUP 4: MODEL REGISTRY & BENCHMARK RUNS
-- =========================================================================

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL, -- visual, object, vlm, nlp
    dimension INTEGER,
    description TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    run_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(query_id),
    pipeline_version TEXT NOT NULL,
    scoring_plan_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES retrieval_runs(run_id),
    query_id TEXT NOT NULL REFERENCES queries(query_id),
    frame_id INTEGER NOT NULL REFERENCES frames(frame_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    visual_rank INTEGER NOT NULL,
    visual_score REAL NOT NULL,
    final_rank INTEGER NOT NULL,
    final_score REAL NOT NULL,
    UNIQUE(run_id, query_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_run ON retrieval_results(run_id, query_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_final_rank ON retrieval_results(run_id, final_rank);

CREATE TABLE IF NOT EXISTS candidate_evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES retrieval_runs(run_id),
    query_id TEXT NOT NULL REFERENCES queries(query_id),
    frame_id INTEGER NOT NULL REFERENCES frames(frame_id),
    siglip_score REAL DEFAULT 0.0,
    object_score REAL DEFAULT 0.0,
    spatial_score REAL DEFAULT 0.0,
    metadata_score REAL DEFAULT 0.0,
    temporal_score REAL DEFAULT 0.0,
    fusion_score REAL DEFAULT 0.0,
    vqa_answer TEXT,
    UNIQUE(run_id, query_id, frame_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_run ON candidate_evidence(run_id, query_id);

-- =========================================================================
-- GROUP 5: TEXT SEGMENTS (OCR / ASR / TITLE / DESCRIPTION)
-- =========================================================================

CREATE TABLE IF NOT EXISTS text_segments (
    segment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id     TEXT    NOT NULL REFERENCES videos(video_id),
    source       TEXT    NOT NULL,   -- 'OCR' | 'ASR' | 'TITLE' | 'DESCRIPTION'
    content      TEXT    NOT NULL,
    start_time   REAL    DEFAULT NULL,
    end_time     REAL    DEFAULT NULL,
    confidence   REAL    DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_text_seg_video ON text_segments(video_id);
CREATE INDEX IF NOT EXISTS idx_text_seg_source ON text_segments(source);

-- FTS5 virtual table for probabilistic (BM25) text search
-- content= links to text_segments.content for tokenization
CREATE VIRTUAL TABLE IF NOT EXISTS fts_text_index
USING fts5(
    content,
    video_id UNINDEXED,
    segment_id UNINDEXED,
    source UNINDEXED,
    start_time UNINDEXED,
    end_time UNINDEXED,
    content="text_segments",
    content_rowid="segment_id",
    tokenize="porter unicode61"
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# MIGRATION STATEMENTS (run after schema init; safe to re-run — errors ignored)
# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: frame_id on text_segments enables direct frame↔OCR association
# without the runtime lookup overhead of the temporal window scan.
# Nullable → existing rows without frame_id remain valid.
MIGRATION_STMTS = [
    "ALTER TABLE text_segments ADD COLUMN frame_id INTEGER REFERENCES frames(frame_id)",
    "CREATE INDEX IF NOT EXISTS idx_text_seg_frame ON text_segments(frame_id)",
    # ocr_score stored separately from metadata_score in evidence table
    "ALTER TABLE candidate_evidence ADD COLUMN ocr_score REAL DEFAULT 0.0",
]
