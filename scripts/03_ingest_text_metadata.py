"""
Script 03: Ingest Title/Description Text Segments (Quick Win - no OCR/ASR needed)
===================================================================================
Ingests video title and description from media-info JSON files into text_segments
and rebuilds the FTS5 index. This is the fastest way to unlock text search:
  - No external models needed.
  - Enables FTS5 to match queries containing video titles, channel names, etc.
  - Provides immediate signal for Gemini-derived text_targets.

Run: venv/Scripts/python.exe scripts/03_ingest_text_metadata.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager
from src.data.text_extraction.ocr_processor import ingest_title_description

def main():
    print("=" * 70)
    print("  PHASE 2 - SCRIPT 03: INGEST TITLE/DESCRIPTION TEXT SEGMENTS")
    print("=" * 70)

    db = DatabaseManager()

    # Check if text_segments already exists (schema may need rebuild)
    try:
        count = db.execute_single("SELECT COUNT(*) as c FROM text_segments")["c"]
        if count > 0:
            print(f"  [Skip] text_segments already has {count:,} rows. Delete DB to re-ingest.")
            return
    except Exception:
        print("  [Init] Re-initializing DB schema (adding text_segments + FTS5)...")
        db._init_pragmas_and_schema()

    # Read all video records from DB
    video_rows = db.execute_query("SELECT video_id, title, description FROM videos")
    print(f"  Found {len(video_rows)} videos to process.")

    videos = [dict(r) for r in video_rows]

    # Supplement with local media-info JSON (has richer description)
    media_info_dir = PROJECT_ROOT / "data" / "raw" / "media-info"
    if media_info_dir.exists():
        for row in videos:
            jf = media_info_dir / f"{row['video_id']}.json"
            if jf.exists():
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    row["title"] = meta.get("title", row.get("title", ""))
                    row["description"] = meta.get("description", row.get("description", ""))
                except Exception:
                    pass

    segments = ingest_title_description(videos)
    print(f"  Generated {len(segments)} text segments (TITLE + DESCRIPTION).")

    inserted = db.insert_text_segments(segments)
    print(f"  Inserted {inserted:,} segments and rebuilt FTS5 index.")

    # Verify
    seg_count = db.execute_single("SELECT COUNT(*) as c FROM text_segments")["c"]
    print(f"\n  Verification:")
    print(f"    text_segments rows: {seg_count:,}")
    
    # Test FTS5
    test_result = db.search_fts("university research robot", top_k=3)
    print(f"    FTS5 test query 'university research robot' → {len(test_result)} hits")
    for r in test_result:
        print(f"      video_id={r['video_id']} | source={r['source']} | bm25={r['bm25_score']:.3f}")
        print(f"      content: {r['content'][:80]}...")

    print("\nDone.")

if __name__ == "__main__":
    main()
