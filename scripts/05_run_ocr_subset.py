"""
Script: OCR Pilot (Phase 3)
===========================
Runs EasyOCR on a subset of 50 videos (keyframes) to extract frame-level text.
Extracted text is deduplicated, timestamped, and ingested into `text_segments`.

Requires: pip install easyocr torch torchvision
"""

import sys
import argparse
import time
from pathlib import Path

# Try importing EasyOCR early so we fail fast if missing
try:
    import easyocr
except ImportError:
    print("[ERROR] easyocr not installed. Please run: pip install easyocr")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager
from src.data.text_extraction.ocr_processor import OCRProcessor

def run_ocr_pilot(num_videos: int = 50):
    print("=" * 70)
    print(f"  PHASE 3 - OCR PILOT ({num_videos} Videos)")
    print("=" * 70)

    db = DatabaseManager()
    
    # 1. Select subset of videos
    rows = db.execute_query("SELECT video_id FROM videos ORDER BY video_id LIMIT ?", (num_videos,))
    videos = [r["video_id"] for r in rows]
    print(f"  Selected {len(videos)} videos for OCR pilot.")

    if not videos:
        print("  [ERROR] No videos found in database.")
        sys.exit(1)

    # 2. Init OCR Processor
    print("  Initializing EasyOCR (this may download model weights)...")
    t0 = time.time()
    # Using easyocr via OCRProcessor (supports deduplication)
    processor = OCRProcessor(langs=['vi', 'en'], use_gpu=False)  # CPU default for safety, switch if GPU avl
    print(f"  Initialization done in {time.time()-t0:.1f}s")

    # 3. Process each video
    total_segments_inserted = 0
    t_start = time.time()

    for i, vid in enumerate(videos, 1):
        print(f"\n  [{i}/{num_videos}] Processing video: {vid}")
        
        # Load frame paths for this video
        frames_meta = db.execute_query("""
            SELECT frame_id, frame_idx, pts_time 
            FROM frames 
            WHERE video_id = ? 
            ORDER BY pts_time
        """, (vid,))

        if not frames_meta:
            print("    - No frames found in DB.")
            continue

        kf_dir = PROJECT_ROOT / "data" / "raw" / "keyframes" / vid
        if not kf_dir.exists():
            print(f"    - [WARN] Keyframe directory not found: {kf_dir}")
            continue
            
        print(f"    - DB contains {len(frames_meta)} frames. Running OCR...")
        
        segments = []
        # Init Remote Loader
        from scripts.remote_data_loader import RemoteDataLoader
        remote_loader = RemoteDataLoader()

        for fm in frames_meta:
            frame_idx = fm['frame_idx']
            img_path = remote_loader.fetch_keyframe(vid, frame_idx)
                
            if not img_path or not img_path.exists():
                print(f"    - [WARN] Could not fetch {vid}:{frame_idx}")
                continue
                
            # Run OCR
            try:
                results = processor.reader.readtext(str(img_path), detail=0, paragraph=True)
                text = " ".join(results).strip()
                if len(text) > 3:  # Only meaningful text
                    segments.append({
                        "video_id": vid,
                        "source": "OCR",
                        "content": text,
                        "start_time": fm["pts_time"],
                        "end_time": fm["pts_time"] + 1.0,  # approximate duration
                        "confidence": 0.8
                    })
            except Exception as e:
                print(f"    - OCR failed on {img_name}: {e}")

        # Deduplicate temporally close identical texts
        deduped = processor.deduplicate_temporal_segments(segments)
        
        # Insert into DB
        if deduped:
            db.insert_text_segments(deduped)
            total_segments_inserted += len(deduped)
            print(f"    - Extracted {len(deduped)} text segments.")
        else:
            print("    - No text found.")

    elapsed = time.time() - t_start
    
    # Generate OCR Pilot Report
    # Note: Utility metrics (meaningful / useful) require manual sampling or NLP classification.
    # For now, we compute raw extraction rates.
    report = {
        "videos_processed": num_videos,
        "total_segments_extracted": total_segments_inserted,
        "avg_segments_per_video": round(total_segments_inserted / max(num_videos, 1), 2),
        "metrics": {
            "temporal_text_utility_note": "Measure manually: (useful segments) / (total extracted) or verify with Query A/B/C tests."
        }
    }
    
    report_path = PROJECT_ROOT / "outputs" / "ocr_pilot_report.json"
    report_path.parent.mkdir(exist_ok=True)
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"  OCR PILOT COMPLETE")
    print(f"  Processed {num_videos} videos in {elapsed:.1f}s")
    print(f"  Inserted {total_segments_inserted} OCR segments into text_segments.")
    print(f"  Report saved to: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of videos to process")
    args = parser.parse_args()
    
    run_ocr_pilot(args.n)
