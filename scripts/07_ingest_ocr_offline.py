"""
Script 07: Offline OCR Ingestion (PaddleOCR)
============================================
Scans extracted frames (e.g. 1 frame per second) and runs PaddleOCR.
Pushes extracted text into SQLite `text_segments`.

Requires: pip install paddleocr paddlepaddle
"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help="Process 1 frame every N frames")
    args = parser.parse_args()

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("ERROR: paddleocr is not installed. Please run: pip install paddleocr paddlepaddle")
        sys.exit(1)

    print("Loading PaddleOCR model...")
    ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
    
    db = DatabaseManager()
    frames = db.execute_query(f"SELECT frame_id, video_id, frame_idx, pts_time FROM frames WHERE frame_idx % {args.interval} = 0")
    
    frames_dir = PROJECT_ROOT / "data" / "raw" / "keyframes"
    
    segments_to_insert = []
    
    total = len(frames)
    print(f"Found {total} frames to process for OCR (interval={args.interval}).")
    
    for i, row in enumerate(frames):
        fid = row["frame_id"]
        vid = row["video_id"]
        idx = row["frame_idx"]
        pts = row["pts_time"]
        
        # Path format: data/raw/keyframes/L01_V001/0000.jpg
        img_path = frames_dir / vid / f"{idx:04d}.jpg"
        if not img_path.exists():
            continue
            
        if i % 100 == 0:
            print(f"Processing frame {i}/{total}...")
            
        try:
            result = ocr.ocr(str(img_path), cls=True)
            if not result or result[0] is None:
                continue
                
            frame_text = " ".join([line[1][0] for line in result[0]])
            if frame_text.strip():
                segments_to_insert.append({
                    "video_id": vid,
                    "source": "OCR",
                    "content": frame_text.strip(),
                    "start_time": pts,
                    "end_time": pts + 1.0,
                    "confidence": 1.0 # Or average confidence from PaddleOCR
                })
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            
    if segments_to_insert:
        print(f"Inserting {len(segments_to_insert)} OCR segments into database...")
        db.insert_text_segments(segments_to_insert)
        print("Done! FTS5 index automatically rebuilt.")
    else:
        print("No OCR segments generated.")

if __name__ == "__main__":
    main()
