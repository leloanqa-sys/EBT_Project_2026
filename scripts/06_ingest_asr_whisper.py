"""
Script 06: Offline ASR Ingestion (YouTube Subtitles / yt-dlp)
=============================================================
Since videos are not stored locally (using remotezip), running Whisper 
is impossible without downloading gigabytes of MP4 files. 
Instead, this script uses `yt-dlp` to download the auto-generated 
YouTube subtitles (.vtt) from the `watch_url` found in media-info.

Requires: pip install yt-dlp webvtt-py
"""
import os
import sys
import json
import argparse
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager

def parse_vtt(file_path):
    import webvtt
    segments = []
    try:
        for caption in webvtt.read(file_path):
            # webvtt times are like "00:00:10.000"
            def to_sec(t_str):
                parts = t_str.split(':')
                return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            segments.append({
                "start": to_sec(caption.start),
                "end": to_sec(caption.end),
                "text": caption.text.replace('\n', ' ').strip()
            })
    except Exception:
        pass
    return segments

def main():
    try:
        import yt_dlp
        import webvtt
    except ImportError:
        print("ERROR: Missing libraries. Please run: pip install yt-dlp webvtt-py")
        sys.exit(1)

    import urllib.parse
    import time
    import random

    db = DatabaseManager()
    media_info_dir = PROJECT_ROOT / "data" / "raw" / "media-info"
    sub_dir = PROJECT_ROOT / "data" / "raw" / "subtitles"
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    videos = db.execute_query("SELECT video_id FROM videos")
    segments_to_insert = []
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['vi'],
        'outtmpl': str(sub_dir / '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,  # Bỏ qua lỗi 429/Private video, đi tiếp
        'sleep_interval': 3,   # Ngủ ngẫu nhiên 3-8 giây giữa các video
        'max_sleep_interval': 8,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for row in videos:
            vid = row["video_id"]
            json_file = media_info_dir / f"{vid}.json"
            if not json_file.exists():
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                
            url = info.get("watch_url")
            if not url:
                continue
                
            # 1. Bóc tách ID Video trực tiếp không cần gọi API (Chống tốn Request)
            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)
            yt_id = qs.get("v", [None])[0]
            if not yt_id:
                yt_id = url.split('/')[-1].split('?')[0] # Fallback cho link youtu.be
                
            vtt_file = sub_dir / f"{yt_id}.vi.vtt"
            
            # 2. Cơ chế Resume: Bỏ qua nếu đã tải rồi
            if vtt_file.exists():
                subs = parse_vtt(str(vtt_file))
                for sub in subs:
                    segments_to_insert.append({
                        "video_id": vid, "source": "ASR", "content": sub["text"],
                        "start_time": sub["start"], "end_time": sub["end"], "confidence": 1.0
                    })
                # print(f"[{vid}] Đã có sẵn trong máy, BỎ QUA tải mới.")
                continue
                
            print(f"[{vid}] Đang tải Subtitle từ YouTube ...")
            try:
                # Ngủ nhẹ chống ban IP nếu phải cào mới
                time.sleep(random.uniform(1.0, 3.0))
                
                dict_meta = ydl.extract_info(url, download=True)
                # Nếu video private/chặn, dict_meta có thể là None
                if not dict_meta:
                    print(f"[{vid}] Bỏ qua vì lỗi chặn/private.")
                    continue
                    
                if vtt_file.exists():
                    subs = parse_vtt(str(vtt_file))
                    for sub in subs:
                        segments_to_insert.append({
                            "video_id": vid, "source": "ASR", "content": sub["text"],
                            "start_time": sub["start"], "end_time": sub["end"], "confidence": 1.0
                        })
                    print(f"[{vid}] Tải thành công {len(subs)} dòng thoại.")
            except Exception as e:
                err_msg = str(e).lower()
                print(f"[{vid}] Lỗi: {str(e)[:80]}")
                # Nếu dính block 429 hoặc IP ban, ngủ đông một lúc rồi chạy tiếp
                if "429" in err_msg or "too many requests" in err_msg or "bot" in err_msg:
                    print(f"⚠️ Bị YouTube chặn! Ngủ đông 60 giây trước khi thử tiếp...")
                    time.sleep(60)

    if segments_to_insert:
        print(f"\nChèn {len(segments_to_insert)} dòng ASR vào Database...")
        db.insert_text_segments(segments_to_insert)
        print("Xong! FTS5 index đã tự động được rebuild.")
    else:
        print("\nKhông tìm thấy hoặc không có Subtitle nào được tải.")

if __name__ == "__main__":
    main()
