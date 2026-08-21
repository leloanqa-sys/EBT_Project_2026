import json
import os
import sys
import glob
import zipfile
import csv
import sqlite3
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.role_a_retrieval.searcher import VectorSearcher

def generate_exact_5_2_from_recorded_analysis():
    with open("outputs/preliminary_submission_analysis.json", "r", encoding="utf-8") as fp:
        analysis_data = json.load(fp)
        
    media_db_path = "data/processed/media.db"
    conn = sqlite3.connect(media_db_path)
    c = conn.cursor()
    
    def get_fenced_keyframes(video_id: str, peak_frame: int, count: int = 5):
        c.execute("""
            SELECT frame_idx FROM keyframes 
            WHERE video_id = ? 
            ORDER BY ABS(frame_idx - ?) ASC 
            LIMIT ?
        """, (video_id, peak_frame, count))
        rows = [r[0] for r in c.fetchall()]
        return rows if rows else [peak_frame]
        
    def get_video_keyframes(video_id: str):
        c.execute("SELECT frame_idx FROM keyframes WHERE video_id = ? ORDER BY frame_idx ASC", (video_id,))
        return [r[0] for r in c.fetchall()]

    searcher = VectorSearcher()
    out_dir = "submission"
    os.makedirs(out_dir, exist_ok=True)
    
    for f in glob.glob(os.path.join(out_dir, "*.csv")):
        try:
            os.remove(f)
        except Exception:
            pass
            
    QA_ANSWERS = {
        "query-p1-15-qa": "Khánh Trung",
        "query-p1-19-qa": "Hỏa hồng Nhật Tảo oanh thiên địa, Kiếm bạt Kiên Giang khóc quỷ thần",
        "query-p1-22-qa": "Bánh bao"
    }
    
    for item in analysis_data:
        qid = item["qid"]
        qtype = item["type"]
        top1 = item["top1"]
        top5_vids = item.get("top5_videos", [])
        
        qfile = f"data/contest_queries/{qid}.txt"
        with open(qfile, "r", encoding="utf-8", errors="replace") as fp:
            qtext = fp.read().strip()
            
        csv_path = os.path.join(out_dir, f"{qid}.csv")
        rows = []
        
        # 1. KIS Queries
        if qtype == "KIS":
            top_vid = str(top1[0]).replace(".mp4", "").strip()
            top_fidx = int(top1[1])
            
            fenced = get_fenced_keyframes(top_vid, top_fidx, count=5)
            for f in fenced:
                rows.append(f"{top_vid},{f}")
                
            seen_vids = {top_vid: len(rows)}
            for vid in top5_vids:
                if vid == top_vid or seen_vids.get(vid, 0) >= 3:
                    continue
                kfs = get_video_keyframes(vid)
                if kfs:
                    for f in kfs[:3]:
                        rows.append(f"{vid},{f}")
                        seen_vids[vid] = seen_vids.get(vid, 0) + 1
                        
            raw_cands = searcher.search_by_text(qtext, top_k=500)
            for cand in raw_cands:
                cv = str(cand.video_id).replace(".mp4", "").strip()
                cf = int(cand.frame_idx)
                if seen_vids.get(cv, 0) >= 3:
                    continue
                cand_kfs = get_fenced_keyframes(cv, cf, count=2)
                for kf in cand_kfs:
                    if len(rows) >= 100:
                        break
                    rows.append(f"{cv},{kf}")
                    seen_vids[cv] = seen_vids.get(cv, 0) + 1
                if len(rows) >= 100:
                    break
                    
            if len(rows) < 100:
                for cand in raw_cands:
                    cv = str(cand.video_id).replace(".mp4", "").strip()
                    cf = int(cand.frame_idx)
                    rows.append(f"{cv},{cf}")
                    if len(rows) >= 100:
                        break
                        
            rows = rows[:100]
            with open(csv_path, "w", encoding="utf-8", newline="") as fp:
                fp.write("\r\n".join(rows) + "\r\n")

        # 2. QA Queries
        elif qtype == "QA":
            top_vid = str(top1[0]).replace(".mp4", "").strip()
            top_fidx = int(top1[1])
            ans = QA_ANSWERS.get(qid, "Không xác định")
            
            fenced = get_fenced_keyframes(top_vid, top_fidx, count=5)
            for f in fenced:
                rows.append(f'{top_vid},{f},"{ans}"')
                
            seen_vids = {top_vid: len(rows)}
            for vid in top5_vids:
                if vid == top_vid or seen_vids.get(vid, 0) >= 3:
                    continue
                kfs = get_video_keyframes(vid)
                if kfs:
                    for f in kfs[:3]:
                        rows.append(f'{vid},{f},"{ans}"')
                        seen_vids[vid] = seen_vids.get(vid, 0) + 1
                        
            raw_cands = searcher.search_by_text(qtext, top_k=500)
            for cand in raw_cands:
                cv = str(cand.video_id).replace(".mp4", "").strip()
                cf = int(cand.frame_idx)
                if seen_vids.get(cv, 0) >= 3:
                    continue
                cand_kfs = get_fenced_keyframes(cv, cf, count=2)
                for kf in cand_kfs:
                    if len(rows) >= 100:
                        break
                    rows.append(f'{cv},{kf},"{ans}"')
                    seen_vids[cv] = seen_vids.get(cv, 0) + 1
                if len(rows) >= 100:
                    break
                    
            if len(rows) < 100:
                for cand in raw_cands:
                    cv = str(cand.video_id).replace(".mp4", "").strip()
                    cf = int(cand.frame_idx)
                    rows.append(f'{cv},{cf},"{ans}"')
                    if len(rows) >= 100:
                        break
                        
            rows = rows[:100]
            with open(csv_path, "w", encoding="utf-8", newline="") as fp:
                fp.write("\r\n".join(rows) + "\r\n")

        # 3. TRAKE Queries
        elif qtype == "TRAKE":
            top_vid = str(top1[0]).replace(".mp4", "").strip()
            raw_frames = [int(x) for x in top1[1:] if str(x).isdigit()]
            num_events = len(raw_frames) if raw_frames else 4
            
            if len(raw_frames) == num_events and all(raw_frames[i] < raw_frames[i+1] for i in range(len(raw_frames)-1)):
                rows.append(f"{top_vid}," + ",".join(map(str, raw_frames)))
                
            for vid in top5_vids:
                kfs = get_video_keyframes(vid)
                n_kfs = len(kfs)
                if n_kfs >= num_events:
                    step = max(1, (n_kfs - 1) // num_events)
                    for start_idx in range(n_kfs):
                        if start_idx + (num_events - 1) * step >= n_kfs:
                            break
                        sub_frames = [kfs[start_idx + i * step] for i in range(num_events)]
                        if all(sub_frames[i] < sub_frames[i+1] for i in range(len(sub_frames)-1)):
                            r_str = f"{vid}," + ",".join(map(str, sub_frames))
                            if r_str not in rows:
                                rows.append(r_str)
                                if len(rows) >= 60:
                                    break
                if len(rows) >= 60:
                    break
                    
            raw_cands = searcher.search_by_text(qtext, top_k=500)
            for cand in raw_cands:
                cv = str(cand.video_id).replace(".mp4", "").strip()
                kfs = get_video_keyframes(cv)
                n_kfs = len(kfs)
                if n_kfs >= num_events:
                    for start_idx in range(n_kfs - num_events):
                        sub_frames = [kfs[start_idx + i] for i in range(num_events)]
                        if all(sub_frames[i] < sub_frames[i+1] for i in range(len(sub_frames)-1)):
                            r_str = f"{cv}," + ",".join(map(str, sub_frames))
                            if r_str not in rows:
                                rows.append(r_str)
                                if len(rows) >= 100:
                                    break
                if len(rows) >= 100:
                    break
                    
            rows = rows[:100]
            with open(csv_path, "w", encoding="utf-8", newline="") as fp:
                fp.write("\r\n".join(rows) + "\r\n")
            
    conn.close()
    
    zip_path = "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for cf in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
            zf.write(cf, arcname=f"submission/{os.path.basename(cf)}")
            
    print(f"\n[Packaging] Successfully updated '{zip_path}'.")
    subprocess.run(["venv\\Scripts\\python", "tools/validate_submission_package.py"], check=True)

if __name__ == '__main__':
    generate_exact_5_2_from_recorded_analysis()
