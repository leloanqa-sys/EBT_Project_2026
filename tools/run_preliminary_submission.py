import os
import sys
import glob
import re
import json
import time
import zipfile
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MVPPipeline
from src.role_c_logic.pipeline_qa import QAPipeline
from src.role_c_logic.pipeline_trake import TRAKEPipeline
from src.role_a_retrieval.searcher import VectorSearcher

def parse_trake_subevents(content: str) -> Tuple[str, List[str]]:
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    main_desc_lines = []
    events = []
    
    for line in lines:
        m = re.match(r'^(?:E\d+|Sự kiện \d+|\d+\.)\s*:\s*(.+)$', line, re.IGNORECASE)
        if m:
            events.append(m.group(1).strip())
        else:
            if not events:
                main_desc_lines.append(line)
            else:
                events[-1] += " " + line
                
    main_query = " ".join(main_desc_lines) if main_desc_lines else (events[0] if events else content)
    if not events:
        events = [content]
    return main_query, events

def parse_qa_query(content: str) -> Tuple[str, str]:
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    full_text = " ".join(lines)
    
    if "Hỏi" in full_text:
        parts = full_text.split("Hỏi")
        event_desc = parts[0].strip()
        question = ("Hỏi " + parts[1]).strip()
    else:
        event_desc = full_text
        question = full_text
    return event_desc, question

def format_csv_answer(ans: str) -> str:
    if not ans:
        return "Không xác định"
    ans = str(ans).strip().replace('\r', ' ').replace('\n', ' ')
    ans = re.sub(r'\s+', ' ', ans)
    if len(ans) > 100:
        ans = ans[:97] + "..."
    return ans

def run_all_queries(queries_dir: str = "data/contest_queries", out_submission_dir: str = "submission"):
    print("=" * 70)
    print("STARTING BATCH RUNNER FOR CONTEST PRELIMINARY ROUND (24 QUERIES)")
    print("=" * 70)
    
    os.makedirs(out_submission_dir, exist_ok=True)
    
    print("[1/4] Loading FAISS index and metadata cache...")
    searcher = VectorSearcher()
    kis_pipe = MVPPipeline(detect_threshold=0.3, top_k_retrieve=500, searcher=searcher)
    qa_pipe = QAPipeline(detect_threshold=0.3, searcher=searcher)
    trake_pipe = TRAKEPipeline(detect_threshold=0.3, searcher=searcher)
    print("  -> Pipelines initialized successfully.\n")
    
    query_files = sorted(glob.glob(os.path.join(queries_dir, "*.txt")))
    print(f"[2/4] Found {len(query_files)} query files in '{queries_dir}'.\n")
    
    summary_results = []
    
    for idx, qpath in enumerate(query_files, 1):
        qfilename = os.path.basename(qpath)
        qid = os.path.splitext(qfilename)[0]
        csv_filename = f"{qid}.csv"
        csv_path = os.path.join(out_submission_dir, csv_filename)
        
        # Check if already computed and valid
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100:
            with open(csv_path, "r", encoding="utf-8") as f:
                row_count = sum(1 for line in f if line.strip())
            if row_count >= 50:
                print(f"[{idx:02d}/{len(query_files):02d}] Skipping {qid} (already generated: {row_count} rows)")
                summary_results.append({"qid": qid, "status": "cached", "rows": row_count})
                continue
        
        with open(qpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
            
        print(f"[{idx:02d}/{len(query_files):02d}] Processing {qid}...")
        t0 = time.time()
        
        if qid.endswith("-kis") or "-kis" in qid:
            res = kis_pipe.run(qid, content, query_type="KIS")
            candidates = res.candidates[:100]
            
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=",")
                for c in candidates:
                    vid_clean = str(c.video_id).replace(".mp4", "").strip()
                    writer.writerow([vid_clean, int(c.frame_idx)])
                    
            elapsed = time.time() - t0
            print(f"  -> Generated KIS CSV: {len(candidates)} rows ({elapsed:.2f}s)")
            summary_results.append({"qid": qid, "type": "KIS", "rows": len(candidates), "time_s": elapsed})
            
        elif qid.endswith("-qa") or "-qa" in qid:
            event_desc, question = parse_qa_query(content)
            res = qa_pipe.run(qid, event_description=event_desc, question=question, top_k=100)
            candidates = res.get("evidence_candidates", [])[:100]
            global_ans = format_csv_answer(res.get("answer", "Không xác định"))
            
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
                for c in candidates:
                    vid_clean = str(c.video_id).replace(".mp4", "").strip()
                    ans = format_csv_answer(getattr(c, "vqa_answer", "") or global_ans)
                    writer.writerow([vid_clean, int(c.frame_idx), ans])
                    
            elapsed = time.time() - t0
            print(f"  -> Generated QA CSV: {len(candidates)} rows | Ans: '{global_ans}' ({elapsed:.2f}s)")
            summary_results.append({"qid": qid, "type": "QA", "rows": len(candidates), "time_s": elapsed, "ans": global_ans})
            
        elif qid.endswith("-trake") or "-trake" in qid:
            main_q, sub_events = parse_trake_subevents(content)
            num_events = len(sub_events)
            print(f"  -> Detected {num_events} sub-events: {sub_events}")
            
            res = trake_pipe.run(qid, main_query=main_q, sub_events=sub_events, top_k=100)
            sequences = res.get("sequences", [])[:100]
            
            if not sequences:
                print("  [Warning] No aligned sequences found, generating fallback sequences...")
                fb_res = kis_pipe.run(qid, main_q, query_type="KIS")
                for c in fb_res.candidates[:100]:
                    vid_clean = str(c.video_id).replace(".mp4", "").strip()
                    base_f = int(c.frame_idx)
                    frames_mock = [base_f + i * 25 for i in range(num_events)]
                    sequences.append({"video_id": vid_clean, "frames": frames_mock})
            
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=",")
                for s in sequences:
                    vid_clean = str(s.get("video_id", "")).replace(".mp4", "").strip()
                    raw_frames = s.get("frames", [])
                    frame_ids = []
                    for f_item in raw_frames:
                        if isinstance(f_item, int):
                            frame_ids.append(f_item)
                        elif isinstance(f_item, dict):
                            frame_ids.append(int(f_item.get("frame_idx", f_item.get("frame_id", 0))))
                        else:
                            frame_ids.append(int(getattr(f_item, "frame_idx", 0)))
                    
                    while len(frame_ids) < num_events:
                        last_f = frame_ids[-1] if frame_ids else 0
                        frame_ids.append(last_f + 25)
                    frame_ids = frame_ids[:num_events]
                    
                    for i in range(1, len(frame_ids)):
                        if frame_ids[i] <= frame_ids[i-1]:
                            frame_ids[i] = frame_ids[i-1] + 25
                            
                    writer.writerow([vid_clean] + frame_ids)
                    
            elapsed = time.time() - t0
            print(f"  -> Generated TRAKE CSV: {len(sequences)} rows ({elapsed:.2f}s)")
            summary_results.append({"qid": qid, "type": "TRAKE", "rows": len(sequences), "time_s": elapsed, "events": num_events})
            
    print("\n" + "=" * 70)
    print("[3/4] PACKAGING SUBMISSION ZIP ARCHIVE")
    print("=" * 70)
    
    zip_output_path = "submission.zip"
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for csv_file in sorted(glob.glob(os.path.join(out_submission_dir, "*.csv"))):
            arcname = f"submission/{os.path.basename(csv_file)}"
            zipf.write(csv_file, arcname=arcname)
            print(f"  + Added: {arcname}")
            
    print(f"\n[4/4] Successfully created contest submission package: '{zip_output_path}'")
    return summary_results

if __name__ == "__main__":
    run_all_queries()
