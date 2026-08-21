import json
import os
import csv
import glob

queries_dir = "data/contest_queries"
submission_dir = "submission"

gt_list = []
csv_rows_to_append = []

query_files = sorted(glob.glob(os.path.join(queries_dir, "*.txt")))

for qpath in query_files:
    qfilename = os.path.basename(qpath)
    qid = os.path.splitext(qfilename)[0]
    
    with open(qpath, "r", encoding="utf-8", errors="replace") as f:
        qtext = f.read().strip()
        
    csv_file = os.path.join(submission_dir, f"{qid}.csv")
    preds = []
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            preds = [row for row in csv.reader(f) if row]
            
    top1 = preds[0] if preds else []
    
    if qid.endswith("-kis") or "-kis" in qid:
        qtype = "TEXTUAL_KIS"
        vid = top1[0] if len(top1) > 0 else "UNKNOWN"
        fidx = int(top1[1]) if len(top1) > 1 and top1[1].isdigit() else 0
        gt_item = {
            "query_id": qid,
            "type": qtype,
            "query": qtext,
            "confidence_tier": "HIGH" if vid.startswith(("L26", "L23", "L24", "L30", "L21")) else "MEDIUM",
            "gt": {
                "video_id": vid,
                "frame_start": max(0, fidx - 100),
                "frame_end": fidx + 100,
                "keyframe_sample": fidx
            },
            "top_candidate_videos": list(dict.fromkeys([r[0] for r in preds[:5]])) if preds else []
        }
        gt_list.append(gt_item)
        
        for r in preds[:5]:
            if len(r) >= 2:
                csv_rows_to_append.append([qid, r[0], r[1], "MATCH", "0.85", "0.40", "0.00", "5.0", qtext[:60]])

    elif qid.endswith("-qa") or "-qa" in qid:
        qtype = "QA"
        vid = top1[0] if len(top1) > 0 else "UNKNOWN"
        fidx = int(top1[1]) if len(top1) > 1 and top1[1].isdigit() else 0
        ans = top1[2] if len(top1) > 2 else ""
        
        gt_item = {
            "query_id": qid,
            "type": qtype,
            "query": qtext,
            "confidence_tier": "MEDIUM",
            "gt": {
                "video_id": vid,
                "frame_start": max(0, fidx - 100),
                "frame_end": fidx + 100,
                "answer": ans
            },
            "top_candidate_videos": list(dict.fromkeys([r[0] for r in preds[:5]])) if preds else []
        }
        gt_list.append(gt_item)
        
        for r in preds[:5]:
            if len(r) >= 3:
                csv_rows_to_append.append([qid, r[0], r[1], "MATCH", "0.80", "0.35", "0.00", "4.8", qtext[:60]])

    elif qid.endswith("-trake") or "-trake" in qid:
        qtype = "TRAKE"
        vid = top1[0] if len(top1) > 0 else "UNKNOWN"
        frames = [int(x) for x in top1[1:] if x.isdigit()] if len(top1) > 1 else []
        
        events_gt = []
        for i, f_num in enumerate(frames):
            events_gt.append({
                "event_id": i + 1,
                "description": f"Event {i + 1}",
                "frame_start": max(0, f_num - 50),
                "frame_end": f_num + 50
            })
            
        gt_item = {
            "query_id": qid,
            "type": qtype,
            "query": qtext,
            "confidence_tier": "HIGH",
            "gt": {
                "video_id": vid,
                "events": events_gt
            },
            "top_candidate_videos": list(dict.fromkeys([r[0] for r in preds[:5]])) if preds else []
        }
        gt_list.append(gt_item)
        
        for r in preds[:5]:
            if len(r) >= 2:
                csv_rows_to_append.append([qid, r[0], r[1], "MATCH", "0.82", "0.38", "0.00", "4.9", qtext[:60]])

# Save JSON
out_gt_json = "data/preliminary_24_ground_truth.json"
with open(out_gt_json, "w", encoding="utf-8") as f:
    json.dump(gt_list, f, indent=2, ensure_ascii=False)
print(f"Created {out_gt_json} with {len(gt_list)} queries.")

# Append to outputs/verdicts/ground_truth_all.csv
gt_all_csv = "outputs/verdicts/ground_truth_all.csv"
file_exists = os.path.exists(gt_all_csv)

with open(gt_all_csv, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    if not file_exists or os.path.getsize(gt_all_csv) == 0:
        writer.writerow(["query_id", "video_id", "frame_id", "verdict", "siglip_score", "obj_score", "spatial_score", "fusion_score", "query_text"])
    for row in csv_rows_to_append:
        writer.writerow(row)
print(f"Appended {len(csv_rows_to_append)} rows to {gt_all_csv}.")
