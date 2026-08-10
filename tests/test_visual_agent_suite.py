"""
Test Suite for Visual Agent & EBT Pipeline Verification
=========================================================
Runs 25 queries across 5 categories against the live pipeline,
harvests Top-5 candidate frames, detected object labels, scores,
and exports structured JSON for Visual QA Agent audit.
"""
import os
import sys
import json
import time
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.routes.kis_routes import _run_real_search

TEST_CATEGORIES = {
    "D1_Single_Object": [
        {"id": "D1_Q1", "query": "Người phụ nữ mặc áo dài đỏ phát biểu", "type": "KIS", "expected_objs": ["person", "clothing"]},
        {"id": "D1_Q2", "query": "Con mèo nằm trên bàn làm việc", "type": "KIS", "expected_objs": ["cat", "table", "laptop"]},
        {"id": "D1_Q3", "query": "Xe máy chạy trên đường phố đông đúc", "type": "KIS", "expected_objs": ["motorcycle", "car", "person"]},
        {"id": "D1_Q4", "query": "Chiếc máy bay chở khách đang hạ cánh", "type": "KIS", "expected_objs": ["airplane", "sky"]},
        {"id": "D1_Q5", "query": "Bát phở nóng trên bàn ăn", "type": "KIS", "expected_objs": ["bowl", "dining table", "food"]},
    ],
    "D2_Multi_Object_Action": [
        {"id": "D2_Q1", "query": "Người đàn ông đi xe đạp bên cạnh dòng sông", "type": "KIS", "expected_objs": ["person", "bicycle"]},
        {"id": "D2_Q2", "query": "Cảnh bác sĩ đeo khẩu trang đang khám bệnh cho bệnh nhân", "type": "KIS", "expected_objs": ["person", "mask", "bed"]},
        {"id": "D2_Q3", "query": "Nhóm học sinh ngồi trong lớp học nghe giảng", "type": "KIS", "expected_objs": ["person", "chair", "desk"]},
        {"id": "D2_Q4", "query": "Hai người bắt tay nhau trước tòa nhà", "type": "KIS", "expected_objs": ["person", "building"]},
        {"id": "D2_Q5", "query": "Cảnh cảnh sát giao thông dừng xe ô tô", "type": "KIS", "expected_objs": ["person", "car"]},
    ],
    "D3_Text_OCR_Signboard": [
        {"id": "D3_Q1", "query": "Biển hiệu cửa hàng có chữ Phở Báo", "type": "KIS", "expected_objs": ["sign", "building", "text"]},
        {"id": "D3_Q2", "query": "Áo phông màu trắng in chữ VIETNAM", "type": "KIS", "expected_objs": ["person", "clothing"]},
        {"id": "D3_Q3", "query": "Xe cứu thương có chữ AMBULANCE", "type": "KIS", "expected_objs": ["car", "truck", "van"]},
        {"id": "D3_Q4", "query": "Bảng hiệu đường phố có chữ Hà Nội", "type": "KIS", "expected_objs": ["sign", "street"]},
        {"id": "D3_Q5", "query": "Bao bì hộp sữa chua ghi 100% nguyên chất", "type": "KIS", "expected_objs": ["bottle", "cup", "box"]},
    ],
    "D4_Temporal_TRAKE": [
        {"id": "D4_Q1", "query": "Bước 1 người mở cửa xe, Bước 2 bước vào xe lái đi", "type": "TRAKE", "expected_objs": ["car", "person"]},
        {"id": "D4_Q2", "query": "Cảnh hoàng hôn xuống biển rồi trời tối hẳn", "type": "TRAKE", "expected_objs": ["sea", "sky", "sun"]},
        {"id": "D4_Q3", "query": "Cầu thủ nhận bóng rồi sút vào lưới", "type": "TRAKE", "expected_objs": ["person", "sports ball"]},
        {"id": "D4_Q4", "query": "Người rót nước vào ly sau đó uống", "type": "TRAKE", "expected_objs": ["person", "cup", "bottle"]},
        {"id": "D4_Q5", "query": "Đội ngũ cắt băng rôn rồi vỗ tay ăn mừng", "type": "TRAKE", "expected_objs": ["person", "ribbon"]},
    ],
    "D5_Visual_QA": [
        {"id": "D5_Q1", "query": "Người đàn ông mặc áo màu gì khi đứng trên sân khấu?", "type": "QA", "question": "Người đàn ông mặc áo màu gì khi đứng trên sân khấu?", "expected_objs": ["person", "clothing", "stage"]},
        {"id": "D5_Q2", "query": "Có bao nhiêu chiếc xe đạp đậu trước cửa nhà?", "type": "QA", "question": "Có bao nhiêu chiếc xe đạp đậu trước cửa nhà?", "expected_objs": ["bicycle", "building"]},
        {"id": "D5_Q3", "query": "Con vật nào đang chạy trong công viên?", "type": "QA", "question": "Con vật nào đang chạy trong công viên?", "expected_objs": ["dog", "cat", "animal", "park"]},
        {"id": "D5_Q4", "query": "Trên bàn làm việc có chiếc máy tính nhãn hiệu gì?", "type": "QA", "question": "Trên bàn làm việc có chiếc máy tính nhãn hiệu gì?", "expected_objs": ["laptop", "table", "keyboard"]},
        {"id": "D5_Q5", "query": "Chiếc ô tô quay đầu ở đoạn đường nào?", "type": "QA", "question": "Chiếc ô tô quay đầu ở đoạn đường nào?", "expected_objs": ["car", "road", "street"]},
    ]
}

def run_suite():
    print("=" * 70)
    print("      EBT SYSTEM 2026 — VISUAL AGENT TEST SUITE EXECUTION")
    print("=" * 70)

    audit_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_categories": len(TEST_CATEGORIES),
        "total_queries": sum(len(q_list) for q_list in TEST_CATEGORIES.values()),
        "categories": {}
    }

    start_suite_time = time.perf_counter()

    for cat_name, query_list in TEST_CATEGORIES.items():
        print(f"\n▶ Category: {cat_name} ({len(query_list)} queries)")
        cat_results = []

        for item in query_list:
            q_id = item["id"]
            raw_q = item["query"]
            q_type = item.get("type", "KIS")
            question = item.get("question")
            expected_objs = item.get("expected_objs", [])

            print(f"  └─ Executing [{q_id}] ({q_type}): '{raw_q}'...")
            q_start = time.perf_counter()
            
            try:
                res = _run_real_search(query=raw_q, query_type=q_type, top_k=5, question=question)
                q_elapsed = (time.perf_counter() - q_start) * 1000
                
                # Assess top 5 results for object matching and query precision
                top5_items = res.get("results", [])[:5]
                top5_audit = []

                for r in top5_items:
                    det_labels = r.get("detected_labels", [])
                    
                    # Evaluate object overlap
                    matched_objs = [obj for obj in expected_objs if any(obj.lower() in label.lower() for label in det_labels)]
                    obj_match_rate = len(matched_objs) / max(len(expected_objs), 1)

                    # Grade verification (True Positive / Partial / Mismatch)
                    clip_score = r.get("clip_score", 0.0)
                    fusion_score = r.get("fusion_score", 0.0)

                    if obj_match_rate >= 0.5 or clip_score > 0.23:
                        verdict = "MATCH"
                    elif obj_match_rate > 0 or clip_score > 0.21:
                        verdict = "PARTIAL"
                    else:
                        verdict = "MISMATCH"

                    frame_url = r.get("frame_url", "")
                    
                    # Check local file existence for frame link verification
                    rel_path = frame_url.lstrip("/")
                    local_frame_path = PROJECT_ROOT / rel_path
                    frame_link_exists = local_frame_path.exists()

                    top5_audit.append({
                        "rank": r.get("rank"),
                        "video_id": r.get("video_id"),
                        "frame_idx": r.get("frame_idx"),
                        "timestamp": r.get("timestamp"),
                        "clip_score": clip_score,
                        "obj_score": r.get("obj_score"),
                        "fusion_score": fusion_score,
                        "frame_url": frame_url,
                        "frame_link_exists": frame_link_exists,
                        "detected_labels": det_labels,
                        "matched_objs": matched_objs,
                        "obj_match_rate": round(obj_match_rate, 2),
                        "verdict": verdict,
                        "watch_url": r.get("watch_url"),
                    })

                # Calculate Precision@5 for query
                matches_count = sum(1 for item in top5_audit if item["verdict"] in ["MATCH", "PARTIAL"])
                precision_at_5 = matches_count / max(len(top5_audit), 1)

                query_record = {
                    "query_id": q_id,
                    "raw_query": raw_q,
                    "query_type": q_type,
                    "search_time_ms": res.get("search_time_ms", round(q_elapsed, 1)),
                    "parsed_info": res.get("parsed_info", {}),
                    "precision_at_5": round(precision_at_5, 2),
                    "top_5_results": top5_audit
                }

                cat_results.append(query_record)
                print(f"     ✅ Search time: {q_elapsed:.1f}ms | Precision@5: {precision_at_5 * 100:.0f}% | Top-1: {top5_audit[0]['video_id']} frame {top5_audit[0]['frame_idx']} ({top5_audit[0]['verdict']})")

            except Exception as e:
                print(f"     ❌ FAILED query [{q_id}]: {e}")
                import traceback
                traceback.print_exc()

        audit_results["categories"][cat_name] = cat_results

    total_suite_time = time.perf_counter() - start_suite_time
    audit_results["total_suite_time_sec"] = round(total_suite_time, 2)

    # Save audit results JSON
    out_json_path = PROJECT_ROOT / "outputs" / "visual_agent_audit_results.json"
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"SUITE COMPLETE in {total_suite_time:.2f}s")
    print(f"Audit results saved to: {out_json_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_suite()
