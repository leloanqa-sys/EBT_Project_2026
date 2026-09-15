import os
import zipfile
import requests
import glob
import shutil
import time
import sys

API_URL = "http://localhost:8000"

def process_queries(zip_path: str):
    if not os.path.exists(zip_path):
        print(f"File không tồn tại: {zip_path}")
        return

    temp_dir = "temp_queries"
    sub_dir = "submission"
    
    # Do NOT delete existing directories so we can resume
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    print(f"Đang giải nén bộ đề từ: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    txt_files = glob.glob(f"{temp_dir}/**/*.txt", recursive=True)
    if not txt_files:
        print("Không tìm thấy file .txt nào trong file zip!")
        return
        
    print(f"Tìm thấy {len(txt_files)} câu truy vấn. Bắt đầu xử lý...")

    for txt_file in sorted(txt_files):
        filename = os.path.basename(txt_file)
        name_only = os.path.splitext(filename)[0]
        csv_filename = f"{name_only}.csv"
        csv_path = os.path.join(sub_dir, csv_filename)
        
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            print(f"\n[{name_only}] Đã có kết quả, bỏ qua...")
            continue
            
        with open(txt_file, "r", encoding="utf-8") as f:
            query_text = f.read().strip()

        print(f"\n[{name_only}] Đang chạy...")
        start_t = time.time()
        
        try:
            if "kis" in name_only.lower():
                payload = {"query": query_text, "query_type": "KIS", "top_k": 100}
                res = requests.post(f"{API_URL}/api/v1/search/kis", json=payload, timeout=300)
                res.raise_for_status()
                data = res.json()
                
                with open(csv_path, "w", encoding="utf-8") as f_csv:
                    for r in data.get("results", []):
                        f_csv.write(f"{r['video_id']},{r['frame_idx']}\n")
                        
            elif "qa" in name_only.lower():
                payload = {"query": query_text, "query_type": "QA", "top_k": 100}
                res = requests.post(f"{API_URL}/api/v1/search/kis", json=payload, timeout=300)
                res.raise_for_status()
                data = res.json()
                
                ans = data.get("qa_answer", "")
                if not ans and data.get("results"):
                    ans = data["results"][0].get("vqa_answer", "")
                ans = ans.replace('"', '""')
                
                with open(csv_path, "w", encoding="utf-8") as f_csv:
                    for r in data.get("results", []):
                        f_csv.write(f"{r['video_id']},{r['frame_idx']},\"{ans}\"\n")
                        
            elif "trake" in name_only.lower():
                payload = {"query": query_text, "top_k": 20}
                res = requests.post(f"{API_URL}/api/v1/search/trake", json=payload, timeout=300)
                res.raise_for_status()
                data = res.json()
                
                with open(csv_path, "w", encoding="utf-8") as f_csv:
                    for seq in data.get("sequences", []):
                        frames = [str(f["frame_idx"]) for f in seq.get("frames", [])]
                        if frames:
                            frames_str = ",".join(frames)
                            f_csv.write(f"{seq['video_id']},{frames_str}\n")
                            
            else:
                print(f"Bỏ qua file không xác định định dạng: {filename}")
                continue
                
            elapsed = time.time() - start_t
            print(f"  -> Xong trong {elapsed:.1f}s. Đã lưu {csv_filename}")
            
        except Exception as e:
            print(f"  -> LỖI KHI CHẠY {filename}: {e}")

    out_zip = "submission"
    print(f"\nĐang nén kết quả thành {out_zip}.zip...")
    shutil.make_archive(out_zip, "zip", ".", "submission")
    
    print(f"HOÀN THÀNH! Bạn có thể nộp file {out_zip}.zip lên hệ thống.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python run_aic_evaluation.py <duong_dan_file_zip_bo_de>")
    else:
        process_queries(sys.argv[1])
