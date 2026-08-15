import os
import sys
import csv
import ssl
import urllib.request
import time

def download_keyframe_zip(keyword: str, csv_path="spreadsheet_data.csv", dest_dir="data/zips"):
    os.makedirs(dest_dir, exist_ok=True)
    
    # Bypass SSL verification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url_map = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    url_map[row[1].strip()] = row[2].strip()
    
    target_files = [f for f in url_map.keys() if keyword.lower() in f.lower() and f.lower().startswith("keyframes")]
    
    if not target_files:
        print(f"❌ Không tìm thấy file Keyframes nào chứa từ khóa '{keyword}'.")
        return

    for fname in target_files:
        url = url_map[fname]
        dest_path = os.path.join(dest_dir, fname)
        
        print(f"\n📥 Đang tải {fname}...")
        existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        if existing_size > 0:
            req.add_header('Range', f'bytes={existing_size}-')
            print(f"🔄 Tiếp tục tải từ {existing_size / (1024*1024):.2f} MB...")
            
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=120) as response:
                mode = 'ab' if existing_size > 0 and response.status == 206 else 'wb'
                if mode == 'wb': existing_size = 0
                
                content_length = response.info().get('Content-Length')
                total_size = existing_size + int(content_length) if content_length else 0
                read_bytes = existing_size
                
                with open(dest_path, mode) as out_file:
                    while True:
                        buffer = response.read(1024 * 256)
                        if not buffer: break
                        out_file.write(buffer)
                        read_bytes += len(buffer)
                        if total_size > 0:
                            sys.stdout.write(f"\rTiến độ: {read_bytes / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({(read_bytes/total_size)*100:.1f}%)")
                            sys.stdout.flush()
                print(f"\n✅ Tải xong {fname}!")
        except Exception as e:
            print(f"\n⚠️ Lỗi khi tải {fname}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
        download_keyframe_zip(keyword)
    else:
        print("Sử dụng: python tools/download_keyframes.py <từ_khóa>")
        print("Ví dụ: python tools/download_keyframes.py L22")
