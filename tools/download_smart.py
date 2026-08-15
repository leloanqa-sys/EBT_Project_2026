import os
import sys
import csv
import zipfile
import urllib.request
import shutil
import time

# Tier 1 Essential Feature Packages
ESSENTIAL_PACKAGES = [
    {"keyword": "clip-features", "subfolder": "clip-features-32", "expected_items": 800},
    {"keyword": "map-keyframes", "subfolder": "map-keyframes", "expected_items": 800},
    {"keyword": "media-info", "subfolder": "media-info", "expected_items": 800},
    {"keyword": "objects", "subfolder": "objects", "expected_items": 800},
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def safe_remove(file_path: str):
    if not os.path.exists(file_path):
        return
    for _ in range(5):
        try:
            os.remove(file_path)
            return
        except Exception:
            time.sleep(0.5)

def is_valid_zip(file_path: str) -> bool:
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
        return False
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            return z.testzip() is None
    except Exception:
        return False

def is_folder_complete(folder_path: str, min_items: int = 800) -> bool:
    if not os.path.exists(folder_path):
        return False
    items = [f for f in os.listdir(folder_path) if f.lower() != 'objects']
    return len(items) >= min_items

def download_with_smart_resume(url: str, dest_path: str):
    """
    Downloads missing bytes using Range header. Resumes gracefully without redundant copies.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if is_valid_zip(dest_path):
        print(f"  ⏩ [ZIP INTEGRITY VERIFIED] {os.path.basename(dest_path)}")
        return

    existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    req_headers = HEADERS.copy()
    
    if existing_size > 0:
        req_headers['Range'] = f'bytes={existing_size}-'
        print(f"  🔄 Resuming download for {os.path.basename(dest_path)} from {existing_size / (1024*1024):.2f} MB...")
    else:
        print(f"  📥 Downloading {os.path.basename(dest_path)}...")

    req = urllib.request.Request(url, headers=req_headers)

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            mode = 'ab' if existing_size > 0 and response.status == 206 else 'wb'
            if mode == 'wb':
                existing_size = 0

            content_length = response.info().get('Content-Length')
            total_size = existing_size + int(content_length) if content_length else 0
            read_bytes = existing_size
            block_size = 1024 * 256  # 256 KB
            
            with open(dest_path, mode) as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    read_bytes += len(buffer)
                    if total_size > 0:
                        pct = (read_bytes / total_size) * 100
                        sys.stdout.write(f"\r    Progress: {read_bytes / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({pct:.1f}%)")
                        sys.stdout.flush()

        print("\n  ✅ Download complete!")
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable -> File fully downloaded or bad size
            if is_valid_zip(dest_path):
                print(f"\n  🎉 ZIP intact: {os.path.basename(dest_path)}")
                return
            else:
                print(f"\n  ⚠️ Invalid range / corrupted zip. Cleaning and restarting download for {os.path.basename(dest_path)}...")
                safe_remove(dest_path)
                download_with_smart_resume(url, dest_path)
        else:
            raise e
    except Exception as e:
        print(f"\n  ⚠️ Download interrupted: {e}")

def extract_and_flatten(zip_path: str, extract_dir: str):
    """
    Extracts zip file and flattens nested folder structure if present.
    """
    print(f"  📦 Extracting {os.path.basename(zip_path)} -> {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Check for redundant nested subfolder (e.g. data/raw/objects/objects)
    subfolder_name = os.path.basename(extract_dir)
    nested = os.path.join(extract_dir, subfolder_name)
    if os.path.exists(nested) and os.path.isdir(nested):
        print(f"    Flattening nested folder: {nested}")
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(extract_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)

    items_count = len([f for f in os.listdir(extract_dir) if f.lower() != 'objects'])
    print(f"  ✅ Extracted successfully! Total items in {subfolder_name}: {items_count}")

def smart_ingest(csv_path: str = "spreadsheet_data.csv", data_dir: str = "data"):
    zips_dir = os.path.join(data_dir, "zips")
    raw_dir = os.path.join(data_dir, "raw")

    os.makedirs(zips_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV manifest not found at: {csv_path}")

    # Read links from spreadsheet_data.csv
    url_map = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                filename = row[1].strip()
                url = row[2].strip()
                url_map[filename] = url

    print("==================================================")
    print("🚀 SMART DATA INGESTION & RESUME PIPELINE")
    print("==================================================")

    for pkg in ESSENTIAL_PACKAGES:
        kw = pkg["keyword"]
        subfolder = pkg["subfolder"]
        min_items = pkg["expected_items"]
        target_raw_dir = os.path.join(raw_dir, subfolder)

        print(f"\n🔍 Package: [{subfolder}]")

        # 1. Check if folder ALREADY HAS all required items extracted
        if is_folder_complete(target_raw_dir, min_items):
            items_count = len([f for f in os.listdir(target_raw_dir) if f.lower() != 'objects'])
            print(f"  ⏩ [ALREADY COMPLETE - SKIPPED] ({items_count} items present in {target_raw_dir})")
            continue

        # 2. Find zip matching this keyword
        zip_filename = next((fname for fname in url_map.keys() if kw in fname.lower()), None)
        if not zip_filename:
            print(f"  ❌ Warning: No matching zip found in CSV for keyword '{kw}'")
            continue

        zip_url = url_map[zip_filename]
        zip_path = os.path.join(zips_dir, zip_filename)

        # 3. Download missing/interrupted zip with Range resume
        if not is_valid_zip(zip_path):
            download_with_smart_resume(zip_url, zip_path)

        # 4. Unzip if zip is valid
        if is_valid_zip(zip_path):
            extract_and_flatten(zip_path, target_raw_dir)

    print("\n==================================================")
    print("🎉 ALL TIER-1 PACKAGES VERIFIED AND FULLY LOADED!")
    print("==================================================")

if __name__ == "__main__":
    smart_ingest()
