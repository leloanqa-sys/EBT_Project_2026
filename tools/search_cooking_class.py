import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

media_info_files = glob.glob("data/raw/media-info/*.json")
for fpath in media_info_files:
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
            data = json.load(fp)
        title = data.get("title", "").lower()
        desc = data.get("description", "").lower()
        vid = os.path.splitext(os.path.basename(fpath))[0]
        
        if "dạy" in title or "lớp" in title or "học nấu ăn" in desc or "thịt nạc xay" in desc:
            print(f"[{vid}] Title: {data.get('title')}")
            print(f"       Desc: {data.get('description')[:150]}")
    except:
        pass
