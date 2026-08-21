import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

media_info_files = glob.glob("data/raw/media-info/*.json")

def search_media(keywords, prefixes=None):
    results = []
    for f in media_info_files:
        vid = os.path.splitext(os.path.basename(f))[0]
        if prefixes and not any(vid.startswith(p) for p in prefixes):
            continue
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                data = json.load(fp)
            corpus = (data.get("title", "") + " " + data.get("description", "") + " " + " ".join(data.get("keywords", []))).lower()
            matched = [k for k in keywords if k in corpus]
            if matched:
                results.append((vid, data.get("title", ""), matched, data.get("description", "")[:120]))
        except:
            pass
    return results

print("=== TRAKE 4: MĂNG TÂY ===")
for r in search_media(["măng tây", "mang tay", "asparagus"], ["L26", "L22"]):
    print(f"  [{r[0]}] {r[1]} -> {r[2]}")

print("\n=== TRAKE 16: MÚA LÂN & CHÀO RỒNG ===")
for r in search_media(["rồng", "múa rồng", "cột", "chào ban giám khảo"], ["L24"]):
    print(f"  [{r[0]}] {r[1]} -> {r[2]}")

print("\n=== TRAKE 18: NẤM, CỦ NĂNG, ĐẬU HỦ ===")
for r in search_media(["củ năng", "nấm", "đậu hũ", "đậu hủ"], ["L26"]):
    print(f"  [{r[0]}] {r[1]} -> {r[2]}")
