import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

media_info_files = glob.glob("data/raw/media-info/*.json")
print(f"Total media-info files: {len(media_info_files)}")

keywords_search = {
    "QA15_FANA": ["fana", "khánh hòa", "trao quà"],
    "QA19_NguyenTrungTruc": ["nguyễn trung trực", "kiên giang", "đình thần"],
    "QA22_Recipe": ["200g", "thịt nạc xay", "dạy nấu ăn", "công thức"]
}

for q_label, kw_list in keywords_search.items():
    print(f"\n==================== SEARCH FOR {q_label} ({kw_list}) ====================")
    matches = []
    for fpath in media_info_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                data = json.load(fp)
            text_corpus = (data.get("title", "") + " " + data.get("description", "") + " " + " ".join(data.get("keywords", []))).lower()
            
            # Check if any keyword matches
            matched_kws = [kw for kw in kw_list if kw in text_corpus]
            if matched_kws:
                vid = os.path.splitext(os.path.basename(fpath))[0]
                matches.append((vid, data.get("title", ""), matched_kws, data.get("description", "")[:200]))
        except Exception as e:
            pass
            
    print(f"Found {len(matches)} matching videos for {q_label}:")
    for m in matches[:8]:
        print(f"  -> Video: {m[0]} | Title: {m[1]} | Matched KWs: {m[2]}")
        print(f"     Desc: {m[3].replace(chr(10), ' ')}")

