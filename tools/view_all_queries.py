import glob
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

files = sorted(glob.glob("data/contest_queries/*.txt"))
for f in files:
    fname = os.path.basename(f)
    with open(f, "r", encoding="utf-8", errors="replace") as fp:
        content = fp.read().strip()
    print(f"=== {fname} ===")
    print(content)
    print("-" * 50)
