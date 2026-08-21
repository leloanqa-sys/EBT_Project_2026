import sqlite3
import os
import glob

print("--- MEDIA.DB ---")
if os.path.exists("data/processed/media.db"):
    conn = sqlite3.connect("data/processed/media.db")
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    print("Tables:", tables)
    for t in tables:
        tname = t[0]
        c.execute(f"PRAGMA table_info({tname})")
        cols = [r[1] for r in c.fetchall()]
        print(f"Table {tname}: {cols}")
        c.execute(f"SELECT * FROM {tname} LIMIT 2")
        print("Sample:", c.fetchall())
    conn.close()

print("\n--- METADATA.DB ---")
if os.path.exists("data/processed/metadata.db"):
    conn = sqlite3.connect("data/processed/metadata.db")
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = c.fetchall()
    print("Tables:", tables)
    for t in tables:
        tname = t[0]
        c.execute(f"PRAGMA table_info({tname})")
        cols = [r[1] for r in c.fetchall()]
        print(f"Table {tname}: {cols}")
        c.execute(f"SELECT * FROM {tname} LIMIT 2")
        print("Sample:", c.fetchall())
    conn.close()

print("\n--- CHECKING LOCAL IMAGES/VIDEOS ---")
img_files = glob.glob("data/raw/**/*.jpg", recursive=True) + glob.glob("data/raw/**/*.png", recursive=True)
vid_files = glob.glob("data/raw/**/*.mp4", recursive=True)
print(f"Found {len(img_files)} images, {len(vid_files)} videos in data/raw.")
if img_files:
    print("Sample image path:", img_files[0])
if vid_files:
    print("Sample video path:", vid_files[0])

