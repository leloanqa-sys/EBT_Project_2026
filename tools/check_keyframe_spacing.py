import sqlite3

conn = sqlite3.connect("data/processed/media.db")
c = conn.cursor()

# Check keyframe step for a few videos (L22_V013, L26_V240, L30_V012)
for vid in ["L22_V013", "L26_V240", "L30_V012", "L26_V387"]:
    c.execute("SELECT frame_idx, pts_time, fps FROM keyframes WHERE video_id = ? ORDER BY frame_idx LIMIT 6", (vid,))
    rows = c.fetchall()
    print(f"\n[{vid}] Sample keyframes:")
    for r in rows:
        print("  Frame:", r[0], "PTS:", r[1], "FPS:", r[2])

conn.close()
