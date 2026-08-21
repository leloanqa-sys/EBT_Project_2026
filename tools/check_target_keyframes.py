import sqlite3

conn = sqlite3.connect("data/processed/media.db")
c = conn.cursor()

for vid in ["L26_V194", "L26_V249", "L24_V003", "L24_V004", "L30_V072", "L30_V078"]:
    c.execute("SELECT count(*), min(frame_idx), max(frame_idx), min(pts_time), max(pts_time) FROM keyframes WHERE video_id = ?", (vid,))
    r = c.fetchone()
    print(f"[{vid}] Total Keyframes: {r[0]} | Frames: {r[1]} -> {r[2]} | PTS: {r[3]:.1f}s -> {r[4]:.1f}s")
conn.close()
