import sqlite3

conn = sqlite3.connect("data/processed/media.db")
c = conn.cursor()

def get_fenced_keyframes(video_id: str, peak_frame: int, n_frames: int = 6):
    # Query actual extracted keyframes for this video sorted by proximity to peak_frame
    c.execute("""
        SELECT frame_idx, pts_time 
        FROM keyframes 
        WHERE video_id = ? 
        ORDER BY ABS(frame_idx - ?) ASC
        LIMIT ?
    """, (video_id, peak_frame, n_frames))
    return c.fetchall()

for vid, peak in [("L22_V013", 17538), ("L30_V012", 6476), ("L26_V240", 2633), ("L26_V387", 2561)]:
    fenced = get_fenced_keyframes(vid, peak, 6)
    print(f"\nFenced keyframes for {vid} around {peak}:")
    for f in fenced:
        print(f"  -> Frame: {f[0]} (PTS: {f[1]:.2f}s)")

conn.close()
