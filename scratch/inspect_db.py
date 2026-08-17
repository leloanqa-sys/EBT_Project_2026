import sqlite3
import numpy as np

print("=== Inspecting metadata.db ===")
conn = sqlite3.connect('data/processed/metadata.db')
cur = conn.cursor()

# Check tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print('Tables:', tables)

# Check schema of detections table
cur.execute("PRAGMA table_info(detections)")
schema = cur.fetchall()
print('\nDetections schema:')
for col in schema:
    print(' ', col)

# Sample rows
cur.execute("SELECT * FROM detections LIMIT 5")
rows = cur.fetchall()
print('\nSample rows:')
for r in rows:
    print(' ', r)

# Check for L22 video
cur.execute("SELECT DISTINCT video_id FROM detections WHERE video_id LIKE '%L22%' LIMIT 5")
l22 = cur.fetchall()
print('\nL22 videos:', l22)

# Check frame_n values for an L22 video
if l22:
    vid = l22[0][0]
    cur.execute("SELECT DISTINCT frame_n FROM detections WHERE video_id=? ORDER BY frame_n LIMIT 20", (vid,))
    frames = cur.fetchall()
    print(f'Frame_n values for {vid}:', [f[0] for f in frames])

# Check total rows
cur.execute("SELECT COUNT(*) FROM detections")
total = cur.fetchone()
print('\nTotal rows in detections:', total[0])

conn.close()

print("\n=== Inspecting mapping_array.npz ===")
data = np.load('data/processed/mapping_array.npz', allow_pickle=True)
print('Keys:', list(data.keys()))
print('video_ids shape:', data['video_ids'].shape)
print('frame_indices shape:', data['frame_indices'].shape)
print('frame_indices sample (first 20):', data['frame_indices'][:20])
print('pts_times sample (first 5):', data['pts_times'][:5])
print('fps sample (first 5):', data['fps'][:5])

# Check frame indices for L22 videos
video_ids = list(data['video_ids'])
video_idx = data['video_idx']
frame_indices = data['frame_indices']
print('\nTotal FAISS vectors:', len(frame_indices))

# Find L22 video entries
for i, v in enumerate(video_ids):
    if 'L22' in str(v):
        mask = video_idx == i
        fi = frame_indices[mask]
        print(f'\nVideo {v} (idx={i}): {len(fi)} frames, frame_idx range: {fi.min()} - {fi.max()}')
        print(f'  First 10 frame_idx: {fi[:10]}')
        break
