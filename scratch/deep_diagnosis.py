import sqlite3
import numpy as np
import json

print("=== Deep Diagnosis: Frame Index Mismatch ===\n")

# Load mapping
data = np.load('data/processed/mapping_array.npz', allow_pickle=True)
video_ids = list(data['video_ids'])
video_idx_arr = data['video_idx']
frame_indices = data['frame_indices']

# Connect DB
conn = sqlite3.connect('data/processed/metadata.db')
cur = conn.cursor()

# Test a few videos: compare what FAISS has vs what DB has
test_videos = ['L22_V001', 'L24_V011', 'L21_V001']

for vid in test_videos:
    if vid not in video_ids:
        print(f"{vid}: NOT in video_ids!")
        continue
    
    vid_i = video_ids.index(vid)
    mask = video_idx_arr == vid_i
    fi = frame_indices[mask]
    
    print(f"\n=== {vid} ===")
    print(f"  FAISS: {len(fi)} keyframes")
    print(f"  FAISS frame_idx first 10: {fi[:10].tolist()}")
    print(f"  FAISS frame_idx last 5: {fi[-5:].tolist()}")
    
    # DB frame_n values (should be 1, 2, 3... if sequential)
    cur.execute("SELECT DISTINCT frame_n FROM detections WHERE video_id=? ORDER BY frame_n LIMIT 20", (vid,))
    db_frames = [r[0] for r in cur.fetchall()]
    print(f"  DB frame_n first 20: {db_frames}")
    
    # Check: does any FAISS frame_idx match DB frame_n?
    matched = [f for f in fi[:20] if f in db_frames]
    print(f"  Overlapping values (first 20 FAISS vs DB): {matched}")
    
    # Key test: query DB using FAISS frame_idx
    test_frame_idx = int(fi[5])  # 6th keyframe
    cur.execute("SELECT COUNT(*) FROM detections WHERE video_id=? AND frame_n=?", (vid, test_frame_idx))
    count_by_faiss = cur.fetchone()[0]
    
    # Query DB using sequential index
    cur.execute("SELECT COUNT(*) FROM detections WHERE video_id=? AND frame_n=?", (vid, 6))  # sequential: 6th frame
    count_by_seq = cur.fetchone()[0]
    
    print(f"\n  Test: query frame_n={test_frame_idx} (FAISS value) -> {count_by_faiss} rows in DB")
    print(f"  Test: query frame_n=6 (sequential) -> {count_by_seq} rows in DB")

# Also check FAISS index type
import faiss
index = faiss.read_index('data/processed/faiss_index/clip_vit_b32.index')
print(f"\n=== FAISS Index Info ===")
print(f"  Type: {type(index).__name__}")
print(f"  ntotal: {index.ntotal}")
print(f"  dimension: {index.d}")
print(f"  metric: {index.metric_type}")  # 0=L2, 1=IP

# Check vector magnitudes for "hubness" candidates
# Reconstruct a few vectors to check their norms
try:
    vecs = np.zeros((5, index.d), dtype=np.float32)
    for i in range(5):
        index.reconstruct(i, vecs[i])
    norms = np.linalg.norm(vecs, axis=1)
    print(f"\n  Sample vector norms (first 5 FAISS vectors): {norms}")
except Exception as e:
    print(f"\n  Could not reconstruct vectors: {e}")

conn.close()
print("\nDone.")
