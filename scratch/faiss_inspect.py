import faiss
import numpy as np
import json

print("=== FAISS Index Deep Inspection ===")
index = faiss.read_index('data/processed/faiss_index/clip_vit_b32.index')
print(f"Type: {type(index).__name__}")
print(f"ntotal: {index.ntotal}")
print(f"d: {index.d}")
# metric_type: METRIC_L2=1, METRIC_INNER_PRODUCT=0  
print(f"metric_type raw value: {index.metric_type}")
print(f"faiss.METRIC_L2 = {faiss.METRIC_L2}")
print(f"faiss.METRIC_INNER_PRODUCT = {faiss.METRIC_INNER_PRODUCT}")

if index.metric_type == faiss.METRIC_INNER_PRODUCT:
    print("-> IndexFlatIP (Inner Product) - CONFIRMED")
elif index.metric_type == faiss.METRIC_L2:
    print("-> IndexFlatL2 - UNEXPECTED!")

# Check vector norms more broadly - sample 1000 vectors
print("\n=== Sampling 1000 vectors for norm distribution ===")
n_sample = min(1000, index.ntotal)
indices_to_check = np.linspace(0, index.ntotal-1, n_sample, dtype=np.int64)
vecs = np.zeros((len(indices_to_check), index.d), dtype=np.float32)
for i, idx in enumerate(indices_to_check):
    index.reconstruct(int(idx), vecs[i])

norms = np.linalg.norm(vecs, axis=1)
print(f"Norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}, std={norms.std():.8f}")
print(f"Vectors with norm < 0.99: {(norms < 0.99).sum()}")
print(f"Vectors with norm > 1.01: {(norms > 1.01).sum()}")

# Now simulate a query to understand hubness
print("\n=== Simulating Hubness Test ===")
# Use a random unit vector as query
query = np.random.randn(1, 512).astype(np.float32)
query /= np.linalg.norm(query)

scores, indices_ret = index.search(query, 100)
print(f"Top 10 scores: {scores[0][:10]}")
print(f"Score range: min={scores[0].min():.4f}, max={scores[0].max():.4f}")

# Check which video each top result belongs to
data = np.load('data/processed/mapping_array.npz', allow_pickle=True)
video_ids_arr = list(data['video_ids'])
video_idx_arr = data['video_idx']

top_video_ids = [video_ids_arr[video_idx_arr[fid]] for fid in indices_ret[0] if fid >= 0]
from collections import Counter
vid_counts = Counter(top_video_ids)
print(f"\nTop videos in results (random query): {vid_counts.most_common(10)}")

# Check the score distribution of specific "hubness" videos
print("\n=== Checking vector norms for known hub videos ===")
with open('data/processed/video_to_faiss_range.json') as f:
    ranges = json.load(f)

for vid in ['L24_V011', 'L22_V001']:
    if vid in ranges:
        start, end = ranges[vid]
        count = end - start + 1
        hub_vecs = np.zeros((count, index.d), dtype=np.float32)
        for i, idx in enumerate(range(start, end+1)):
            index.reconstruct(idx, hub_vecs[i])
        hub_norms = np.linalg.norm(hub_vecs, axis=1)
        print(f"{vid}: {count} vecs, norm min={hub_norms.min():.6f}, max={hub_norms.max():.6f}, mean={hub_norms.mean():.6f}")
