import sys, os
sys.path.append(os.path.abspath('.'))

from src.role_c_logic.pipeline_trake import TRAKEPipeline
pipeline = TRAKEPipeline(detect_threshold=0.3)

print('=== Testing TRAKE Pipeline ===')
res = pipeline.run('trake_test_1', ['người đàn ông vẫy tay', 'người đàn ông bước lên xe buýt'], top_k=3)
print('Latency:', res.get('latency_ms'))
print('Sequences found:', len(res.get('sequences', [])))
for idx, seq in enumerate(res.get('sequences', [])):
    print(f"Sequence {idx+1}: Video {seq['video_id']} (avg_score: {seq['avg_score']:.4f})")
    for f in seq['frames']:
         print(f"  - Frame {f.frame_idx} | PTS: {f.pts_time:.2f}s | CLIP: {f.clip_score:.4f} | Fusion: {f.fusion_score:.4f}")
