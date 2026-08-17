import os
import sys
import time
from typing import List

# Ensure the root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import MVPPipeline

def run_benchmark():
    print("=" * 60)
    print("🚀 BẮT ĐẦU CHẠY BENCHMARK - GEMINI VLM INTEGRATION")
    print("=" * 60)
    
    # Init Pipeline
    pipeline = MVPPipeline()
    
    # Define Test Queries
    queries = [
        # Query 1: Simple object (Should NOT escalate if confident)
        {"id": "BM_001", "text": "a red car", "type": "KIS"},
        
        # Query 2: Complex Event/Action (Should ALWAYS escalate to VLM based on our rules)
        {"id": "BM_002", "text": "a person is jumping over a fence", "type": "KIS"},
        
        # Query 3: Text/OCR (Should ALWAYS escalate to VLM)
        {"id": "BM_003", "text": "a sign with text STOP", "type": "KIS"}
    ]
    
    start_time = time.time()
    results = pipeline.run_batch(queries)
    end_time = time.time()
    
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT BENCHMARK")
    print("=" * 60)
    
    total_latency = end_time - start_time
    print(f"Tổng thời gian chạy (3 Queries): {total_latency:.2f} giây")
    print(f"Throughput trung bình: {total_latency / len(queries):.2f} giây/query")
    
    for res in results:
        print(f"\nQuery ID: {res.query_id} | Text: '{res.query_text}'")
        print(f"  - Latency: {res.latency_ms:.1f} ms")
        if res.candidates:
            top1 = res.candidates[0]
            print(f"  - Top 1 Candidate: Video {top1.video_id} - Frame {top1.frame_idx} (Score: {top1.fusion_score:.2f})")
        else:
            print("  - Không tìm thấy candidate nào.")
            
    print("\nLưu ý: API Gọi lên Gemini có thể được cache. Nếu chạy lần 2, thời gian sẽ giảm đáng kể.")
    
if __name__ == "__main__":
    run_benchmark()
