import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(os.getcwd())
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MVPPipeline
from api.main import _get_searcher

def main():
    queries = [
        ("Q1_EASY", "Cảnh bãi biển vào buổi sáng"),
        ("Q2_COLOR", "Một chiếc xe ô tô màu đỏ đang chạy trên đường"),
        ("Q3_SPATIAL", "Người đàn ông đứng bên cạnh chiếc xe đạp"),
        ("Q4_COLOR_SPATIAL", "Người phụ nữ mặc áo xanh đang ngồi trên ghế"),
        ("Q5_ACTION", "Một nhóm người đang tập thể dục trong công viên")
    ]
    
    print("Loading Searcher...")
    searcher = _get_searcher()
    pipeline = MVPPipeline(detect_threshold=0.3, top_k_retrieve=500, searcher=searcher)
    
    for qid, text in queries:
        print(f"\n=====================================")
        print(f"Query: [{qid}] {text}")
        print(f"=====================================")
        
        t0 = time.time()
        result = pipeline.run(qid, text, "KIS")
        t1 = time.time()
        
        trace = result.operator_trace
        ir = trace.get("ir_graph", {}) if trace else {}
        print(f"--> IR Intent: {ir.get('intent', 'N/A')}")
        print(f"--> IR Entities: {[e.get('label') for e in ir.get('entities', [])]}")
        print(f"--> Latency: {t1-t0:.2f}s")
        
        print(f"--> Top 3 Results:")
        for i, c in enumerate(result.candidates[:3]):
            vqa_ans = f" | VLM Match: {c.vqa_answer}" if c.vqa_answer else ""
            fusion = getattr(c, 'fusion_score', 0.0)
            print(f"    Rank {i+1}: {c.video_id}:{c.frame_idx} (Fusion: {fusion:.4f}, CLIP: {c.clip_score:.4f}, OBJ: {c.obj_score:.4f}){vqa_ans}")

if __name__ == "__main__":
    main()
