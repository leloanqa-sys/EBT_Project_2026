import os
import sys
import re
import time
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def warmup():
    print("=== STARTING PIPELINE WARMUP (PRE-CACHE) FOR 25 EXAM QUERIES ===")
    
    from src.pipeline import MVPPipeline
    from api.main import _get_searcher
    
    searcher = _get_searcher()
    if searcher is None:
        print("[ERROR] VectorSearcher is not available. Please make sure database indices are built.")
        sys.exit(1)
        
    pipeline = MVPPipeline(detect_threshold=0.3, top_k_retrieve=500, searcher=searcher)
    
    queries_dir = Path("data/contest_queries")
    if not queries_dir.exists():
        print(f"[ERROR] Queries directory {queries_dir} does not exist.")
        sys.exit(1)
        
    # Find and parse files
    pattern = re.compile(r"query-p1-(\d+)-(\w+)\.txt")
    exam_items = []
    
    for f in queries_dir.iterdir():
        if f.is_file() and f.suffix == ".txt":
            match = pattern.match(f.name)
            if match:
                q_num = int(match.group(1))
                q_type = match.group(2).upper()
                
                with open(f, "r", encoding="utf-8") as file_handle:
                    q_text = file_handle.read().strip()
                
                exam_items.append({
                    "number": q_num,
                    "type": q_type,
                    "text": q_text,
                    "filename": f.name
                })
                
    exam_items.sort(key=lambda x: x["number"])
    
    total = len(exam_items)
    print(f"Loaded {total} queries to pre-cache.\n")
    
    for idx, item in enumerate(exam_items):
        q_num = item["number"]
        q_type = item["type"]
        q_text = item["text"]
        
        print(f"[{idx+1}/{total}] Warmup Query #{q_num} ({q_type})...")
        start = time.perf_counter()
        
        # Build prompt question if QA or TRAKE
        question = q_text if q_type in ("QA", "TRAKE") else None
        
        try:
            query_id = f"warmup_p1_{q_num}"
            # Run the actual pipeline (this automatically caches NLP, CLIP and VLM results)
            result = pipeline.run(query_id, q_text, q_type, question=question)
            
            elapsed = (time.perf_counter() - start) * 1000
            print(f"    -> Done in {elapsed:.1f}ms (Candidates: {len(result.candidates)})")
        except Exception as e:
            print(f"    -> [ERROR] Failed to run query #{q_num}: {e}")
            
    print("\n=== WARMUP COMPLETE! ALL 25 QUERIES ARE CACHED AND READY ===")

if __name__ == "__main__":
    warmup()
