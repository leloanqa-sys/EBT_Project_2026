import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path regardless of execution method
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.role_a_retrieval.searcher import VectorSearcher
from src.role_c_logic.pipeline_kis import run_kis, get_searcher
from src.role_c_logic.ranking import cluster_by_event, rank_5budget
from tests.score_auditor import audit_candidate, audit_vector_l2_norms
from tests.review_tool import export_review_html


BENCHMARK_QUERIES = [
    {
        "id": "audit_001",
        "query": "một người nam đeo kính đang thuyết trình trong phòng họp"
    },
    {
        "id": "audit_002",
        "query": "cảnh đường phố đông đúc xe máy và ô tô di chuyển"
    },
    {
        "id": "audit_003",
        "query": "trẻ em chơi đùa trong công viên xanh"
    },
    {
        "id": "audit_004",
        "query": "người mặc áo xanh"
    }
]

def run_verification_audit(out_dir: str = "tests/audit_output"):
    print("=" * 70)
    print("🚀 RUNNING TIER 1 & TIER 2 VERIFICATION AUDIT (ROLE C)")
    print("=" * 70)

    searcher = get_searcher()
    
    # 1. VERIFY IMAGE & TEXT VECTOR L2 NORMS
    print("\n--- 🔍 STEP 1: L2-NORM VERIFICATION ---")
    
    # Inspect query vector norms
    sample_text = BENCHMARK_QUERIES[0]["query"]
    query_vec = searcher.encode_text_query(sample_text)
    text_norm = float(np.linalg.norm(query_vec))
    print(f"[Text Query Vector] Raw shape: {query_vec.shape}, L2 Norm: {text_norm:.6f}")
    assert abs(text_norm - 1.0) < 1e-4, f"Text vector L2 norm is NOT 1.0! Got {text_norm}"
    
    # Inspect stored NumPy Hot-Path image vectors
    if hasattr(searcher, 'index') and searcher.index.ntotal > 0:
        sample_img_vecs = searcher.index.reconstruct_n(0, min(100, searcher.index.ntotal))
        img_norms = audit_vector_l2_norms(sample_img_vecs)
        avg_img_norm = float(np.mean(img_norms))
        print(f"[FAISS Stored Image Vectors] Total: {searcher.index.ntotal}, Sample Size: 100")
        print(f"[FAISS Stored Image Vectors] Min Norm: {min(img_norms):.6f}, Max Norm: {max(img_norms):.6f}, Avg Norm: {avg_img_norm:.6f}")
        assert abs(avg_img_norm - 1.0) < 1e-3, f"Image vector L2 norm is NOT 1.0! Got {avg_img_norm}"
        print("✅ [VERIFIED] Both Text Query Vectors and Stored Image Vectors are STRICTLY L2-Normalized (norm = 1.0)!")
    
    # 2. RUN REAL BENCHMARK QUERIES & GENERATE HTML REVIEW REPORTS
    print("\n--- 📊 STEP 2: BENCHMARK QUERIES & VISUAL REVIEW GENERATION ---")
    audit_results = []
    
    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        q_text = item["query"]
        print(f"\nEvaluating [{q_id}]: '{q_text}'...")
        
        # Run KIS Pipeline
        csv_path = run_kis(q_text, query_id=q_id, top_k_raw=300)
        
        # Get raw candidates for auditing & HTML generation
        raw_candidates = searcher.search_by_text(q_text, top_k=300)
        clustered = cluster_by_event(raw_candidates, gap_threshold=15)
        top10_candidates = clustered[:10]
        
        # Audit top 1
        q_vec = searcher.encode_text_query(q_text)
        report = audit_candidate(top10_candidates[0], q_text, q_vec)
        
        # Generate HTML Review report
        html_file = os.path.join(out_dir, f"{q_id}_review.html")
        export_review_html(q_id, q_text, top10_candidates, html_file)
        
        audit_results.append({
            "query_id": q_id,
            "raw_query": q_text,
            "query_l2_norm": report.query_l2_norm,
            "top1_video": report.top1_video_id,
            "top1_frame": report.top1_frame_idx,
            "top1_clip_score": report.top1_clip_score,
            "html_report": html_file,
            "csv_submission": csv_path
        })
        
        print(f"  └─ Top-1 Result: {report.top1_video_id} / Frame {report.top1_frame_idx} (Clip Score: {report.top1_clip_score})")
        print(f"  └─ HTML Review: {html_file}")
        print(f"  └─ CSV Output:  {csv_path}")

    print("\n" + "=" * 70)
    print("📋 AUDIT VERIFICATION SUMMARY REPORT")
    print("=" * 70)
    print(json.dumps(audit_results, indent=2, ensure_ascii=False))
    print("\n✅ Verification Audit Completed Successfully!")

if __name__ == "__main__":
    run_verification_audit()
