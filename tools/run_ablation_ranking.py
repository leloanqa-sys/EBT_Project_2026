"""Run ablation studies on 5 ranking configurations for KIS queries."""

import csv
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.schemas import QueryType, CandidateFrame
from src.role_a_retrieval.searcher import VectorSearcher
from src.role_b_nlp.gemini_nlp_engine import compile_to_visual_ir
from src.role_c_logic.ranking import rank_confidence_portfolio, rank_5budget
from tools.aic_evaluator import AICEvaluator, compute_top_k_recalls
from src.role_c_logic.vlm_client import GeminiVisionClient
from src.role_c_logic.prompt_generator import ir_graph_to_prompt

def run_ablation() -> None:
    searcher = VectorSearcher()
    evaluator = AICEvaluator()
    vlm_client = GeminiVisionClient()
    
    configs = [
        {"name": "4. RRF (Raw 0.5, Strict 1.0, Relaxed 0.5) + Portfolio", "weights": {"raw": 0.5, "strict": 1.0, "relaxed": 0.5}, "ranking": "portfolio", "use_vlm": False},
        {"name": "5. RRF (0.5, 1.0, 0.5) + Portfolio + VLM Top 20", "weights": {"raw": 0.5, "strict": 1.0, "relaxed": 0.5}, "ranking": "portfolio", "use_vlm": True},
    ]
    
    kis_queries = {qid: gt for qid, gt in evaluator.ground_truths.items() if 'KIS' in gt.get('type', 'KIS')}
    print(f"Evaluating {len(kis_queries)} KIS queries on {len(configs)} configurations...")
    
    results_matrix = []
    
    for config in configs:
        print(f"\n--- Running Config: {config['name']} ---")
        all_preds = {}
        for qid, gt in kis_queries.items():
            query_path = PROJECT_ROOT / "data" / "contest_queries" / f"{qid}.txt"
            query_text = query_path.read_text(encoding="utf-8", errors="replace").strip()
            
            ir_graph = compile_to_visual_ir(qid, query_text, QueryType.KIS.value)
            
            raw_cands = searcher.search_by_text(ir_graph.raw_text, top_k=500)
            english_strict_text = ir_graph.clip_query_en if ir_graph.clip_query_en else ir_graph.raw_text
            strict_cands = searcher.search_by_text(english_strict_text, top_k=500)
            english_relaxed_text = ir_graph.relaxed_query_en if ir_graph.relaxed_query_en else english_strict_text
            relaxed_cands = searcher.search_by_text(english_relaxed_text, top_k=500)
            
            w_raw = config["weights"]["raw"]
            w_strict = config["weights"]["strict"]
            w_relaxed = config["weights"]["relaxed"]
            
            merged_dict = {}
            for rank, c in enumerate(raw_cands):
                key = (c.video_id, c.frame_idx)
                merged_dict[key] = merged_dict.get(key, 0.0) + w_raw * (1.0 / (60 + rank))
                
            for rank, c in enumerate(strict_cands):
                key = (c.video_id, c.frame_idx)
                merged_dict[key] = merged_dict.get(key, 0.0) + w_strict * (1.0 / (60 + rank))
                
            for rank, c in enumerate(relaxed_cands):
                key = (c.video_id, c.frame_idx)
                merged_dict[key] = merged_dict.get(key, 0.0) + w_relaxed * (1.0 / (60 + rank))
                
            # Reconstruct candidates with fused score
            fused_cands = []
            for (vid, f_idx), score in merged_dict.items():
                fused_cands.append(CandidateFrame(
                    faiss_id=-1, video_id=vid, frame_idx=f_idx, pts_time=0.0, fps=25.0, siglip_score=score, fusion_score=score
                ))
            
            # 4. Rank
            if config["ranking"] == "portfolio":
                final_ranked = rank_confidence_portfolio(fused_cands, top_k=100)
            else:
                final_ranked_items = rank_5budget(fused_cands, strategy="diversify")
                final_ranked = [CandidateFrame(faiss_id=-1, video_id=it.video_id, frame_idx=it.frame_id, pts_time=0.0, fps=25.0, siglip_score=it.confidence_score, fusion_score=it.confidence_score) for it in final_ranked_items]
            
            if config.get("use_vlm"):
                vlm_prompt = ir_graph_to_prompt(ir_graph)
                
                # Get distinct top 20 candidates
                top_distinct_cands = []
                seen_videos = set()
                for c in final_ranked:
                    if c.video_id not in seen_videos:
                        top_distinct_cands.append(c)
                        seen_videos.add(c.video_id)
                    if len(top_distinct_cands) >= 20:
                        break
                        
                vlm_results = vlm_client.verify_candidates_batch(
                    top_distinct_cands, 
                    prompt=vlm_prompt, 
                    raw_text=ir_graph.raw_text, 
                    prompt_version="1.0"
                )
                
                # Apply bonus
                bonus_applied = False
                for c, res in zip(top_distinct_cands, vlm_results):
                    if res.get("match") is True:
                        # Find original in final_ranked and apply bonus
                        for fc in final_ranked:
                            if fc.video_id == c.video_id and fc.frame_idx == c.frame_idx:
                                fc.fusion_score += 5.0
                                fc.siglip_score += 5.0
                                bonus_applied = True
                                break
                
                if bonus_applied:
                    final_ranked = rank_confidence_portfolio(final_ranked, top_k=100)
            
            # Convert to dict for evaluator
            preds = [{'video_id': c.video_id, 'frame_id': c.frame_idx} for c in final_ranked]
            all_preds[qid] = preds
            
        res = evaluator.evaluate_all(all_preds)
        print(f"  R@1:   {res['mean_R@1']:.4f}")
        print(f"  R@5:   {res['mean_R@5']:.4f}")
        print(f"  R@20:  {res['mean_R@20']:.4f}")
        print(f"  R@50:  {res['mean_R@50']:.4f}")
        print(f"  R@100: {res['mean_R@100']:.4f}")
        print(f"  Final Score: {res['overall_final_score']:.4f}")
        results_matrix.append(res)
        
    print("\n--- Summary ---")
    for idx, config in enumerate(configs):
        print(f"Config {idx+1}: {config['name']} -> Score: {results_matrix[idx]['overall_final_score']:.4f}")

if __name__ == "__main__":
    run_ablation()
