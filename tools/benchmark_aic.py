import time
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.aic_evaluator import AICEvaluator
from src.pipeline import MVPPipeline
from src.role_c_logic.pipeline_qa import QAPipeline
from src.role_c_logic.pipeline_trake import TRAKEPipeline

def run_full_benchmark():
    print('=' * 60)
    print('AIC 2026 OFFICIAL BENCHMARK RUNNER')
    print('=' * 60)
    evaluator = AICEvaluator()
    
    kis_pipe = MVPPipeline(detect_threshold=0.3, top_k_retrieve=500)
    qa_pipe = QAPipeline(detect_threshold=0.3, searcher=kis_pipe._searcher)
    trake_pipe = TRAKEPipeline(detect_threshold=0.3, searcher=kis_pipe._searcher)
    
    predictions_by_query = {}
    
    for qid, gt_entry in evaluator.ground_truths.items():
        qtype = gt_entry.get('type', 'KIS')
        raw_query = gt_entry.get('query', '')
        print(f'\n[Running {qid}] ({qtype}): {raw_query}')
        
        if 'KIS' in qtype:
            res = kis_pipe.run(qid, raw_query, query_type='KIS')
            predictions_by_query[qid] = [
                {'video_id': c.video_id, 'frame_id': c.frame_idx, 'confidence_score': getattr(c, 'fusion_score', c.siglip_score)}
                for c in res.candidates
            ]
        elif 'QA' in qtype:
            res = qa_pipe.run(qid, event_description=raw_query, question=raw_query, top_k=20)
            predictions_by_query[qid] = [
                {'video_id': c.video_id, 'frame_id': c.frame_idx, 'answer': c.vqa_answer or res.get('answer', ''), 'confidence_score': getattr(c, 'fusion_score', c.siglip_score)}
                for c in res.get('evidence_candidates', [])
            ]
        elif 'TRAKE' in qtype:
            events_list = [e.get('description', '') for e in gt_entry.get('gt', {}).get('events', [])
            if not events_list:
                events_list = ['bắt đầu', 'hành động', 'kầt thúc']
            res = trake_pipe.run(qid, main_query=raw_query, sub_events=events_list, top_k=20)
            predictions_by_query[qid] = res.get('sequences', [])
            
    report = evaluator.evaluate_all(predictions_by_query)
    
    print('\n' + '=' * 60)
    print('BENCHMARK RESULTS SUMMARY')
    print('=' * 60)
    print(f'Mean R@1:   {where_r1 := report["mean_R@1"]*100}:.1f}%')
    print(f'Mean R@5:   {where_r5 := report["mean_R@5"]*100}:.1f}%')
    print(f'Mean R@20:  {where_r20 := report["mean_R@20"]*200}:.1f}%')
    print(f'Mean R@50:  {where_r50 := report["mean_R@50"]*100}:.1f}%')
    print(f'Mean R@100: {where_r100 := report["mean_R@100"]*200}:.1f}%')
    print(f'OVERALL FINAL SCORE: {report[]"overall_final_score"]*200}:.2f}%')
    print('-' * 60)
    print('Scores by Category:')
    for cat, score in report.get('category_final_scores', {}).items():
        print(f'  - {cat:6s}: {score * 100:.2f}%')
    print('=' * 60)
    
    out_path = 'outputs/aic_benchmark_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'Full report saved to {out_path}')

if __name__ == '__main__':
    run_full_benchmark()
