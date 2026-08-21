import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize_answer(text: Optional[str]) -> str:
    if not text:
        return ''
    text = str(text).strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    synonyms = {
        'đỏ': 'red', 'màu đỏ': 'red',
        'trắng': 'white', 'màu trắng': 'white',
        'đen': 'black', 'màu đen': 'black',
        'xanh': 'blue', 'màu xanh': 'blue',
        'vàng': 'yellow', 'màu vàng': 'yellow',
        'máy tính': 'laptop', 'laptop': 'laptop', 'máy tính xách tay': 'laptop',
        'một': '1', 'hai': '2', 'ba': '3', 'bốn': '4', 'năm': '5',
        'sáu': '6', 'bảy': '7', 'tám': '8', 'chín': '9', 'mười': '10'
    }
    for k, v in synonyms.items():
        if text == k or text.startswith(k + ' ') or text.endswith(' ' + k):
            return v
    return text


def evaluate_kis_candidate(c: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    c_vid = str(c.get('video_id', '')).strip()
    gt_vid = str(gt.get('video_id', '')).strip()
    if c_vid != gt_vid:
        return False
    c_frame = int(c.get('frame_id', c.get('frame_idx', -1)))
    f_start = int(gt.get('frame_start', 0))
    f_end = int(gt.get('frame_end', f_start))
    return f_start <= c_frame <= f_end


def evaluate_qa_candidate(c: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    if not evaluate_kis_candidate(c, gt):
        return False
    c_ans = normalize_answer(c.get('answer', c.get('vqa_answer', '')))
    gt_ans = normalize_answer(gt.get('answer', ''))
    if not gt_ans:
        return True
    return c_ans == gt_ans or gt_ans in c_ans or c_ans in gt_ans


def evaluate_trake_sequence(seq: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    seq_vid = str(seq.get('video_id', '')).strip()
    gt_vid = str(gt.get('video_id', '')).strip()
    if seq_vid != gt_vid:
        return False
    events_gt = gt.get('events', [])
    frames = seq.get('frames', [])
    if len(frames) < len(events_gt):
        return False
    for i, e_gt in enumerate(events_gt):
        f_obj = frames[i]
        f_idx = int(f_obj.get('frame_idx', f_obj.get('frame_id', -1)) if isinstance(f_obj, dict) else getattr(f_obj, 'frame_idx', -1))
        f_start = int(e_gt.get('frame_start', 0))
        f_end = int(e_gt.get('frame_end', f_start))
        if not (f_start <= f_idx <= f_end):
            return False
    return True


def compute_top_k_recalls(predictions: List[Dict[str, Any]], gt: Dict[str, Any], qtype: str = 'KIS', k_list: List[int] = [1, 5, 20, 50, 100]) -> Dict[str, float]:
    recalls = {}
    for k in k_list:
        sub_preds = predictions[:k]
        hit = False
        for pred in sub_preds:
            if qtype in ('KIS', 'TEXTUAL_KIS', 'Textual KIS'):
                if evaluate_kis_candidate(pred, gt):
                    hit = True
                    break
            elif qtype in ('QA', 'Q&A'):
                if evaluate_qa_candidate(pred, gt):
                    hit = True
                    break
            elif qtype == 'TRAKE':
                if evaluate_trake_sequence(pred, gt):
                    hit = True
                    break
            else:
                if evaluate_kis_candidate(pred, gt):
                    hit = True
                    break
        recalls[f'R@{k}'] = 1.0 if hit else 0.0
    recalls['Final_Score'] = sum(recalls[f'R@{k}'] for k in k_list) / len(k_list)
    return recalls


class AICEvaluator:
    def __init__(self, gt_path: Optional[str] = None):
        if gt_path is None:
            gt_path = str(PROJECT_ROOT / 'data' / 'official_preliminary_gt.json')
            if not os.path.exists(gt_path):
                gt_path = str(PROJECT_ROOT / 'data' / 'm2_10_human_gt.json')
        self.gt_path = gt_path
        self.ground_truths = self._load_gt()

    def _load_gt(self) -> Dict[str, Dict[str, Any]]:
        with open(self.gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {item['query_id']: item for item in data}

    def evaluate_all(self, predictions_by_query: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        results_by_query = {}
        category_scores: Dict[str, List[float]] = {}
        all_final_scores, all_r1, all_r5, all_r20, all_r50, all_r100 = [], [], [], [], [], []
        for qid, gt_entry in self.ground_truths.items():
            preds = predictions_by_query.get(qid, [])
            qtype = gt_entry.get('type', 'KIS')
            cat = 'KIS' if 'KIS' in qtype else ('QA' if 'QA' in qtype else 'TRAKE')
            scores = compute_top_k_recalls(preds, gt_entry.get('gt', {}), qtype=cat)
            results_by_query[qid] = {'type': cat, 'query': gt_entry.get('query', ''), 'scores': scores}
            category_scores.setdefault(cat, []).append(scores['Final_Score'])
            all_final_scores.append(scores['Final_Score'])
            all_r1.append(scores['R@1'])
            all_r5.append(scores['R@5'])
            all_r20.append(scores['R@20'])
            all_r50.append(scores['R@50'])
            all_r100.append(scores['R@100'])
        return {
            'mean_R@1': sum(all_r1) / len(all_r1) if all_r1 else 0.0,
            'mean_R@5': sum(all_r5) / len(all_r5) if all_r5 else 0.0,
            'mean_R@20': sum(all_r20) / len(all_r20) if all_r20 else 0.0,
            'mean_R@50': sum(all_r50) / len(all_r50) if all_r50 else 0.0,
            'mean_R@100': sum(all_r100) / len(all_r100) if all_r100 else 0.0,
            'overall_final_score': sum(all_final_scores) / len(all_final_scores) if all_final_scores else 0.0,
            'category_final_scores': {cat: sum(sc)/len(sc) for cat, sc in category_scores.items()},
            'queries': results_by_query
        }

if __name__ == '__main__':
    evaluator = AICEvaluator()
    print(f'[AICEvaluator] Successfully loaded {len(evaluator.ground_truths)} GT queries.')