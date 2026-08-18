import time
import uuid
import sys
import os
import base64
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(PROJECT_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
from tools.review_tool import resolve_keyframe_b64

from src.common.schemas import CandidateFrame
from src.pipeline import MVPPipeline, PipelineResult
from src.role_c_logic.vlm_client import GeminiVisionClient

class QAPipeline:
    def __init__(self, detect_threshold: float = 0.3):
        self.kis_pipeline = MVPPipeline(detect_threshold=detect_threshold, top_k_retrieve=500)
        self.vqa_engine = GeminiVisionClient()
        
    def run(self, query_id: str, event_description: str, question: str, top_k: int = 5) -> Dict:
        """
        Runs the Q&A Pipeline:
        1. Retrieve top-K frames based on event description.
        2. Feed frames + question to VLM to get answer.
        """
        start_time = time.time()
        
        # 1. Retrieve candidates using existing KIS pipeline logic
        kis_result = self.kis_pipeline.run(query_id, event_description, query_type="QA")
        
        # We only need the top candidate(s) to answer the question.
        # NMS is already applied in executor. We take top_k.
        candidates = kis_result.candidates[:top_k]
        
        if not candidates:
            return {
                "answer": "Không tìm thấy video/khung hình nào phù hợp với mô tả sự kiện.",
                "evidence_candidates": []
            }
            
        # Tìm candidate tốt nhất được VLM xác nhận (hoặc fallback về candidate[0])
        # Hệ thống KIS pipeline chạy trước đó đã gọi VLM verify và cộng điểm 5.0 (VLM Match) nếu khớp
        best_candidate = candidates[0]
        for c in candidates:
            # Nếu frame có điểm fusion_score cao bất thường (>= 5.0 do được cộng RANK_1_BONUS từ VLM verify)
            if c.fusion_score >= 5.0:
                best_candidate = c
                break
                
        image_bytes = None
        b64_str = None
        
        try:
            vid = best_candidate.video_id
            fidx = best_candidate.frame_idx
            
            b64_str, keyframe_n, expected_fname = resolve_keyframe_b64(vid, fidx, keyframes_root=os.path.join(PROJECT_ROOT, "data", "raw", "keyframes"))
            if b64_str:
                image_bytes = base64.b64decode(b64_str.split(",")[1])
                        
        except Exception as e:
            print(f"[QA] Error extracting image for VLM: {e}")
            
        # 2. Get Answer from VLM
        if b64_str:
            # We prepend the instruction to make it focus on QA
            prompt = f"Dựa vào hình ảnh này, hãy trả lời câu hỏi ngắn gọn: {question}"
            answer = self.vqa_engine._call_api(prompt, [b64_str])
        else:
            answer = "Lỗi: Không trích xuất được hình ảnh để đưa vào VLM."
            
        return {
            "answer": answer,
            "evidence_candidates": candidates,
            "latency_ms": (time.time() - start_time) * 1000
        }
