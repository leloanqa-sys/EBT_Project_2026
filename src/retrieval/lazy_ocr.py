
import os
import io
import base64
import logging
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
import easyocr

from tools.review_tool import resolve_keyframe_b64

logger = logging.getLogger(__name__)

class LazyOCREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LazyOCREngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            logger.info('[LazyOCR] Initializing EasyOCR Reader (vi, en)...')
            self.reader = easyocr.Reader(['vi', 'en'], gpu=False) # Fallback to CPU if no GPU
            self._initialized = True

    def _get_image_numpy(self, video_id: str, frame_idx: int) -> np.ndarray:
        try:
            b64_str, _, _ = resolve_keyframe_b64(
                video_id, 
                frame_idx, 
                keyframes_root=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'keyframes'),
                allow_remote=False
            )
            if not b64_str:
                return None
            
            img_data = base64.b64decode(b64_str.split(',')[1])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.error(f'[LazyOCR] Error loading image {video_id}/{frame_idx}: {e}')
            return None

    def evaluate_frames(self, candidates: List[Tuple[int, str, int]], text_queries: List[str]) -> Dict[int, float]:
        if not text_queries or not candidates:
            return {}

        results = {}
        queries_lower = [q.lower() for q in text_queries]

        logger.info(f'[LazyOCR] Running OCR on {len(candidates)} frames for terms: {queries_lower}')
        
        for frame_id, video_id, frame_idx in candidates:
            img_np = self._get_image_numpy(video_id, frame_idx)
            if img_np is None:
                results[frame_id] = 0.0
                continue

            try:
                detected_texts = self.reader.readtext(img_np, detail=0)
                full_text = ' '.join(detected_texts).lower()
                
                score = 0.0
                for q in queries_lower:
                    if q in full_text:
                        score += 1.0
                
                if score > 0:
                    logger.info(f'  [LazyOCR] FOUND match in {video_id}/{frame_idx}: {full_text}')
                
                results[frame_id] = score
            except Exception as e:
                logger.error(f'[LazyOCR] EasyOCR error on {video_id}/{frame_idx}: {e}')
                results[frame_id] = 0.0

        return results

