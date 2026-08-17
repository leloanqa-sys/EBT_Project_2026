import os
import json
import hashlib
import time
import requests
import threading
from typing import List

from src.common.schemas import CandidateFrame

CACHE_DIR = os.path.join("data", "cache", "gemini_vlm")
os.makedirs(CACHE_DIR, exist_ok=True)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class GeminiVisionClient:
    """
    Agent 2: Gemini Vision Client for VLM Verification.
    """
    GEMINI_CASCADE = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    def __init__(self):
        self.api_key = self._load_api_key()
        self._current_model_idx = 0
        self._lock = threading.Lock()
        self.total_api_calls = 0

    def _load_api_key(self) -> str:
        for key_name in ["API_GEMINI", "GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"]:
            val = os.getenv(key_name)
            if val and val.strip():
                return val.strip()

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for env_file in [os.path.join(PROJECT_ROOT, ".env"), os.path.join(PROJECT_ROOT, ".env.example")]:
            if os.path.exists(env_file):
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k in ["API_GEMINI", "GEMINI_API_KEY", "GEMINI_KEY", "GOOGLE_API_KEY"] and v:
                                return v
        return ""

    def _get_next_model(self):
        with self._lock:
            model = self.GEMINI_CASCADE[self._current_model_idx]
            self._current_model_idx = (self._current_model_idx + 1) % len(self.GEMINI_CASCADE)
            return model

    def _get_cache_path(self, raw_text: str, prompt_version: str, video_id: str, frame_idx: int) -> str:
        # Bug #4 Fix: Include video_id in cache key.
        # Without it, frame_idx=449 of L25_V005 and L25_V022 would hash identically.
        key_str = f"{raw_text}_{prompt_version}_{video_id}_{frame_idx}"
        h = hashlib.sha256(key_str.encode('utf-8')).hexdigest()
        return os.path.join(CACHE_DIR, f"{h}.json")

    def _read_cache(self, raw_text: str, prompt_version: str, video_id: str, frame_idx: int):
        p = self._get_cache_path(raw_text, prompt_version, video_id, frame_idx)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return None

    def _write_cache(self, raw_text: str, prompt_version: str, video_id: str, frame_idx: int, data: dict):
        p = self._get_cache_path(raw_text, prompt_version, video_id, frame_idx)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _call_api(self, prompt: str, base64_images: List[str]) -> str:
        if not self.api_key:
            return ""
            
        parts = [{"text": prompt}]
        for b64 in base64_images:
            raw_b64 = b64.split(",")[-1] if "," in b64 else b64
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": raw_b64
                }
            })
            
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.0}
        }

        for attempt in range(3):
            model_name = self._get_next_model()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            try:
                self.total_api_calls += 1
                resp = requests.post(url, json=payload, timeout=60.0, verify=False)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                elif resp.status_code == 429:
                    time.sleep(2.0 * (2 ** attempt))
                    continue
            except:
                time.sleep(2.0 * (2 ** attempt))
                continue
        return ""

    def verify_candidates_batch(self, candidates: List[CandidateFrame], prompt: str, raw_text: str, prompt_version: str) -> List[dict]:
        """
        Sends a batch of images to Gemini to verify if they match the prompt.
        Returns a list of dicts: {"match": bool, "answer": Optional[str]} in the same order.
        """
        results = [{"match": False, "answer": None} for _ in range(len(candidates))]
        
        # We need to resolve image paths or B64 for candidates
        import os
        from tools.review_tool import resolve_keyframe_b64
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        keyframes_root = os.path.join(PROJECT_ROOT, "data", "raw", "keyframes")
        
        uncached_indices = []
        b64_images_to_send = []
        
        for idx, candidate in enumerate(candidates):
            cached_res = self._read_cache(raw_text, prompt_version, candidate.video_id, candidate.frame_idx)
            if cached_res is not None:
                results[idx] = cached_res
            else:
                b64_str, _, _ = resolve_keyframe_b64(candidate.video_id, candidate.frame_idx, keyframes_root=keyframes_root)
                if b64_str:
                    uncached_indices.append(idx)
                    b64_images_to_send.append(b64_str)
                    
        if not uncached_indices:
            return results
            
        BATCH_SIZE = 5
        for i in range(0, len(uncached_indices), BATCH_SIZE):
            batch_idx = uncached_indices[i:i+BATCH_SIZE]
            batch_b64 = b64_images_to_send[i:i+BATCH_SIZE]
            
            batch_prompt = (
                f"You are given {len(batch_b64)} images. For each image, verify: {prompt}\n\n"
                "Return a JSON array of objects. Each object must have a 'match' boolean field, and an 'answer' string/null field. "
                "The array length must exactly match the number of images. "
                "Example format: [{\"match\": true, \"answer\": \"red\"}, {\"match\": false, \"answer\": null}]"
            )
            
            vlm_response = self._call_api(batch_prompt, batch_b64)
            parsed_array = [{"match": False, "answer": None} for _ in range(len(batch_idx))]
            try:
                clean_resp = vlm_response.strip()
                if clean_resp.startswith("```json"): clean_resp = clean_resp[7:]
                if clean_resp.endswith("```"): clean_resp = clean_resp[:-3]
                parsed_json = json.loads(clean_resp)
                if isinstance(parsed_json, list):
                    for j, val in enumerate(parsed_json):
                        if j < len(parsed_array) and isinstance(val, dict):
                            parsed_array[j] = {
                                "match": bool(val.get("match", False)),
                                "answer": str(val.get("answer")) if val.get("answer") is not None else None
                            }
            except Exception as e:
                print(f"[VLM] Failed to parse JSON array: {e}")
                
            for j, original_idx in enumerate(batch_idx):
                match_val = parsed_array[j]
                results[original_idx] = match_val
                c = candidates[original_idx]
                self._write_cache(raw_text, prompt_version, c.video_id, c.frame_idx, match_val)
                
        return results
