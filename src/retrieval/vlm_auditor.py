import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import google.generativeai as genai
from PIL import Image

# Import schemas and data loader
from src.common.schemas import RankedCandidate
from scripts.remote_data_loader import RemoteDataLoader
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("OFFLINE_CACHE_DIR", "data/processed/cache")) / "vlm_rerank"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class VLMAuditor:
    """
    Role C: VLM Reranker and Temporal Predictor using Gemini 3.5 Flash Lite.
    Intercepts Top-K candidates for visual validation and temporal bound generation.
    """
    def __init__(self, db: DatabaseManager):
        # 1) Try environment variables
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("API_GEMINI")
        
        # 2) Fallback to .env and .env.example
        if not api_key:
            for env_filename in [".env", ".env.example"]:
                env_path = Path(env_filename)
                if env_path.exists():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            if k.strip() in ["API_GEMINI", "GEMINI_API_KEY"]:
                                api_key = v.strip().strip('"').strip("'")
                                break
                if api_key:
                    break

        if not api_key:
            logger.warning("[VLMAuditor] No API Key found. VLM features will be disabled.")
            self.enabled = False
            return
            
        genai.configure(api_key=api_key, transport='rest')
        # Prefer lite for speed/cost, fallback to flash if needed
        self.model = genai.GenerativeModel("gemini-3.5-flash-lite")
        self.enabled = True
        self.loader = RemoteDataLoader()
        self.db = db

    def _get_cache(self, cache_key: str) -> Optional[dict]:
        path = CACHE_DIR / f"{cache_key}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _set_cache(self, cache_key: str, data: dict):
        path = CACHE_DIR / f"{cache_key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def rerank_top_k(self, query: str, candidates: List[RankedCandidate], top_k: int = 10) -> List[RankedCandidate]:
        if not self.enabled or not candidates:
            return candidates

        targets = candidates[:top_k]
        
        # Build cache key
        cands_hash = "_".join([f"{c.candidate.video_id}:{c.candidate.frame_idx}" for c in targets])
        prompt_hash = hashlib.sha1(f"RERANK_{query}_{cands_hash}".encode()).hexdigest()
        
        cached = self._get_cache(prompt_hash)
        if cached:
            logger.info("[VLMAuditor] Rerank cache hit.")
            scores = cached.get("scores", [])
        else:
            logger.info(f"[VLMAuditor] Fetching {len(targets)} images for VLM Reranking...")
            images = []
            valid_targets = []
            
            for c in targets:
                img_path = self.loader.fetch_keyframe(c.candidate.video_id, c.candidate.frame_idx)
                if img_path and img_path.exists():
                    try:
                        images.append(Image.open(img_path))
                        valid_targets.append(c)
                    except Exception as e:
                        logger.error(f"Image open error: {e}")
            
            if not images:
                return candidates

            prompt = f"""You are a strict Visual QA judge. The user is searching for: "{query}".
I am providing {len(images)} images in order. 
Score each image from 0.0 to 10.0 based on how perfectly it matches the query.
Be very harsh. If the query asks for "red shirt" and it's a blue shirt, score 0.
Output a JSON array of floats exactly matching the number of images. 
Example output: [9.5, 0.0, 3.2]"""
            
            try:
                contents = [prompt] + images
                response = self.model.generate_content(
                    contents,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                scores = json.loads(response.text)
                if len(scores) != len(valid_targets):
                    scores = [0.0] * len(valid_targets) # fallback if parsing fails
                
                self._set_cache(prompt_hash, {"scores": scores})
            except Exception as e:
                logger.error(f"[VLMAuditor] API Call Failed: {e}")
                scores = [0.0] * len(valid_targets)

            targets = valid_targets

        # Apply boost
        for i, cand in enumerate(targets):
            if i < len(scores):
                # Normalize VLM score (0-10) to a weight (0-1.0)
                vlm_normalized = scores[i] / 10.0
                # VLM heavily dictates the final order among Top 10 (80% VLM, 20% Original)
                cand.score.final = (cand.score.final * 0.2) + (vlm_normalized * 0.8)
                logger.debug(f"[VLMAuditor] Reranked: {cand.candidate.video_id} -> VLM Score: {scores[i]:.1f}, New Final: {cand.score.final:.4f}")

        # Re-sort only the affected subset, then combine
        candidates[:len(targets)] = targets
        candidates.sort(key=lambda r: r.score.final, reverse=True)
        return candidates

    def predict_temporal_bounds(self, query: str, peak_cand: RankedCandidate) -> Tuple[float, float]:
        """
        Fetches 7 surrounding frames from the DB, streams them to VLM, and predicts [s, e].
        Returns (start_pts, end_pts).
        """
        if not self.enabled:
            return max(0.0, peak_cand.candidate.pts_time - 2.5), peak_cand.candidate.pts_time + 2.5

        vid = peak_cand.candidate.video_id
        pts = peak_cand.candidate.pts_time

        # Get nearby frames from DB
        rows = self.db.execute_query(
            "SELECT frame_idx, pts_time FROM frames WHERE video_id = ? AND pts_time BETWEEN ? AND ? ORDER BY pts_time",
            (vid, max(0.0, pts - 5.0), pts + 5.0)
        )
        
        # Sample up to 7 frames to represent the sequence
        step = max(1, len(rows) // 7)
        sampled_rows = rows[::step][:7]
        
        if len(sampled_rows) < 2:
            return max(0.0, pts - 2.5), pts + 2.5

        cache_key = hashlib.sha1(f"TEMPORAL_{query}_{vid}_{pts}".encode()).hexdigest()
        cached = self._get_cache(cache_key)
        if cached:
            logger.info("[VLMAuditor] Temporal cache hit.")
            return cached.get("start_time", pts), cached.get("end_time", pts)

        images = []
        pts_mapping = []
        for r in sampled_rows:
            ipath = self.loader.fetch_keyframe(vid, r["frame_idx"])
            if ipath and ipath.exists():
                try:
                    images.append(Image.open(ipath))
                    pts_mapping.append(r["pts_time"])
                except Exception:
                    pass

        if len(images) < 2:
             return max(0.0, pts - 2.5), pts + 2.5

        prompt = f"""You are a temporal localization AI. The user is searching for: "{query}".
I am providing a sequence of {len(images)} consecutive frames from a video.
Determine the start and end of the action.
Respond with a JSON object containing two integers ('start_idx' and 'end_idx') from 0 to {len(images)-1}.
Example: {{"start_idx": 1, "end_idx": 4}}"""

        try:
            contents = [prompt] + images
            response = self.model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            res_json = json.loads(response.text)
            s_idx = min(max(0, res_json.get("start_idx", 0)), len(pts_mapping)-1)
            e_idx = min(max(0, res_json.get("end_idx", len(pts_mapping)-1)), len(pts_mapping)-1)
            
            # Ensure start <= end
            if s_idx > e_idx:
                s_idx, e_idx = e_idx, s_idx
                
            start_time = pts_mapping[s_idx]
            end_time = pts_mapping[e_idx]
            
            self._set_cache(cache_key, {"start_time": start_time, "end_time": end_time})
            return start_time, end_time
        except Exception as e:
            logger.error(f"[VLMAuditor] Temporal Bound API Failed: {e}")
            return max(0.0, pts - 2.5), pts + 2.5

    def answer_question(self, question: str, cand: RankedCandidate) -> str:
        """
        Ask a specific question about a candidate frame.
        """
        if not self.enabled:
            return "VLM is disabled."
            
        vid = cand.candidate.video_id
        pts = cand.candidate.pts_time
        
        cache_key = hashlib.sha1(f"QA_{question}_{vid}_{cand.candidate.frame_idx}".encode()).hexdigest()
        cached = self._get_cache(cache_key)
        if cached:
            logger.info("[VLMAuditor] QA cache hit.")
            return cached.get("answer", "No answer")

        img_path = self.loader.fetch_keyframe(vid, cand.candidate.frame_idx)
        if not img_path or not img_path.exists():
            return "Không tìm thấy ảnh để trả lời."
            
        try:
            image = Image.open(img_path)
        except Exception as e:
            logger.error(f"Image open error for QA: {e}")
            return "Lỗi mở ảnh."
            
        prompt = f"""You are a Visual QA assistant. Please answer the following question based on the image provided.
Question: "{question}"
If the image does not contain enough information to answer, say "Không tìm thấy đáp án rõ ràng từ hình ảnh."
Keep your answer concise and in Vietnamese."""

        try:
            response = self.model.generate_content([prompt, image])
            answer = response.text.strip()
            self._set_cache(cache_key, {"answer": answer})
            return answer
        except Exception as e:
            logger.error(f"[VLMAuditor] QA API Failed: {e}")
            return "Lỗi kết nối VLM."
