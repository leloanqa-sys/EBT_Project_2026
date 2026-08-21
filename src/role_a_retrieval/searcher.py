import os
import sys
import json
import hashlib
import numpy as np
import faiss
from typing import List, Optional, Union

# Suppress benign HuggingFace Hub symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Force offline mode to prevent SSL connection hanging on Windows (assuming model is cached)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import requests
import warnings
import urllib3
warnings.simplefilter('ignore', urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification globally for HuggingFace Hub
_orig_request = requests.Session.request
def _unverified_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, *args, **kwargs)
requests.Session.request = _unverified_request

from src.common.schemas import CandidateFrame
from src.role_a_retrieval.feature_store import l2_normalize

class VectorSearcher:
    """
    Role A Vector Searcher Engine:
    - Performs Startup Validation (Fail-Fast).
    - Uses FAISS Index (IndexFlatIP) and NumPy Hot-Path Columnar Lookup O(1).
    - Exposes search_by_text, search_by_vector, search_by_vectors (Batch Search).
    - Supports disk-based query embedding cache and fast video_ids filtering.
    """
    def __init__(self,
                 index_path: str = "data/processed/faiss_index/siglip2.index",
                 npz_path: str = "data/processed/mapping_array.npz",
                 manifest_path: str = "data/processed/faiss_index/index_manifest.json",
                 range_json_path: str = "data/processed/video_to_faiss_range.json",
                 cache_dir: str = "data/processed/cache/query_cache"):
        self.index_path = index_path
        self.npz_path = npz_path
        self.manifest_path = manifest_path
        self.range_json_path = range_json_path
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Startup Validation & Load FAISS Index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Build index first.")
        
        self.index = faiss.read_index(self.index_path)

        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)

            if self.manifest.get("model") != "siglip2-base-patch16-224":
                raise RuntimeError(f"Startup Fail-Fast: Expected model 'siglip2-base-patch16-224', manifest has '{self.manifest.get('model')}'")
            if self.manifest.get("dimension") != 768:
                raise RuntimeError(f"Startup Fail-Fast: Expected dimension 768, manifest has {self.manifest.get('dimension')}")
            if self.index.ntotal != self.manifest.get("ntotal"):
                raise RuntimeError(f"Startup Fail-Fast: Index ntotal ({self.index.ntotal}) != Manifest ntotal ({self.manifest.get('ntotal')})")

        # 2. Load NumPy Hot-Path Columnar Arrays
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"Hot-Path mapping NPZ not found at {self.npz_path}. Run build_global_mapping first.")

        npz_data = np.load(self.npz_path, allow_pickle=True)
        self.video_ids = list(npz_data["video_ids"])
        self.video_idx = npz_data["video_idx"]
        self.frame_indices = npz_data["frame_indices"]
        self.pts_times = npz_data["pts_times"]
        self.fps = npz_data["fps"]

        if self.index.ntotal != len(self.frame_indices):
            raise RuntimeError(f"Startup Fail-Fast: FAISS ntotal ({self.index.ntotal}) != Mapping length ({len(self.frame_indices)})")

        # 3. Load Video Ranges for Filtering
        self.video_range_map = {}
        if os.path.exists(self.range_json_path):
            with open(self.range_json_path, "r", encoding="utf-8") as f:
                self.video_range_map = json.load(f)

        self._text_model = None
        self._text_tokenizer = None

    def _init_siglip_text_encoder(self):
        """Khởi tạo SigLIP2 Text Encoder (768-dim)."""
        if self._text_model is None:
            try:
                import ssl
                # Bypass local SSL verification issues when downloading from HF
                ssl._create_default_https_context = ssl._create_unverified_context
            except Exception:
                pass

            try:
                from transformers import AutoProcessor, AutoModel
                import torch
                
                model_name = "google/siglip2-base-patch16-224"
                self._processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model.eval()
                
                # Chuyển sang GPU nếu có
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(self._device)
                
                self._text_model = model
            except Exception as e:
                raise RuntimeError(f"SigLIP2 loading failed ({e}). Kiểm tra cài đặt transformers/torch.")

    def encode_text_query(self, text_query: str) -> np.ndarray:
        """
        Encode text query thành vector 768-dim đã L2-normalized.
        """
        import torch
        norm_query = text_query.strip().lower()
        # Đổi key cache sang siglip2 để không bị xung đột với cache cũ của CLIP
        cache_key = hashlib.sha256(f"siglip2-base-patch16-224::full-query-v2::{norm_query}".encode("utf-8")).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_file):
            return np.load(cache_file)

        self._init_siglip_text_encoder()

        inputs = self._processor(text=[norm_query], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._text_model.get_text_features(**inputs)
            # Handle return formats from different transformers/model versions
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            # L2 Normalization chuẩn
            vector = l2_normalize(text_features.cpu().numpy().astype(np.float32))

        np.save(cache_file, vector)
        return vector

    def _get_id_selector(self, video_ids: Optional[List[str]]) -> Optional[faiss.IDSelector]:
        """Constructs FAISS IDSelector for video filtering."""
        if not video_ids or not self.video_range_map:
            return None

        matching_faiss_ids = []
        for vid in video_ids:
            if vid in self.video_range_map:
                start_id, end_id = self.video_range_map[vid]
                matching_faiss_ids.extend(range(start_id, end_id + 1))

        if not matching_faiss_ids:
            return None

        faiss_id_arr = np.array(matching_faiss_ids, dtype=np.int64)
        return faiss.IDSelectorBatch(faiss_id_arr)

    def search_by_vector(self,
                         vector: np.ndarray,
                         top_k: int = 100,
                         video_ids: Optional[List[str]] = None) -> List[CandidateFrame]:
        """
        Executes ANN search for a single query vector (1, 512).
        Returns List[CandidateFrame] mapped via NumPy Hot-Path arrays O(1).
        """
        results = self.search_by_vectors(np.atleast_2d(vector), top_k=top_k, video_ids=video_ids)
        return results[0] if results else []

    def search_by_vectors(self,
                          vectors: np.ndarray,
                          top_k: int = 100,
                          video_ids: Optional[List[str]] = None) -> List[List[CandidateFrame]]:
        """
        Executes fast Batch Search for Q query vectors (Q, 512).
        Returns List[List[CandidateFrame]] for each query vector.
        """
        vectors_norm = l2_normalize(np.atleast_2d(vectors))
        q_count = vectors_norm.shape[0]

        selector = self._get_id_selector(video_ids)
        params = faiss.SearchParameters()
        if selector is not None:
            params.sel = selector

        scores, indices = self.index.search(vectors_norm, top_k, params=params)

        all_batch_candidates = []

        for q_idx in range(q_count):
            q_scores = scores[q_idx]
            q_indices = indices[q_idx]

            candidates = []
            for score, fid in zip(q_scores, q_indices):
                if fid < 0 or fid >= len(self.frame_indices):
                    continue

                vid_name = self.video_ids[self.video_idx[fid]]
                f_idx = int(self.frame_indices[fid])
                p_time = float(self.pts_times[fid])
                fps_val = float(self.fps[fid])

                candidates.append(CandidateFrame(
                    faiss_id=int(fid),
                    video_id=vid_name,
                    frame_idx=f_idx,
                    pts_time=p_time,
                    fps=fps_val,
                    siglip_score=float(score)
                ))

            all_batch_candidates.append(candidates)

        return all_batch_candidates

    def search_by_text(self,
                       query: str,
                       top_k: int = 100,
                       video_ids: Optional[List[str]] = None) -> List[CandidateFrame]:
        """
        Encodes text query string and searches index for top_k CandidateFrames.
        """
        query_vec = self.encode_text_query(query)
        return self.search_by_vector(query_vec, top_k=top_k, video_ids=video_ids)
