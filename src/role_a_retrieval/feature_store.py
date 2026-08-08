import os
import json
import glob
import numpy as np
from typing import Generator, List, Tuple

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Performs in-place/controlled-buffer L2 normalization on float32 matrix.
    Guarantees L2 norm = 1.0 per vector for Inner Product ≡ Cosine Similarity.
    """
    vectors = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms

class FeatureStore:
    """
    Role A FeatureStore: Provides chunked loading, mmap reading, and vector normalization
    to prevent memory spikes during index building and inspection.
    """
    def __init__(self, features_dir: str = "data/raw/clip-features-32",
                 order_json_path: str = "data/processed/video_id_order.json"):
        self.features_dir = features_dir
        self.order_json_path = order_json_path

        if os.path.exists(order_json_path):
            with open(order_json_path, "r", encoding="utf-8") as f:
                self.video_ids = json.load(f)
        else:
            npy_files = glob.glob(os.path.join(features_dir, "*.npy"))
            self.video_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in npy_files])

    def iterate_chunks(self, chunk_size: int = 10000) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        Yields (normalized_vectors_chunk, start_faiss_id, end_faiss_id)
        Iterates over video .npy files using mmap and buffer management.
        """
        buffer_vectors = []
        current_start_id = 0
        current_faiss_id = 0

        for vid_id in self.video_ids:
            npy_path = os.path.join(self.features_dir, f"{vid_id}.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"Feature file missing for video {vid_id}: {npy_path}")

            feats = np.load(npy_path, mmap_mode="r")
            n_rows = feats.shape[0]

            buffer_vectors.append(feats)
            current_faiss_id += n_rows

            buffered_total = sum(b.shape[0] for b in buffer_vectors)

            if buffered_total >= chunk_size:
                concat_feats = np.vstack(buffer_vectors)
                norm_feats = l2_normalize(concat_feats)

                chunk_end_id = current_start_id + norm_feats.shape[0] - 1
                yield norm_feats, current_start_id, chunk_end_id

                current_start_id = chunk_end_id + 1
                buffer_vectors = []

        if buffer_vectors:
            concat_feats = np.vstack(buffer_vectors)
            norm_feats = l2_normalize(concat_feats)
            chunk_end_id = current_start_id + norm_feats.shape[0] - 1
            yield norm_feats, current_start_id, chunk_end_id

    def load_single_video_features(self, vid_id: str, normalize: bool = True) -> np.ndarray:
        """
        Loads and optionally L2-normalizes features for a single video.
        """
        npy_path = os.path.join(self.features_dir, f"{vid_id}.npy")
        feats = np.load(npy_path)
        if normalize:
            return l2_normalize(feats)
        return feats.astype(np.float32)
