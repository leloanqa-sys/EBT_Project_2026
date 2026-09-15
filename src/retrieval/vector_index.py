"""
FAISS Vector Index Abstraction Layer
====================================
Pure vector retrieval engine.
Manages:
- FAISS Index (search / add / save / load)
- Manifest validation
- In-memory mapping from internal FAISS ID to Canonical Frame ID.

Phase 4 update: Multi-batch support.
  FAISSIndex can now hold N (index, mapping, manifest) tuples.
  search_vectors merges results from all batches and re-sorts by score.
  Batch 1 index is loaded by default; Batch 2 is registered via add_batch_index().
"""

import os
import json
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "faiss_index", "siglip2.index"
)
DEFAULT_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "faiss", "mapping_batch1.npy"
)
DEFAULT_MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "manifests", "siglip_batch1_v1.json"
)


class _BatchIndex:
    """Internal: one (index, mapping, manifest) tuple for a single batch."""

    def __init__(self, index_path: str, mapping_path: str, manifest_path: Optional[str] = None):
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.index: Optional[faiss.Index] = None
        self.mapping: Optional[np.ndarray] = None
        self.manifest: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        logger.info(f"[FAISSIndex] Loading index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Mapping array not found: {self.mapping_path}. "
                "Run the ingest script first."
            )
        self.mapping = np.load(str(self.mapping_path))
        logger.info(
            f"[FAISSIndex] Loaded mapping: {len(self.mapping):,} entries | "
            f"index ntotal={self.index.ntotal:,}"
        )

        if self.manifest_path and self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            logger.info(f"[FAISSIndex] Manifest: {self.manifest.get('index_name')}")

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (scores, faiss_ids, frame_ids) each of length top_k.
        faiss_ids are LOCAL to this batch index.
        frame_ids are canonical frame IDs from the mapping array.
        """
        scores, indices = self.index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]
        # Map faiss_id → canonical frame_id (O(1) RAM lookup)
        frame_ids = np.where(
            (indices >= 0) & (indices < len(self.mapping)),
            self.mapping[np.clip(indices, 0, len(self.mapping) - 1)],
            -1
        )
        return scores, indices, frame_ids


class FAISSIndex:
    """
    Decoupled FAISS Index manager with manifest and in-memory frame_id mapping.

    Phase 4 multi-batch:
      - Primary batch loaded in __init__ (Batch 1 by default).
      - Additional batches registered via add_batch_index().
      - search_vectors merges all batches and returns unified top_k results.
    """

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        mapping_path: str = DEFAULT_MAPPING_PATH,
        manifest_path: Optional[str] = DEFAULT_MANIFEST_PATH,
    ):
        self._batches: List[_BatchIndex] = []
        # Load primary (Batch 1) index
        self._batches.append(_BatchIndex(index_path, mapping_path, manifest_path))

    # ── Public API ─────────────────────────────────────────────────────────

    def add_batch_index(
        self,
        index_path: str,
        mapping_path: str,
        manifest_path: Optional[str] = None,
    ) -> None:
        """
        Register an additional batch index (e.g. Batch 2).
        Call this AFTER validating provenance (07_validate_batch2_provenance.py).
        """
        logger.info(f"[FAISSIndex] Registering extra batch index: {index_path}")
        self._batches.append(_BatchIndex(index_path, mapping_path, manifest_path))
        logger.info(f"[FAISSIndex] Total batches loaded: {len(self._batches)}")

    @property
    def total_vectors(self) -> int:
        return sum(b.index.ntotal for b in self._batches if b.index)

    @property
    def dimension(self) -> int:
        return self._batches[0].index.d if self._batches and self._batches[0].index else 0

    def search_vectors(
        self,
        query_vec: np.ndarray,
        top_k: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Searches query vector across ALL registered batch indices.
        Results are merged and re-sorted by score (descending).

        Returns:
            scores:    np.ndarray shape (top_k,) — cosine similarity
            faiss_ids: np.ndarray shape (top_k,) — local FAISS IDs (batch-local, for debugging)
            frame_ids: np.ndarray shape (top_k,) — canonical frame_ids from mapping
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        # Normalize once before searching all batches
        faiss.normalize_L2(query_vec)

        all_scores: List[float] = []
        all_faiss_ids: List[int] = []
        all_frame_ids: List[int] = []

        for batch in self._batches:
            scores, indices, frame_ids = batch.search(query_vec, top_k)
            for sc, fid, frame_id in zip(scores, indices, frame_ids):
                if sc > -1e9 and frame_id > 0:  # Filter sentinel values
                    all_scores.append(float(sc))
                    all_faiss_ids.append(int(fid))
                    all_frame_ids.append(int(frame_id))

        if not all_scores:
            return np.array([]), np.array([]), np.array([])

        # Sort by score descending, take top_k
        order = np.argsort(all_scores)[::-1][:top_k]
        return (
            np.array(all_scores)[order],
            np.array(all_faiss_ids)[order],
            np.array(all_frame_ids)[order],
        )
