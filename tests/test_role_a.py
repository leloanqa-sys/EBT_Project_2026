import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.schemas import CandidateFrame

from src.role_a_retrieval.mapping_utils import run_sanity_check, build_global_mapping, get_sorted_video_ids
from src.role_a_retrieval.feature_store import FeatureStore, l2_normalize
from src.role_a_retrieval.build_db import build_sqlite_db
from src.role_a_retrieval.build_index import build_faiss_index
from src.role_a_retrieval.searcher import VectorSearcher

def test_01_sanity_check():
    """Verify 100% sanity check pass on raw dataset."""
    result = run_sanity_check(features_dir="data/raw/clip-features-32", map_dir="data/raw/map-keyframes")
    assert result["total_videos"] > 0
    assert result["total_keyframes"] > 0

def test_02_global_mapping():
    """Verify global_mapping.csv and derived NPZ/JSON artifacts."""
    df = build_global_mapping(features_dir="data/raw/clip-features-32", map_dir="data/raw/map-keyframes")
    assert os.path.exists("data/processed/global_mapping.csv")
    assert os.path.exists("data/processed/video_id_order.json")
    assert os.path.exists("data/processed/video_to_faiss_range.json")
    assert os.path.exists("data/processed/mapping_array.npz")
    assert len(df) > 0

def test_03_sqlite_db():
    """Verify cold-path SQLite media.db creation and row count."""
    build_sqlite_db(mapping_csv_path="data/processed/global_mapping.csv", db_path="data/processed/media.db")
    assert os.path.exists("data/processed/media.db")
    
    import sqlite3
    conn = sqlite3.connect("data/processed/media.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM keyframes;")
    count = cursor.fetchone()[0]
    conn.close()

    df = pd.read_csv("data/processed/global_mapping.csv")
    assert count == len(df)

def test_04_faiss_index_and_manifest():
    """Verify FAISS index build and index_manifest.json."""
    index = build_faiss_index(features_dir="data/raw/clip-features-32")
    assert os.path.exists("data/processed/faiss_index/clip_vit_b32.index")
    assert os.path.exists("data/processed/faiss_index/index_manifest.json")
    
    df = pd.read_csv("data/processed/global_mapping.csv")
    assert index.ntotal == len(df)

def test_05_vector_sync():
    """Verify vector sync between original .npy and FAISS index via np.allclose."""
    searcher = VectorSearcher()
    mapping_df = pd.read_csv("data/processed/global_mapping.csv")
    
    sample_ids = np.random.choice(len(mapping_df), size=min(10, len(mapping_df)), replace=False)
    
    for fid in sample_ids:
        row = mapping_df.iloc[fid]
        vid_id = row["video_id"]
        n_val = int(row["n"]) # 1-indexed line
        
        # Load from .npy
        npy_path = f"data/raw/clip-features-32/{vid_id}.npy"
        feats = np.load(npy_path)
        vec_npy = l2_normalize(feats[n_val - 1 : n_val])
        
        # Reconstruct from FAISS
        vec_faiss = searcher.index.reconstruct(int(fid)).reshape(1, -1)
        
        assert np.allclose(vec_npy, vec_faiss, atol=1e-5), f"Mismatch at faiss_id {fid} for video {vid_id}"

def test_06_searcher_interface_and_batch():
    """Verify VectorSearcher text search, vector search, batch search, and filtering."""
    searcher = VectorSearcher()
    
    # 1. Text Search
    candidates = searcher.search_by_text("a photo of a person", top_k=10)
    assert len(candidates) > 0
    assert isinstance(candidates[0], CandidateFrame)
    assert candidates[0].clip_score >= -1.0 and candidates[0].clip_score <= 1.0

    # 2. Vector Search (Self search test)
    sample_vec = searcher.index.reconstruct(0).reshape(1, -1)
    vec_candidates = searcher.search_by_vector(sample_vec, top_k=5)
    assert len(vec_candidates) > 0
    assert vec_candidates[0].faiss_id == 0
    assert np.isclose(vec_candidates[0].clip_score, 1.0, atol=1e-4)

    # 3. Batch Search
    batch_vecs = np.vstack([
        searcher.index.reconstruct(0),
        searcher.index.reconstruct(1)
    ])
    batch_results = searcher.search_by_vectors(batch_vecs, top_k=5)
    assert len(batch_results) == 2
    assert batch_results[0][0].faiss_id == 0
    assert batch_results[1][0].faiss_id == 1

    # 4. Video Filtering Search
    first_vid = searcher.video_ids[0]
    filtered_results = searcher.search_by_text("car", top_k=10, video_ids=[first_vid])
    for cand in filtered_results:
        assert cand.video_id == first_vid

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
