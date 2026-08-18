import os
import sys
import json
import hashlib
import datetime
import faiss
import numpy as np
import pandas as pd
from src.role_a_retrieval.mapping_utils import build_global_mapping
from src.role_a_retrieval.feature_store import FeatureStore

def calculate_md5(file_path: str) -> str:
    """Calculates MD5 hash of a file."""
    if not os.path.exists(file_path):
        return "FILE_NOT_FOUND"
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def build_faiss_index(features_dir: str = "data/raw/siglip2-features",
                     mapping_csv_path: str = "data/processed/global_mapping.csv",
                     output_dir: str = "data/processed/faiss_index",
                     dimension: int = 768) -> faiss.Index:
    """
    Builds exact FAISS IndexFlatIP index sequentially using FeatureStore chunked iterator.
    Writes index_manifest.json and records MD5 checksums for artifact locking.
    """
    if not os.path.exists(mapping_csv_path):
        print(f"Mapping CSV not found at {mapping_csv_path}, building global mapping first...")
        build_global_mapping(features_dir=features_dir)

    order_json_path = "data/processed/video_id_order.json"
    feature_store = FeatureStore(features_dir=features_dir, order_json_path=order_json_path)

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "siglip2.index")
    manifest_path = os.path.join(output_dir, "index_manifest.json")

    print(f"--- [FAISS Build] Initializing IndexFlatIP(dim={dimension}) ---")
    index = faiss.IndexFlatIP(dimension)

    total_added = 0
    for chunk_feats, start_id, end_id in feature_store.iterate_chunks(chunk_size=10000):
        if chunk_feats.shape[1] != dimension:
            print(f"CRITICAL ERROR: Vector dimension mismatch! Expected {dimension}, got {chunk_feats.shape[1]}")
            sys.exit(1)

        index.add(chunk_feats)
        total_added += chunk_feats.shape[0]
        print(f"Added chunk IDs {start_id} -> {end_id} (Accumulated: {index.ntotal})")

    # Save FAISS Index
    faiss.write_index(index, index_path)

    # Build and save Index Manifest
    manifest_data = {
        "model": "siglip2-base-patch16-224",
        "dimension": dimension,
        "metric": "inner_product",
        "normalized": True,
        "index_type": "IndexFlatIP",
        "ntotal": index.ntotal,
        "build_timestamp": datetime.datetime.now().isoformat()
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Compute MD5 Checksums
    index_md5 = calculate_md5(index_path)
    db_md5 = calculate_md5("data/processed/media.db")

    print(f"--- [FAISS Build Success] Total vectors indexed: {index.ntotal} ---")
    print(f"Index saved to: {index_path}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"MD5 Checksum (siglip2.index): {index_md5}")
    print(f"MD5 Checksum (media.db): {db_md5}")

    # Append to docs/decisions.md for audit trail
    decisions_path = "docs/decisions.md"
    os.makedirs(os.path.dirname(decisions_path), exist_ok=True)
    with open(decisions_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Artifact Lock Checksum ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"- `siglip2.index` ntotal={index.ntotal} MD5: `{index_md5}`\n")
        f.write(f"- `media.db` MD5: `{db_md5}`\n")

    return index

if __name__ == "__main__":
    build_faiss_index()
