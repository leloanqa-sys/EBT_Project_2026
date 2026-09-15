"""
Script 02: Ingest Batch 1 Metadata & Keyframes
==============================================
Populates aic2026.db with:
- batch1 registration
- videos from data/raw/media-info
- frames from data/processed/media.db
- saves mapping_batch1.npy & manifest.json
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_manager import DatabaseManager
from src.data.ingestion import BatchIngestionPipeline

def main():
    print("=" * 70)
    print("  PHASE 1 - SCRIPT 02: INGEST BATCH 1 METADATA & KEYFRAMES")
    print("=" * 70)

    db = DatabaseManager()
    ingestion = BatchIngestionPipeline(db, PROJECT_ROOT)

    # 1. Register Batch 1
    ingestion.register_batch("batch1", "AIC 2026 Batch 1 Dataset", "Preliminary round dataset with 873 videos.")

    # 2. Ingest Video Metadata
    media_info_dir = PROJECT_ROOT / "data" / "raw" / "media-info"
    ingestion.ingest_videos_metadata("batch1", str(media_info_dir))

    # 3. Migrate Keyframes
    legacy_media_db = PROJECT_ROOT / "data" / "processed" / "media.db"
    mapping = ingestion.migrate_keyframes(str(legacy_media_db))

    # 4. Save Mapping & Manifest
    faiss_dir = PROJECT_ROOT / "data" / "processed" / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = faiss_dir / "mapping_batch1.npy"
    np.save(mapping_path, mapping)
    print(f"  [Mapping] Saved FAISS mapping array ({len(mapping):,} entries) to {mapping_path}")

    # Build manifest
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifests" / "siglip_batch1_v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "index_name": "siglip_batch1_v1",
        "model": "google/siglip2-base-patch16-224",
        "model_version": "v1.0",
        "dimension": 768,
        "metric": "inner_product (cosine)",
        "batch_id": "batch1",
        "vector_count": len(mapping),
        "mapping_file": str(mapping_path.relative_to(PROJECT_ROOT)),
        "created_at": "2026-08-25"
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  [Manifest] Saved index manifest to {manifest_path}")

    # Verification counts
    v_count = db.execute_single("SELECT COUNT(*) as c FROM videos")["c"]
    f_count = db.execute_single("SELECT COUNT(*) as c FROM frames")["c"]
    print(f"\nVerification:")
    print(f"  - Total Videos in SQLite: {v_count}")
    print(f"  - Total Canonical Frames in SQLite: {f_count:,}")
    print("\nIngestion completed successfully!")

if __name__ == "__main__":
    main()
