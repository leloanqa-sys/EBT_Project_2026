import os
import sqlite3
import pandas as pd
from src.role_a_retrieval.mapping_utils import build_global_mapping

def create_schema(db_path: str = "data/processed/media.db"):
    """
    Creates SQLite schema with indexes on key attributes.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS keyframes;")
    cursor.execute("""
        CREATE TABLE keyframes (
            faiss_id INTEGER PRIMARY KEY,
            video_id TEXT NOT NULL,
            n INTEGER NOT NULL,
            frame_idx INTEGER NOT NULL,
            pts_time REAL NOT NULL,
            fps REAL NOT NULL
        );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON keyframes (video_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_idx ON keyframes (video_id, frame_idx);")
    conn.commit()
    conn.close()

def build_sqlite_db(mapping_csv_path: str = "data/processed/global_mapping.csv",
                    db_path: str = "data/processed/media.db"):
    """
    Cold-Path SQLite builder. Bulk inserts global_mapping.csv into media.db.
    """
    if not os.path.exists(mapping_csv_path):
        print(f"Mapping CSV not found at {mapping_csv_path}, building global mapping first...")
        mapping_df = build_global_mapping()
    else:
        mapping_df = pd.read_csv(mapping_csv_path)

    create_schema(db_path)

    conn = sqlite3.connect(db_path)
    print(f"--- [SQLite Build] Bulk inserting {len(mapping_df)} rows into {db_path} ---")

    mapping_df.to_sql("keyframes", conn, if_exists="append", index=False, chunksize=5000, method="multi")

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM keyframes;")
    count = cursor.fetchone()[0]
    conn.close()

    print(f"--- [SQLite Build] Success! Table 'keyframes' verified with {count} rows. ---")

if __name__ == "__main__":
    build_sqlite_db()
