import sqlite3
import os
import json

conn = sqlite3.connect("data/processed/metadata.db")
c = conn.cursor()

queries_test = [
    ("query-p1-6-kis", ["flower", "dish", "plate", "food"]),
    ("query-p1-10-kis", ["bookcase", "shelf", "musical instrument", "drum"]),
    ("query-p1-14-kis", ["skateboard", "roller skate", "sculpture", "statue"]),
    ("query-p1-24-kis", ["bicycle", "helmet", "person"])
]

for qid, entities in queries_test:
    placeholders = ",".join(["?"] * len(entities))
    query = f"""
        SELECT video_id, frame_n, count(DISTINCT class_entity) as matched_count, group_concat(DISTINCT class_entity) as matched_classes
        FROM detections
        WHERE class_entity IN ({placeholders}) AND score > 0.3
        GROUP BY video_id, frame_n
        HAVING matched_count >= 2
        ORDER BY matched_count DESC
        LIMIT 5
    """
    c.execute(query, entities)
    rows = c.fetchall()
    print(f"\n[{qid}] Entity search {entities}:")
    for r in rows:
        print("  -> Video:", r[0], "Frame_N:", r[1], "Matched:", r[2], "Classes:", r[3])

conn.close()
