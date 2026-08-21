import sqlite3

conn = sqlite3.connect("data/processed/metadata.db")
c = conn.cursor()
c.execute("PRAGMA index_list('detections')")
print("Indices:", c.fetchall())

# Test single indexed query
c.execute("SELECT video_id, frame_n, score FROM detections WHERE class_entity = 'skateboard' LIMIT 5")
print("Skateboard samples:", c.fetchall())

c.execute("SELECT video_id, frame_n, score FROM detections WHERE class_entity = 'bicycle' LIMIT 5")
print("Bicycle samples:", c.fetchall())
conn.close()
