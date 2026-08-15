import sqlite3, time

conn = sqlite3.connect('data/processed/metadata.db')
rows = conn.execute('SELECT COUNT(*) FROM detections').fetchone()[0]
videos = conn.execute('SELECT COUNT(DISTINCT video_id) FROM detections').fetchone()[0]
print(f'Total rows: {rows:,}')
print(f'Unique videos: {videos}')

t = time.time()
result = conn.execute("SELECT class_entity, score, ymin, xmin, ymax, xmax FROM detections WHERE video_id=? AND frame_n=?", ('L26_V444', 1)).fetchall()
elapsed = (time.time() - t) * 1000
print(f'Single frame lookup: {elapsed:.1f}ms, {len(result)} rows')
print('Sample:', result[:3])
conn.close()
