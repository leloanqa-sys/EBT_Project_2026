import sqlite3

conn = sqlite3.connect('data/processed/metadata.db')

man_syns = {'person', 'man', 'human face', 'human head', 'clothing'}
shirt_syns = {'shirt', 'clothing'}

print('=== obj_score simulation (current logic, no color check) ===')
print('Query: man + green shirt')
print()

rows_31 = conn.execute('SELECT class_entity, score FROM detections WHERE video_id=? AND frame_n=?',
                       ('L21_V031', 10682)).fetchall()

man_best = max((r[1] for r in rows_31 if r[0].lower() in man_syns), default=0.0)
shirt_best = max((r[1] for r in rows_31 if r[0].lower() in shirt_syns), default=0.0)
obj_combined = (man_best + shirt_best) / 2
print(f'L21_V031 #10682: man_best={man_best:.4f}, shirt_best={shirt_best:.4f}, obj_score={obj_combined:.4f}')
print()

print('=== ROOT CAUSE: DOUBLE DIPPING VIA SHARED ALIAS ===')
print('man_syns intersection shirt_syns:', man_syns.intersection(shirt_syns))
print('-> clothing score 0.477 is counted BOTH for man AND shirt')
print('-> man_best driven by human_face=0.972 (face != green shirt)')
print('-> obj_score of 0.97 means NOTHING about green color')
print()

print('=== BEACH scenario: what labels exist in DB? ===')
rows = conn.execute("""
    SELECT DISTINCT class_entity FROM detections
    WHERE class_entity LIKE '%beach%' OR class_entity LIKE '%sea%'
       OR class_entity LIKE '%ocean%' OR class_entity LIKE '%sand%'
       OR class_entity LIKE '%water%'
    LIMIT 20
""").fetchall()
print('Beach-related classes in DB:', [r[0] for r in rows])
print()

# Distribution of person vs environment
total_rows = conn.execute('SELECT COUNT(*) FROM detections').fetchone()[0]
person_rows = conn.execute(
    "SELECT COUNT(*) FROM detections WHERE class_entity IN ('person','man','woman','human face','human head','clothing')"
).fetchone()[0]
print(f'Total detections: {total_rows:,}')
print(f'Person-class detections: {person_rows:,} ({person_rows/total_rows*100:.1f}%)')
print()

# Top 20 most common labels in DB
print('=== TOP 20 most frequent labels in DB ===')
top = conn.execute(
    'SELECT class_entity, COUNT(*) as cnt FROM detections GROUP BY class_entity ORDER BY cnt DESC LIMIT 20'
).fetchall()
for label, cnt in top:
    pct = cnt / total_rows * 100
    print(f'  {pct:5.1f}%  {cnt:>8,}  {label}')

conn.close()
