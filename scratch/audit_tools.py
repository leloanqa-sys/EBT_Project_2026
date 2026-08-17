import sqlite3, os, json, glob, statistics

conn = sqlite3.connect('data/processed/metadata.db')

print('=== SCORE DISTRIBUTION by class ===')
for cls in ['human face', 'man', 'woman', 'clothing', 'person']:
    rows = conn.execute('SELECT score FROM detections WHERE class_entity=? LIMIT 100000', (cls,)).fetchall()
    s = [r[0] for r in rows]
    if s:
        print(f'  {cls:<16} n={len(s):>8,}  mean={statistics.mean(s):.3f}  p90={sorted(s)[int(len(s)*0.9)]:.3f}  max={max(s):.3f}')

print()
print('=== UNSUPPORTED plan_steps in recent traces ===')
op_freq = {}
for f in glob.glob('outputs/traces/*.json'):
    try:
        d = json.load(open(f, encoding='utf-8'))
        for s in d.get('plan_steps', []):
            status = s.get('status', '')
            op = s.get('op', '')
            if status in ('UNSUPPORTED', 'EXPERIMENTAL', 'UNTESTABLE', 'SKIPPED'):
                op_freq[op] = op_freq.get(op, 0) + 1
    except:
        pass
for op, cnt in sorted(op_freq.items(), key=lambda x: -x[1]):
    print(f'  {op:<30} x{cnt}')

print()
print('=== CAPABILITY REGISTRY ===')
cap_file = 'src/role_c_logic/capability_registry.py'
if os.path.exists(cap_file):
    for line in open(cap_file, encoding='utf-8'):
        if 'UNSUPPORTED' in line or 'EXPERIMENTAL' in line or 'READY' in line:
            print(' ', line.rstrip())

conn.close()
