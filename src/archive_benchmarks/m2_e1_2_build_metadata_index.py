import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Sử dụng orjson nếu có để parse JSON cực nhanh, nếu không fallback về json thuần
try:
    import json as json_parser
    def load_json(fp):
        with open(fp, "rb") as f:
            return json_parser.loads(f.read())
except ImportError:
    import json as json_parser
    def load_json(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return json_parser.load(f)

OBJ_DIR = "data/raw/objects"
DB_PATH = "data/processed/metadata.db"

def process_single_video(args):
    """Hàm xử lý độc lập cho 1 video (chạy trên 1 core CPU)"""
    video_id, video_path = args
    rows = []
    frames_count = 0

    # Lấy danh sách file json
    try:
        filenames = sorted(os.listdir(video_path))
    except Exception:
        return video_id, [], 0

    for fname in filenames:
        if not fname.endswith(".json"):
            continue

        frame_n = int(fname[:-5]) # Nhanh hơn .replace(".json", "")
        fpath = os.path.join(video_path, fname)

        try:
            data = load_json(fpath)
        except Exception:
            continue

        entities = data.get("detection_class_entities", [])
        scores = data.get("detection_scores", [])
        boxes = data.get("detection_boxes", [])

        for i, ent in enumerate(entities):
            score = float(scores[i]) if i < len(scores) else 0.0
            box = boxes[i] if i < len(boxes) else [None, None, None, None]
            try:
                ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            except Exception:
                ymin = xmin = ymax = xmax = None

            rows.append((video_id, frame_n, str(ent).lower(), score, ymin, xmin, ymax, xmax))

        frames_count += 1

    return video_id, rows, frames_count


def build_index():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print(f"Building metadata index at {DB_PATH}...")

    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-512000")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            video_id     TEXT NOT NULL,
            frame_n      INTEGER NOT NULL,
            class_entity TEXT NOT NULL,
            score        REAL NOT NULL,
            ymin REAL, xmin REAL, ymax REAL, xmax REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS build_progress (
            video_id TEXT PRIMARY KEY
        )
    """)
    conn.isolation_level = None  # autocommit mode - explicit BEGIN/COMMIT control
    try:
        conn.execute("ROLLBACK")  # close any open txn from prev crash, safe to ignore
    except Exception:
        pass

    done = set(r[0] for r in conn.execute("SELECT video_id FROM build_progress"))
    print(f"Resuming: {len(done)} videos already indexed.")

    # Chuẩn bị danh sách video chưa làm
    all_videos = []
    for vid in sorted(os.listdir(OBJ_DIR)):
        vpath = os.path.join(OBJ_DIR, vid)
        if os.path.isdir(vpath) and vid not in done:
            all_videos.append((vid, vpath))

    total_files = 0
    video_count = 0
    start_t = time.time()

    # Sử dụng Multi-processing để đọc/parse file song song trên tất cả các nhân CPU
    max_workers = os.cpu_count() or 4
    print(f"Processing using {max_workers} CPU workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit công việc theo nhóm (chunk)
        futures = [executor.submit(process_single_video, item) for item in all_videos]

        for future in as_completed(futures):
            video_id, rows, frames_count = future.result()

            conn.execute("BEGIN")
            if rows:
                conn.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?)", rows)
            conn.execute("INSERT OR IGNORE INTO build_progress VALUES (?)", (video_id,))
            conn.execute("COMMIT")

            total_files += frames_count
            video_count += 1

            if video_count % 100 == 0 or video_count == len(all_videos):
                elapsed = time.time() - start_t
                print(f"  [{video_count}/{len(all_videos)} videos | {total_files} frames | {elapsed:.0f}s]")

    print("Building indexes & Analyzing...")
    conn.execute("PRAGMA synchronous=NORMAL") # Trả về NORMAL an toàn
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vid_frame ON detections(video_id, frame_n)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_class ON detections(class_entity)")
    conn.execute("ANALYZE;") # Giúp SQLite tối ưu query speed dựa trên thống kê dữ liệu
    conn.commit()
    conn.close()

    elapsed = time.time() - start_t
    print(f"Done. {total_files} frames indexed in {elapsed:.1f}s. DB: {DB_PATH}")


if __name__ == "__main__":
    build_index()