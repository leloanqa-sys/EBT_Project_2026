"""
Plan B Fix: Migrate metadata.db frame_n từ sequential index → absolute frame_idx

Chạy:
  --dry-run : Kiểm tra nhưng không thay đổi gì
  --apply   : Thực sự migrate
  --verify  : Verify sau khi đã apply
"""
import os
import sys
import shutil
import sqlite3
import numpy as np
import argparse
from datetime import datetime

NPZ_PATH = "data/processed/mapping_array.npz"
DB_PATH  = "data/processed/metadata.db"
BACKUP_PATH = f"data/processed/metadata.db.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BATCH_SIZE = 50_000


def load_npz_mapping():
    """Trả về dict: {video_id: [abs_frame_idx_0, abs_frame_idx_1, ...]} (0-based thứ tự keyframe)."""
    data = np.load(NPZ_PATH, allow_pickle=True)
    video_ids = list(data['video_ids'])
    video_idx_arr = data['video_idx']
    frame_indices = data['frame_indices']

    mapping = {}
    for vid_i, vid_id in enumerate(video_ids):
        mask = video_idx_arr == vid_i
        frames = frame_indices[mask].tolist()  # ordered by faiss_id (sequential)
        mapping[str(vid_id)] = frames
    return mapping


def dry_run(mapping):
    """Kiểm tra tính nhất quán giữa DB và NPZ. Không thay đổi gì."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print(f"\n{'='*60}")
    print("DRY RUN — Kiểm tra nhất quán DB vs NPZ")
    print(f"{'='*60}")
    print(f"{'Video ID':<20} {'n_faiss':>8} {'n_db':>8} {'Status':>12}")
    print("-" * 55)

    mismatches = []
    ok_count = 0

    for vid_id, frames in sorted(mapping.items()):
        n_faiss = len(frames)
        cur.execute("SELECT COUNT(DISTINCT frame_n) FROM detections WHERE video_id=?", (vid_id,))
        n_db = cur.fetchone()[0]

        if n_faiss != n_db:
            status = "❌ MISMATCH"
            mismatches.append((vid_id, n_faiss, n_db))
        else:
            status = "✅ OK"
            ok_count += 1

        # Only print first 20 and mismatches
        if ok_count <= 20 or n_faiss != n_db:
            print(f"{vid_id:<20} {n_faiss:>8} {n_db:>8} {status:>12}")

    conn.close()

    print(f"\n{'='*60}")
    print(f"TỔNG KẾT: {ok_count}/{len(mapping)} videos OK")
    if mismatches:
        print(f"⚠️  {len(mismatches)} video(s) bị MISMATCH — xem chi tiết trên")
        print("\nDETAIL MISMATCHES:")
        for vid, nf, nd in mismatches:
            print(f"  {vid}: faiss={nf} db={nd}")
        print("\n❌ KHÔNG AN TOÀN để migrate! Cần điều tra trước.")
        return False
    else:
        print(f"\n✅ Tất cả {len(mapping)} videos đều nhất quán. An toàn để --apply.")
        return True


def backup_db():
    """Backup DB trước khi migrate."""
    db_size_gb = os.path.getsize(DB_PATH) / (1024**3)
    free_space = shutil.disk_usage(os.path.dirname(os.path.abspath(DB_PATH))).free / (1024**3)

    print(f"\nDB size  : {db_size_gb:.2f} GB")
    print(f"Free disk: {free_space:.2f} GB")

    if free_space < db_size_gb * 1.2:
        print(f"⚠️  Không đủ disk space để backup! Cần ít nhất {db_size_gb*1.2:.1f} GB.")
        resp = input("Tiếp tục mà không backup? (yes/NO): ").strip().lower()
        if resp != "yes":
            sys.exit(1)
        return None

    print(f"Backing up → {BACKUP_PATH} ...")
    shutil.copy2(DB_PATH, BACKUP_PATH)
    print(f"✅ Backup xong: {BACKUP_PATH}")
    return BACKUP_PATH


def apply_migration(mapping):
    """Thực sự migrate frame_n trong detections."""
    print(f"\n{'='*60}")
    print("APPLYING MIGRATION...")
    print(f"{'='*60}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-131072")  # 128MB cache
    cur = conn.cursor()

    total_videos = len(mapping)
    total_updated = 0

    for vid_idx, (vid_id, frames) in enumerate(sorted(mapping.items())):
        # Build map: {seq_n (1-based) -> abs_frame_idx}
        seq_to_abs = {(i + 1): abs_idx for i, abs_idx in enumerate(frames)}

        try:
            conn.execute("BEGIN")

            # Lấy tất cả frame_n distinct của video này
            cur.execute(
                "SELECT DISTINCT frame_n FROM detections WHERE video_id=? ORDER BY frame_n",
                (vid_id,)
            )
            db_frame_ns = [r[0] for r in cur.fetchall()]

            # Validate: mỗi frame_n trong DB phải có mapping
            for fn in db_frame_ns:
                if fn not in seq_to_abs:
                    conn.rollback()
                    print(f"❌ ROLLBACK: {vid_id} frame_n={fn} không có trong NPZ mapping! Dừng lại.")
                    conn.close()
                    sys.exit(1)

            # Batch UPDATE: đổi frame_n từ sequential → absolute
            # Dùng temporary rename trick để tránh collision (1→90, 2→239, etc.)
            # Bước 1: Rename sang negative (tránh conflict nếu new value trùng old value)
            batch_neg = []
            for fn in db_frame_ns:
                batch_neg.append((-fn, vid_id, fn))

            for i in range(0, len(batch_neg), BATCH_SIZE):
                chunk = batch_neg[i:i+BATCH_SIZE]
                cur.executemany(
                    "UPDATE detections SET frame_n=? WHERE video_id=? AND frame_n=?",
                    chunk
                )

            # Bước 2: Rename từ negative → absolute frame_idx
            batch_abs = []
            for fn in db_frame_ns:
                abs_val = seq_to_abs[fn]
                batch_abs.append((abs_val, vid_id, -fn))

            for i in range(0, len(batch_abs), BATCH_SIZE):
                chunk = batch_abs[i:i+BATCH_SIZE]
                cur.executemany(
                    "UPDATE detections SET frame_n=? WHERE video_id=? AND frame_n=?",
                    chunk
                )

            conn.commit()
            total_updated += len(db_frame_ns)

            if (vid_idx + 1) % 50 == 0 or (vid_idx + 1) == total_videos:
                pct = (vid_idx + 1) / total_videos * 100
                print(f"  [{vid_idx+1:4d}/{total_videos}] {vid_id:<20} — {pct:.1f}% done, updated {total_updated} frame_n keys")

        except Exception as e:
            conn.rollback()
            print(f"❌ ERROR on {vid_id}: {e}. ROLLBACK applied.")
            conn.close()
            sys.exit(1)

    # Rebuild index after mass update
    print("\nRebuilding SQLite index...")
    conn.execute("ANALYZE")
    conn.close()

    print(f"\n✅ MIGRATION COMPLETE! Updated {total_updated} unique frame_n keys across {total_videos} videos.")


def verify(mapping):
    """Verify: query bằng absolute frame_idx phải trả về data."""
    print(f"\n{'='*60}")
    print("VERIFY — Kiểm tra ngẫu nhiên 10 videos")
    print(f"{'='*60}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    import random
    sample_vids = random.sample(list(mapping.keys()), min(10, len(mapping)))

    all_ok = True
    for vid_id in sample_vids:
        frames = mapping[vid_id]
        # Lấy frame thứ 3 (nếu có)
        test_frame = frames[min(2, len(frames)-1)]

        cur.execute(
            "SELECT COUNT(*) FROM detections WHERE video_id=? AND frame_n=?",
            (vid_id, test_frame)
        )
        count = cur.fetchone()[0]
        status = "✅" if count > 0 else "❌"
        if count == 0:
            all_ok = False
        print(f"  {status} {vid_id:<20} frame_n={test_frame:6d} → {count} rows")

    conn.close()

    if all_ok:
        print("\n✅ Verify PASSED! metadata.db đã được migrate thành công.")
    else:
        print("\n❌ Verify FAILED! Một số frame vẫn không có data.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Fix metadata.db frame_n mapping")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra, không thay đổi")
    parser.add_argument("--apply", action="store_true", help="Thực sự migrate")
    parser.add_argument("--verify", action="store_true", help="Verify sau khi apply")
    args = parser.parse_args()

    print(f"Loading NPZ mapping from {NPZ_PATH}...")
    mapping = load_npz_mapping()
    print(f"Loaded {len(mapping)} videos from NPZ.")

    if args.dry_run:
        dry_run(mapping)
    elif args.apply:
        ok = dry_run(mapping)
        if not ok:
            print("\n❌ Dry-run failed. Hãy sửa mismatch trước khi apply.")
            sys.exit(1)
        backup_db()
        apply_migration(mapping)
        verify(mapping)
    elif args.verify:
        verify(mapping)
    else:
        print("Dùng --dry-run, --apply, hoặc --verify")
        parser.print_help()


if __name__ == "__main__":
    main()
