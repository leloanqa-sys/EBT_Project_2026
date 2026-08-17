import os
import json
import sqlite3
import logging
from typing import Set

logger = logging.getLogger(__name__)

DB_PATH = "data/processed/metadata.db"
CACHE_PATH = "data/taxonomy/detected_classes_cache.json"

def load_detected_taxonomy(db_path: str = DB_PATH, force_reload: bool = False) -> Set[str]:
    """
    Tầng 1 (Toàn tập Ground Truth): Truy xuất trực tiếp các class_entity đã từng xuất hiện thật
    trong metadata.db do BTC cung cấp. Trả về set các nhãn đã lower().strip().
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    
    if not os.path.exists(db_path):
        logger.error(f"Không tìm thấy DB {db_path}. Fallback về set rỗng.")
        return set()
        
    try:
        conn = sqlite3.connect(db_path)
        current_row_count = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    except Exception as e:
        logger.error(f"Lỗi kết nối DB {db_path}: {e}")
        return set()

    prev_classes_count = 0
    # Đọc cache nếu có và không bị force_reload
    if not force_reload and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cached_row_count = data.get("db_row_count", -1)
                
                if cached_row_count == current_row_count:
                    # Cache hợp lệ
                    return set(data.get("classes", []))
                else:
                    prev_classes_count = data.get("count", 0)
                    logger.info(f"DB row count thay đổi ({cached_row_count} -> {current_row_count}). Tiến hành quét lại taxonomy.")
        except Exception as e:
            logger.warning(f"Lỗi đọc taxonomy cache: {e}. Tiến hành quét lại DB.")
            
    # Quét lại từ metadata.db
    try:
        # Quét DISTINCT
        rows = conn.execute("SELECT DISTINCT class_entity FROM detections").fetchall()
        
        classes_set = set()
        skipped_count = 0
        for r in rows:
            val = r[0]
            if val is not None and str(val).strip() != "":
                classes_set.add(str(val).lower().strip())
            else:
                skipped_count += 1
                
        if skipped_count > 0:
            logger.debug(f"Skipped {skipped_count} empty/null class_entity rows")
            
        current_classes_count = len(classes_set)
        if prev_classes_count > 0:
            delta = current_classes_count - prev_classes_count
            logger.info(f"Taxonomy updated - previous: {prev_classes_count} classes, now: {current_classes_count} classes — delta: {delta:+} classes")
            
        # Lưu cache
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                "classes": list(classes_set), 
                "count": current_classes_count,
                "db_row_count": current_row_count
            }, f, indent=2)
            
        logger.info(f"Đã load và cache {current_classes_count} unique taxonomy classes từ {db_path}")
        return classes_set
    except Exception as e:
        logger.error(f"Lỗi quét taxonomy từ DB: {e}")
        return set()

# Khởi tạo singleton lúc load module
TAXONOMY_CLASSES_SET = load_detected_taxonomy()
