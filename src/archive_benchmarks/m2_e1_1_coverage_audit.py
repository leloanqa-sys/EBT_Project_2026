import os
import json
import time
from concurrent.futures import ProcessPoolExecutor

def process_file(file_path):
    unique_classes = set()
    total_objects = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            entities = data.get("detection_class_entities", [])
            total_objects = len(entities)
            for ent in entities:
                unique_classes.add(str(ent).lower())
        return total_objects, unique_classes, True
    except Exception:
        return 0, set(), False

def audit_coverage_fast():
    obj_dir = "data/raw/objects"
    start_t = time.time()
    
    # Gom tất cả file path trước
    file_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(obj_dir)
        for file in files if file.endswith(".json")
    ]
    
    total_files = len(file_paths)
    total_objects = 0
    unique_classes = set()
    
    # Chạy đa nhân CPU (Multi-processing)
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file, file_paths, chunksize=500)
        
        for objs, classes, success in results:
            if success:
                total_objects += objs
                unique_classes.update(classes)

    elapsed = time.time() - start_t
    print(f"--- Fast Coverage Audit ---")
    print(f"Total JSON Files Scanned: {total_files}")
    print(f"Total Detected Objects: {total_objects}")
    print(f"Unique Object Classes: {len(unique_classes)}")
    print(f"Audit Time: {elapsed:.1f} seconds")

if __name__ == "__main__":
    audit_coverage_fast()
