import os
import json
import numpy as np
import pandas as pd

def audit_id_integrity(npz_path: str = "data/processed/mapping_array.npz", 
                      csv_dir: str = "data/raw/map-keyframes", 
                      obj_dir: str = "data/raw/objects",
                      sample_size: int = 5):
    print(f"--- M2-D.0: Dataset/ID Integrity Audit ---")
    print(f"Loading mapping array from {npz_path}...")
    
    if not os.path.exists(npz_path):
        print(f"[FAIL] {npz_path} not found.")
        return False
        
    npz_data = np.load(npz_path, allow_pickle=True)
    video_ids = npz_data["video_ids"]
    video_idx = npz_data["video_idx"]
    frame_indices = npz_data["frame_indices"]
    pts_times = npz_data["pts_times"]
    
    total_vectors = len(frame_indices)
    print(f"Total FAISS vectors (frames) mapped: {total_vectors}")
    
    # Pick random FAISS vector IDs
    np.random.seed(42) # For reproducibility
    test_faiss_ids = np.random.choice(total_vectors, min(sample_size, total_vectors), replace=False)
    
    all_passed = True
    
    for faiss_id in test_faiss_ids:
        print(f"\n[Audit Chain] FAISS vector_id: {faiss_id}")
        
        # 1. FAISS -> NPZ Mapping
        v_idx = video_idx[faiss_id]
        vid_id = video_ids[v_idx]
        frame_idx = frame_indices[faiss_id]
        pts = pts_times[faiss_id]
        print(f"  -> Maps to Video: {vid_id}, Frame ID: {frame_idx}, PTS: {pts}")
        
        # 2. Check CSV
        csv_path = os.path.join(csv_dir, f"{vid_id}.csv")
        if not os.path.exists(csv_path):
            print(f"  -> [FAIL] CSV mapping not found: {csv_path}")
            all_passed = False
            continue
            
        try:
            df = pd.read_csv(csv_path)
            # Find the row with this frame_idx
            matching_rows = df[df['frame_idx'] == frame_idx]
            if matching_rows.empty:
                print(f"  -> [FAIL] Frame {frame_idx} NOT FOUND in CSV: {csv_path}")
                all_passed = False
                continue
            else:
                csv_pts = matching_rows.iloc[0]['pts_time']
                print(f"  -> [PASS] Frame {frame_idx} exists in CSV. CSV PTS: {csv_pts:.2f} | NPZ PTS: {pts:.2f}")
                # check round-trip
                csv_n = matching_rows.iloc[0]['n']
        except Exception as e:
            print(f"  -> [FAIL] Error reading CSV: {e}")
            all_passed = False
            continue
            
        # JSON object files are named by the 'n' column padded with 3 zeros (e.g. 001.json)
        obj_path_n = os.path.join(obj_dir, vid_id, f"{int(csv_n):03d}.json")
        
        json_path_found = None
        if os.path.exists(obj_path_n):
            json_path_found = obj_path_n
            
        if json_path_found:
            print(f"  -> [PASS] JSON object exists: {json_path_found}")
        else:
            print(f"  -> [WARN] JSON object not found at {obj_path_n}")
            # we don't necessarily fail on JSON because some frames might not have detected objects
            
    print(f"\n--- Audit Summary ---")
    if all_passed:
        print("[VERDICT: PASS] ID Chain Integrity is verified.")
    else:
        print("[VERDICT: FAIL] Broken ID mapping chain detected. STOP M2-D.")
        
    return all_passed

if __name__ == "__main__":
    audit_id_integrity()
