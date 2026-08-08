import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def get_sorted_video_ids(features_dir: str = "data/raw/clip-features-32") -> List[str]:
    """
    Scans features_dir for .npy files and returns a deterministically sorted list of video IDs.
    """
    npy_files = glob.glob(os.path.join(features_dir, "*.npy"))
    video_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in npy_files])
    return video_ids

def run_sanity_check(features_dir: str = "data/raw/clip-features-32",
                     map_dir: str = "data/raw/map-keyframes",
                     min_videos: int = 1) -> Dict[str, int]:
    """
    Performs full sanity check across all video datasets.
    Asserts features.shape[0] == len(df) for 100% of videos.
    If any mismatch or missing file occurs, prints details and performs sys.exit(1).
    """
    print(f"--- [Sanity Check] Scanning {features_dir} & {map_dir} ---")
    npy_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    csv_files = sorted(glob.glob(os.path.join(map_dir, "*.csv")))

    if len(npy_files) < min_videos or len(csv_files) < min_videos:
        print(f"CRITICAL ERROR: Found {len(npy_files)} .npy files and {len(csv_files)} .csv files. Expected at least {min_videos}.")
        sys.exit(1)

    npy_videos = {os.path.splitext(os.path.basename(f))[0]: f for f in npy_files}
    csv_videos = {os.path.splitext(os.path.basename(f))[0]: f for f in csv_files}

    mismatched = []
    missing = []
    total_keyframes = 0

    for vid_id in sorted(npy_videos.keys()):
        if vid_id not in csv_videos:
            missing.append(vid_id)
            continue

        npy_path = npy_videos[vid_id]
        csv_path = csv_videos[vid_id]

        try:
            feats = np.load(npy_path, mmap_mode="r")
            df = pd.read_csv(csv_path)

            n_npy = feats.shape[0]
            n_csv = len(df)

            if n_npy != n_csv:
                mismatched.append((vid_id, n_npy, n_csv))
            else:
                total_keyframes += n_npy

        except Exception as e:
            print(f"ERROR reading {vid_id}: {e}")
            sys.exit(1)

    if missing or mismatched:
        print("CRITICAL SANITY CHECK FAILURES:")
        if missing:
            print(f"Missing CSV files for video IDs ({len(missing)}): {missing[:5]}...")
        if mismatched:
            print(f"Mismatched keyframe counts ({len(mismatched)} videos):")
            for vid_id, n_npy, n_csv in mismatched[:10]:
                print(f"  - {vid_id}: .npy has {n_npy} rows | .csv has {n_csv} rows")
        sys.exit(1)

    print(f"--- [Sanity Check] PASS 100%! Tested {len(npy_videos)} videos, total {total_keyframes} keyframes. ---")
    return {"total_videos": len(npy_videos), "total_keyframes": total_keyframes}

def build_global_mapping(features_dir: str = "data/raw/clip-features-32",
                         map_dir: str = "data/raw/map-keyframes",
                         output_dir: str = "data/processed") -> pd.DataFrame:
    """
    Builds single source of truth global mapping and derived artifacts:
    1. global_mapping.csv
    2. video_id_order.json
    3. video_to_faiss_range.json
    4. mapping_array.npz (NumPy Columnar Hot-Path Lookup)
    """
    run_sanity_check(features_dir=features_dir, map_dir=map_dir)
    os.makedirs(output_dir, exist_ok=True)

    video_ids = get_sorted_video_ids(features_dir)
    order_json_path = os.path.join(output_dir, "video_id_order.json")
    with open(order_json_path, "w", encoding="utf-8") as f:
        json.dump(video_ids, f, indent=2)

    all_rows = []
    video_range_map = {}

    current_faiss_id = 0

    video_idx_list = []
    frame_idx_list = []
    pts_time_list = []
    fps_list = []

    for vid_idx, vid_id in enumerate(video_ids):
        csv_path = os.path.join(map_dir, f"{vid_id}.csv")
        df = pd.read_csv(csv_path)

        n_rows = len(df)
        start_id = current_faiss_id
        end_id = current_faiss_id + n_rows - 1

        video_range_map[vid_id] = [start_id, end_id]

        for idx, row in df.iterrows():
            faiss_id = current_faiss_id
            frame_idx = int(row.get("frame_idx", row.name))
            pts_time = float(row.get("pts_time", 0.0))
            fps = float(row.get("fps", 0.0))
            n_val = int(row.get("n", idx + 1))

            all_rows.append({
                "faiss_id": faiss_id,
                "video_id": vid_id,
                "n": n_val,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "fps": fps
            })

            video_idx_list.append(vid_idx)
            frame_idx_list.append(frame_idx)
            pts_time_list.append(pts_time)
            fps_list.append(fps)

            current_faiss_id += 1

    mapping_df = pd.DataFrame(all_rows)

    # Save global_mapping.csv
    csv_out_path = os.path.join(output_dir, "global_mapping.csv")
    mapping_df.to_csv(csv_out_path, index=False)

    # Save video_to_faiss_range.json
    range_json_path = os.path.join(output_dir, "video_to_faiss_range.json")
    with open(range_json_path, "w", encoding="utf-8") as f:
        json.dump(video_range_map, f, indent=2)

    # Save NumPy Columnar Hot-Path NPZ
    npz_out_path = os.path.join(output_dir, "mapping_array.npz")
    np.savez_compressed(
        npz_out_path,
        video_ids=np.array(video_ids),
        video_idx=np.array(video_idx_list, dtype=np.uint16),
        frame_indices=np.array(frame_idx_list, dtype=np.int32),
        pts_times=np.array(pts_time_list, dtype=np.float32),
        fps=np.array(fps_list, dtype=np.float32)
    )

    print(f"--- [Global Mapping Built] Total rows: {len(mapping_df)} ---")
    print(f"Saved artifacts to {output_dir}: global_mapping.csv, video_id_order.json, video_to_faiss_range.json, mapping_array.npz")
    return mapping_df

if __name__ == "__main__":
    build_global_mapping()
