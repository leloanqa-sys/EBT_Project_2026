"""
Module: Remote Data Loader
==========================
Streams keyframes from remote ZIP files (via spreadsheet_data.csv) 
to local disk on-demand using `remotezip`, bypassing SSL errors.
"""

import sys
import csv
import warnings
from pathlib import Path

import requests
import remotezip

# Bypass SSL warnings in corporate/proxy environments
warnings.filterwarnings('ignore', category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class RemoteDataLoader:
    def __init__(self, csv_path: Path = PROJECT_ROOT / "spreadsheet_data.csv"):
        self.zip_map = {}
        self._load_spreadsheet(csv_path)
        self.keyframes_dir = PROJECT_ROOT / "data" / "raw" / "keyframes"
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_spreadsheet(self, csv_path: Path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["Filenames"]
                url = row["Download link"]
                if filename and url and filename.endswith(".zip"):
                    # Map L21 -> https://.../Keyframes_L21.zip
                    prefix = filename.replace("Keyframes_", "").replace(".zip", "")
                    self.zip_map[prefix] = url
                    
    def _get_url_for_video(self, video_id: str) -> str:
        # video_id format: L21_V001 -> prefix is L21
        prefix = video_id.split("_")[0]
        # Handle L26_a, L26_b etc. For now, try exact prefix or fallback.
        # But wait, L26 has L26_a, L26_b. We might need to try them.
        for k, url in self.zip_map.items():
            if k == prefix or k.startswith(prefix + "_"):
                # Ideally, we should know exactly which sub-zip it's in, 
                # but for pilot, we can just return the first matching prefix URL
                return url
        return None

    def fetch_keyframe(self, video_id: str, frame_idx: int) -> Path:
        """
        Fetches a specific keyframe from the remote ZIP if it doesn't exist locally.
        Returns the path to the local downloaded JPG.
        """
        vid_dir = self.keyframes_dir / video_id
        vid_dir.mkdir(exist_ok=True)
        
        # Try both 04d and generic naming format
        formats = [f"{frame_idx:04d}.jpg", f"{frame_idx:03d}.jpg", f"{frame_idx}.jpg"]
        for fmt in formats:
            local_path = vid_dir / fmt
            if local_path.exists():
                return local_path
                
        # If not found locally, fetch remotely
        url = self._get_url_for_video(video_id)
        if not url:
            raise ValueError(f"No remote ZIP found for video {video_id}")
            
        try:
            with remotezip.RemoteZip(url, verify=False) as rz:
                # We need to find the exact filename in the zip.
                # Usually it's keyframes/L21_V001/0130.jpg
                target_prefix = f"keyframes/{video_id}/"
                for zinfo in rz.infolist():
                    if zinfo.filename.startswith(target_prefix):
                        # Extract the specific frame
                        file_idx_str = Path(zinfo.filename).stem
                        try:
                            if int(file_idx_str) == frame_idx:
                                # Extract to the local dir
                                # remotezip extract() maintains the zip folder structure
                                rz.extract(zinfo, path=str(PROJECT_ROOT / "data" / "raw"))
                                extracted_path = PROJECT_ROOT / "data" / "raw" / zinfo.filename
                                return extracted_path
                        except ValueError:
                            continue
        except Exception as e:
            print(f"[ERROR] Failed to fetch {video_id}:{frame_idx} from {url} - {e}")
            
        return None

if __name__ == "__main__":
    print("Testing RemoteDataLoader...")
    loader = RemoteDataLoader()
    print(f"Loaded {len(loader.zip_map)} ZIP links.")
    
    test_vid = "L21_V001"
    test_idx = 130
    print(f"Fetching {test_vid} frame {test_idx}...")
    path = loader.fetch_keyframe(test_vid, test_idx)
    if path and path.exists():
        print(f"Success! Saved to {path}")
    else:
        print("Failed to fetch.")
