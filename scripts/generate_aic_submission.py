import os
import sys
import zipfile
import shutil
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env or .env.example manually to bypass dotenv dependency
for env_file in [".env", ".env.example"]:
    env_path = PROJECT_ROOT / env_file
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

# Force Disable SSL Verification globally for the script
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_old_request = requests.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _old_request(self, method, url, **kwargs)
requests.Session.request = _new_request
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Import the actual API runner to bypass old pipeline code
from api.routes.kis_routes import _run_real_search

def process_query_file(file_path: Path, submission_dir: Path):
    file_name = file_path.stem  # e.g. query-1-kis
    
    # Parse query type from file name
    if file_name.endswith("-kis"):
        query_type = "KIS"
    elif file_name.endswith("-qa"):
        query_type = "QA"
    elif file_name.endswith("-trake"):
        query_type = "TRAKE"
    else:
        query_type = "KIS" # Fallback
        
    with open(file_path, "r", encoding="utf-8") as f:
        query_text = f.read().strip()
        
    print(f"Processing {file_name} ({query_type}): {query_text[:50]}...")
    
    # Run via API flow
    try:
        # For QA/TRAKE, we pass the query_text as question/query
        result_dict = _run_real_search(query=query_text, query_type=query_type, top_k=100, question=query_text)
        results = result_dict.get("results", [])
    except Exception as e:
        print(f"Error running search for {file_name}: {e}")
        results = []
    
    # Format and save to CSV
    csv_path = submission_dir / f"{file_name}.csv"
    
    # Write CSV without headers
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        
        # Max 100 rows
        for c in results[:100]:
            # Remove .mp4 from video name if present
            video_name = c["video_id"].replace(".mp4", "")
            frame_idx = c["frame_idx"]
            
            if query_type == "KIS":
                writer.writerow([video_name, frame_idx])
            elif query_type == "QA":
                answer = c.get("vqa_answer") or ""
                writer.writerow([video_name, frame_idx, answer])
            elif query_type == "TRAKE":
                # Count events based on (1), (2) or 1., 2. in query_text
                events_matches = re.findall(r'(\(\d+\)|\d+\.)', query_text)
                num_events = len(events_matches) if events_matches else 2
                num_events = max(2, num_events) # Minimum 2 events
                
                start_frame = c.get("start_frame")
                end_frame = c.get("end_frame")
                
                if start_frame is not None and end_frame is not None and start_frame != end_frame:
                    # Interpolate frames between start and end
                    step = max(1, (end_frame - start_frame) // (num_events - 1))
                    frames = [start_frame + i * step for i in range(num_events)]
                else:
                    # Fallback mock sequence
                    frames = [frame_idx + i * 25 for i in range(num_events)]
                
                writer.writerow([video_name] + [int(f) for f in frames])
                
    print(f"Saved {csv_path}")

def main():
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
    else:
        zip_path = PROJECT_ROOT / "SOTUYEN2-bo-de-thi.zip"
            
    if not zip_path.exists():
        print(f"Error: Could not find zip file at {zip_path}")
        return

    import uuid
    # Create new output directory with unique ID to avoid lock issues
    run_id = uuid.uuid4().hex[:8]
    temp_workspace = PROJECT_ROOT / f"aic_submission_temp_{run_id}"
    temp_workspace.mkdir(parents=True, exist_ok=True)
    
    submission_dir = temp_workspace / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    # Temp extract dir
    temp_dir = PROJECT_ROOT / f"temp_extract_{run_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    print("Running Search API backend...")
    
    # Find all txt files
    txt_files = list(temp_dir.rglob("*.txt"))
    if not txt_files:
        print("No .txt files found in the zip.")
        
    for txt_file in txt_files:
        process_query_file(txt_file, submission_dir)
        
    # Zip the submission directory
    submission_zip = PROJECT_ROOT / "submission.zip"
    if submission_zip.exists():
        try:
            submission_zip.unlink()
        except:
            print("Warning: Could not delete old submission.zip. Will create submission_new.zip")
            submission_zip = PROJECT_ROOT / "submission_new.zip"
        
    print(f"Creating {submission_zip}...")
    # Zip the temp_workspace which contains the 'submission' folder
    shutil.make_archive(str(submission_zip).replace('.zip', ''), 'zip', str(temp_workspace))
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print("Done! Submit submission.zip")

if __name__ == "__main__":
    main()
