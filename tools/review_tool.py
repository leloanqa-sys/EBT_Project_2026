import os
import sys
import glob
import base64
import json
import zipfile
from typing import List, Optional, Tuple, Dict, Set
import pandas as pd
from jinja2 import Template
from src.common.schemas import CandidateFrame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Interactive Visual Audit — {{ query_id }}</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 20px; padding-bottom: 80px; }
        h1 { color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; margin-bottom: 15px; }
        
        /* Control Bar */
        .control-bar { position: sticky; top: 10px; z-index: 100; background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #38bdf8; box-shadow: 0 4px 20px rgba(0,0,0,0.5); display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; flex-wrap: wrap; gap: 10px; }
        .shortcuts { display: flex; gap: 12px; font-size: 0.88em; flex-wrap: wrap; }
        .key { background: #334155; color: #38bdf8; padding: 2px 8px; border-radius: 4px; font-weight: bold; border: 1px solid #475569; }
        
        .stats { display: flex; gap: 12px; font-weight: bold; font-size: 0.9em; align-items: center; }
        .stat-box { background: #0f172a; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; }
        
        .btn-export { background: #0284c7; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        .btn-export:hover { background: #0369a1; }
        
        .query-box { background: #1e293b; padding: 15px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 25px; }
        
        /* Grid Layout */
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }
        .card { background: #1e293b; border-radius: 10px; padding: 14px; border: 2px solid #334155; transition: all 0.2s; position: relative; }
        .card.active { border-color: #38bdf8; box-shadow: 0 0 15px rgba(56, 189, 248, 0.4); }
        .card.verdict-match { border-color: #22c55e; background: #064e3b22; }
        .card.verdict-uncertain { border-color: #f59e0b; background: #78350f22; }
        .card.verdict-mismatch { border-color: #ef4444; background: #7f1d1d22; }
        
        /* Image Preview Box */
        .img-container { width: 100%; height: 190px; background: #0f172a; border-radius: 6px; margin: 10px 0; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid #334155; }
        .img-container img { width: 100%; height: 100%; object-fit: cover; }
        .img-placeholder { color: #94a3b8; font-size: 0.82em; text-align: center; padding: 12px; line-height: 1.5; }
        .img-placeholder code { background: #1e293b; color: #38bdf8; padding: 2px 5px; border-radius: 4px; }

        /* Object Pills */
        .obj-box { margin: 8px 0; display: flex; flex-wrap: wrap; gap: 5px; }
        .obj-pill { background: #0369a1; color: #e0f2fe; padding: 2px 7px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }

        .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; background: #0284c7; color: white; margin-bottom: 8px; }
        .verdict-tag { float: right; font-weight: bold; font-size: 0.85em; }
        .tag-match { color: #4ade80; }
        .tag-uncertain { color: #fbbf24; }
        .tag-mismatch { color: #f87171; }
        
        .card-header { font-weight: bold; color: #f1f5f9; margin-bottom: 4px; font-size: 1.05em; }
        .meta { font-size: 0.85em; color: #94a3b8; }
        .score-row { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; background: #0f172a; padding: 6px 8px; border-radius: 6px; margin: 6px 0; font-size: 0.82em; border: 1px solid #334155; }
        .score-val { color: #38bdf8; font-weight: bold; }
        .score-fusion { color: #4ade80; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🔍 Smart Visual Audit Tool (Visual Keyframe Preview + 3-State Verdict)</h1>

    <div class="control-bar">
        <div class="shortcuts">
            <span><span class="key">1</span>/<span class="key">Y</span>: Khớp (MATCH ✅)</span>
            <span><span class="key">2</span>/<span class="key">U</span>: Không chắc (UNCERTAIN ⚠️)</span>
            <span><span class="key">0</span>/<span class="key">N</span>: Sai (MISMATCH ❌)</span>
            <span><span class="key">J</span>/<span class="key">K</span>: Kế/Trước</span>
            <span><span class="key">Ctrl + S</span>: Xuất CSV</span>
        </div>
        <div class="stats">
            <div class="stat-box">Đã Chấm: <span id="stat-count" style="color:#38bdf8">0/{{ items|length }}</span></div>
            <div class="stat-box">✅ <span id="stat-match" style="color:#4ade80">0</span></div>
            <div class="stat-box">⚠️ <span id="stat-uncertain" style="color:#fbbf24">0</span></div>
            <div class="stat-box">❌ <span id="stat-mismatch" style="color:#f87171">0</span></div>
            <button class="btn-export" onclick="exportVerdictCSV()">💾 Xuất CSV Verdict</button>
        </div>
    </div>

    <div class="query-box">
        <h3>Query ID: {{ query_id }}</h3>
        <p><strong>Raw Query:</strong> {{ query_text }}</p>
    </div>
    
    <div class="grid" id="card-grid">
    {% for item in items %}
        <div class="card {% if loop.first %}active{% endif %}" 
             data-index="{{ loop.index0 }}" 
             data-video="{{ item.candidate.video_id }}" 
             data-frame="{{ item.candidate.frame_idx }}" 
             data-clip-score="{{ item.candidate.clip_score }}"
             data-obj-score="{{ item.candidate.obj_score }}"
             data-spatial-score="{{ item.candidate.spatial_score }}"
             data-fusion-score="{{ item.candidate.fusion_score }}">
            <span class="badge">Rank #{{ loop.index }}</span>
            <span class="verdict-tag" id="tag-{{ loop.index0 }}">⏳ Chờ chấm</span>
            <div class="card-header">{{ item.candidate.video_id }} — Frame {{ item.candidate.frame_idx }}</div>
            
            <!-- Keyframe Image Rendering -->
            <div class="img-container">
            {% if item.image_b64 %}
                <img src="{{ item.image_b64 }}" alt="{{ item.candidate.video_id }}_n_{{ item.keyframe_n }}">
            {% else %}
                <div class="img-placeholder">
                    ⚡ <strong>[Smart On-Demand Extractor Ready]</strong><br>
                    Video: <code>{{ item.candidate.video_id }}</code> | Keyframe <code>n={{ item.keyframe_n or 'N/A' }}</code><br>
                    Cần thả file <code>keyframes_{{ item.candidate.video_id[:3] }}.zip</code> vào <code>data/zips/</code>
                </div>
            {% endif %}
            </div>

            <!-- Detected Object Pills -->
            {% if item.detected_objects %}
            <div class="obj-box">
                {% for obj in item.detected_objects %}
                    <span class="obj-pill">🏷️ {{ obj }}</span>
                {% endfor %}
            </div>
            {% endif %}

            <div class="score-row">
                <div>CLIP: <span class="score-val">{{ "%.4f"|format(item.candidate.clip_score) }}</span></div>
                <div>OBJ: <span class="score-val">{{ "%.2f"|format(item.candidate.obj_score) }}</span></div>
                <div>SPATIAL: <span class="score-val">{{ "%.2f"|format(item.candidate.spatial_score) }}</span></div>
                <div>FUSION: <span class="score-fusion">{{ "%.4f"|format(item.candidate.fusion_score) }}</span></div>
            </div>

            <div class="meta">
                <p>Mapped Keyframe: <strong>n = {{ item.keyframe_n or 'N/A' }}</strong> | PTS: {{ "%.2f"|format(item.candidate.pts_time) }}s</p>
            </div>
        </div>
    {% endfor %}
    </div>

    <script>
        let activeIndex = 0;
        const totalCards = {{ items|length }};
        // verdicts: 'MATCH', 'UNCERTAIN', 'MISMATCH', or null
        const verdicts = new Array(totalCards).fill(null);

        function updateCardFocus() {
            document.querySelectorAll('.card').forEach((card, idx) => {
                if (idx === activeIndex) {
                    card.classList.add('active');
                    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    card.classList.remove('active');
                }
            });
        }

        function setVerdict(index, status) {
            verdicts[index] = status;
            const card = document.querySelector(`.card[data-index="${index}"]`);
            const tag = document.getElementById(`tag-${index}`);

            card.classList.remove('verdict-match', 'verdict-uncertain', 'verdict-mismatch');
            if (status === 'MATCH') {
                card.classList.add('verdict-match');
                tag.innerHTML = '✅ KHỚP (MATCH)';
                tag.className = 'verdict-tag tag-match';
            } else if (status === 'UNCERTAIN') {
                card.classList.add('verdict-uncertain');
                tag.innerHTML = '⚠️ KHÔNG CHẮC (UNCERTAIN)';
                tag.className = 'verdict-tag tag-uncertain';
            } else if (status === 'MISMATCH') {
                card.classList.add('verdict-mismatch');
                tag.innerHTML = '❌ SAI (MISMATCH)';
                tag.className = 'verdict-tag tag-mismatch';
            }
            updateStats();
        }

        function updateStats() {
            const rated = verdicts.filter(v => v !== null).length;
            const matches = verdicts.filter(v => v === 'MATCH').length;
            const uncertains = verdicts.filter(v => v === 'UNCERTAIN').length;
            const mismatches = verdicts.filter(v => v === 'MISMATCH').length;
            
            document.getElementById('stat-count').innerText = `${rated}/${totalCards}`;
            document.getElementById('stat-match').innerText = `${matches}`;
            document.getElementById('stat-uncertain').innerText = `${uncertains}`;
            document.getElementById('stat-mismatch').innerText = `${mismatches}`;
        }

        function exportVerdictCSV() {
            let csvContent = "data:text/csv;charset=utf-8,query_id,rank,video_id,frame_id,verdict,clip_score,obj_score,spatial_score,fusion_score\\n";
            document.querySelectorAll('.card').forEach((card, idx) => {
                const vid = card.getAttribute('data-video');
                const frame = card.getAttribute('data-frame');
                const clip = card.getAttribute('data-clip-score') || '0.0';
                const obj = card.getAttribute('data-obj-score') || '0.0';
                const spatial = card.getAttribute('data-spatial-score') || '0.0';
                const fusion = card.getAttribute('data-fusion-score') || '0.0';
                const v = verdicts[idx] || "UNRATED";
                csvContent += `{{ query_id }},${idx + 1},${vid},${frame},${v},${clip},${obj},${spatial},${fusion}\\n`;
            });

            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `human_verdict_{{ query_id }}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === '1' || e.key.toLowerCase() === 'y') {
                setVerdict(activeIndex, 'MATCH');
                if (activeIndex < totalCards - 1) activeIndex++;
                updateCardFocus();
            } else if (e.key === '2' || e.key.toLowerCase() === 'u') {
                setVerdict(activeIndex, 'UNCERTAIN');
                if (activeIndex < totalCards - 1) activeIndex++;
                updateCardFocus();
            } else if (e.key === '0' || e.key.toLowerCase() === 'n') {
                setVerdict(activeIndex, 'MISMATCH');
                if (activeIndex < totalCards - 1) activeIndex++;
                updateCardFocus();
            } else if (e.key.toLowerCase() === 'j' || e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                if (activeIndex < totalCards - 1) activeIndex++;
                updateCardFocus();
            } else if (e.key.toLowerCase() === 'k' || e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                if (activeIndex > 0) activeIndex--;
                updateCardFocus();
            } else if (e.ctrlKey && e.key.toLowerCase() === 's') {
                e.preventDefault();
                exportVerdictCSV();
            }
        });
    </script>
</body>
</html>
"""

# Global memory cache to prevent re-scanning zip files repeatedly
_REMOTE_URL_MAP = None

def get_remote_zip_urls(video_id: str) -> List[str]:
    global _REMOTE_URL_MAP
    if _REMOTE_URL_MAP is None:
        _REMOTE_URL_MAP = {}
        csv_path = os.path.join(os.path.dirname(__file__), "..", "spreadsheet_data.csv")
        try:
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3 and "Keyframes" in row[1]:
                        fname = row[1].strip()
                        url = row[2].strip()
                        _REMOTE_URL_MAP[fname] = url
        except Exception:
            pass
            
    prefix = video_id.split("_")[0]
    expected_zip = f"Keyframes_{prefix}.zip"
    if expected_zip in _REMOTE_URL_MAP:
        return [_REMOTE_URL_MAP[expected_zip]]
    
    # Fallback to parts (e.g. Keyframes_L26_a.zip, Keyframes_L26_b.zip)
    matches = []
    for k, v in _REMOTE_URL_MAP.items():
        if k.startswith(f"Keyframes_{prefix}"):
            matches.append(v)
    return matches

def extract_single_file_from_remote_zip(video_id: str, fname: str) -> Optional[bytes]:
    """Smart Extraction: Fetches ONLY the required image bytes via HTTP Range from remote ZIP."""
    urls = get_remote_zip_urls(video_id)
    if not urls: return None
    
    target_suffix = f"{video_id}/{fname}"
    try:
        import remotezip
        import requests
        from functools import partial
        session = requests.Session()
        session.verify = False
        session.request = partial(session.request, timeout=10.0)
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        for url in urls:
            try:
                with remotezip.RemoteZip(url, session=session) as rz:
                    for inner_name in rz.namelist():
                        if inner_name.endswith(target_suffix):
                            print(f"  [RemoteZip] Streamed {inner_name} directly from {url}")
                            return rz.read(inner_name)
            except Exception as e:
                # 404 means the file doesn't exist in this specific zip, ignore and try next
                continue
    except Exception as e:
        print(f"  [RemoteZip] Global Error: {e}")
    return None

_ZIP_INDEX_CACHE: Dict[str, Dict[str, str]] = {}
_DISCOVERED_ZIPS: Optional[List[str]] = None

def map_frame_idx_to_keyframe_n(video_id: str, frame_idx: int, map_root: str = "data/raw/map-keyframes") -> Optional[int]:
    """Translates frame_idx to actual keyframe index n using map-keyframes CSV."""
    csv_p = os.path.join(map_root, f"{video_id}.csv")
    if not os.path.exists(csv_p):
        return None
    try:
        df = pd.read_csv(csv_p)
        match = df[df["frame_idx"] == frame_idx]
        if not match.empty:
            return int(match["n"].iloc[0])
        df["diff"] = (df["frame_idx"] - frame_idx).abs()
        min_row = df.loc[df["diff"].idxmin()]
        if min_row["diff"] <= 45:
            return int(min_row["n"])
    except Exception:
        pass
    return None

def get_detected_objects_for_card(video_id: str, keyframe_n: Optional[int], frame_idx: int, objects_root: str = "data/raw/objects") -> List[str]:
    """Retrieves detected object entities from Faster R-CNN JSON."""
    vid_dir = os.path.join(objects_root, video_id)
    if not os.path.exists(vid_dir):
        return []
    
    candidates = []
    if keyframe_n is not None:
        candidates.extend([f"{keyframe_n:03d}.json", f"{keyframe_n:04d}.json", f"{keyframe_n}.json"])
    candidates.extend([f"{frame_idx:03d}.json", f"{frame_idx}.json"])

    for fname in candidates:
        json_p = os.path.join(vid_dir, fname)
        if os.path.exists(json_p):
            try:
                with open(json_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entities = data.get("detection_class_entities", [])
                    seen = set()
                    unique_objs = []
                    for item in entities:
                        if item not in seen and item.strip():
                            seen.add(item)
                            unique_objs.append(item)
                    return unique_objs[:7]
            except Exception:
                pass
    return []

def discover_zip_files() -> List[str]:
    """Safely discovers zip files in specific data directories only."""
    global _DISCOVERED_ZIPS
    if _DISCOVERED_ZIPS is not None:
        return _DISCOVERED_ZIPS

    search_dirs = [
        os.path.join(PROJECT_ROOT, "data", "zips"),
        os.path.join(PROJECT_ROOT, "data", "zip"),
        os.path.join(PROJECT_ROOT, "data", "raw", "zip"),
        os.path.join(PROJECT_ROOT, "data", "raw"),
        os.path.join(PROJECT_ROOT, "data", "archive_zips"),
        "data/zips", "data/zip", "data/raw/zip", "data/raw"
    ]
    found = []
    for sdir in search_dirs:
        if os.path.exists(sdir):
            try:
                for fname in os.listdir(sdir):
                    if fname.endswith(".zip"):
                        found.append(os.path.abspath(os.path.join(sdir, fname)))
            except Exception:
                pass

    _DISCOVERED_ZIPS = list(set(found))
    return _DISCOVERED_ZIPS

def extract_single_file_from_zip(video_id: str, fname: str) -> Optional[bytes]:
    """Extracts ONLY the requested single image file from any .zip file in memory with O(1) lookup."""
    global _ZIP_INDEX_CACHE
    zip_files = discover_zip_files()
    if not zip_files:
        return None

    target_suffix = f"{video_id}/{fname}".lower()

    for zip_path in zip_files:
        try:
            if zip_path not in _ZIP_INDEX_CACHE:
                index_map = {}
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for inner_name in zf.namelist():
                        if inner_name.endswith(".jpg") or inner_name.endswith(".png"):
                            norm = inner_name.replace("\\", "/")
                            parts = norm.split("/")
                            if len(parts) >= 2:
                                key = f"{parts[-2]}/{parts[-1]}".lower()
                                index_map[key] = inner_name
                            index_map[norm.lower()] = inner_name
                _ZIP_INDEX_CACHE[zip_path] = index_map

            cached_map = _ZIP_INDEX_CACHE[zip_path]
            matched_inner = cached_map.get(target_suffix)

            if matched_inner:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    return zf.read(matched_inner)
        except Exception:
            continue
    return None

def extract_frame_from_video_mp4(video_id: str, frame_idx: int, video_dir: str = "data/raw/videos") -> Optional[bytes]:
    """Seeks and extracts ONLY frame_idx from video MP4 in-memory using OpenCV."""
    mp4_p = os.path.join(video_dir, f"{video_id}.mp4")
    if not os.path.exists(mp4_p):
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(mp4_p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                return buffer.tobytes()
    except Exception:
        pass
    return None

def resolve_keyframe_b64(video_id: str, frame_idx: int, keyframes_root: str = "data/raw/keyframes", allow_remote: bool = False) -> Tuple[Optional[str], Optional[int], str]:
    """Smart Multi-Strategy Image Resolver."""
    keyframe_n = map_frame_idx_to_keyframe_n(video_id, frame_idx)
    possible_filenames = []
    if keyframe_n is not None:
        possible_filenames.extend([
            f"{keyframe_n:03d}.jpg", f"{keyframe_n:04d}.jpg", f"{keyframe_n:05d}.jpg", f"{keyframe_n}.jpg", f"{keyframe_n:03d}.png"
        ])
    possible_filenames.extend([f"{frame_idx}.jpg", f"{frame_idx:04d}.jpg"])
    expected_filename = possible_filenames[0] if possible_filenames else f"{frame_idx}.jpg"
    
    # Strategy 1: Check Local Unzipped Folder
    vid_dir = os.path.join(keyframes_root, video_id)
    if os.path.exists(vid_dir):
        for fname in possible_filenames:
            full_p = os.path.join(vid_dir, fname)
            if os.path.exists(full_p):
                try:
                    with open(full_p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                        mime = "image/png" if fname.endswith(".png") else "image/jpeg"
                        return f"data:{mime};base64,{b64}", keyframe_n, fname
                except Exception:
                    pass

    # Strategy 2: Smart On-Demand Zip Extraction (In Memory O(1))
    for fname in possible_filenames:
        img_bytes = extract_single_file_from_zip(video_id, fname)
        if img_bytes:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            mime = "image/png" if fname.endswith(".png") else "image/jpeg"
            return f"data:{mime};base64,{b64}", keyframe_n, fname

    # Strategy 3: Smart OpenCV MP4 Video Seek
    mp4_bytes = extract_frame_from_video_mp4(video_id, frame_idx)
    if mp4_bytes:
        b64 = base64.b64encode(mp4_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}", keyframe_n, expected_filename

    # Strategy 4: Smart Remote ZIP HTTP Stream (Only if allow_remote=True)
    if allow_remote:
        for fname in possible_filenames:
            img_bytes = extract_single_file_from_remote_zip(video_id, fname)
            if img_bytes:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                mime = "image/png" if fname.endswith(".png") else "image/jpeg"
                return f"data:{mime};base64,{b64}", keyframe_n, fname

    return None, keyframe_n, expected_filename

def export_review_html(query_id: str, query_text: str, candidates: List[CandidateFrame], out_filepath: str) -> str:
    """Renders Jinja2 HTML review report with Object Tags + Smart Image Extraction."""
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
    
    template_items = []
    for c in candidates:
        b64_str, k_n, exp_fname = resolve_keyframe_b64(c.video_id, c.frame_idx)
        detected_objs = get_detected_objects_for_card(c.video_id, k_n, c.frame_idx)
        template_items.append({
            "candidate": c,
            "image_b64": b64_str,
            "keyframe_n": k_n,
            "expected_filename": exp_fname,
            "detected_objects": detected_objs
        })

    template = Template(HTML_TEMPLATE)
    html_content = template.render(query_id=query_id, query_text=query_text, items=template_items)
    
    with open(out_filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return out_filepath

if __name__ == "__main__":
    import argparse
    import sys
    import webbrowser

    parser = argparse.ArgumentParser(description="🔍 Review Tool - Tự tạo query và chấm điểm trực quan (3-State Verdict)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Nội dung câu truy vấn cần tìm kiếm")
    parser.add_argument("--id", "-i", type=str, default=None, help="Query ID (Ví dụ: Q001, Test_01)")
    parser.add_argument("--top_k", "-k", type=int, default=20, help="Số lượng kết quả hiển thị để chấm (mặc định: 20)")
    parser.add_argument("--open", action="store_true", default=True, help="Tự động mở file HTML trên trình duyệt")
    args = parser.parse_args()

    # If no query provided via args, prompt interactively
    query_text = args.query
    query_id = args.id

    if not query_text:
        print("=" * 60)
        print("🎯 SMART VISUAL AUDIT & VERIFICATION TOOL")
        print("=" * 60)
        query_text = input("👉 Nhập câu truy vấn của bạn (VD: a person riding a bicycle): ").strip()
        if not query_text:
            print("❌ Không có nội dung truy vấn. Đang thoát.")
            sys.exit(0)

    if not query_id:
        import time
        query_id = f"Q_{int(time.time()) % 10000:04d}"

    print(f"\n🚀 Đang chạy pipeline cho query [{query_id}]: '{query_text}'...")
    from src.pipeline import MVPPipeline
    
    pipeline = MVPPipeline()
    result = pipeline.run(query_id=query_id, raw_text=query_text)
    
    out_dir = os.path.join("outputs", "reviews")
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"review_{query_id}.html")
    
    candidates_to_review = result.candidates[:args.top_k]
    export_review_html(query_id, query_text, candidates_to_review, out_html)
    
    abs_path = os.path.abspath(out_html)
    file_uri = f"file:///{abs_path.replace(os.sep, '/')}"
    print(f"\n✅ Đã tạo xong file Review HTML ({len(candidates_to_review)} candidates)!")
    print(f"👉 Đường dẫn mở trình duyệt: {file_uri}")
    print(f"💡 Hướng dẫn chấm:")
    print("   - Phím 1 / Y : Khớp (MATCH ✅)")
    print("   - Phím 2 / U : Không chắc chắn (UNCERTAIN ⚠️)")
    print("   - Phím 0 / N : Sai (MISMATCH ❌)")
    print("   - Phím J / K : Sang thẻ kế tiếp / Lùi lại")
    print("   - Phím Ctrl + S : Tải file CSV kết quả (hãy lưu vào outputs/verdicts/)")
    print("=" * 60)

    if args.open:
        try:
            webbrowser.open(file_uri)
        except Exception:
            pass

