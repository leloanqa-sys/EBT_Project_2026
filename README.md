# EBT_Project_2026 — AI Challenge 2026 System

Hệ thống truy vết khoảnh khắc video (Textual KIS), truy vấn câu hỏi hình ảnh (Visual Q&A) và căn chỉnh sự kiện thời gian (TRAKE) cho cuộc thi AI Challenge 2026.

---

## 🏗️ Cấu Trúc Dự Án (Project Architecture)

```text
EBT_Project_2026/
├── config/                     # File cấu hình yaml chung
├── data/
│   ├── raw/
│   │   ├── clip-features-32/   # Dữ liệu vector CLIP ViT-B/32 gốc (.npy)
│   │   └── map-keyframes/      # Dữ liệu CSV map khung hình gốc (.csv)
│   └── processed/
│       ├── global_mapping.csv  # Single Source of Truth Global Mapping (177,321 rows)
│       ├── video_id_order.json # Thứ tự duy nhất của 873 video
│       ├── video_to_faiss_range.json # Dải FAISS ID [start, end] per-video
│       ├── mapping_array.npz   # NumPy Columnar Hot-Path arrays O(1) lookup
│       ├── media.db            # SQLite persistent Cold-Path Database
│       └── faiss_index/
│           ├── clip_vit_b32.index  # FAISS IndexFlatIP (177,321 vectors)
│           └── index_manifest.json # Build manifest metadata
├── src/
│   ├── common/
│   │   ├── schemas.py          # Data contract: CandidateFrame, RankedCandidate...
│   │   └── logger.py
│   ├── role_a_retrieval/       # ROLE A: Data Scientist / Vector Retrieval Engine
│   │   ├── feature_store.py    # Chunked loading mmap .npy & L2-normalize
│   │   ├── mapping_utils.py    # Dynamic sanity check & global mapping builder
│   │   ├── build_db.py         # SQLite cold-path persistent storage builder
│   │   ├── build_index.py      # FAISS IndexFlatIP builder & MD5 checksum locker
│   │   ├── benchmark_index.py  # Benchmark R@1/5/20/50/100 & Latency p50/p95/p99
│   │   ├── encode_features.py  # Offline CLIP extractor cho video thô
│   │   └── searcher.py         # Entry-point API: search_by_text, search_by_vector, search_by_vectors
│   ├── role_b_nlp/             # ROLE B: AI Engineer (NLP & Object Fusion)
│   └── role_c_logic/           # ROLE C: Logic Engineer (Alignment & Ranking)
├── tests/
│   └── test_role_a.py          # Pytest integration suite cho Role A
├── docs/
│   └── decisions.md            # Nơi khóa mã băm MD5 Checksum artifacts
├── requirements.txt            # Thư viện phụ thuộc
└── README.md                   # Báo cáo tiến độ & tài liệu hướng dẫn
```

---

## ⚡ Báo Cáo Tiến Độ Role A (Data Scientist / Retrieval Engine)

### 1. Data Invariant & Sanity Check 100% Pass
- **Kết quả quét dữ liệu thô Batch 1**:
  - Đã kiểm tra **873 file `.npy`** và **873 file `.csv`**.
  - **100% PASS (0 video bị lệch)**.
  - Tổng số keyframes đã index: **177,321 keyframes**.
- **Quy tắc bảo vệ**: Hàm `run_sanity_check()` tự động phát hiện số video động (không hardcode 873) và dừng cứng `sys.exit(1)` lập tức nếu bất kỳ video nào bị lệch số lượng vector vs CSV.

### 2. Kiến Trúc 2 Tầng: Hot-Path vs Cold-Path
- **Online Hot-Path (NumPy Columnar Arrays `mapping_array.npz`)**:
  - Sau khi FAISS trả về `faiss_id`, tra cứu metadata $O(1)$ trực tiếp qua mảng NumPy đơn cột:
    ```python
    video_id = video_ids[video_idx[faiss_ids]]
    frame_idx = frame_indices[faiss_ids]
    pts_time = pts_times[faiss_ids]
    fps = fps_arr[faiss_ids]
    ```
  - Độ trễ tra cứu metadata cho hàng nghìn candidates **< 1 ms** (không gọi SQLite, không gọi Pandas trên hot-path).
- **Offline Cold-Path (SQLite `media.db`)**:
  - Lưu trữ 177,321 dòng vào bảng `keyframes` phục vụ audit, offline tooling và debugging.

### 3. FAISS Index & Lock Artifact Checksum
- Xây dựng FAISS Index phẳng **`IndexFlatIP(512)`** đảm bảo chính xác tuyệt đối (Recall = 100%).
- Đã xuất file `data/processed/faiss_index/index_manifest.json` ghi nhận chính xác model `clip-ViT-B-32`, dimension `512`, `ntotal=177321`.
- Đã khóa mã băm MD5 Checksum vào `docs/decisions.md`:
  - `clip_vit_b32.index` MD5: `b95908dfc23053e8313b9c7d89de8f50`
  - `media.db` MD5: `473ce4c72855aa8e14789b293596ad32`

### 4. Vector Searcher API Chuẩn Cho Role B & C
Đã triển khai class `VectorSearcher` trong `src/role_a_retrieval/searcher.py` với các tính năng:
- **Startup Validation (Fail-Fast)**: Tự động kiểm tra `manifest.model == "clip-ViT-B-32"` và `index.ntotal == len(mapping)` khi khởi tạo.
- **Search by Text**: `search_by_text(query: str, top_k=100) -> List[CandidateFrame]`
- **Search by Vector**: `search_by_vector(vector: np.ndarray, top_k=100, video_ids=None) -> List[CandidateFrame]`
- **Batch Search**: `search_by_vectors(vectors: np.ndarray, top_k=100, video_ids=None) -> List[List[CandidateFrame]]` (Phục vụ truy vấn chuỗi nhiều sub-events cho TRAKE).
- **Video Filtering**: Hỗ trợ lọc nhanh theo danh sách `video_ids` thông qua dải FAISS ID `[start, end]` liên tục per-video.
- **Disk Embedding Cache**: Tự động lưu cache embedding của câu query text vào `data/processed/cache/query_cache/<sha256>.npy`.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Chạy Sanity Check & Build Global Mapping
```bash
python -m src.role_a_retrieval.mapping_utils
```

### 3. Build SQLite Persistent Storage (Cold-Path)
```bash
python -m src.role_a_retrieval.build_db
```

### 4. Build FAISS Index & Ghi Manifest
```bash
python -m src.role_a_retrieval.build_index
```

### 5. Chạy Test Suite Tự Động (Pytest)
```bash
pytest tests/test_role_a.py -v
```

---

## 📋 Data Contract Giao Tiếp Giữa Các Roles

```python
# --- ROLE A OUTPUT CONTRACT (src/common/schemas.py) ---
@dataclass
class CandidateFrame:
    faiss_id: int
    video_id: str
    frame_idx: int
    pts_time: float
    fps: float
    clip_score: float

# --- ROLE B FUSION CONTRACT ---
@dataclass
class RankedCandidate:
    candidate: CandidateFrame
    obj_score: float = 0.0
    meta_score: float = 0.0
    fusion_score: float = 0.0
```