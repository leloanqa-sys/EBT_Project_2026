# EBT_Project_2026 — AI Challenge 2026 System

Hệ thống truy vết khoảnh khắc video (Textual KIS), truy vấn câu hỏi hình ảnh (Visual Q&A) và căn chỉnh sự kiện thời gian (TRAKE) cho cuộc thi AI Challenge 2026.

> 📘 **Tài liệu Hướng dẫn Sử dụng & Vận hành:** Xem [USER_GUIDE.md](file:///c:/Users/Admin/Downloads/EBT_Project_2026/docs/USER_GUIDE.md)  
> 📐 **Tài liệu Kiến trúc & Bàn giao Kỹ thuật:** Xem [ARCHITECTURE_HANDOVER.md](file:///c:/Users/Admin/Downloads/EBT_Project_2026/docs/ARCHITECTURE_HANDOVER.md)  
> 📦 **Hồ sơ Đóng băng Bản V0 Baseline:** Xem [BASELINE_V0_RELEASE.md](file:///c:/Users/Admin/Downloads/EBT_Project_2026/docs/BASELINE_V0_RELEASE.md)

---

## 🏗️ Cấu Trúc Dự Án (Project Architecture)

```text
EBT_Project_2026/
├── config/                     # File cấu hình yaml chung
├── data/
│   ├── raw/
│   │   ├── clip-features-32/   # Dữ liệu vector CLIP ViT-B/32 gốc (.npy)
│   │   ├── map-keyframes/      # Dữ liệu CSV map khung hình gốc (.csv)
│   │   ├── keyframes/          # Ảnh keyframe trích từ video (*.jpg)
│   │   └── objects/            # JSON Faster R-CNN detection results
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
│   │   └── searcher.py         # Entry-point API: search_by_text, search_by_vector
│   ├── role_b_nlp/             # ROLE B: AI Engineer (NLP & Object Fusion)
│   │   ├── text_normalizer.py  # Chuẩn hóa text (Unicode, lowercase, strip)
│   │   ├── query_parser.py     # Parse raw query → ParsedQuery (Gemini + Fallback)
│   │   ├── gemini_nlp_engine.py # Gemini 1.5 Flash integration (extract objects, decompose events)
│   │   ├── question_classifier.py # Phân loại câu hỏi QA
│   │   ├── object_matcher.py   # Faster R-CNN JSON parser + object overlap scoring
│   │   └── fusion_score.py     # Score fusion: 0.7×CLIP + 0.2×Object + 0.1×Meta
│   └── role_c_logic/           # ROLE C: Logic Engineer (Alignment & Ranking)
│       ├── pipeline_kis.py     # End-to-end KIS pipeline
│       ├── pipeline_qa.py      # Q&A pipeline (WIP)
│       ├── pipeline_trake.py   # TRAKE temporal alignment (WIP)
│       ├── vqa_model.py        # VQA model wrapper (WIP)
│       ├── ranking.py          # 5-Budget ranking strategy
│       └── output_formatter.py # CSV submission formatter
├── api/                        # API Gateway & Frontend UI
│   ├── main.py                 # FastAPI app entry point
│   ├── routes/
│   │   └── kis_routes.py       # POST /api/v1/search/kis endpoint
│   └── static/
│       ├── index.html          # Search UI (Dark mode, Visualizer)
│       ├── algorithm.html      # Thuật toán & Cơ chế chấm điểm (public)
│       ├── style.css           # Premium dark mode stylesheet
│       └── app.js              # Frontend application logic
├── tests/
│   └── test_role_a.py          # Pytest integration suite cho Role A
├── docs/
│   ├── EBT-2026.txt            # Concept & phân tích đề bài
│   ├── decisions.md            # Log quyết định kỹ thuật & MD5 checksums
│   └── API_DOCUMENTATION.md    # Tài liệu API endpoints
├── outputs/                    # Submission CSV files
├── requirements.txt            # Thư viện phụ thuộc
└── README.md                   # Báo cáo tiến độ & tài liệu hướng dẫn
```

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

### 6. 🌐 Khởi chạy API Gateway & Giao diện UI
```bash
# Từ thư mục gốc EBT_Project_2026/
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Truy cập:
| URL | Mô tả |
|-----|--------|
| http://localhost:8000/ | **Giao diện tìm kiếm (Visualizer)** |
| http://localhost:8000/static/algorithm.html | **Thuật toán & Cơ chế chấm điểm** |
| http://localhost:8000/docs | **Swagger API Documentation** |
| http://localhost:8000/redoc | **ReDoc API Documentation** |

> **Lưu ý:** Nếu FAISS index chưa được build hoặc thiếu dữ liệu, hệ thống sẽ tự động chạy ở chế độ **DEMO** với dữ liệu mẫu. Giao diện UI vẫn hoạt động bình thường.

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
  - Sau khi FAISS trả về `faiss_id`, tra cứu metadata $O(1)$ trực tiếp qua mảng NumPy đơn cột.
  - Độ trễ tra cứu metadata cho hàng nghìn candidates **< 1 ms**.
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
- **Batch Search**: `search_by_vectors(vectors: np.ndarray, top_k=100, video_ids=None) -> List[List[CandidateFrame]]`
- **Video Filtering**: Hỗ trợ lọc nhanh theo danh sách `video_ids` thông qua dải FAISS ID `[start, end]` liên tục per-video.
- **Disk Embedding Cache**: Tự động lưu cache embedding của câu query text vào `data/processed/cache/query_cache/<sha256>.npy`.

---

## 🧠 Báo Cáo Tiến Độ Role B (AI Engineer / NLP & Object Fusion)

### 1. Tái Kiến Trúc NLP Engine
- **Loại bỏ technical debt**: Dọn dẹp thư mục rác và từ điển gán cứng `LABEL_SYNONYMS`.
- **Tích hợp Gemini 1.5 Flash**: Sử dụng JSON Structured Outputs cho 2 nhiệm vụ:
  - `extract_target_objects()`: Dịch object tiếng Việt → English Faster R-CNN labels
  - `decompose_events()`: Bẻ gãy query phức tạp thành chuỗi sub-events cho TRAKE

### 2. Cơ Chế Fallback & Caching
- **Fallback 2 tầng**: Gemini → Regex → Null safe (hệ thống "không bao giờ sập")
- **SHA-256 Caching**: Cache kết quả NLP vào `data/cache/gemini_nlp/<sha256>.json`

### 3. Score Fusion
- **Công thức**: `fusion_score = 0.70 × CLIP + 0.20 × Object + 0.10 × Meta`
- **Object Matching**: Parse JSON Faster R-CNN → tính overlap ratio với target objects từ NLP

---

## 🎯 Báo Cáo Tiến Độ Role C (Logic Engineer / Ranking)

### 1. KIS Pipeline (Done)
- Pipeline end-to-end: Retrieval → NLP Parse → Fusion → Temporal Clustering → 5-Budget Ranking → CSV Export
- **Temporal Clustering**: Gom cụm frame cùng video theo gap_threshold (15 frames)
- **5-Budget Strategy**: Rank 1 = an toàn nhất; Ranks 2-5 = đa dạng video_id; 6-100 = lưới an toàn

### 2. API Gateway & Visualizer UI (Done)
- **FastAPI Backend**: Endpoint `POST /api/v1/search/kis` kết nối pipeline → JSON response
- **Giao diện Visualizer**: Dark mode UI hiển thị kết quả với hình ảnh + score bars + labels
- **Trang Thuật toán**: Giải thích công khai cơ chế chấm, 5-budget, cơ chế đặt cược
- **Demo Mode**: Tự động fallback dữ liệu mẫu khi pipeline chưa sẵn sàng

### 3. Q&A và TRAKE Pipelines (WIP)
- `pipeline_qa.py` và `pipeline_trake.py` đang ở giai đoạn thiết kế

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