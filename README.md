# EBT Project 2026 - Video Retrieval System (AI Challenge 2026)

Hệ thống truy xuất video lai (Hybrid Video Retrieval) tốc độ cao, được thiết kế để giải quyết bài toán tìm kiếm KIS (Known-Item Search) và TRAKE (Temporal Action Tracking) với nguồn tài nguyên phần cứng giới hạn (tối ưu cho máy 8GB RAM).

---

## 1. Cấu trúc Thư mục & Luồng gọi (Directory Structure)

Dự án được chia thành các module phân tán. Tầng API (Web) sẽ gọi xuống tầng Retrieval (Tìm kiếm), tầng Retrieval sẽ điều phối NLP (Xử lý ngôn ngữ) và Database.

```text
EBT_Project_2026/
│
├── api/                        # Tầng Giao tiếp (FastAPI Gateway)
│   ├── main.py                 # Khởi tạo Global State (Database, FAISS) để tiết kiệm RAM.
│   └── routes/                 # Tiếp nhận Request từ Web UI
│       ├── kis_routes.py       # Xử lý tìm kiếm KIS, gọi HybridSearcher.
│       └── trake_routes.py     # Xử lý tìm kiếm chuỗi sự kiện TRAKE.
│
├── src/                        # LÕI LOGIC HỆ THỐNG
│   ├── common/                 # Dữ liệu dùng chung
│   │   ├── schemas.py          # Single Source of Truth chứa `QueryIR`, `CandidateFrame`...
│   │   └── config.py           # Cấu hình trọng số và tham số.
│   │
│   ├── nlp/                    # Tầng Xử lý Ngôn ngữ Tự nhiên
│   │   └── gemini_parser.py    # Dịch câu hỏi Tiếng Việt -> JSON `QueryIR` (Bóc tách Visual, Text, Temporal).
│   │
│   ├── retrieval/              # Tầng Truy xuất & Chấm điểm (Trái tim hệ thống)
│   │   ├── hybrid_searcher.py  # Thuật toán Late Fusion: Gộp điểm FAISS và SQLite. Gọi NLP và Database.
│   │   ├── text_retriever.py   # Tìm kiếm bằng chữ (BM25) qua SQLite FTS5.
│   │   ├── vector_index.py     # Quản lý FAISS Index (Search Vector).
│   │   ├── vlm_auditor.py      # LLM Vision: Cắt Top 10 ảnh thật gửi Gemini Vision để chấm điểm lại.
│   │   └── trake_formatter.py  # Thuật toán Dynamic Programming gộp chuỗi sự kiện TRAKE.
│   │
│   └── database/               # Tầng Giao tiếp Dữ liệu Đĩa
│       ├── db_manager.py       # Kết nối SQLite (media.db, aic2026.db).
│       └── legacy_detection_store.py # Kết nối metadata.db (2.2GB bouding boxes).
│
├── scripts/                    # Các script chạy độc lập để chuẩn bị dữ liệu (Data Ingestion)
│   ├── 01_init_db.py           # Khởi tạo schema cho SQLite.
│   ├── 02_ingest_all_data.py   # Nạp dữ liệu cơ bản.
│   └── ...                     
│
└── data/                       # (Thư mục này không up lên Git)
    ├── raw/keyframes/          # Chứa hàng trăm ngàn ảnh cắt từ video gốc.
    └── processed/              # Chứa faiss index (.index) và SQLite (.db).
```

---

## 2. Tổng quan Pipeline (Hệ thống hoạt động ra sao?)

1. **Nhận truy vấn:** Người dùng nhập Tiếng Việt (VD: *"người đàn ông áo đỏ lướt sóng"*). `kis_routes.py` tiếp nhận.
2. **Phân rã NLP:** `gemini_parser.py` (Gemini 3.5 Flash lite) dịch sang tiếng Anh và chẻ thành các mảng: `dense_caption_en` (cho hình ảnh), `text_targets` (cho OCR/Phụ đề).
3. **Tìm kiếm Phân tán (Scatter):** 
   - `HybridSearcher` đưa tiếng Anh vào mô hình **SigLIP2** biến thành Vector, quét **FAISS** lấy Top 500 ID hình ảnh giống nhất.
   - Quét **SQLite FTS5** bằng từ khóa để lấy các ID có chứa chữ.
4. **Gom tụ (Gather & Late Fusion):** Gộp các ID lại. Dùng 1 lệnh SQL duy nhất kéo thông tin thời gian của toàn bộ ID. Chấm điểm chéo `Final = w_visual * SigLIP + w_ocr * Text`.
5. **Giám khảo VLM (Optional):** Gửi 10 ảnh tốt nhất cho Gemini Vision nhìn lại lần cuối để Rerank.
6. **Xếp chuỗi (TRAKE):** Nếu là câu hỏi chuỗi, chạy Quy hoạch động (DP) để nối các frame theo đúng thứ tự thời gian.
7. **Trả kết quả:** Chuyển thành JSON gửi về UI.

---

## 3. Đánh giá Hiện trạng (Status & Limitations)

### 🟢 Những gì Đã Chạy Ổn Định (Stable)
*   **Kiến trúc DI (Dependency Injection):** RAM được tối ưu triệt để. Database và FAISS chỉ load 1 lần lúc bật server, xử lý Request cực nhanh, phù hợp cho cấu hình 8GB RAM.
*   **Luồng Visual (SigLIP2 + FAISS):** Chạy cực kỳ chính xác. Đã mở khóa tính năng dịch Tiếng Anh tự động từ Gemini để SigLIP hiểu đúng context hơn là dùng Tiếng Việt.
*   **Luồng TRAKE (Thuật toán DP):** Hoạt động hoàn hảo trong việc khâu nối các sự kiện rời rạc thành dòng thời gian liên tục mà không cần Model Video nặng nề.

### 🟡 Những Giới Hạn Cần Cải Thiện (Limitations / WIP)
*   **Database OCR Đang Trống:** Bảng `text_segments` trong `aic2026.db` hiện có 0 dòng do chưa từng chạy script Ingest OCR cho full dataset. Luồng tìm kiếm bằng chữ tạm thời vô tác dụng.
*   **Dữ liệu Vật thể (20M boxes) Bị Bỏ Xó:** Hệ thống nạp file `metadata.db` 2.2GB nhưng `w_object` đang khóa ở `0.0` do nhãn bị nhiễu. **Hướng giải quyết tới:** Dùng nó làm "Điểm Thưởng" (Bonus Score) cho các truy vấn đếm số lượng ("2 chiếc xe") hoặc không gian ("bên trái").
*   **Mù chuyển động (Motion Blindness):** Do quét ảnh tĩnh, SigLIP dễ bắt nhầm các frame lỗi. Cần bổ sung thuật toán Làm mượt theo thời gian (Temporal Smoothing / Sliding Window 3s) ở `HybridSearcher`.

---

## 4. Hướng dẫn Setup Data

Hệ thống yêu cầu cấu trúc thư mục `data/` như sau (không đẩy lên Git):
```text
data/
├── raw/
│   └── keyframes/              # Chứa các thư mục video (L01_V001/0001.jpg)
└── processed/
    ├── aic2026.db              # DB chứa OCR/ASR (Cần chạy scripts 06, 07 để tạo)
    ├── media.db                # DB chứa map frame_id, video_id, pts_time (Cần chạy scripts 01, 02)
    ├── metadata.db             # DB chứa 20M object bounding boxes
    └── faiss/
        └── siglip2.index       # File index FAISS đã huấn luyện
```
*Lưu ý: Chạy lần lượt các file trong thư mục `scripts/` (01 đến 07) để build lại Database nếu bạn tải về bộ data trắng.*

---

## 5. Hướng dẫn Chạy (Run Instructions)

1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Cấu hình Môi trường:
   - Tạo file `.env` (copy từ `.env.example`).
   - Điền API Key của Google Gemini vào.
3. Chạy Server FastAPI:
   ```bash
   # Khởi động server ở port 8000
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Truy cập giao diện Web tĩnh tại: `http://localhost:8000/static/index.html` (Hoặc mở gốc `http://localhost:8000/`)
