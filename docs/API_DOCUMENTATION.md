# 📘 API Documentation — EBT Vision Search 2026

Tài liệu mô tả chi tiết toàn bộ các RESTful API endpoints của hệ thống **EBT Vision Search** (AI Challenge 2026).

---

## 🌐 Base URL & Khởi chạy

```
http://localhost:8000
```

* **Swagger UI (Interactive Docs):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Web Visualizer UI:** [http://localhost:8000/](http://localhost:8000/)

### Khởi động API Server:
```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📌 Danh sách Endpoints

### 1. `POST /api/v1/search/kis` — Truy vấn Known Item Search (KIS)
Nhận câu truy vấn ngôn ngữ tự nhiên (Tiếng Việt hoặc Tiếng Anh), chạy qua pipeline **NLP → Vector Retrieval → SQLite Soft Filtering → 5-Budget Ranking** và trả về danh sách khung hình kèm điểm số phân rã.

#### Request Body:
```json
{
  "query": "người đàn ông mặc áo đỏ đang phát biểu",
  "query_type": "KIS",
  "top_k": 20
}
```

| Trường | Kiểu | Bắt buộc | Mặc định | Mô tả |
| :--- | :--- | :---: | :---: | :--- |
| `query` | string | ✅ | — | Câu truy vấn (Tiếng Việt hoặc English) |
| `query_type` | string | ❌ | `"KIS"` | Loại truy vấn: `KIS`, `QA`, `TRAKE` |
| `top_k` | int | ❌ | `100` | Số lượng kết quả trả về (1–500) |

#### Response (200 OK):
```json
{
  "query_id": "api_8f3a1b2c",
  "query": "người đàn ông mặc áo đỏ đang phát biểu",
  "query_type": "KIS",
  "total_results": 20,
  "search_time_ms": 235.4,
  "cache_hit": true,
  "parsed_info": {
    "normalized_text": "nguoi dan ong mac ao do dang phat bieu",
    "extracted_objects": ["person", "clothing"],
    "sub_events": []
  },
  "results": [
    {
      "rank": 1,
      "video_id": "L25_V086",
      "frame_idx": 14550,
      "clip_score": 0.2854,
      "obj_score": 0.50,
      "spatial_score": 0.00,
      "fusion_score": 0.5354,
      "pts_time": 145.5,
      "timestamp": "02:25",
      "frame_url": "/api/v1/image/L25_V086/14550",
      "detected_labels": ["person", "clothing", "microphone"],
      "watch_url": "https://youtube.com/watch?v=...?t=145"
    }
  ]
}
```

---

### 2. `POST /api/v1/search/qa` — Truy vấn Hỏi-Đáp Video (Q&A)
Truy xuất khung hình sự kiện và trả lời câu hỏi chi tiết về nội dung thị giác.

#### Request Body:
```json
{
  "event_description": "người đàn ông đứng bên cạnh xe ô tô",
  "question": "Người đó mặc áo màu gì?",
  "top_k": 5
}
```

#### Response (200 OK):
```json
{
  "answer": "Người đàn ông đang mặc áo sơ mi màu trắng.",
  "evidence_candidates": [
    {
      "video_id": "L01_V005",
      "frame_idx": 3420,
      "clip_score": 0.312,
      "fusion_score": 0.624
    }
  ],
  "latency_ms": 350.2
}
```

---

### 3. `POST /api/v1/feedback` — Chấm điểm Ground Truth (3-State Verdict)
Lưu trữ quyết định đánh giá của người dùng/tester phục vụ việc chạy `ml_tuner.py` tối ưu hóa trọng số.

#### Request Body:
```json
{
  "query_id": "Q001",
  "video_id": "L25_V086",
  "frame_idx": 14550,
  "verdict": "MATCH",
  "clip_score": 0.2854,
  "obj_score": 0.50,
  "spatial_score": 0.00,
  "fusion_score": 0.5354
}
```

*Giá trị `verdict` hỗ trợ:* `"MATCH"` (1), `"UNCERTAIN"` (2), `"MISMATCH"` (0).

#### Response (200 OK):
```json
{
  "status": "success",
  "message": "Verdict 'MATCH' saved successfully",
  "verdict": "MATCH"
}
```

---

### 4. `GET /api/v1/feedback/stats` — Thống kê Ground Truth
Xem số lượng mẫu đã chấm trong hệ thống:
```json
{
  "total_samples": 45,
  "matches": 15,
  "uncertains": 8,
  "mismatches": 22
}
```

---

### 5. `GET /api/v1/image/{video_id}/{frame_idx}` — Trích xuất ảnh On-Demand
Tự động giải mã và phục vụ ảnh JPEG trực tiếp từ:
1. Thư mục `data/raw/keyframes/{video_id}/`
2. File nén `data/zips/keyframes_{video_id[:3]}.zip` (In-memory, không tốn ổ đĩa)
3. Trích xuất frame trực tiếp từ video `.mp4` qua OpenCV.

---

### 6. `GET /api/health` — Kiểm tra trạng thái hệ thống
```json
{
  "status": "ok",
  "searcher_loaded": true,
  "project_root": "C:/Users/.../EBT_Project_2026",
  "keyframes_available": true
}
```
