# API Documentation — EBT Vision Search

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### 1. `POST /api/v1/search/kis` — KIS Search

Thực hiện truy vấn Known Item Search (KIS). Nhận câu truy vấn tiếng Việt, chạy qua pipeline NLP → Retrieval → Fusion → Ranking và trả về kết quả JSON kèm giải thích.

**Request Body:**

```json
{
  "query": "người đàn ông mặc áo đỏ đang phát biểu",
  "query_type": "KIS",
  "top_k": 20,
  "question": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Câu truy vấn tiếng Việt (min 1 ký tự) |
| `query_type` | string | ❌ | `"KIS"` | Loại truy vấn: `KIS`, `QA`, `TRAKE` |
| `top_k` | int | ❌ | `20` | Số kết quả trả về (1–100) |
| `question` | string | ❌ | `null` | Câu hỏi đi kèm (chỉ dùng cho QA) |

**Response:**

```json
{
  "query_id": "api_a1b2c3d4",
  "query": "người đàn ông mặc áo đỏ đang phát biểu",
  "query_type": "KIS",
  "total_results": 20,
  "search_time_ms": 245.3,
  "cache_hit": false,
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
      "clip_score": 0.2469,
      "obj_score": 0.5,
      "fusion_score": 0.2728,
      "frame_url": "/keyframes/L25_V086/14550.jpg",
      "detected_labels": ["person", "clothing", "building"]
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `query_id` | ID duy nhất của truy vấn |
| `search_time_ms` | Thời gian xử lý (ms) |
| `cache_hit` | `true` nếu kết quả NLP được lấy từ cache |
| `parsed_info.normalized_text` | Văn bản đã chuẩn hóa |
| `parsed_info.extracted_objects` | Các object được Gemini trích xuất & dịch sang English labels |
| `results[].clip_score` | Điểm tương đồng CLIP (cosine similarity) |
| `results[].obj_score` | Điểm khớp Object Detection (0.0–1.0) |
| `results[].fusion_score` | Điểm tổng hợp `0.7×CLIP + 0.2×Object + 0.1×Meta` |
| `results[].frame_url` | URL tương đối để load ảnh keyframe |
| `results[].detected_labels` | Labels Faster R-CNN phát hiện trong frame |

---

### 2. `GET /api/health` — Health Check

Kiểm tra trạng thái hệ thống.

**Response:**

```json
{
  "status": "ok",
  "searcher_loaded": true,
  "project_root": "C:/Users/.../EBT_Project_2026",
  "keyframes_available": true
}
```

---

### 3. `GET /api/v1/search/status` — Pipeline Status

Kiểm tra trạng thái pipeline tìm kiếm.

**Response:**

```json
{
  "pipeline_ready": true,
  "mode": "LIVE"
}
```

Khi `mode` = `"DEMO"`, API sẽ trả về dữ liệu mẫu thay vì kết quả thực.

---

## Serving Static Files

| Path | Mô tả |
|------|--------|
| `/static/*` | Frontend UI (HTML, CSS, JS) |
| `/keyframes/{video_id}/{frame_idx}.jpg` | Ảnh keyframe từ `data/raw/keyframes/` |

---

## Khởi chạy Server

```bash
# Từ thư mục gốc EBT_Project_2026/
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Truy cập:
- **UI**: http://localhost:8000/
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc
