# 📘 Hướng Dẫn Sử Dụng & Vận Hành Hệ Thống EBT Vision Search 2026

> **Dành cho:** Người dùng, Giám khảo Kỹ thuật, Operators và Devs.
> **Phiên bản:** v0.1.0-baseline (V0 Stable Release)

---

## 📑 Mục Lục
1. [Khởi Chạy Hệ Thống Nhanh (Quick Start)](#1-khởi-chạy-hệ-thống-nhanh-quick-start)
2. [Hướng Dẫn Sử Dụng Giao Diện Web UI Visualizer](#2-hướng-dẫn-sử-dụng-giao-diện-web-ui-visualizer)
3. [Hướng Dẫn Sử Dụng REST API (Dành cho Devs)](#3-hướng-dẫn-sử-dụng-rest-api-dành-cho-devs)
4. [Hướng Dẫn Thêm Ảnh Keyframes JPG Thực Tế](#4-hướng-dẫn-thêm-ảnh-keyframes-jpg-thực-tế)
5. [Giải Thích Thông Số & Minh Bạch Hóa (Explainable AI)](#5-giải-thích-thông-số--minh-bạch-hóa-explainable-ai)
6. [Xử Lý Lỗi Thường Gặp (Troubleshooting)](#6-xử-lý-lỗi-thường-gặp-troubleshooting)

---

## 1. Khởi Chạy Hệ Thống Nhanh (Quick Start)

### Bước 1: Mở Terminal tại thư mục dự án
```powershell
cd C:\Users\Admin\Downloads\EBT_Project_2026
```

### Bước 2: Kích hoạt môi trường ảo (nếu có)
```powershell
.\venv\Scripts\activate
```

### Bước 3: Chạy lệnh khởi động Server
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. Hướng Dẫn Sử Dụng Giao Diện Web UI Visualizer

Mở trình duyệt bất kỳ (Chrome, Edge, Firefox) và truy cập đường dẫn:
👉 **`http://localhost:8000/`**

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 🔍 EBT VISION SEARCH 2026 — VISUALIZER UI                              │
 ├────────────────────────────────────────────────────────────────────────┤
 │ [ KIS ]  [ Q&A ]  [ TRAKE ]                                            │
 │ ┌──────────────────────────────────────────────────┐ ┌───────────────┐ │
 │ │ Nhập truy vấn (Tiếng Việt / English)...          │ │   TÌM KIẾM    │ │
 │ └──────────────────────────────────────────────────┘ └───────────────┘ │
 └────────────────────────────────────────────────────────────────────────┘
```

### Các thao tác chính:
1. **Chọn Dạng Truy Vấn (Query Type):**
   - **KIS (Textual Known Item Search):** Tìm kiếm vị trí khung hình cụ thể theo mô tả.
   - **Q&A (Visual Question Answering):** Tìm kiếm kèm giải đáp câu hỏi.
   - **TRAKE (Temporal Retrieval):** Truy vết chuỗi sự kiện có thứ tự thời gian.
2. **Nhập Câu Truy Vấn (Hỗ trợ Song ngữ VIE/ENG):**
   - *Ví dụ Tiếng Việt:* `người đàn ông mặc áo đỏ đang phát biểu`
   - *Ví dụ Tiếng Anh:* `a man in a red shirt giving a speech outdoors`
   - *Mẹo:* Hệ thống tự động lọc bỏ các từ khẩu lệnh nhiễu như `"FIND A"`, `"SEARCH FOR"`, `"TÌM KIẾM"`.
3. **Xem Kết Quả Trực Quan:**
   - **Mốc thời gian ⏱️:** Hiển thị chính xác thời lượng `Phút:Giây` trong video MP4 (Ví dụ: `⏱️ 02:52`).
   - **Thẻ nhãn Faster R-CNN:** Hiển thị các nhãn vật thể thực tế được AI phát hiện.
   - **3 Thanh Điểm Số:** Phân rã điểm tổng hợp `Fusion`, điểm ngữ nghĩa `CLIP`, và điểm vật thể `Object`.

---

## 3. Hướng Dẫn Sử Dụng REST API (Dành cho Devs)

Hệ thống cung cấp chuẩn OpenAPI Swagger tương tác tại:
👉 **`http://localhost:8000/docs`**

### Endpoint Tìm kiếm KIS:
* **URL:** `POST /api/v1/search/kis`
* **Headers:** `Content-Type: application/json`
* **Request Body Mẫu:**
```json
{
  "query": "FIND A SOCCER BALL",
  "top_k": 10,
  "query_type": "KIS"
}
```
* **Response Output Mẫu:**
```json
{
  "query_id": "kis_query_1723218580",
  "query": "FIND A SOCCER BALL",
  "total_results": 10,
  "search_time_ms": 142.5,
  "results": [
    {
      "rank": 1,
      "video_id": "L28_V016",
      "frame_idx": 4320,
      "pts_time": 172.8,
      "timestamp": "02:52",
      "clip_score": 0.1327,
      "obj_score": 1.0,
      "fusion_score": 0.4796,
      "frame_url": "/keyframes/L28_V016/4320.jpg",
      "detected_labels": ["Ball", "Football", "Person"]
    }
  ]
}
```

---

## 4. Hướng Dẫn Thêm Ảnh Keyframes JPG Thực Tế

Hiện tại hệ thống sử dụng thẻ đệm trực quan (Placeholder) để tối ưu dung lượng đĩa. Nếu muốn hiển thị ảnh JPG thật trên giao diện Web:

1. **Bước 1:** Tải bộ zip ảnh keyframe từ link Google Drive của BTC.
2. **Bước 2:** Giải nén vào thư mục `data/raw/keyframes/` theo cấu trúc:
   ```text
   data/raw/keyframes/
   ├── L21_V001/
   │    ├── 0001.jpg
   │    └── ...
   └── L28_V016/
        └── 4320.jpg
   ```
3. **Kết quả:** Ngay sau khi chép ảnh vào, Web UI sẽ tự động load ảnh thật mà không cần restart server!

---

## 5. Giải Thích Thông Số & Minh Bạch Hóa (Explainable AI)

Hệ thống được thiết kế theo nguyên tắc minh bạch (Unboxing the Black Box):

* **Tại sao điểm Fusion lại cao/thấp?**
  - Công thức tính điểm khi có nhãn vật thể:
    $$S_{\text{fusion}} = 0.65 \times S_{\text{obj}} + 0.25 \times S_{\text{clip}} + 0.10 \times S_{\text{meta}}$$
  - Bằng chứng vật thể $S_{\text{obj}}$ đóng vai trò ưu tiên hàng đầu (65%), CLIP đóng vai trò bọc lót (25%).

---

## 6. Xử Lý Lỗi Thường Gặp (Troubleshooting)

| Sự Cố | Nguyên Nhân | Cách Khắc Phục |
| :--- | :--- | :--- |
| **Lỗi `Errno 10048` khi chạy server** | Port 8000 đang được sử dụng bởi một tiến trình background khác. | Server thực ra đã chạy sẵn! Bạn truy cập thẳng vào `http://localhost:8000/`. Nếu muốn reset, tắt terminal cũ đi. |
| **Ảnh bị 404/Hiện thẻ đệm màu tối** | Chưa giải nén bộ ảnh JPG vào `data/raw/keyframes/`. | Đây là tính năng bình thường. Xem Mục 4 để ném ảnh JPG thật vào nếu muốn. |
| **Tìm kiếm bị chậm (> 3s)** | Lần đầu gọi Gemini API chưa tạo Cache. | Các lần tìm tiếp theo với câu query tương tự sẽ có tốc độ 0ms nhờ đĩa Cache SHA-256. |

---
*Tài liệu này được lưu trữ tại `docs/USER_GUIDE.md`.*
