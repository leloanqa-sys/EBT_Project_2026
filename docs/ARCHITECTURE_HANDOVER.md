# 📐 Tài Liệu Bàn Giao Kiến Trúc Kỹ Thuật (Tech Lead Handover Document)
## Hệ Thống Truy Vết Video AI Challenge 2026 — EBT Project

> **Dành cho:** Đội ngũ Kỹ thuật tiếp quản, Devs Frontend/Backend, DevOps và Giám khảo Kỹ thuật.
> **Tác giả:** Tech Lead Team Dev
> **Ngày cập nhật:** 09/08/2026

---

## 📑 Mục Lục
1. [Tổng Quan Kiến Trúc & Luồng Dữ Liệu (Architecture & Data Flow)](#1-tổng-quan-kiến-trúc--luồng-dữ-liệu)
2. [Chi Tiết 3 Role & Lý Do Lựa Chọn Công Nghệ (Technical Justifications)](#2-chi-tiết-3-role--lý-do-lựa-chọn-công-nghệ)
3. [Cơ Chế Giải Thích (Explainable AI - XAI) & Minh Bạch Hóa](#3-cơ-chế-giải-thích-explainable-ai---xai--minh-bạch-hóa)
4. [Chiến Lược Đặt Cược 5-Budget & Tối Ưu Điểm Số R@k](#4-chiến-lược-đặt-cược-5-budget--tối-ưu-điểm-số-rk)
5. [Sơ Đồ Ánh Xạ Dữ Liệu (Data Mapping Layout)](#5-sơ-đồ-ánh-xạ-dữ-liệu-data-mapping-layout)
6. [Hướng Dẫn Setup & Vận Hành Cho Đội Tiếp Quản (Operations Guide)](#6-hướng-dẫn-setup--vận-hành-cho-đội-tiếp-quản)

---

## 1. Tổng Quan Kiến Trúc & Luồng Dữ Liệu

Hệ thống được thiết kế theo kiến trúc **Phân tầng Trách nhiệm (Decoupled Multi-Role Architecture)** chia làm 3 Role độc lập kết nối qua Data Contract chuẩn hóa (`schemas.py`), đóng gói bởi **FastAPI Gateway**.

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         PRESENTATION LAYER                             │
 │            Web UI (Vanilla JS/CSS)  <───>  FastAPI Gateway             │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (JSON Request)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      ROLE B: NLP ENGINE & FUSION                       │
 │  • Gemini 1.5 Flash (Structured Output) ──> Dịch Object & Decompose    │
 │  • SHA-256 Disk Cache ──> Giảm Latency 0ms cho query lặp               │
 │  • Faster R-CNN Overlap Scoring ──> S_obj                               │
 └─────────────────┬───────────────────────────────────┬──────────────────┘
                   │ Query Vector                      │ Parsed Query
                   ▼                                   ▼
 ┌───────────────────────────────────┐   ┌────────────────────────────────┐
 │     ROLE A: RETRIEVAL ENGINE      │   │    ROLE C: LOGIC & RANKING     │
 │  • FAISS IndexFlatIP (177,321)    │   │  • Event Temporal Clustering   │
 │  • Hot-Path NumPy Array O(1) < 1ms├───┼─>• 5-Budget Betting Strategy   │
 │  • Cold-Path SQLite Persistent    │   │  • CSV Submission Generator    │
 └───────────────────────────────────┘   └────────────────────────────────┘
```

---

## 2. Chi Tiết 3 Role & Lý Do Lựa Chọn Công Nghệ

### 🔹 ROLE A: Retrieval Engine (Truy Xuất Vector Tốc Độ Cao)

*   **Nhiệm vụ:** Quản lý 177,321 vector CLIP ViT-B/32 của 873 video. Trả về Top K ứng viên thô trong thời gian $< 50\text{ms}$.
*   **Công nghệ sử dụng:** `FAISS (IndexFlatIP)` + `NumPy Columnar Arrays (.npz)` + `SQLite (media.db)`.
*   **Lý do lựa chọn & Giải thích kiến trúc:**
    1. **Tại sao chọn `IndexFlatIP` thay vì IVF/HNSW?**
       - Số lượng vector là 177,321 (khoảng 350MB RAM). Ở quy mô này, `IndexFlatIP` chạy tìm kiếm Brute-force Cosine Similarity trên CPU chỉ mất **~15-30ms**, đạt **độ chính xác 100% (Recall = 1.0)** mà không bị mất mát thông tin như các chỉ mục xấp xỉ (IVF/HNSW).
    2. **Tại sao dùng Kiến trúc 2 Tầng (Hot-Path vs Cold-Path)?**
       - **Online Hot-Path (`mapping_array.npz`)**: Khi FAISS trả về `faiss_id`, ta tra cứu trực tiếp qua mảng NumPy đơn cột $O(1)$. Độ trễ tra cứu metadata $< 1\text{ms}$, hoàn toàn loại bỏ I/O đọc ổ đĩa hay truy vấn cơ sở dữ liệu trên luồng thời gian thực.
       - **Offline Cold-Path (`media.db` - SQLite)**: Phục vụ kiểm duyệt (Audit), ghi log và debug offline.

---

### 🔹 ROLE B: NLP Engine & Object Fusion (Xử Lý Ngôn Ngữ & Khớp Vật Thể)

*   **Nhiệm vụ:** Hiểu câu truy vấn tiếng Việt, bóc tách vật thể physical, dịch thuật ngữ cảnh và cộng gộp điểm tin cậy thô (`fusion_score`).
*   **Công nghệ sử dụng:** `Gemini 1.5 Flash (Structured JSON)` + `Faster R-CNN OpenImages V4` + `SHA-256 Disk Cache`.
*   **Lý do lựa chọn & Giải thích kiến trúc:**
    1. **Tại sao chọn Gemini 1.5 Flash?**
       - Tốc độ phản hồi cực nhanh (~200-400ms), chi phí thấp, hỗ trợ tiếng Việt xuất sắc.
       - Sử dụng tính năng **JSON Structured Outputs** để ép Gemini trả về đúng định dạng JSON mong muốn mà không bị dính văn bản thừa.
    2. **Tại sao dùng Cơ chế Fallback 2 Tầng (Fault Tolerance)?**
       - *Tầng 1:* Gọi Gemini API.
       - *Tầng 2 (Fallback):* Nếu mất mạng hoặc API lỗi, tự động chuyển sang mô hình Regex chuẩn hóa tiếng Việt không dấu.
       - *Cam kết:* Hệ thống "không bao giờ chết" (Crash-proof) trong môi trường thi đấu.
    3. **Tại sao phải kết hợp Object Detection (Faster R-CNN)?**
       - Mô hình CLIP rất mạnh về ngữ nghĩa tổng thể nhưng **yếu trong việc nhận diện các vật thể nhỏ cụ thể** (như áo đỏ, xe máy, cái cốc).
       - Công thức Fusion ghép nối:
         $$\text{fusion\_score} = 0.70 \times \text{clip\_score} + 0.20 \times \text{obj\_score} + 0.10 \times \text{meta\_score}$$

---

### 🔹 ROLE C: Logic Engineer & Ranking (Xếp Hạng & Chiến Lược Đặt Cược)

*   **Nhiệm vụ:** Gom cụm thời gian (Clustering), áp dụng chiến thuật 5-Budget để tối đa hóa điểm số R@k, và xuất file kết quả CSV.
*   **Lý do lựa chọn & Giải thích kiến trúc:**
    1. **Tại sao phải Gom Cụm Thời Gian (Temporal Event Clustering)?**
       - Các frame liên tiếp trong cùng 1 video (vd: frame 100, 101, 102) có điểm CLIP gần như giống hệt nhau. Nếu không gom cụm, Top 5 kết quả sẽ bị chiếm trọn bởi 1 video duy nhất $\rightarrow$ Rủi ro cực cao nếu video đó sai.
       - Thuật toán gom cụm gộp các frame cách nhau $\le 15$ frames thành 1 sự kiện đại diện.

---

## 3. Cơ Chế Giải Thích (Explainable AI - XAI) & Minh Bạch Hóa

Để giải quyết bài toán "Hộp Kín" (Black Box) khi chưa có ảnh JPG, hệ thống cung cấp 3 tham số giải thích minh bạch trên từng thẻ kết quả UI:

1. **⏱️ Mốc thời gian Video (`timestamp` & `pts_time`):** Hiển thị chính xác thời lượng `MM:SS` (Ví dụ `02:15`). Giám khảo chỉ cần mở video MP4 gốc tua đến đúng giây `02:15` là đối chiếu được ngay.
2. **🏷️ Nhãn Faster R-CNN thực tế (`detected_labels`):** Bóc tách các nhãn vật thể thực tế đọc từ file JSON của BTC (VD: `Person`, `Clothing`, `Tree`, `Car`).
3. **📊 Phân rã 3 thanh điểm số:** Tách biệt rõ điểm ngữ nghĩa (`CLIP`), điểm vật thể (`Object`), và điểm tổng hợp (`Fusion`).

---

## 4. Chiến Lược Đặt Cược 5-Budget & Tối Ưu Điểm Số R@k

Công thức tính điểm của BTC dựa trên trung bình cộng của $R@1, R@5, R@20, R@50, R@100$. Qua phân tích toán học:

$$\text{Trọng số vị trí Rank 1} = 1.0 \quad \text{gấp 5 lần} \quad \text{Trọng số vị trí Rank 51-100} = 0.2$$

### Chiến thuật Phân bổ 100 Đáp Án (5-Budget Strategy):
*   **Rank 1 (Vị trí Vàng):** Dành cho ứng viên có `fusion_score` cao nhất toàn bộ hệ thống (An toàn tuyệt đối).
*   **Ranks 2 – 5 (Vùng Đa Dạng Hóa):** Bắt buộc chọn từ **4 video_id KHÁC NHAU**. Lý do: Nếu video #1 bị sai, 4 video ở Rank 2-5 sẽ "bọc lót" để ăn trọn điểm từ ngưỡng $R@5$ đến $R@100$.
*   **Ranks 6 – 20:** Lưới an toàn cấp 1 (mở rộng thêm các video tiềm năng).
*   **Ranks 21 – 100:** Lưới an toàn cấp 2 (thử nghiệm các giả thuyết frame khác nhau).

---

## 5. Sơ Đồ Ánh Xạ Dữ Liệu (Data Mapping Layout)

Để đội ngũ tiếp quản không bị nhầm lẫn giữa các chỉ số:

```text
[Tên Video: L24_V011]
   │
   ├── CSV Mapping: data/raw/map-keyframes/L24_V011.csv
   │     Dòng n=180  ──>  frame_idx = 14321  ──>  pts_time = 572.84s (09:32)
   │
   ├── JSON Objects: data/raw/objects/L24_V011/180.json
   │     Chứa Bounding Box & Class Entities: ["Person", "Clothing", "Building"]
   │
   └── Keyframe Image: data/raw/keyframes/L24_V011/14321.jpg (nếu có ảnh thô)
```

> ⚠️ **Lưu ý quan trọng:** File JSON trong `objects/` được đặt tên theo chỉ số dòng $n$ (`180.json`), trong khi URL hình ảnh và submission CSV dùng chỉ số `frame_idx` (`14321`). Module `object_matcher.py` đã có hàm `get_n_from_map_csv()` tự động chuyển đổi minh bạch.

---

## 6. Hướng Dẫn Setup & Vận Hành Cho Đội Tiếp Quản

### 1. Yêu cầu Môi trường
* Python 3.10+
* RAM: Tối thiểu 4GB (Khuyên dùng 8GB+)
* Ô cứng: ~2GB cho bộ dữ liệu processed

### 2. Cài đặt Phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Các lệnh Build Offline (Chỉ chạy 1 lần khi có data mới)
```bash
# 1. Kiểm tra toàn vẹn dữ liệu & build global mapping
python -m src.role_a_retrieval.mapping_utils

# 2. Build SQLite DB cold-path
python -m src.role_a_retrieval.build_db

# 3. Build FAISS Index
python -m src.role_a_retrieval.build_index
```

### 4. Khởi chạy Server Online
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---
*Tài liệu này được đóng băng và lưu trữ tại `docs/ARCHITECTURE_HANDOVER.md`.*
