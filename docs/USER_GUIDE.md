# 📘 Hướng Dẫn Vận Hành & Sử Dụng Hệ Thống EBT Vision Search 2026

Tài liệu hướng dẫn toàn diện dành cho Tester, Người dùng và Kỹ sư vận hành hệ thống truy xuất video **EBT Project 2026**.

---

## 📑 Mục Lục
1. [Khởi Chạy Hệ Thống Nhanh](#1-khởi-chạy-hệ-thống-nhanh)
2. [Giao Diện Web Visualizer (Tìm kiếm & Tương tác)](#2-giao-diện-web-visualizer)
3. [Giao Diện Chấm Điểm Trực Quan (Smart Visual Audit Tool)](#3-giao-diện-chấm-điểm-trực-quan)
4. [Tự Động Tìm Bộ Trọng Số Lý Tưởng (ML Tuner)](#4-tự-động-tìm-bộ-trọng-số-lý-tưởng-ml-tuner)
5. [Quy Tắc Quản Lý Dữ Liệu & Giải Nén On-Demand](#5-quy-tắc-quản-lý-dữ-liệu--giải-nén-on-demand)

---

## 1. Khởi Chạy Hệ Thống Nhanh

### Bước 1: Kích hoạt môi trường ảo
```powershell
cd C:\Users\Admin\Downloads\EBT_Project_2026
.\venv\Scripts\activate
```

### Bước 2: Khởi động API Server & Giao diện Web
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Truy cập ngay trên trình duyệt:* **`http://localhost:8000/`**

---

## 2. Giao Diện Web Visualizer

Trang chủ cung cấp khả năng tìm kiếm video theo 3 chế độ:
* **KIS (Known Item Search):** Tìm kiếm cảnh quay theo mô tả hành động, bối cảnh, vật thể.
* **Q&A (Visual Question Answering):** Trả lời câu hỏi chi tiết về nội dung cảnh quay.
* **TRAKE (Temporal Tracking):** Tìm chuỗi sự kiện theo thứ tự thời gian.

### Tính năng chấm điểm nhanh trên từng thẻ:
Trên mỗi kết quả tìm kiếm, bạn có thể bấm trực tiếp các nút:
* **`✅ Match`**: Xác nhận khung hình khớp đúng với truy vấn.
* **`⚠️ Không chắc`**: Đánh dấu khung hình có chi tiết đúng nhưng chưa trọn vẹn hoặc góc máy mờ.
* **`❌ Mismatch`**: Khung hình không liên quan.

*Dữ liệu chấm sẽ tự động lưu vào `outputs/verdicts/`.*

---

## 3. Giao Diện Chấm Điểm Trực Quan (Smart Visual Audit Tool)

Khi bạn muốn tự tạo một câu truy vấn riêng và chấm hàng loạt bằng phím tắt cực nhanh:

### Lệnh khởi chạy:
```powershell
python tools/review_tool.py --query "người đi xe máy trên đường phố" --id Q001
```
*(Nếu không truyền tham số, công cụ sẽ hiển thị dấu nhắc để bạn nhập trực tiếp).*

### Bảng phím tắt chấm bài:
| Phím tắt | Thao tác | Mô tả |
| :--- | :--- | :--- |
| <kbd>1</kbd> hoặc <kbd>Y</kbd> | **MATCH (✅)** | Đánh dấu Khớp đúng (Viền xanh lá) |
| <kbd>2</kbd> hoặc <kbd>U</kbd> | **UNCERTAIN (⚠️)** | Đánh dấu Không chắc chắn / Vừa đúng vừa sai (Viền vàng) |
| <kbd>0</kbd> hoặc <kbd>N</kbd> | **MISMATCH (❌)** | Đánh dấu Sai hoàn toàn (Viền đỏ) |
| <kbd>J</kbd> / <kbd>K</kbd> | **Next / Prev** | Chuyển sang thẻ kế tiếp / Lùi lại |
| <kbd>Ctrl + S</kbd> | **Export CSV** | Tải file `human_verdict_Qxxx.csv` về máy |

---

## 4. Tự Động Tìm Bộ Trọng Số Lý Tưởng (ML Tuner)

Sau khi bạn đã chấm từ **5 đến 15 câu truy vấn** và lưu các file CSV vào thư mục `outputs/verdicts/`:

```powershell
python tools/ml_tuner.py
```

### Kết quả đầu ra:
* Hệ thống hiển thị bảng thống kê số mẫu `MATCH`, `UNCERTAIN`, `MISMATCH`.
* Tính toán hàm mất mát phân tầng 3 mức (3-Tier Ranking Loss).
* Đưa ra bộ trọng số đề xuất `(w_clip, w_obj, w_spatial)`.
* Tự động lưu cấu hình tối ưu vào `outputs/tuning_results.json`.

---

## 5. Quy Tắc Quản Lý Dữ Liệu & Giải Nén On-Demand

Hệ thống được trang bị bộ trích xuất ảnh thông minh **Multi-Strategy Image Resolver**:
* Bạn **không cần giải nén thủ công hàng chục GB ảnh**.
* Chỉ cần thả các file zip keyframes (Ví dụ `keyframes_L25.zip`) vào thư mục:
  ```
  data/zips/
  ```
* Hệ thống sẽ tự động đọc trực tiếp từng ảnh từ trong file `.zip` vào RAM trong ~0.05ms mà không tốn dung lượng ổ đĩa.
