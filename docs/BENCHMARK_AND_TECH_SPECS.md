# 📊 Tài Liệu Đặc Tả Kiến Trúc & Benchmark Kỹ Thuật (V1.1 - Gemini VLM)

Tài liệu này mô tả chi tiết quá trình nâng cấp kiến trúc thay thế mô hình xử lý hình ảnh cục bộ (Qwen2-VL) bằng hệ thống gọi API đám mây (Gemini Vision API) nhằm giải quyết triệt để vấn đề cạn kiệt tài nguyên RAM/VRAM, đồng thời chuẩn hóa luồng xử lý (Code Flow) và tối ưu hóa (Optimization Flow).

---

## 1. 📈 Benchmark: Đo lường gì và để làm gì?

Trong kiến trúc mới, Benchmark (Agent 4) không chỉ đo lường điểm số thuần túy mà còn đo lường **Hiệu năng và Chi phí thực thi API**.

*   **Đo lường Final Score ($\Delta R@k$):** So sánh tỷ lệ Recall@k ($k \in \{1, 5, 20, 50, 100\}$) giữa hệ thống trước khi tích hợp Gemini (Baseline V0) và sau khi tích hợp. Đảm bảo rằng việc gửi ảnh lên Gemini không làm giảm điểm của những câu truy vấn vốn đã đúng.
*   **Tách biệt nhóm Escalation:** Phân lập kết quả Benchmark thành 2 nhóm:
    *   *Nhóm KHÔNG Escalate:* Các ứng viên rõ ràng, điểm số cao. Điểm phải được **giữ nguyên 100%**.
    *   *Nhóm CÓ Escalate:* Các ứng viên mập mờ (ambiguous), OCR, hoặc Event_Action. Điểm phải **tăng lên hoặc giữ nguyên**, không được phép giảm.
*   **Đo lường Throughput (Thời gian & Ngân sách API):** 
    *   Theo dõi `total_api_calls` thực tế sinh ra (đã trừ đi các lần Cache Hit).
    *   Tính toán tổng thời gian trễ (latency) sinh ra do network overhead.
    *   *Mục đích:* Nội suy (extrapolate) xem với số lượng câu hỏi thi thật, hệ thống có hoàn thành trong ngân sách thời gian 3 tiếng của BTC hay không.

---

## 2. 🗂️ Cập nhật cấu trúc File và Chức năng

Quá trình chuyển đổi từ Qwen2-VL sang Gemini VLM đã thay đổi các file sau:

| File | Hành động | Chức năng mới / Thay đổi |
| :--- | :--- | :--- |
| `src/role_c_logic/vqa_model.py` | **Xóa bỏ** | Gỡ bỏ hoàn toàn việc tải mô hình Qwen2-VL bằng thư viện `transformers` do gây tràn RAM (OS Error 1455). |
| `src/role_c_logic/vlm_client.py` | **Tạo mới** | Class `GeminiVisionClient`. Chịu trách nhiệm gọi API Gemini đám mây. Tích hợp cơ chế Fallback model cascade (`3.5-flash-lite` -> `3.1-flash-lite`), quản lý Rate Limit (429 Backoff), và lưu Cache JSONL/SQLite để tiết kiệm chi phí. Thực hiện gửi ảnh theo Batch (tối đa 5 ảnh). |
| `src/role_c_logic/prompt_generator.py` | **Tạo mới** | Class/Hàm `ir_graph_to_prompt`. Dịch biểu diễn ngữ nghĩa `VisualIRGraph` thành câu hỏi tiếng Anh chuẩn mực dạng Yes/No cho VLM. Ép đầu ra của VLM thành mảng JSON boolean, không cho phép mô hình sinh văn xuôi tự do. |
| `src/role_c_logic/executor.py` | **Cập nhật** | Bổ sung logic `should_escalate_to_vlm`. Đây là chốt chặn (Gateway) quyết định Candidate nào đáng bị gửi lên API (Dựa trên 4 điều kiện: tính mập mờ không gian `is_ambiguous`, điểm truy xuất CLIP `< 0.5`, có yêu cầu Event hoặc OCR). Giúp tiết kiệm 90% số lượng API call vô ích. |
| `src/pipeline.py` | **Cập nhật** | Xóa logic gọi Qwen2-VL cũ. Tại **Bước 5 (Role D)**, pipeline lấy top-20 frames, lọc qua hàm escalate, sau đó gọi `vlm_client.verify_candidates_batch()`. Những frame được Gemini đánh giá là MATCH (`True`) sẽ được cộng `RANK_1_BONUS = 5.0` điểm để đẩy lên Top 1. |

---

## 3. 🔄 Khai báo Luồng Code (Code Flow)

Luồng chạy truy vấn từ End-to-End được định nghĩa như sau:

1.  **Tiếp nhận Query (API Gateway):** Người dùng gửi câu truy vấn text.
2.  **M1 - NLP Parsing (`gemini_nlp_engine.py`):** Dịch câu truy vấn thành cấu trúc `VisualIRGraph` (bóc tách Entity, Relation, Attribute).
3.  **M2 - Vector Retrieval (`searcher.py`):** Nhúng vector câu query (CLIP) và tìm Top-500 khung hình thô qua FAISS Index (chạy dưới 10ms).
4.  **M3 - Deterministic Planning & Execution (`deterministic_planner.py` & `executor.py`):** Lập kế hoạch lọc bằng Bounding Box SQLite. Các frame thỏa mãn các điều kiện không gian sẽ được cộng điểm `spatial_score`.
5.  **M4 - VLM Escalation (Agent 3 & Gemini):**
    *   Lọc Top-20 candidates tốt nhất.
    *   Những candidate có cờ `is_ambiguous = True` hoặc điểm thấp sẽ bị đánh dấu `escalate=True`.
    *   Chuyển IR Graph thành câu hỏi qua `prompt_generator.py`.
    *   Gửi Base64 image + Prompt lên Gemini qua `vlm_client.py` theo từng Batch 5 ảnh.
6.  **Ranking (`ranking.py`):** 
    *   Nhận kết quả True/False từ Gemini, cộng thưởng `+5.0` vào `fusion_score`.
    *   Sắp xếp lại toàn bộ danh sách, cắt lấy Top-100.
7.  **Phản hồi:** Trả JSON Top-100 về cho Web UI.

---

## 4. ⚙️ Khai báo Luồng Tối ưu (Optimization Flow)

Luồng tối ưu hóa của hệ thống tập trung vào **tốc độ (Latency)** và **chi phí API (Cost/Throughput)**:

1.  **Tối ưu Mạng & Gọi API (Batching):** Thay vì gửi 1 ảnh / 1 request, hệ thống gộp tối đa 5 ảnh vào 1 payload API duy nhất và ép Gemini trả về mảng JSON `[true, false, true, ...]`. Giảm overhead HTTP.
2.  **Tối ưu Chi phí bằng Cache 3 Lớp:**
    *   *Khóa Cache:* `SHA-256(RawText + PromptVersion + FrameIdx)`.
    *   Nếu gặp lại frame ảnh trong cùng 1 câu lệnh, lấy ngay kết quả Local Cache, $0 API Call.
3.  **Tối ưu Fallback Thông minh:** 
    *   Ưu tiên gọi model nhanh gọn nhẹ: `gemini-3.5-flash-lite`. Nếu sập, rớt xuống `3.1-flash-lite`. Không dùng bản Pro để tiết kiệm thời gian chờ.
4.  **Tối ưu Băng thông Disk I/O:** 
    *   Extract ảnh base64 trực tiếp từ trong file `keyframes_Lxx.zip` bằng buffer RAM (`Multi-Strategy Image Resolver`), hoàn toàn không cần giải nén ra ổ cứng ssd.

---

## 5. 🛠️ Công Nghệ & Thư Viện Sử Dụng

| Công Nghệ / Thư Viện | Nơi Sử Dụng | Chức Năng Cụ Thể |
| :--- | :--- | :--- |
| **FastAPI & Uvicorn** | `api/main.py` | Cung cấp Web Server và RESTful API asynchronous hiệu năng cao. Phục vụ giao diện tĩnh và định tuyến HTTP. |
| **FAISS (IndexFlatIP)** | `role_a_retrieval/searcher.py` | Tìm kiếm Vector KNN siêu tốc trên không gian nhớ (In-memory). Khớp embedding của text với 177K frames video trong < 10ms. |
| **NumPy** | Xuyên suốt | Xử lý mảng (Array processing) cho Hot-Path mapping (nối ID FAISS ra Frame ID và Tọa độ thời gian) ở độ phức tạp $O(1)$. |
| **SQLite3** | `executor.py` | Cơ sở dữ liệu nhẹ để tra cứu Bounding box và metadata của vật thể, thay thế cho việc đọc hàng ngàn file JSON làm nghẽn I/O. |
| **Requests & Urllib3** | `vlm_client.py` | Tạo HTTP Client giao tiếp với Google Gemini REST API. Có cơ chế Disable SSL Warning trên máy cục bộ và thiết lập timeout. |
| **Gemini API** | `role_b_nlp` & `vlm_client.py` | Trí tuệ nhân tạo đám mây thay thế cho GPU cục bộ. Dùng để bóc tách ý định người dùng (NLP) và nhận diện, thẩm định thị giác chéo (VLM). |
