# 📦 Đóng Băng Mã Nguồn Bản V0 (Baseline V0 Release)
## EBT Vision Search 2026 — Stable Baseline Release

> **Trạng thái:** BẢN GỐC CHUẨN ĐÃ ĐÓNG BẰNG (STABLE V0)
> **Tác giả:** Team Lead & Product Owner
> **Ngày đóng băng:** 09/08/2026

---

## 📑 1. Tổng Quan Bản V0 Baseline

Bản V0 là **phiên bản khả dụng ổn định tuyệt đối (Deterministic Stable MVP)** được tách bạch hoàn toàn với các thử nghiệm đa tầng AI (V1 Multi-Agent/Rerank) sắp tới.

### Các đặc trưng của V0:
1. **Role A (Retrieval):** FAISS `IndexFlatIP` trên 177,321 vectors (873 videos), hot-path NumPy array $O(1) < 1\text{ms}$.
2. **Role B (NLP Engine):** 
   - Hỗ trợ song ngữ (Tiếng Việt / Tiếng Anh).
   - Tự động gọt bỏ từ khẩu lệnh nhiễu (`clean_instruction_words`).
   - Gemini API 2.0/2.5 + Bộ lọc trích xuất nhãn tại chỗ (`extract_target_objects_fallback`).
   - Sửa 100% bug mapping `n` & `frame_idx` cho Faster R-CNN OpenImages V4 JSON.
   - Hàm Score Fusion trọng số: $S_{\text{fusion}} = 0.65 S_{\text{obj}} + 0.25 S_{\text{clip}} + 0.10 S_{\text{meta}}$ khi có nhãn NLP; và $S_{\text{fusion}} = 0.90 S_{\text{clip}} + 0.10 S_{\text{meta}}$ khi không có nhãn.
3. **Role C (Ranking & Output):**
   - Temporal Event Clustering (Gom cụm thời gian $\le 15$ frames).
   - Chiến lược đặt cược 5-Budget baseline (Rank 1 an toàn, Rank 2-5 đa dạng hóa 4 Video IDs).
   - Xuất file kết quả chuẩn CSV cho BTC.
4. **Presentation Layer (API Gateway & Web UI):**
   - FastAPI Gateway (`/api/v1/search/kis`, `/api/health`).
   - Dark Mode UI Visualizer (`http://localhost:8000/`) minh bạch hóa ⏱️ Timestamp `MM:SS`, nhãn Faster R-CNN thực tế và phân rã 3 thanh điểm số.

---

## 🛡️ 2. Quy Tắc Đóng Băng & Nguyên Tắc Benchmark Bản Thử Nghiệm V1

Để tránh hiện tượng **AI Stacking, False Consensus & Echo Chamber**, các thử nghiệm V1 sắp tới phải tuân thủ nguyên tắc lạnh lùng:

> **"Không có mô hình AI nào được giữ lại chỉ vì nghe nó thông minh. Nó phải tạo ra chỉ số $\Delta R@k$ dương so với Bản V0 Baseline."**

### Bảng tiêu chuẩn đánh giá thử nghiệm V1:

| Pipeline Thử Nghiệm | R@1 | R@5 | R@20 | R@50 | R@100 | Quyết Định Keep/Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V0 Baseline** *(CLIP + Object + 5-Budget)* | Benchmark | Benchmark | Benchmark | Benchmark | Benchmark | **Gốc tham chiếu** |
| **V1.1 (+ Gemini Visual Judge)** | ? | ? | ? | ? | ? | Chỉ giữ nếu $\Delta R@k > 0$ |
| **V1.2 (+ Ollama Context Judge)** | ? | ? | ? | ? | ? | Chỉ giữ nếu $\Delta R@k > 0$ |
| **V1.3 (+ Selective DeepSeek Verifier)** | ? | ? | ? | ? | ? | Chỉ thử trên Hard Cases |
| **V1.4 (+ Adaptive Greedy Betting)** | ? | ? | ? | ? | ? | Benchmark vs Simple Sort |

---

## 🔒 3. Cam Kết Đóng Băng

Mã nguồn bản V0 được lưu vết lịch sử trên branch `dev` / tag `v0.1.0-baseline` làm mốc phục hồi an toàn bất cứ lúc nào.
