# 📐 Tài Liệu Kiến Trúc & Hướng Dẫn Vận Hành (Unified Technical Architecture)
## Hệ Thống Truy Vết Video AI Challenge 2026 — EBT Project

> **Trạng thái:** DỰ ÁN BẢN V0 BASELINE ĐÃ DỰNG THÀNH CÔNG (STABLE V0) & ĐỊNH HƯỚNG BẢN THỬ NGHIỆM V1
> **Tác giả:** Team Lead & Product Owner
> **Ngày cập nhật:** 09/08/2026

---

## 📑 Mục Lục
1. [Kiến Trúc Đã Đóng Băng (Baseline V0 Architecture)](#1-kiến-trúc-đã-đóng-bằng-baseline-v0-architecture)
2. [Định Hướng Bản Thử Nghiệm V1 (Disciplined V1 Architecture)](#2-định-hướng-bản-thử-nghiệm-v1-disciplined-v1-architecture)
3. [Quy Trình Kiểm Tra & Tiêu Chi Nâng Cấp (Delta R@k Benchmark)](#3-quy-trình-kiểm-tra--tiêu-chi-nâng-cấp-delta-rk-benchmark)
4. [Sơ Đồ Ánh Xạ Dữ Liệu (Data Mapping Layout)](#4-sơ-đồ-ánh-xạ-dữ-liệu-data-mapping-layout)
5. [Hướng Dẫn Onboarding & Vận Hành Cho Đội Kỹ Thuật (Operations Guide)](#5-hướng-dẫn-onboarding--vận-hành-cho-đội-kỹ-thuật)

---

## 1. Kiến Trúc Đã Đóng Băng (Baseline V0 Architecture)

Bản V0 là **phiên bản khả dụng ổn định tuyệt đối (Deterministic Stable MVP)** được đóng băng tại Git Tag `v0.1.0-baseline`.

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         PRESENTATION LAYER                             │
 │            Web UI (Vanilla JS/CSS)  <───>  FastAPI Gateway             │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (JSON Request)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      ROLE B: NLP ENGINE & FUSION                       │
 │  • Gemini 2.0/2.5 API + Fallback từ điển tại chỗ                       │
 │  • Gọt bỏ từ khẩu lệnh nhiễu ('FIND A', 'TÌM KIẾM')                    │
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

*   **Công thức Score Fusion V0:**
    *   Khi có vật thể từ NLP: $S_{\text{fusion}} = 0.65 S_{\text{obj}} + 0.25 S_{\text{clip}} + 0.10 S_{\text{meta}}$
    *   Khi không có vật thể: $S_{\text{fusion}} = 0.90 S_{\text{clip}} + 0.10 S_{\text{meta}}$

---

## 2. Định Hướng Bản Thử Nghiệm V1 (Disciplined V1 Architecture)

> 📘 **Đặc tả Toán học Chi tiết:** Tham khảo tài liệu [PROPOSED_V1_MATHEMATICAL_SPEC.md](file:///c:/Users/Admin/Downloads/EBT_Project_2026/docs/PROPOSED_V1_MATHEMATICAL_SPEC.md) chứa 7 công thức toán học chi tiết.

Để tránh hiện tượng **AI Stacking & False Consensus (Đồng thuận giả tạo)**, kiến trúc V1 thử nghiệm tuân thủ 5 nguyên tắc:

1. **AI Scores là Evidence, không phải Truth:** Không tin tưởng tuyệt đối vào độ tự tin của AI; bắt buộc qua bước **Validation Calibration** trước khi xếp hạng.
2. **Data Contract Tinh Gọn (`CandidateFrame`):** Bỏ chuỗi văn bản `llm_reasoning`, chỉ giữ `score`, `confidence`, `flags`.
3. **Temporal Grouping Di Chuyển Sớm:** Gom cụm ngay sau FAISS Top 300 ($\text{Top 300} \rightarrow \text{30-60 Groups}$).
4. **Selective Semantic Verifier (DeepSeek via DSAPI):** Chỉ kích hoạt DeepSeek khi có xung đột dữ liệu giữa Gemini và Ollama hoặc câu hỏi Q&A phức tạp.
5. **Adaptive Greedy Betting (Có Phanh `MAX_GROUP_ALLOCATION`):** Tự động cược theo Expected Marginal Utility $EMU = P(\text{hit}) \times U(r)$, giới hạn 1 group không chiếm quá 35% Top 100.

```text
                                USER QUERY
                                     │
                                     ▼
                           ┌──────────────────┐
                           │     ROLE B       │
                           │  Query Parsing   │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │  CLIP RETRIEVAL  │
                           │  FAISS (177K)    │
                           └────────┬─────────┘
                                    │ Top 300
                                    ▼
                           ┌──────────────────┐
                           │ Temporal Grouping│
                           │ 30 - 60 Groups   │
                           └────────┬─────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
                 Gemini Visual             Ollama Context
                       │                         │
                       └────────────┬────────────┘
                                    │
                              Conflict Check
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                     Low Conflict       High Conflict
                          │                   │
                          │            DeepSeek Verifier
                          │            (Via DSAPI Gateway)
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                           ROLE C FUSION ENGINE
                                    │
                                    ▼
                         Probability Calibration
                                    │
                                    ▼
                         Adaptive Greedy Betting
                                    │
                                    ▼
                                 TOP 100
```

---

## 3. Quy Trình Kiểm Tra & Tiêu Chí Nâng Cấp (Delta R@k Benchmark)

Mọi mô hình AI nâng cấp ở bản V1 phải vượt qua thử nghiệm độc lập:

> **"Không có mô hình nào được giữ lại chỉ vì nó nghe thông minh. Nó phải tạo ra $\Delta R@k > 0$ so với Bản V0 Baseline."**

*   **P0 (Baseline V0):** CLIP + Object JSON + 5-Budget.
*   **P1 (+ Gemini Visual Judge):** Chỉ giữ nếu $\Delta R@k > 0$.
*   **P2 (+ Ollama Context Judge):** Chỉ giữ nếu $\Delta R@k > 0$.
*   **P3 (+ Selective DeepSeek Verifier):** Chỉ dùng cho Hard Cases.
*   **P4 (+ Adaptive Greedy Betting):** Benchmark trực tiếp với Simple Sort.

---

## 4. Sơ Đồ Ánh Xạ Dữ Liệu (Data Mapping Layout)

```text
[Video ID: L24_V011]
   │
   ├── CSV Mapping: data/raw/map-keyframes/L24_V011.csv
   │     Row n=180  ──>  frame_idx = 14321  ──>  pts_time = 572.84s (09:32)
   │
   ├── JSON Objects: data/raw/objects/L24_V011/180.json
   │     Bounding Box & Entities: ["Person", "Clothing", "Building"]
   │
   └── Keyframe Image: data/raw/keyframes/L24_V011/14321.jpg (nếu có)
```

---

## 5. Hướng Dẫn Onboarding & Vận Hành Cho Đội Kỹ Thuật

### 1. Khởi chạy Server Bản V0 Baseline
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Các Lệnh Build Data Offline (Chỉ chạy khi có dữ liệu mới)
```bash
python -m src.role_a_retrieval.mapping_utils
python -m src.role_a_retrieval.build_db
python -m src.role_a_retrieval.build_index
```
