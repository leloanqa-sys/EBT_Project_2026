# 🚀 Đề Xuất Kiến Trúc Nâng Cấp: Multi-Stage Retrieval & Semantic Reranking
## Hệ Thống Truy Vết Video AI Challenge 2026

> **Tài liệu Thảo luận & Định hướng Kỹ thuật**
> **Tác giả:** Product Owner & Lead Tech Team Dev
> **Ngày lập:** 09/08/2026

---

## 🎯 Triết Lý Cốt Lõi (Core Principles)

> **"Cheap model làm Recall, Expensive model làm Precision."**
> **"CLIP chịu trách nhiệm Recall, Object/Metadata cung cấp Evidence, LLM chịu trách nhiệm Semantic Reasoning, Vision LLM chỉ xử lý Hard Cases, và Role C tối ưu theo luật điểm R@k của BTC."**

---

## 📐 Sơ Đồ Kiến Trúc Luồng Dữ Liệu (Multi-Stage Pipeline)

```text
                                USER QUERY
                                     │
                                     ▼
                           ┌──────────────────┐
                           │     ROLE B       │
                           │  Query Analysis  │
                           ├──────────────────┤
                           │ Normalize        │
                           │ Translate        │
                           │ Object extraction│
                           │ Event extraction │
                           │ Query hypotheses │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
                 CLIP Query      Object Query     Event Query
                    │               │                │
                    └───────────────┼────────────────┘
                                    ▼
                           ┌──────────────────┐
                           │     ROLE A       │
                           │    RETRIEVAL     │
                           │                  │
                           │ FAISS IndexFlatIP│
                           │ 177,321 vectors  │
                           └────────┬─────────┘
                                    │
                               Top 200~300
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │ Candidate Filtering  │
                        ├──────────────────────┤
                        │ Object JSON          │
                        │ Metadata             │
                        │ Temporal locality    │
                        │ Duplicate removal    │
                        └──────────┬───────────┘
                                   │
                                Top 30~50
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 CLIP Score    DeepSeek LLM    Vision LLM
                    │              │              │
                    │         Semantic Judge      │
                    │              │          Hard Cases
                    └──────────────┼──────────────┘
                                   ▼
                           ┌──────────────────┐
                           │   ROLE C         │
                           │  FUSION ENGINE   │
                           ├──────────────────┤
                           │ CLIP             │
                           │ LLM              │
                           │ Object           │
                           │ Metadata         │
                           │ Temporal         │
                           └────────┬─────────┘
                                    │
                              Candidate Pool
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │ COMPETITION          │
                        │ OPTIMIZER            │
                        ├──────────────────────┤
                        │ Rank 1 optimization  │
                        │ Rank 2-5 diversity   │
                        │ Rank 6-20 coverage   │
                        │ Rank 21-50 expansion │
                        │ Rank 51-100 explore  │
                        └──────────┬───────────┘
                                   │
                                   ▼
                              TOP 100
```

---

## 🔬 Chi Tiết 6 Giai Đoạn Vận Hành

### Giai Đoạn 1: Role B — Query Understanding & Hypotheses Generation
* **Đầu vào:** Câu truy vấn thô (Tiếng Việt / Tiếng Anh).
* **Nhiệm vụ:** Không tìm kiếm. Bóc tách thành `ParsedQuery` và tạo nhiều giả thuyết tìm kiếm (`Q1` ngữ nghĩa gốc, `Q2` dịch tiếng Anh, `Q3` hướng vật thể, `Q4` hướng sự kiện).
* **Mục tiêu:** Tối đa hóa **Recall** trước khi đi vào giai đoạn xếp hạng.

### Giai Đoạn 2: Role A — Retrieval (FAISS 177K Vectors)
* **Nhiệm vụ:** FAISS `IndexFlatIP` quét toàn bộ 177,321 vector trong $< 30\text{ms}$.
* **Đầu ra:** Top 200 – 300 ứng viên thô.

### Giai Đoạn 3: Candidate Filtering & Deduplication
* **Nhiệm vụ:** Lọc nhanh dựa trên Faster R-CNN Object JSON, Metadata và Gom cụm trùng lặp thời gian (Temporal Clustering).
* **Đầu ra:** Co hẹp còn **Top 30 – 50 ứng viên chất lượng cao**.

### Giai Đoạn 4: DeepSeek LLM — Semantic Judge (Độc Lập)
* **Nhiệm vụ:** Trả lời câu hỏi *"Candidate này có hợp lý về mặt lý luận ngữ nghĩa (Semantic Feasibility) hay không?"*
* **Chế độ độc lập (Mode A - Baseline):** Không truyền điểm CLIP cho DeepSeek để tránh hiện tượng suy luận thiên lệch (Anchoring Bias).

### Giai Đoạn 5: Vision LLM (Ollama / BLIP) — Chỉ Dành Cho Hard Cases
* **Tiêu chí kích hoạt (Trigger Criteria):**
  1. Khi có sự xung đột điểm số: $\text{CLIP Score} \ge 0.85$ nhưng $\text{DeepSeek Score} \le 0.50$ (hoặc ngược lại).
  2. Khi câu hỏi thuộc dạng bài Q&A đếm số lượng, nhận diện màu sắc chi tiết hoặc quan hệ không gian phức tạp.

### Giai Đoạn 6: Role C — Competition Optimizer & 5-Budget Strategy
* **Nhiệm vụ:** Tối ưu hóa thứ tự 100 đáp án nộp bài theo công thức tính điểm $R@1, R@5, R@20, R@50, R@100$ của BTC.

---

## 📋 Extended Data Contract (`CandidateFrame`)

```python
@dataclass
class CandidateFrame:
    video_id: str
    frame_id: int
    pts_time: float
    fps: float
    
    # Stage 1: Retrieval
    clip_score: float = 0.0
    
    # Stage 2: Evidence
    detected_objects: List[str] = field(default_factory=list)
    metadata_match: float = 0.0
    
    # Stage 3: Semantic Reasoning
    llm_score: float = 0.0
    llm_reasoning: str = ""
    vision_score: float = 0.0
    
    # Stage 4: Fusion & Competition
    fusion_score: float = 0.0
    competition_score: float = 0.0
    final_rank: int = 0
```
