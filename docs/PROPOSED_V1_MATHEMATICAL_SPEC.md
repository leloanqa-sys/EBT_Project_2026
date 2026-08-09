# 📐 Đề Xuất Chi Tiết Kiến Trúc V1 & Các Hàm Toán Học (Detailed V1 Mathematical Specification)

## Hệ Thống Truy Vết Video AI Challenge 2026 — EBT Project

> **Tài liệu Chi tiết Kỹ thuật & Đặc tả Hàm Toán học Bản Thử Nghiệm V1**
> **Tác giả:** Team Lead & Product Owner
> **Ngày cập nhật:** 09/08/2026

---

## 📑 Mục Lục
1. [Sơ Đồ Kiến Trúc Đề Xuất V1 (Proposed V1 Architecture)](#1-sơ-đồ-kiến-trúc-đề-xuất-v1-proposed-v1-architecture)
2. [Chi Tiết 7 Hàm Toán Học Cốt Lõi (Core Mathematical Specifications)](#2-chi-tiết-7-hàm-toán-học-cốt-lõi-core-mathematical-specifications)
3. [Tích Hợp Mã Nguồn Mở DSAPI & Rate Limiting Guardrails](#3-tích-hợp-mã-nguồn-mở-dsapi--rate-limiting-guardrails)
4. [Data Contract Chi Tiết (`CandidateGroup` & `CandidateFrame`)](#4-data-contract-chi-tiết-candidategroup--candidateframe)
5. [Quy Trình Kiểm Tra & Tiêu Chí Nâng Cấp ($\Delta R@k$ Benchmark)](#5-quy-trình-kiểm-tra--tiêu-chí-nâng-cấp-delta-rk-benchmark)

---

## 1. Sơ Đồ Kiến Trúc Đề Xuất V1 (Proposed V1 Architecture)

Kiến trúc V1 nâng cấp hệ thống từ "Single-pass Fusion" thành **"Multi-Stage Pipeline + Selective Verification + Adaptive Greedy Betting"**:

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
                 Frame Judge                Scene Judge
                       │                         │
                       └────────────┬────────────┘
                                    │
                              Conflict Check
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                     Low Conflict       High Conflict
                     (|S_G - S_O| ≤ 0.3) (|S_G - S_O| > 0.3)
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
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   Exploitation            Exploration
                      (90%)                   (10%)
                        │                       │
                        └───────────┬───────────┘
                                    ▼
                                 TOP 100
```

---

## 2. Chi Tiết 7 Hàm Toán Học Cốt Lõi (Core Mathematical Specifications)

### 2.1 Hàm 1: Đồng Thuận Cảm Biến AI (Multi-Sensor Weighted Consensus)

Dành cho Candidate Group $g$:
$$S_{\text{AI}}(g) = \frac{w_G \cdot c_G(g) \cdot S_G(g) + w_O \cdot c_O(g) \cdot S_O(g) + w_D \cdot c_D(g) \cdot S_D(g)}{w_G \cdot c_G(g) + w_O \cdot c_O(g) + w_D \cdot c_D(g)}$$

*   $S_G(g), S_O(g), S_D(g) \in [0, 1]$: Điểm số từ 3 cảm biến Gemini, Ollama, DeepSeek.
*   $c_G(g), c_O(g), c_D(g) \in [0, 1]$: Độ tự tin (confidence) của từng cảm biến.
*   Trọng số cơ sở được benchmark: $w_G = 0.45, w_O = 0.25, w_D = 0.30$.
*   *Lưu ý:* Nếu DeepSeek không được kích hoạt (Low Conflict), $w_D = 0$.

---

### 2.2 Hàm 2: Điểm Bằng Chứng Tổng Hợp (Evidence Fusion Score)

$$S_{\text{fusion}}(g) = \mu \cdot S_{\text{AI}}(g) + (1 - \mu) \left( w_{\text{obj}} \cdot S_{\text{obj}}(g) + w_{\text{clip}} \cdot S_{\text{clip}}(g) + w_{\text{meta}} \cdot S_{\text{meta}}(g) \right)$$

*   Khi NLP trích xuất được vật thể mục tiêu ($\text{has\_target\_objects} = \text{True}$):
    $$\mu = 0.40, \quad w_{\text{obj}} = 0.65 \text{ (Primary)}, \quad w_{\text{clip}} = 0.25 \text{ (Fallback)}, \quad w_{\text{meta}} = 0.10$$
*   Khi không trích xuất được vật thể ($\text{has\_target\_objects} = \text{False}$):
    $$\mu = 0.20, \quad w_{\text{obj}} = 0.00, \quad w_{\text{clip}} = 0.90 \text{ (Fallback)}, \quad w_{\text{meta}} = 0.10$$

---

### 2.3 Hàm 3: Hiệu Chỉnh Xác Suất Thực Tế (Probability Calibration Engine)

Chuyển đổi điểm số tổng hợp $S_{\text{fusion}}(g) \in [0, 1]$ thành Xác suất thực tế $P(g) \in [0, 1]$ thông qua hàm **Sigmoidal Temperature Calibration**:

$$P(g) = \sigma\left( \frac{S_{\text{fusion}}(g) - S_0}{T} \right) = \frac{1}{1 + \exp\left( -\frac{S_{\text{fusion}}(g) - S_0}{T} \right)}$$

*   $S_0 = 0.50$: Mốc điểm trung hòa (Neutral threshold).
*   $T = 0.15$: Hệ số nhiệt độ (Temperature parameter) kiểm soát độ dốc của đường cong xác suất.

---

### 2.4 Hàm 4: Trọng Số Ích Lợi Vị Trí Hạng BTC (Competition Rank Utility Function)

Theo quy chế tính điểm trung bình $R@1, R@5, R@20, R@50, R@100$ của BTC:

$$U(r) = \begin{cases} 
1.0 & \text{khi } r = 1 \\ 
0.8 & \text{khi } 2 \le r \le 5 \\ 
0.6 & \text{khi } 6 \le r \le 20 \\ 
0.4 & \text{khi } 21 \le r \le 50 \\ 
0.2 & \text{khi } 51 \le r \le 100 
\end{cases}$$

---

### 2.5 Hàm 5: Ích Lợi Kỳ Vọng Biên (Expected Marginal Utility - EMU)

Tại slot xếp hạng thứ $r$ (từ Rank 1 đến Rank 100):

$$EMU(g_i, f_j \mid r) = R_{\text{rem}}(g_i) \cdot P(f_j \mid g_i) \cdot U(r)$$

*   $R_{\text{rem}}(g_i)$: Lượng khối lượng xác suất còn lại (Remaining Probability Mass) của Candidate Group $g_i$. Ban đầu $R_{\text{rem}}(g_i) = P(g_i)$.
*   $P(f_j \mid g_i)$: Xác suất cục bộ frame $f_j$ đại diện cho Group $g_i$.
*   **Cơ chế suy giảm tương quan (Correlation Decay):** Sau khi chọn 1 frame $f_j$ của Group $g_i$, khối lượng xác suất của Group $g_i$ được cập nhật giảm đi:
$$R_{\text{rem}}(g_i) \leftarrow R_{\text{rem}}(g_i) \cdot (1 - \gamma \cdot P(f_j \mid g_i)) \quad \text{với } \gamma = 0.85$$

---

### 2.6 Hàm 6: Quy Tắc Khám Phá Có Trọng Số Xác Suất (Probability-Weighted Exploration Floor)

*   **Tỷ lệ phân bổ:** 90% ngân sách Exploitation (Tham lam theo EMU), 10% ngân sách Exploration (Khám phá bảo vệ).
*   **Xác suất chọn Group khi Khám phá (Exploration Selection):**
$$P_{\text{explore}}(g_i) = \frac{P(g_i)^\alpha}{\sum_k P(g_k)^\alpha} \quad \text{với } \alpha = 1.5$$

---

### 2.7 Hàm 7: Phanh Khống Chế Số Lượng Slot Cực Đại (Anti-Overconfidence Allocation Limit)

Để ngăn hiện tượng 1 mô hình quá tự tin nhầm và đổ toàn bộ 100 slots vào 1 video duy nhất:

$$N_{\text{alloc}}(g_i) \le \min\left( |g_i|, \lfloor 0.35 \times 100 \rfloor \right) = 35 \text{ frames}$$

---

## 3. Tích Hợp Mã Nguồn Mở DSAPI & Rate Limiting Guardrails

### 3.1 Cấu hình Docker cho DSAPI
Dùng `cjackhwang/ds2api` đóng vai trò Proxy Bridge kết nối với DeepSeek Web Chat:
```yaml
version: '3.8'
services:
  dsapi:
    image: cjackhwang/ds2api:latest
    container_name: ebt_dsapi_gateway
    ports:
      - "8080:8080"
    environment:
      - SERVER_PORT=8080
      - DEEPSEEK_WEB_TOKEN=${DEEPSEEK_WEB_COOKIE}
    restart: always
```

### 3.2 Luồng 3-Tier Fail-Safe cho LLM Client
```python
class SmartDeepSeekClient:
    def generate_semantic_score(self, prompt: str) -> dict:
        # Tier 1: DSAPI Local Proxy (Free & Unlimited Token)
        try:
            return self._call_dsapi(prompt, timeout=2.0)
        except Exception:
            pass

        # Tier 2: Official DeepSeek API Key (Paid Backup)
        try:
            return self._call_official_api(prompt, timeout=2.0)
        except Exception:
            pass

        # Tier 3: Local Rule-based Dictionary (Crash-proof 100% Offline)
        return self._local_rule_fallback(prompt)
```

---

## 4. Data Contract Chi Tiết (`CandidateGroup` & `CandidateFrame`)

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CandidateFrame:
    video_id: str
    frame_id: int
    pts_time: float
    local_p: float = 0.90

@dataclass
class CandidateGroup:
    group_id: str
    video_id: str
    start_frame: int
    end_frame: int
    frames: List[CandidateFrame]
    
    # Evidence Scores
    clip_score: float = 0.0
    obj_score: float = 0.0
    meta_score: float = 0.0
    
    # AI Sensor Scores, Confidences & Flags (No long reasoning string)
    gemini_score: Optional[float] = None;   gemini_conf: float = 1.0
    ollama_score: Optional[float] = None;   ollama_conf: float = 1.0
    deepseek_score: Optional[float] = None; deepseek_conf: float = 1.0
    llm_flags: List[str] = field(default_factory=list)
    
    # Calibrated Probabilities & Dynamic States
    s_fusion: float = 0.0
    p_correct: float = 0.0
    remaining_mass: float = 0.0
    expected_utility: float = 0.0
    final_rank: int = 0
```

---

## 5. Quy Trình Kiểm Tra & Tiêu Chí Nâng Cấp ($\Delta R@k$ Benchmark)

Tất cả các thử nghiệm ở bản V1 phải vượt qua bài kiểm tra benchmark độc lập so với **Bản V0 Baseline (`v0.1.0-baseline`)**:

> **"Mô hình AI nào không mang lại $\Delta R@k > 0$ so với Bản V0 Baseline sẽ bị loại bỏ ngay lập tức."**

```text
               BẢN V0 BASELINE (FAISS + Object + 5-Budget)
                                 │
                                 ▼
                     THỬ NGHIỆM V1 (P1 -> P4)
                                 │
                       Benchmark vs Baseline
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
      ΔR@k > 0 (Tăng điểm)                  ΔR@k ≤ 0 (Không tăng)
              │                                     │
              ▼                                     ▼
        KEEP (Giữ lại)                        DROP (Bỏ ngay lập tức 🗑️)
```
