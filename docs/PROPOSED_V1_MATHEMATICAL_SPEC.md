# 📐 Đặc Tả Chi Tiết Kiến Trúc V1 & Các Hàm Toán Học (V1 Mathematical Specification)

> **Trạng thái:** ACCEPTED WITH MATHEMATICAL CORRECTIONS (ĐÃ DUYỆT CÓ ĐIỀU CHỈNH TOÁN HỌC)
> **Tác giả:** Team Lead & Product Owner
> **Ngày phê duyệt:** 09/08/2026

---

## 📑 1. Sửa Đổi Kỹ Thuật & Sửa Bug Toán Học (5 Core Corrections)

1. **Sửa Bug Toán Học Renormalization (Sensor Consensus):**
   Khi DeepSeek không được kích hoạt ($w_D = 0$), mẫu số phải được chia lại (renormalize) để tổng trọng số luôn bằng $1.0$, tránh việc `S_AI` tự động bị tụt điểm.
2. **Tách Biệt Điểm Mô Hình (Score) và Độ Tự Tin (Confidence):**
   Không nhân trực tiếp $c \cdot S$ vào điểm số. $c$ dùng để phân luồng (Routing) hoặc điều chỉnh trọng số tin cậy, không trực tiếp ép điểm $0.90 \rightarrow 0.45$.
3. **Không Hard-code Tham Số Calibration:**
   Trước khi có tập Validation lớn, gọi chỉ số này là `confidence_score`. Tham số $S_0$ và $T$ phải được fit từ dữ liệu thực tế trước khi coi là xác suất $P(\text{correct})$.
4. **Tách Biệt Thuật Toán & Hạ Tầng (Infrastructure Decoupling):**
   Lõi xếp hạng Role C hoàn toàn độc lập với DSAPI. Chi tiết Docker DSAPI và Retry Gateway được quản lý riêng tại `src/common/llm_gateway.py`.
5. **Phân Tầng Lộ Trình Triển Khai Thực Chiến (V1-A $\rightarrow$ V1-B $\rightarrow$ V1-C):**
   - **V1-A (Safe Baseline):** CLIP $\rightarrow$ Object $\rightarrow$ Temporal Grouping $\rightarrow$ Gemini/Ollama $\rightarrow$ Fusion $\rightarrow$ Simple Sort.
   - **V1-B (Betting Engine):** Calibration $\rightarrow$ Expected Utility $\rightarrow$ Greedy Allocation $\rightarrow$ 35-frame cap.
   - **V1-C (Robust Betting):** Correlation decay $\rightarrow$ 90/10 Exploration $\rightarrow$ Selective DeepSeek.

---

## 🧮 2. Hệ Thống 7 Hàm Toán Học Chuẩn Hóa

### 2.1 Hàm 1: Đồng Thuận Cảm Biến AI Có Chuẩn Hóa Mẫu Số (Renormalized Consensus)

Dành cho Candidate Group $g$:

$$S_{\text{AI}}(g) = \frac{w_G \cdot S_G(g) + w_O \cdot S_O(g) + w_D \cdot S_D(g)}{w_G \cdot \mathbb{I}_G + w_O \cdot \mathbb{I}_O + w_D \cdot \mathbb{I}_D}$$

Trong đó $\mathbb{I}_X \in \{0, 1\}$ là biến chỉ thị mô hình $X$ có được kích hoạt hay không.
*   Trọng số cơ sở: $w_G = 0.45, w_O = 0.25, w_D = 0.30$.
*   Nếu DeepSeek không chạy ($\mathbb{I}_D = 0$), mẫu số tự động renormalize về $w_G + w_O = 0.70$.

---

### 2.2 Hàm 2: Điểm Bằng Chứng Tổng Hợp (Evidence Fusion Score)

$$S_{\text{fusion}}(g) = \mu \cdot S_{\text{AI}}(g) + (1 - \mu) \left( w_{\text{obj}} \cdot S_{\text{obj}}(g) + w_{\text{clip}} \cdot S_{\text{clip}}(g) + w_{\text{meta}} \cdot S_{\text{meta}}(g) \right)$$

*   Khi có nhãn vật thể ($\text{has\_target\_objects} = \text{True}$):
    $$\mu = 0.40, \quad w_{\text{obj}} = 0.65, \quad w_{\text{clip}} = 0.25, \quad w_{\text{meta}} = 0.10$$
*   Khi không có nhãn vật thể ($\text{has\_target\_objects} = \text{False}$):
    $$\mu = 0.20, \quad w_{\text{obj}} = 0.00, \quad w_{\text{clip}} = 0.90, \quad w_{\text{meta}} = 0.10$$

---

### 2.3 Hàm 3: Hiệu Chỉnh Điểm Tin Cậy (Confidence & Probability Calibration)

$$P_{\text{correct}}(g) = \sigma\left( \frac{S_{\text{fusion}}(g) - S_0}{T} \right) = \frac{1}{1 + \exp\left( -\frac{S_{\text{fusion}}(g) - S_0}{T} \right)}$$

*   $S_0, T$: Các tham số hiệu chỉnh được học từ tập Validation (không hard-code).

---

### 2.4 Hàm 4: Trọng Số Ích Lợi Vị Trí Hạng BTC (Competition Rank Utility)

$$U(r) = \begin{cases} 
1.0 & \text{khi } r = 1 \\ 
0.8 & \text{khi } 2 \le r \le 5 \\ 
0.6 & \text{khi } 6 \le r \le 20 \\ 
0.4 & \text{khi } 21 \le r \le 50 \\ 
0.2 & \text{khi } 51 \le r \le 100 
\end{cases}$$

---

### 2.5 Hàm 5: Ích Lợi Kỳ Vọng Biên (Expected Marginal Utility - EMU)

 Tại vị trí slot $r$:
$$EMU(g_i, f_j \mid r) = R_{\text{rem}}(g_i) \cdot P(f_j \mid g_i) \cdot U(r)$$

*   **Cơ chế suy giảm tương quan (Correlation Decay - V1-C):**
$$R_{\text{rem}}(g_i) \leftarrow R_{\text{rem}}(g_i) \cdot (1 - \gamma \cdot P(f_j \mid g_i)) \quad (\gamma = 0.85 \text{ - Tunable})$$

---

### 2.6 Hàm 6: Khám Phá Có Trọng Số Xác Suất (Exploration Floor - V1-C)

$$P_{\text{explore}}(g_i) = \frac{P(g_i)^\alpha}{\sum_k P(g_k)^\alpha} \quad (\alpha = 1.5 \text{ - Tunable})$$

---

### 2.7 Hàm 7: Phanh Khống Chế Số Lượng Slot Cực Đại (Anti-Overconfidence Circuit Breaker)

Giới hạn tối đa số frames chọn từ 1 Candidate Group:
$$\text{MAX\_GROUP\_ALLOCATION} = 35 \text{ frames}$$

---

## 📋 3. Data Contract Tinh Gọn V1 (`CandidateFrame` & `CandidateGroup`)

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CandidateFrame:
    video_id: str
    frame_id: int
    pts_time: float
    local_score: float = 0.0

@dataclass
class CandidateGroup:
    group_id: str
    video_id: str
    start_frame: int
    end_frame: int
    frames: List[CandidateFrame]
    
    # Evidence
    clip_score: float = 0.0
    obj_score: float = 0.0
    meta_score: float = 0.0
    
    # AI Judges (Bỏ llm_reasoning, tách score & confidence)
    gemini_score: Optional[float] = None;   gemini_conf: float = 1.0
    ollama_score: Optional[float] = None;   ollama_conf: float = 1.0
    deepseek_score: Optional[float] = None; deepseek_conf: float = 1.0
    llm_flags: List[str] = field(default_factory=list)
    
    # Derived
    s_fusion: float = 0.0
    p_correct: float = 0.0
    expected_utility: float = 0.0
    final_rank: int = 0
```
