# 📐 Kế Hoạch Kỹ Thuật & Công Thức Toán Học Cho Probabilistic Adaptive Greedy Betting Engine

## Hệ Thống Truy Vết Video AI Challenge 2026 — EBT Project

> **Tài liệu Chi tiết Kỹ thuật, Công thức Toán học & Cấu hình DSAPI**
> **Tác giả:** Team Lead & Product Owner
> **Ngày cập nhật:** 09/08/2026

---

## 📑 Mục Lục
1. [Cấu Hình & Tích Hợp Mã Nguồn Mở DSAPI (DS2API Bridge)](#1-cấu-hình--tích-hợp-mã-nguồn-mở-dsapi-ds2api-bridge)
2. [Chiến Lược Quản Lý Hạn Ngạch & Chống Sập (Quota & Rate Limiting Guardrails)](#2-chiến-lược-quản-lý-hạn-ngạch--chống-sập-quota--rate-limiting-guardrails)
3. [Hệ Thống Các Hàm Toán Học Cụ Thể (Mathematical Specifications)](#3-hệ-thống-các-hàm-toán-học-cụ-thể-mathematical-specifications)
4. [Kiến Trúc Module Code Chi Tiết (Code Layout & Function Signatures)](#4-kiến-trúc-module-code-chi-tiết-code-layout--function-signatures)
5. [Kế Hoạch Triển Khai Theo Giai Đoạn (Implementation Timeline)](#5-kế-hoạch-triển-khai-theo-giai-đoạn-implementation-timeline)

---

## 1. Cấu Hình & Tích Hợp Mã Nguồn Mở DSAPI (DS2API Bridge)

### 1.1 Nguyên lý hoạt động
DSAPI (DS2API) là một mã nguồn mở đóng vai trò **Middleware Reverse-Proxy** kết nối trực tiếp với giao thức Web Chat của DeepSeek, tự động chuyển đổi thành API chuẩn tương thích OpenAI tại địa chỉ:
`http://localhost:8080/v1/chat/completions`

### 1.2 Cấu hình Docker cho DSAPI
Tạo file `docker-compose.dsapi.yml`:
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

### 1.3 Batch Prompting (Gửi 1 Request cho Top 30 Candidates)
Để tránh quá tải và tối ưu thời gian phản hồi $< 1.5\text{s}$, hệ thống **không gọi 30 lần API riêng lẻ**, mà gom 30 Candidate Groups vào **1 đòn Batch Prompt duy nhất**:

```json
{
  "query": "A man in a red shirt speaking outdoors",
  "candidates": [
    {"group_id": "G1", "video_id": "L24_V011", "objects": ["Person", "Clothing"], "timestamp": "02:15"},
    {"group_id": "G2", "video_id": "L21_V005", "objects": ["Car", "Tree"], "timestamp": "01:10"}
  ]
}
```

---

## 2. Chiến Lược Quản Lý Hạn Ngạch & Chống Sập (Quota & Rate Limiting Guardrails)

Để đảm bảo hệ thống **tuyệt đối không bị treo hoặc cắn hết quota** khi thi đấu:

```text
               ┌────────────────────────────────────────────────────────┐
               │                REQUEST ĐẦU VÀO TRUY VẤN                │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │           KIỂM TRA SHA-256 DISK CACHE                  │
               │  • Nếu có trong Cache -> Lấy kết quả 0ms (Quota = 0)   │
               └───────────────────────────┬────────────────────────────┘
                                           │ (Cache Miss)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         RATE LIMITER GUARD (Tối đa 15 RPM)             │
               │  • Nếu vượt ngưỡng RPM -> Chuyển thẳng về Tier 3      │
               └───────────────────────────┬────────────────────────────┘
                                           │ (Dưới ngưỡng RPM)
                                           ▼
         ┌────────────────────────────────────────────────────────────────────┐
         │             LUỒNG GỌI 3-TIER API FAIL-SAFE (TIMEOUT 2.5s)          │
         ├────────────────────────────────────────────────────────────────────┤
         │ Tier 1: DSAPI (Local Docker Free Proxy)  ──> If Fail/Timeout 2.5s  │
         │ Tier 2: Official DeepSeek API Key        ──> If Fail/Quota 429     │
         │ Tier 3: Local Rule-based Dictionary      ──> Always Success (100%)│
         └────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hệ Thống Các Hàm Toán Học Cụ Thể (Mathematical Specifications)

### 3.1 Hàm Toán Học Đồng Thuận Cảm Biến AI (Multi-Sensor Weighted Consensus)

Dành cho Candidate Group $g$:
$$S_{\text{AI}}(g) = \frac{w_G \cdot c_G(g) \cdot S_G(g) + w_O \cdot c_O(g) \cdot S_O(g) + w_D \cdot c_D(g) \cdot S_D(g)}{w_G \cdot c_G(g) + w_O \cdot c_O(g) + w_D \cdot c_D(g)}$$

*   $S_G(g), S_O(g), S_D(g) \in [0, 1]$: Điểm số từ Gemini, Ollama, DeepSeek (qua DSAPI).
*   $c_G(g), c_O(g), c_D(g) \in [0, 1]$: Độ tự tin (confidence) tương ứng của từng AI.
*   Trọng số cơ sở: $w_G = 0.45, w_O = 0.25, w_D = 0.30$.

---

### 3.2 Hàm Phối Hợp Điểm Bằng Chứng (Evidence Fusion Score)

$$S_{\text{fusion}}(g) = \mu \cdot S_{\text{AI}}(g) + (1 - \mu) \left( w_{\text{obj}} \cdot S_{\text{obj}}(g) + w_{\text{clip}} \cdot S_{\text{clip}}(g) + w_{\text{meta}} \cdot S_{\text{meta}}(g) \right)$$

*   Khi NLP trích xuất được vật thể ($\text{has\_target\_objects} = \text{True}$):
    $$\mu = 0.40, \quad w_{\text{obj}} = 0.65, \quad w_{\text{clip}} = 0.25, \quad w_{\text{meta}} = 0.10$$
*   Khi không có vật thể ($\text{has\_target\_objects} = \text{False}$):
    $$\mu = 0.20, \quad w_{\text{obj}} = 0.00, \quad w_{\text{clip}} = 0.90, \quad w_{\text{meta}} = 0.10$$

---

### 3.3 Hàm Hiệu Chỉnh Xác Suất Khớp Thực Tế (Probability Calibration Engine)

Chuyển đổi điểm số $S_{\text{fusion}}(g) \in [0, 1]$ thành Xác suất thực tế $P(g) \in [0, 1]$ thông qua hàm **Sigmoidal Temperature Calibration**:

$$P(g) = \sigma\left( \frac{S_{\text{fusion}}(g) - S_0}{T} \right) = \frac{1}{1 + \exp\left( -\frac{S_{\text{fusion}}(g) - S_0}{T} \right)}$$

*   $S_0 = 0.50$: Mốc điểm trung hòa.
*   $T = 0.15$: Hệ số nhiệt độ (Temperature parameter) kiểm soát độ dốc của đường cong xác suất.

---

### 3.4 Hàm Trọng Số Ích Lợi Vị Trí Hạng BTC (Competition Rank Utility Function)

Theo quy chế tính điểm trung bình $R@1, R@5, R@20, R@50, R@100$ của BTC:

$$U(r) = \begin{cases} 
1.0 & \text{khi } r = 1 \\ 
0.8 & \text{khi } 2 \le r \le 5 \\ 
0.6 & \text{khi } 6 \le r \le 20 \\ 
0.4 & \text{khi } 21 \le r \le 50 \\ 
0.2 & \text{khi } 51 \le r \le 100 
\end{cases}$$

---

### 3.5 Hàm Tính Ích Lợi Kỳ Vọng Biên (Expected Marginal Utility - EMU)

Tại slot xếp hạng thứ $r$ (từ Rank 1 đến Rank 100):

$$EMU(g_i, f_j \mid r) = R_{\text{rem}}(g_i) \cdot P(f_j \mid g_i) \cdot U(r)$$

*   $R_{\text{rem}}(g_i)$: Lượng khối lượng xác suất còn lại (Remaining Probability Mass) của Candidate Group $g_i$. Ban đầu $R_{\text{rem}}(g_i) = P(g_i)$.
*   $P(f_j \mid g_i)$: Xác suất frame $f_j$ đại diện cho Group $g_i$.
*   **Cơ chế suy giảm tương quan (Correlation Decay):** Sau khi chọn 1 frame $f_j$ của Group $g_i$, khối lượng xác suất của Group $g_i$ được cập nhật giảm đi:
$$R_{\text{rem}}(g_i) \leftarrow R_{\text{rem}}(g_i) \cdot (1 - \gamma \cdot P(f_j \mid g_i)) \quad \text{với } \gamma = 0.85$$

---

### 3.6 Quy Tắc Khám Phá Có Trọng Số Xác Suất (Probability-Weighted Exploration Floor)

*   **Tỷ lệ phân bổ:** 90% ngân sách Exploitation (Tham lam theo EMU), 10% ngân sách Exploration (Khám phá bảo vệ).
*   **Xác suất chọn Group khi Khám phá (Exploration Selection):**
$$P_{\text{explore}}(g_i) = \frac{P(g_i)^\alpha}{\sum_k P(g_k)^\alpha} \quad \text{với } \alpha = 1.5$$

---

## 4. Kiến Trúc Module Code Chi Tiết (Code Layout & Function Signatures)

Các module mã nguồn mới sẽ được đặt tại `src/role_c_logic/betting_engine.py`:

```python
import math
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class FrameCandidate:
    frame_id: int
    pts_time: float
    local_p: float = 0.90

@dataclass
class CandidateGroup:
    group_id: str
    video_id: str
    start_frame: int
    end_frame: int
    frames: List[FrameCandidate]
    
    # Evidence Scores
    clip_score: float = 0.0
    obj_score: float = 0.0
    meta_score: float = 0.0
    
    # AI Sensor Scores & Confidences
    gemini_score: float = 0.0;    gemini_conf: float = 1.0
    ollama_score: float = 0.0;    ollama_conf: float = 1.0
    deepseek_score: float = 0.0;  deepseek_conf: float = 1.0
    
    # Calibrated Probabilities & Dynamic States
    s_fusion: float = 0.0
    p_correct: float = 0.0
    remaining_mass: float = 0.0

def compute_rank_utility(r: int) -> float:
    """Returns BTC Rank Utility U(r)."""
    if r == 1: return 1.0
    elif 2 <= r <= 5: return 0.8
    elif 6 <= r <= 20: return 0.6
    elif 21 <= r <= 50: return 0.4
    else: return 0.2

def calibrate_probability(s_fusion: float, s0: float = 0.50, t: float = 0.15) -> float:
    """Computes Sigmoidal Temperature Calibrated Probability P(g)."""
    val = (s_fusion - s0) / t
    return 1.0 / (1.0 + math.exp(-val))

def run_adaptive_greedy_betting(
    groups: List[CandidateGroup],
    total_slots: int = 100,
    gamma: float = 0.85,
    explore_ratio: float = 0.10
) -> List[Tuple[int, str, int]]:
    """
    Executes the Probabilistic Adaptive Greedy Betting algorithm to fill Top 100 submission slots.
    Returns: List of (rank, video_id, frame_id)
    """
    # 1. Initialize remaining probability mass for all groups
    for g in groups:
        g.remaining_mass = g.p_correct

    selected_slots = []
    
    for r in range(1, total_slots + 1):
        u_r = compute_rank_utility(r)
        
        # Check if exploration slot (10% exploration floor)
        is_explore = (r % 10 == 0) and (r > 5)
        
        if is_explore:
            # Probability-weighted exploration
            masses = np.array([max(1e-4, g.remaining_mass ** 1.5) for g in groups])
            probs = masses / masses.sum()
            chosen_group = np.random.choice(groups, p=probs)
        else:
            # Exploitation: Select frame with Maximum Expected Marginal Utility (EMU)
            best_emu = -1.0
            chosen_group = groups[0]
            best_frame = groups[0].frames[0]
            
            for g in groups:
                if g.remaining_mass <= 1e-4:
                    continue
                for f in g.frames:
                    emu = g.remaining_mass * f.local_p * u_r
                    if emu > best_emu:
                        best_emu = emu
                        chosen_group = g
                        best_frame = f
                        
        # Record selection
        target_frame = chosen_group.frames[0] if is_explore else best_frame
        selected_slots.append((r, chosen_group.video_id, target_frame.frame_id))
        
        # Correlation Decay: Decay remaining mass of selected group
        chosen_group.remaining_mass *= (1.0 - gamma * target_frame.local_p)
        
    return selected_slots
```

---

## 5. Kế Hoạch Triển Khai Theo Giai Đoạn (Implementation Timeline)

| Giai Đoạn | Công Việc Cụ Thể | Sản Phẩm Đầu Ra |
| :--- | :--- | :--- |
| **Giai đoạn 1** | • Dựng Docker `DSAPI` (`docker-compose.dsapi.yml`) trên port `8080`.<br>• Tạo module `SmartDeepSeekClient` với luồng 3-Tier Fail-Safe + Rate Limiter 15 RPM. | `api/dsapi_client.py` chạy ổn định |
| **Giai đoạn 2** | • Tạo module `src/role_c_logic/betting_engine.py` chứa các hàm toán học $S_{\text{AI}}, S_{\text{fusion}}, P(g), U(r), EMU$. | Module Betting Engine độc lập với unit tests |
| **Giai đoạn 3** | • Tích hợp `betting_engine.py` vào `pipeline_kis.py`.<br>• Cập nhật giao diện UI để hiển thị thông số `p_correct` và `emu_score`. | Hệ thống chính thức chạy thuật toán cược thích ứng |
