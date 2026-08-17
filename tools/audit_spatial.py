import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
from src.role_c_logic.executor import DeterministicExecutor
from src.common.thresholds import SPATIAL_MIN_DIST
from src.common.schemas import CandidateFrame

def mock_get_metadata(video_id: str, frame_idx: int):
    # Trả về các boxes cho video "test"
    if frame_idx == 1:
        # Tình huống 2D: Lệch trục Y
        # Người ở góc trên trái, Xe ở góc dưới phải
        return [
            {"class_entity": "person", "score": 0.9, "ymin": 0.0, "xmin": 0.0, "ymax": 0.2, "xmax": 0.2},
            {"class_entity": "car", "score": 0.9, "ymin": 0.8, "xmin": 0.8, "ymax": 1.0, "xmax": 1.0}
        ]
    elif frame_idx == 2:
        # Tình huống 3D: Mâu thuẫn (XOR conflict)
        # Person A: ymax lớn hơn (đáy chạm đất hơn -> trước), nhưng diện tích nhỏ hơn -> sau
        # Ymax A = 0.9, Area A = 0.2 * 0.2 = 0.04
        # Person B: ymax nhỏ hơn (đáy cao hơn -> sau), nhưng diện tích lớn hơn -> trước
        # Ymax B = 0.8, Area B = 0.4 * 0.4 = 0.16
        return [
            {"class_entity": "person", "score": 0.9, "ymin": 0.7, "xmin": 0.1, "ymax": 0.9, "xmax": 0.3}, # Person A
            {"class_entity": "person", "score": 0.9, "ymin": 0.4, "xmin": 0.5, "ymax": 0.8, "xmax": 0.9}  # Person B
        ]
    return []

def run_spatial_audit():
    print("🚀 Bắt đầu quá trình Audit Toán tử SPATIAL (Giai đoạn 2)...\n")
    
    # Setup mock executor
    executor = DeterministicExecutor(searcher=None)
    # Monkey-patch get_metadata
    class MockCache:
        def get_metadata(self, vid, fidx):
            return mock_get_metadata(vid, fidx)
    executor.meta_cache = MockCache()

    # ---------------------------------------------------------
    # TEST 1: Negative Test 2D - Lệch trục Y
    # Cần chứng minh relation "left_of" trả False vì x_overlap không thỏa mãn Axis Alignment
    # ---------------------------------------------------------
    print("▶️ BÀI ĐO 1: KIỂM TOÁN LỆCH TRỤC 2D (NEGATIVE TEST)")
    candidates_1 = [CandidateFrame(faiss_id=1, video_id="test", frame_idx=1)]
    args_1 = {"source_label": "person", "target_label": "car"}
    # Person tâm x = 0.1, Car tâm x = 0.9 -> Person left_of Car là True về mặt center_x.
    # NHƯNG Y-overlap: min(ymax) = 0.2, max(ymin) = 0.8 -> 0.2 > 0.8 là False.
    result_1 = executor._op_spatial(candidates_1, args_1, "left_of")
    
    c1 = result_1[0]
    assert c1.spatial_score == 0.0, f"Lỗi: Rào lỗi Axis Alignment 2D thất bại! Điểm = {c1.spatial_score}"
    assert c1.is_ambiguous == False, "Lỗi: is_ambiguous vô tình bị bật ở nhánh 2D!"
    print("  ✅ 2D AXIS ALIGNMENT PASSED: Đã chặn thành công False Positive do lệch trục.")

    # ---------------------------------------------------------
    # TEST 2: Negative Test 3D - Mâu thuẫn tín hiệu XOR
    # Cần chứng minh is_ambiguous = True khi ymax và area cãi nhau
    # ---------------------------------------------------------
    print("\n▶️ BÀI ĐO 2: KIỂM TOÁN ESCALATION 3D (NEGATIVE TEST)")
    candidates_2 = [CandidateFrame(faiss_id=2, video_id="test", frame_idx=2)]
    # Person vs Person -> lấy tất cả các cặp
    args_2 = {"source_label": "person", "target_label": "person"}
    result_2 = executor._op_spatial(candidates_2, args_2, "front")
    
    c2 = result_2[0]
    assert c2.is_ambiguous == True, "Lỗi: Heuristics 3D không phất cờ is_ambiguous khi có mâu thuẫn!"
    print(f"DEBUG: c2.spatial_score = {c2.spatial_score}")  # kỳ vọng 0.5, không phải 1.0
    print("  ✅ 3D ESCALATION PASSED: Cờ is_ambiguous đã bật chính xác để đẩy cho VLM.")
    
    print("\n✅ HOÀN THÀNH AUDIT SPATIAL: Toán tử Không gian đã vững như bàn thạch!")

if __name__ == "__main__":
    run_spatial_audit()
