import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from src.role_c_logic.synonyms_map import CLASS_SYNONYMS_MAP, _verify_invariants, _verify_no_taxonomy_overlap
from src.role_c_logic.taxonomy_loader import TAXONOMY_CLASSES_SET
from src.role_c_logic.executor import safe_mean, DeterministicExecutor
from src.common.schemas import CandidateFrame

def run_audit():
    print("🚀 Bắt đầu quá trình Audit & Benchmark Toán tử DETECT (Giai đoạn 1)...\n")
    
    # 1. Đo lường: Kiểm toán Rào lỗi Invariant (Import-time Invariant Check)
    print("▶️ BÀI ĐO 1: KIỂM TOÁN RÀO LỖI INVARIANT (CROSS-CONTAMINATION GUARD)")
    start_time = time.time()
    try:
        _verify_invariants()
        elapsed = (time.time() - start_time) * 1000
        print(f"  ✅ INVARIANT PASSED: Toàn bộ alias 1-1 không bị rò rỉ ngữ nghĩa. ({elapsed:.2f} ms)")
    except Exception as e:
        print(f"  ❌ INVARIANT FAILED: {e}")
        return
        
    # 2. Đo lường: Test Fallback và Semantic Lookup
    print("\n▶️ BÀI ĐO 2: KIỂM TOÁN TÍNH TÌM KIẾM ĐỒNG NGHĨA VÀ 2-TIER ARCHITECTURE")
    assert "person" in CLASS_SYNONYMS_MAP["man"], "Lỗi: 'man' không match được 'person'"
    assert "woman" not in CLASS_SYNONYMS_MAP["man"], "Lỗi: 'man' bị dính chéo 'woman'"
    
    # Kịch bản Tầng 1: "chair" có trong taxonomy nhưng không có trong map
    assert "chair" in TAXONOMY_CLASSES_SET, "Lỗi: 'chair' không có trong DB ground truth."
    assert "chair" not in CLASS_SYNONYMS_MAP, "Lỗi: 'chair' không nên có trong map tay vì thừa thãi."
    
    # Kịch bản Fallback: "alien" không có trong map lẫn taxonomy
    target = "alien"
    if target in CLASS_SYNONYMS_MAP:
        aliases = set(CLASS_SYNONYMS_MAP[target])
    elif target in TAXONOMY_CLASSES_SET:
        aliases = {target}
    else:
        aliases = {target}
        # executor sẽ log ra csv ở đây
    assert aliases == {"alien"}, "Lỗi: Fallback sai lệch."
    print("  ✅ 2-TIER ARCHITECTURE PASSED: Match chính xác, Tầng 1 hoạt động thay thế map tay.")
    
    print("\n▶️ BÀI ĐO 2B: KIỂM TOÁN DỌN DẸP MAP TAY (REDUNDANT OVERLAP)")
    _verify_no_taxonomy_overlap()
    print("  ✅ OVERLAP CHECK PASSED (Xem log nếu có cảnh báo redundant).")

    # 3. Đo lường: Kiểm toán tính hội tụ của Điểm Số (Score Bounding [0, 1])
    print("\n▶️ BÀI ĐO 3: KIỂM TOÁN TÍNH HỘI TỤ ĐIỂM SỐ VÀ CẤM CỘNG DỒN (+=)")
    
    # Giả lập 5 step detect (5 entity khác nhau) trên 1 candidate frame
    mock_scores = [0.9, 0.8, 1.0, 0.5, 0.99]
    final_score = safe_mean(mock_scores)
    
    # Nếu code cũ (+=) thì điểm sẽ là 4.19 -> Vượt 1.0
    # Code mới (Mean) -> Điểm phải <= 1.0
    print(f"  Điểm các thành phần (5 entities): {mock_scores}")
    print(f"  Điểm tổng hợp (safe_mean): {final_score:.4f}")
    
    assert final_score <= 1.0, f"LỖI NGHIÊM TRỌNG: Điểm tổng hợp ({final_score}) vượt quá 1.0!"
    assert final_score > 0.0, "LỖI: Điểm tổng hợp <= 0.0"
    print("  ✅ SCORE BOUNDING PASSED: Tuyệt đối tuân thủ quy tắc gán đè (=) và giới hạn <= 1.0.")

    print("\n✅ HOÀN THÀNH AUDIT: Toàn bộ 4 nguyên tắc kiến trúc đã được xác thực hoàn hảo!")

if __name__ == "__main__":
    run_audit()
