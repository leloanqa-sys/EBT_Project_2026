"""
Chẩn đoán: Tại sao L25_V017 / "người cầm cờ" / "trường SISU" xuất hiện liên tiếp?

Phân tích pipeline.py - Temporal Interpolation (STEP 5.5)
"""
import numpy as np
import json

# Simulate the pipeline bug
# Giả sử VLM trả về match=True cho một frame của L25_V017
# Pipeline sẽ tạo synthetic candidates như sau:

def simulate_temporal_interpolation(anchor_video_id, anchor_frame_idx, anchor_pts_time, anchor_fusion_score):
    INTERPOLATE_WINDOW_SEC = 3.0
    betting_interval = 8 if anchor_fusion_score >= 5.0 else 5
    
    fps = anchor_frame_idx / anchor_pts_time if anchor_pts_time > 0.1 else 25.0
    if fps < 10 or fps > 60:
        fps = 25.0
    
    frames_to_span = int(INTERPOLATE_WINDOW_SEC * fps)
    start_frame = max(0, anchor_frame_idx - frames_to_span)
    end_frame = anchor_frame_idx + frames_to_span
    
    synthetics = []
    
    # Left
    curr_frame = anchor_frame_idx - betting_interval
    while curr_frame >= start_frame:
        dist_sec = abs(anchor_frame_idx - curr_frame) / fps
        synthetics.append({
            "video_id": anchor_video_id,
            "frame_idx": int(curr_frame),
            "direction": "LEFT",
            "fusion_score": anchor_fusion_score - 0.01 * dist_sec
        })
        curr_frame -= betting_interval
    
    # Right
    curr_frame = anchor_frame_idx + betting_interval
    while curr_frame <= end_frame:
        dist_sec = abs(curr_frame - anchor_frame_idx) / fps
        synthetics.append({
            "video_id": anchor_video_id,
            "frame_idx": int(curr_frame),
            "direction": "RIGHT",
            "fusion_score": anchor_fusion_score - 0.01 * dist_sec
        })
        curr_frame += betting_interval
    
    return synthetics, fps, frames_to_span

print("=== BUG SIMULATION: Temporal Interpolation ===\n")

# Case 1: VLM Match (fusion_score >= 5.0) -> betting_interval = 8
anchor_frame_idx = 5000
anchor_pts_time = 166.67  # 2.78 min
anchor_fusion_score = 5.5  # VLM match + bonus

synthetics, fps, span = simulate_temporal_interpolation(
    "L25_V017", anchor_frame_idx, anchor_pts_time, anchor_fusion_score
)

print(f"Anchor: L25_V017 frame={anchor_frame_idx}, pts={anchor_pts_time}s, fusion={anchor_fusion_score}")
print(f"Detected FPS: {fps:.1f}")
print(f"frames_to_span = int({INTERPOLATE_WINDOW_SEC} * {fps:.1f}) = {span}")
INTERPOLATE_WINDOW_SEC = 3.0
print(f"Window: [{anchor_frame_idx - span}, {anchor_frame_idx + span}]")
print(f"betting_interval = {8 if anchor_fusion_score >= 5.0 else 5}")
print(f"\nSynthetic frames generated: {len(synthetics)}")
print(f"Score range: {min(s['fusion_score'] for s in synthetics):.4f} - {max(s['fusion_score'] for s in synthetics):.4f}")

for s in synthetics[:5]:
    print(f"  {s['direction']:5s} frame={s['frame_idx']:6d} score={s['fusion_score']:.4f}")
print("  ...")
for s in synthetics[-3:]:
    print(f"  {s['direction']:5s} frame={s['frame_idx']:6d} score={s['fusion_score']:.4f}")

# ===== BUG ANALYSIS =====
print("\n=== BUG ANALYSIS ===")
print("""
BUG 1: FPS Estimation Formula sai tệ!
    fps = anchor_frame_idx / anchor_pts_time
    VD: frame_idx=5000, pts=166.67s → fps = 5000/166.67 = 30.0 (may work)
    NHƯNG: frame_idx=0 (frame đầu tiên), pts=0 → pts_time <= 0.1 → fps = 25.0 fallback
    VẤN ĐỀ: frame_idx KHÔNG phải là "số frame từ đầu video"
             mà là ABSOLUTE frame index trong toàn bộ keyframe sequence
    → FPS bị tính SAI cho hầu hết trường hợp!
""")

# Case 2: FPS = 25 fallback, frame_idx nhỏ
anchor_frame_idx2 = 750
anchor_pts_time2 = 30.0
anchor_fusion_score2 = 5.2

synthetics2, fps2, span2 = simulate_temporal_interpolation(
    "L25_V017", anchor_frame_idx2, anchor_pts_time2, anchor_fusion_score2
)
print(f"Case 2: frame_idx={anchor_frame_idx2}, pts={anchor_pts_time2}s")
print(f"  fps = {anchor_frame_idx2}/{anchor_pts_time2} = {anchor_frame_idx2/anchor_pts_time2:.1f}")
print(f"  frames_to_span = int(3.0 * {anchor_frame_idx2/anchor_pts_time2:.1f}) = {int(3.0 * anchor_frame_idx2/anchor_pts_time2)}")
print(f"  Synthetics generated: {len(synthetics2)} frames")

print("""
BUG 2: Không có giới hạn tổng số synthetic frames trong Top 100!
    Nếu anchor fusion_score >= 4.0, TẤT CẢ synthetics sẽ có score gần với anchor
    → Chúng chiếm hết Top 100 slots vì score cao hơn các candidates thật từ video khác
    → "L25_V017 chiếm hàng loạt slots"

BUG 3: NMS (Non-Maximum Suppression) trong executor.py chỉ penalize -0.05
    Với RANK_1_BONUS = +5.0, penalty -0.05 không có tác dụng gì
    → Các frame cận nhau 1-8 frames vẫn giữ được score cao

BUG 4: "người cầm cờ" xuất hiện liên tiếp
    Vì betting_interval = 8 frames (khoảng 0.32 giây với fps=25)
    → Các synthetic frames liên tiếp nhau, tất cả hiển thị cùng cảnh
    → UI thấy ~50-75 ảnh liên tiếp từ cùng 1 đoạn video
""")

print("=== FAISS NMS Logic hiện tại ===")
print("""
executor.py _apply_nms():
    min_seconds = 5.0
    penalty = -0.05
    
VỚI VLM BONUS +5.0:
    fusion_score của synthetic = ~5.4
    Penalty = -0.05
    Effective score sau NMS = ~5.35 (vẫn cao hơn bất kỳ candidate thật nào!)
→ NMS hoàn toàn vô hiệu với VLM-boosted candidates
""")
