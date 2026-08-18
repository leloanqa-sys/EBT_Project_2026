import os
import sys
import glob
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def load_human_verdicts(search_dirs=None):
    """Loads all human_verdict_*.csv files across specified directories."""
    if search_dirs is None:
        search_dirs = [
            os.path.join("outputs", "verdicts"),
            os.path.join("data", "verdicts"),
            "."
        ]
        
    csv_files = []
    for d in search_dirs:
        if os.path.exists(d):
            found = glob.glob(os.path.join(d, "human_verdict_*.csv"))
            csv_files.extend(found)
            
    # Deduplicate files by basename
    unique_files = {}
    for f in csv_files:
        bname = os.path.basename(f)
        if bname not in unique_files:
            unique_files[bname] = f
            
    csv_files = list(unique_files.values())
    if not csv_files:
        print("❌ Không tìm thấy file human_verdict nào!")
        print("👉 Hãy mở Web Review (tools/review_tool.py), chấm bài và lưu file CSV vào thư mục 'outputs/verdicts/'.")
        return pd.DataFrame()
        
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Extract or ensure query_id
            qid = os.path.basename(f).replace("human_verdict_", "").replace(".csv", "")
            if "query_id" not in df.columns or df["query_id"].isnull().all():
                df["query_id"] = qid
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Không đọc được file {f}: {e}")
        
    if not df_list:
        return pd.DataFrame()
        
    combined = pd.concat(df_list, ignore_index=True)
    print(f"📁 Đã nạp thành công {len(csv_files)} file CSV ({len(combined)} khung hình đã chấm).")
    return combined

def scoring_function(weights, df):
    """
    3-Tier Ranking Loss Function:
    - MATCH items: Goal is Rank 0..4 (Penalty: 1.0 * rank)
    - UNCERTAIN items: Goal is Middle Rank (Penalty: 0.5 * rank)
    - MISMATCH items: Goal is Bottom (Penalty if ranked ahead of MATCH)
    """
    w_clip, w_obj, w_spatial = weights
    
    # Calculate synthetic fusion score using norm_clip
    df['synthetic_fusion'] = (w_clip * df['norm_clip']) + (w_obj * df['obj_score']) + (w_spatial * df['spatial_score'])
    
    total_penalty = 0.0
    queries = df['query_id'].unique()
    
    for q in queries:
        q_df = df[df['query_id'] == q].sort_values(by='synthetic_fusion', ascending=False).reset_index(drop=True)
        
        # 1. MATCH Loss
        match_idx = q_df.index[q_df['verdict'] == 'MATCH'].tolist()
        if match_idx:
            # Average rank penalty for MATCH items (ideal rank is 0, 1, 2...)
            match_rank_penalty = sum(match_idx) / len(match_idx)
            total_penalty += 1.0 * match_rank_penalty
            
        # 2. UNCERTAIN Loss (Soft Penalty)
        uncertain_idx = q_df.index[q_df['verdict'] == 'UNCERTAIN'].tolist()
        if uncertain_idx:
            uncertain_rank_penalty = sum(uncertain_idx) / len(uncertain_idx)
            total_penalty += 0.5 * uncertain_rank_penalty
            
        # 3. Inversion Penalty: Count how many MISMATCH items are placed above any MATCH item
        mismatch_idx = q_df.index[q_df['verdict'] == 'MISMATCH'].tolist()
        if match_idx and mismatch_idx:
            min_match_idx = min(match_idx)
            # Count mismatches ahead of the best match
            mismatches_above_match = sum(1 for m in mismatch_idx if m < min_match_idx)
            total_penalty += 2.0 * mismatches_above_match
            
    return total_penalty

def optimize_weights_numpy(loss_fn, init_weights, df, bounds=(0.0, 4.0), step=0.2):
    """
    Fast, robust Coordinate Descent + Grid Search in pure NumPy (Zero extra dependencies).
    """
    best_weights = list(init_weights)
    best_loss = loss_fn(best_weights, df)
    
    # 1. Coarse Grid Search
    clip_vals = np.arange(bounds[0], bounds[1] + step, step)
    obj_vals = np.arange(bounds[0], bounds[1] + step, step)
    spatial_vals = np.arange(bounds[0], bounds[1] + step, step)
    
    for wc in clip_vals:
        for wo in obj_vals:
            for ws in spatial_vals:
                if wc == 0 and wo == 0 and ws == 0:
                    continue
                loss = loss_fn([wc, wo, ws], df)
                if loss < best_loss:
                    best_loss = loss
                    best_weights = [float(wc), float(wo), float(ws)]
                    
    # 2. Fine-grained Coordinate Descent around best point
    fine_step = step / 4.0
    improved = True
    for _ in range(5):
        if not improved:
            break
        improved = False
        for i in range(3):
            for delta in [-fine_step, fine_step]:
                candidate = list(best_weights)
                candidate[i] = max(bounds[0], min(bounds[1], candidate[i] + delta))
                loss = loss_fn(candidate, df)
                if loss < best_loss:
                    best_loss = loss
                    best_weights = candidate
                    improved = True
                    
    return best_weights, best_loss

def run_ml_tuner(output_json="outputs/tuning_results.json"):
    print("=" * 60)
    print("🤖 ML TUNER - TỐI ƯU HÓA TRỌNG SỐ FUSION DỰA TRÊN HUMAN VERDICTS")
    print("=" * 60)
    
    df = load_human_verdicts()
    if df.empty:
        return
        
    # Ensure numeric columns
    for col in ['clip_score', 'obj_score', 'spatial_score']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    # Normalize clip_score per query to match ranking.py behavior
    def normalize_clip_series(x):
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            return pd.Series(1.0, index=x.index)
        return (x - xmin) / (xmax - xmin)
        
    df['norm_clip'] = df.groupby('query_id')['clip_score'].transform(normalize_clip_series)
        
    # Summary of verdicts
    counts = df['verdict'].value_counts().to_dict()
    print("📊 Thống kê Ground Truth hiện có:")
    print(f"   - Khớp (MATCH)        : {counts.get('MATCH', 0)}")
    print(f"   - Không chắc (UNCERTAIN): {counts.get('UNCERTAIN', 0)}")
    print(f"   - Sai (MISMATCH)      : {counts.get('MISMATCH', 0)}")
    print(f"   - Chưa chấm (UNRATED) : {counts.get('UNRATED', 0)}")
    
    # Baseline with current default weights: [1.0, 0.5, 0.5]
    init_weights = [1.0, 0.5, 0.5]
    baseline_loss = scoring_function(init_weights, df.copy())
    print(f"\n📉 Mức phạt trước tối ưu (Baseline Loss với [1.0, 0.5, 0.5]): {baseline_loss:.4f}")
    
    bounds = (0.0, 4.0)
    print("⏳ Đang tính toán tìm kiếm điểm trọng số tối ưu...")
    
    if HAS_SCIPY:
        res = minimize(
            scoring_function, 
            init_weights, 
            args=(df.copy(),), 
            method='L-BFGS-B', 
            bounds=[(bounds[0], bounds[1])] * 3
        )
        opt_w_clip, opt_w_obj, opt_w_spatial = res.x
        opt_loss = res.fun
    else:
        best_w, opt_loss = optimize_weights_numpy(scoring_function, init_weights, df.copy(), bounds=bounds)
        opt_w_clip, opt_w_obj, opt_w_spatial = best_w
    
    print("\n" + "=" * 60)
    print("✅ TỐI ƯU HÓA HOÀN TẤT!")
    print(f"📉 Mức phạt tối ưu đạt được (Optimal Loss): {opt_loss:.4f} (Giảm: {baseline_loss - opt_loss:.4f})")
    print("\n🎯 BỘ TRỌNG SỐ ĐỀ XUẤT (Candidate Optimal Hyperparameters):")
    print(f"   - w_clip    = {opt_w_clip:.4f}")
    print(f"   - w_obj     = {opt_w_obj:.4f}")
    print(f"   - w_spatial = {opt_w_spatial:.4f}")
    print("=" * 60)
    
    # Save results to JSON file for benchmarking / later use
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    results_payload = {
        "baseline_loss": float(baseline_loss),
        "optimal_loss": float(opt_loss),
        "optimal_weights": {
            "w_clip": round(float(opt_w_clip), 4),
            "w_obj": round(float(opt_w_obj), 4),
            "w_spatial": round(float(opt_w_spatial), 4)
        },
        "verdict_counts": {k: int(v) for k, v in counts.items()},
        "total_evaluated_frames": len(df)
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
        
    print(f"💾 Đã lưu kết quả cấu hình tối ưu vào: {output_json}")
    print("📌 Ghi chú: Trọng số trong ranking.py vẫn được giữ nguyên mặc định cho đến khi bạn hoàn tất chấm toàn bộ batch lớn.")

if __name__ == "__main__":
    run_ml_tuner()
