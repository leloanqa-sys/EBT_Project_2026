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

def classify_query_context(query_text: str) -> str:
    """Classifies a query into a domain category for specialized weight tuning."""
    if not query_text:
        return "general_scene"
    t = query_text.lower()
    
    color_keywords = ["đỏ", "xanh", "vàng", "trắng", "đen", "tím", "cam", "hồng", "nâu", "xám", "áo", "quần", "mũ", "nón", "red", "blue", "green", "yellow", "black", "white", "shirt", "pants", "hat", "dress"]
    spatial_keywords = ["bên trái", "bên phải", "phía sau", "đằng trước", "ở giữa", "cạnh", "kế bên", "gần", "trên", "dưới", "left", "right", "behind", "front", "middle", "next to", "above", "below"]
    action_keywords = ["chạy", "nhảy", "đi bộ", "cầm", "nắm", "lái xe", "ăn", "uống", "nói chuyện", "đá", "ném", "chơi", "ngã", "ôm", "hát", "run", "walk", "jump", "drive", "eat", "drink", "talk", "kick", "throw", "play", "fall", "hold"]
    object_heavy_keywords = ["ô tô", "xe hơi", "xe máy", "xe đạp", "con chó", "con mèo", "cái bàn", "cái ghế", "người", "car", "motorcycle", "bicycle", "dog", "cat", "table", "chair", "person"]
    
    if any(k in t for k in color_keywords):
        return "color_attribute"
    if any(k in t for k in spatial_keywords):
        return "spatial_heavy"
    if any(k in t for k in action_keywords):
        return "action_event"
    if any(k in t for k in object_heavy_keywords):
        return "object_heavy"
    return "general_scene"

def run_ml_tuner(output_json="outputs/tuning_results.json"):
    print("=" * 60)
    print("🤖 ML TUNER - TỐI ƯU HÓA TRỌNG SỐ FUSION DỰA TRÊN HUMAN VERDICTS")
    print("=" * 60)
    
    df = load_human_verdicts()
    if df.empty:
        return
        
    # Support both siglip_score and legacy clip_score
    if 'siglip_score' not in df.columns and 'clip_score' in df.columns:
        df['siglip_score'] = df['clip_score']
    elif 'siglip_score' not in df.columns:
        df['siglip_score'] = 0.0
        
    for col in ['siglip_score', 'obj_score', 'spatial_score']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    if 'query_text' not in df.columns:
        df['query_text'] = ""
    df['query_text'] = df['query_text'].fillna("")
    
    # Classify each query
    df['context_type'] = df['query_text'].apply(classify_query_context)
        
    # Normalize siglip_score per query to match ranking.py behavior
    def normalize_siglip_series(x):
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            return pd.Series(1.0, index=x.index)
        return (x - xmin) / (xmax - xmin)
        
    df['norm_clip'] = df.groupby('query_id')['siglip_score'].transform(normalize_siglip_series)
        
    # Summary of verdicts
    counts = df['verdict'].value_counts().to_dict()
    print("📊 Thống kê Ground Truth hiện có:")
    print(f"   - Khớp (MATCH)          : {counts.get('MATCH', 0)}")
    print(f"   - Không chắc (UNCERTAIN): {counts.get('UNCERTAIN', 0)}")
    print(f"   - Sai (MISMATCH)        : {counts.get('MISMATCH', 0)}")
    print(f"   - Phân loại ngữ cảnh    : {df['context_type'].value_counts().to_dict()}")
    
    # 1. Global Optimization
    init_weights = [1.0, 0.5, 0.5]
    baseline_loss = scoring_function(init_weights, df.copy())
    print(f"\n📉 Mức phạt toàn cục trước tối ưu (Baseline Loss): {baseline_loss:.4f}")
    
    bounds = (0.0, 4.0)
    print("⏳ Đang tối ưu hóa trọng số toàn cục (Global Weights)...")
    
    if HAS_SCIPY:
        res = minimize(scoring_function, init_weights, args=(df.copy(),), method='L-BFGS-B', bounds=[(bounds[0], bounds[1])] * 3)
        opt_w_siglip, opt_w_obj, opt_w_spatial = res.x
        opt_loss = res.fun
    else:
        best_w, opt_loss = optimize_weights_numpy(scoring_function, init_weights, df.copy(), bounds=bounds)
        opt_w_siglip, opt_w_obj, opt_w_spatial = best_w
    
    print("\n🎯 TRỌNG SỐ TOÀN CỤC ĐỀ XUẤT (Global Optimal Hyperparameters):")
    print(f"   - w_siglip  = {opt_w_siglip:.4f}")
    print(f"   - w_obj     = {opt_w_obj:.4f}")
    print(f"   - w_spatial = {opt_w_spatial:.4f}")
    
    # 2. Per-Category Optimization
    category_weights = {}
    print("\n🔬 TỐI ƯU HÓA CHI TIẾT THEO TỪNG DẠNG TRUY VẤN (Per-Category Tuning):")
    for ctype, cdf in df.groupby('context_type'):
        if len(cdf['query_id'].unique()) < 2:
            print(f"   ℹ️ Ngữ cảnh '{ctype}' có ít hơn 2 truy vấn, dùng fallback toàn cục.")
            category_weights[ctype] = {
                "w_siglip": round(float(opt_w_siglip), 4),
                "w_obj": round(float(opt_w_obj), 4),
                "w_spatial": round(float(opt_w_spatial), 4)
            }
            continue
            
        if HAS_SCIPY:
            cres = minimize(scoring_function, init_weights, args=(cdf.copy(),), method='L-BFGS-B', bounds=[(bounds[0], bounds[1])] * 3)
            cw_siglip, cw_obj, cw_spatial = cres.x
        else:
            cbest_w, _ = optimize_weights_numpy(scoring_function, init_weights, cdf.copy(), bounds=bounds)
            cw_siglip, cw_obj, cw_spatial = cbest_w
            
        print(f"   ✨ Ngữ cảnh [{ctype}]: w_siglip={cw_siglip:.2f}, w_obj={cw_obj:.2f}, w_spatial={cw_spatial:.2f}")
        category_weights[ctype] = {
            "w_siglip": round(float(cw_siglip), 4),
            "w_obj": round(float(cw_obj), 4),
            "w_spatial": round(float(cw_spatial), 4)
        }
        
    # Save results to JSON file
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    results_payload = {
        "baseline_loss": float(baseline_loss),
        "optimal_loss": float(opt_loss),
        "global_weights": {
            "w_siglip": round(float(opt_w_siglip), 4),
            "w_obj": round(float(opt_w_obj), 4),
            "w_spatial": round(float(opt_w_spatial), 4)
        },
        "category_weights": category_weights,
        "verdict_counts": {k: int(v) for k, v in counts.items()},
        "total_evaluated_frames": len(df)
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\n💾 Đã lưu cấu hình đa tầng vào: {output_json}")

if __name__ == "__main__":
    run_ml_tuner()
