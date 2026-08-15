import os
import glob
import pandas as pd
from scipy.optimize import minimize

def load_human_verdicts(csv_dir="."):
    """Loads all human_verdict_*.csv files."""
    csv_files = glob.glob(os.path.join(csv_dir, "human_verdict_*.csv"))
    if not csv_files:
        print("❌ Không tìm thấy file human_verdict nào! Yêu cầu tester chấm bài trước qua review_tool.py")
        return pd.DataFrame()
        
    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Extract query_id from filename: human_verdict_Q001.csv -> Q001
        qid = os.path.basename(f).replace("human_verdict_", "").replace(".csv", "")
        df["query_id"] = qid
        df_list.append(df)
        
    return pd.concat(df_list, ignore_index=True)

def scoring_function(weights, df):
    """
    Computes a synthetic fusion score using weights, then calculates how well
    the MATCH items are ranked compared to MISMATCH items.
    We want to MINIMIZE this function (so we return a negative score or penalty).
    """
    w_clip, w_obj, w_spatial = weights
    
    # Calculate fusion_score
    # Note: Currently the human_verdict.csv only saves clip_score. 
    # To properly tune w_obj and w_spatial, the review_tool.py needs to export obj_score and spatial_score as well.
    # We will assume they are present in the df for the future.
    if 'obj_score' not in df.columns:
        df['obj_score'] = 0.0
    if 'spatial_score' not in df.columns:
        df['spatial_score'] = 0.0
        
    df['fusion_score'] = (w_clip * df['clip_score']) + (w_obj * df['obj_score']) + (w_spatial * df['spatial_score'])
    
    # Simple Loss Metric: Average rank of MATCH items (lower is better)
    # We sort by fusion_score descending, and find the rank of MATCH items
    total_penalty = 0.0
    queries = df['query_id'].unique()
    
    for q in queries:
        q_df = df[df['query_id'] == q].sort_values(by='fusion_score', ascending=False).reset_index()
        # Find index of MATCH
        match_indices = q_df.index[q_df['verdict'] == 'MATCH'].tolist()
        if not match_indices:
            continue
            
        # Penalty is the average rank of MATCH items
        # Ideal rank is 0, 1, 2...
        avg_rank = sum(match_indices) / len(match_indices)
        total_penalty += avg_rank
        
    return total_penalty

def run_ml_tuner():
    print("🚀 Bắt đầu quá trình ML Tuning dựa trên Human Verdicts...")
    df = load_human_verdicts()
    if df.empty:
        return
        
    # Initial weights: [w_clip, w_obj, w_spatial]
    init_weights = [1.0, 0.5, 0.5]
    bounds = [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)] # Trọng số từ 0 đến 5
    
    print(f"Đang tối ưu hóa trên {len(df)} khung hình đã được chấm...")
    
    res = minimize(
        scoring_function, 
        init_weights, 
        args=(df,), 
        method='L-BFGS-B', 
        bounds=bounds
    )
    
    print("\n✅ Tối ưu hóa hoàn tất!")
    print(f"📉 Mức phạt thấp nhất đạt được (Loss): {res.fun:.4f}")
    print("\n🎯 BỘ TRỌNG SỐ TỐI ƯU (Optimal Weights):")
    print(f"  w_clip    = {res.x[0]:.4f}")
    print(f"  w_obj     = {res.x[1]:.4f}")
    print(f"  w_spatial = {res.x[2]:.4f}")
    
    print("\n👉 Hãy cập nhật các trọng số này vào hàm compute_fusion_scores trong src/role_c_logic/ranking.py")

if __name__ == "__main__":
    run_ml_tuner()
