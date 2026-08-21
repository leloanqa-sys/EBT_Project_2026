import os
import sys
import json
import numpy as np
import pandas as pd

def optimize_weights_numpy(loss_func, init_weights, df, bounds=(0.0, 3.0), steps=25):
    best_weights = init_weights
    best_loss = loss_func(init_weights, df)
    
    w_clip_vals = np.linspace(bounds[0], bounds[1], steps)
    w_obj_vals = np.linspace(0.0, 2.0, steps)
    w_spatial_vals = np.linspace(0.0, 1.0, 10)
    
    for wc in w_clip_vals:
        for wo in w_obj_vals:
            for ws in w_spatial_vals:
                if wc == 0 and wo == 0 and ws == 0:
                    continue
                w = [wc, wo, ws]
                current_loss = loss_func(w, df)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_weights = w
                    
    return best_weights, best_loss

def tune_preliminary_weights(csv_path="outputs/verdicts/human_verdict_preliminary_24.csv", output_json="outputs/tuning_results.json"):
    print("=" * 70)
    print("NUMPY GRID HYPERPARAMETER OPTIMIZATION ON 24 PRELIMINARY QUERIES")
    print("=" * 70)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} verdict items across {df['query_id'].nunique()} unique queries.")
    
    for col in ["siglip_score", "obj_score", "spatial_score"]:
        if col not in df.columns:
            df[col] = 0.0
            
    def scoring_loss(weights, data):
        w_siglip, w_obj, w_spatial = weights
        data["synthetic_score"] = (w_siglip * data["siglip_score"]) + (w_obj * data["obj_score"]) + (w_spatial * data["spatial_score"])
        
        total_loss = 0.0
        for qid, qdf in data.groupby("query_id"):
            sorted_qdf = qdf.sort_values(by="synthetic_score", ascending=False).reset_index(drop=True)
            
            match_indices = sorted_qdf.index[sorted_qdf["verdict"] == "MATCH"].tolist()
            if match_indices:
                total_loss += sum(match_indices) / len(match_indices)
                
            unc_indices = sorted_qdf.index[sorted_qdf["verdict"] == "UNCERTAIN"].tolist()
            if unc_indices:
                total_loss += 0.3 * (sum(unc_indices) / len(unc_indices))
                
        return total_loss

    init_w = [1.0, 0.5, 0.2]
    base_loss = scoring_loss(init_w, df.copy())
    print(f"\nBaseline Ranking Loss: {base_loss:.4f}")
    
    print("Optimizing multi-dimensional weights via NumPy grid...")
    best_w, opt_loss = optimize_weights_numpy(scoring_loss, init_w, df.copy(), bounds=(0.5, 2.5), steps=20)
    
    opt_siglip, opt_obj, opt_spatial = best_w
    
    impr = f"{((base_loss - opt_loss)/base_loss)*100:.2f}%" if base_loss > 0 else "0.00%"
    print("\nOPTIMIZED HYPERPARAMETERS:")
    print(f"  - w_siglip:  {opt_siglip:.4f}")
    print(f"  - w_obj:     {opt_obj:.4f}")
    print(f"  - w_spatial: {opt_spatial:.4f}")
    print(f"  - Optimal Loss: {opt_loss:.4f} (Improvement: {impr})")
    
    payload = {
        "dataset": "official_preliminary_24",
        "baseline_loss": float(base_loss),
        "optimal_loss": float(opt_loss),
        "optimal_weights": {
            "w_siglip": round(float(opt_siglip), 4),
            "w_obj": round(float(opt_obj), 4),
            "w_spatial": round(float(opt_spatial), 4)
        },
        "global_weights": {
            "w_siglip": round(float(opt_siglip), 4),
            "w_obj": round(float(opt_obj), 4),
            "w_spatial": round(float(opt_spatial), 4)
        },
        "category_weights": {
            "action_event": {"w_siglip": round(float(opt_siglip * 1.15), 4), "w_obj": round(float(opt_obj * 0.8), 4), "w_spatial": round(float(opt_spatial), 4)},
            "color_attribute": {"w_siglip": round(float(opt_siglip * 0.85), 4), "w_obj": round(float(opt_obj * 1.25), 4), "w_spatial": round(float(opt_spatial * 1.2), 4)},
            "scene_context": {"w_siglip": round(float(opt_siglip * 1.2), 4), "w_obj": round(float(opt_obj * 0.5), 4), "w_spatial": round(float(opt_spatial * 0.5), 4)}
        }
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved optimal configuration to '{output_json}'.")

if __name__ == '__main__':
    tune_preliminary_weights()
