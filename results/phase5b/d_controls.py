import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/phase5b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_section_d():
    print("=" * 70)
    print("SECTION D: CASCADE ABLATION CONTROLS (IV, VI, VII)")
    print("=" * 70)
    
    df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
    df_a500 = df2a[(df2a["preset"] == "ALIKED") & (df2a["N"] == 500)].sort_values(["fname", "i", "j"]).reset_index(drop=True)
    df_s4000 = df2a[(df2a["preset"] == "SIFT") & (df2a["N"] == 4000)].sort_values(["fname", "i", "j"]).reset_index(drop=True)
    
    pairs = md.list_pairs()
    n_pairs = len(pairs)
    
    p_a500 = df_a500["p_epi"].values
    err_a500 = df_a500["pose_err"].values
    act_a500 = df_a500["actual_kp"].values
    inl_count_a500 = df_a500["n_inliers"].values
    inl_ratio_a500 = df_a500["inlier_ratio"].values
    # Reprojection factor from Sampson error: exp(-mean_sampson / 2.0)
    mean_sampson = df_a500["mean_sampson"].values
    reproj_a500 = np.exp(-np.clip(mean_sampson, 0.0, 50.0) / 2.0)
    
    p_s4000 = df_s4000["p_epi"].values
    err_s4000 = df_s4000["pose_err"].values
    act_s4000 = df_s4000["actual_kp"].values
    
    # Scene masks
    mask_0015 = np.array([p[0].startswith("0015") for p in pairs])
    mask_0022 = np.array([p[0].startswith("0022") for p in pairs])
    
    # Stopping mask for Cascade C* (tau=0.40)
    stop_c_star = (p_a500 >= 0.40)
    stop_rate_pooled = float(np.mean(stop_c_star))
    esc_rate_pooled = 1.0 - stop_rate_pooled
    stop_rate_0015 = float(np.mean(stop_c_star[mask_0015]))
    stop_rate_0022 = float(np.mean(stop_c_star[mask_0022]))
    
    print(f"Cascade C* (tau=0.40) Stop Rate: Pooled={stop_rate_pooled*100:.2f}%, 0015={stop_rate_0015*100:.2f}%, 0022={stop_rate_0022*100:.2f}%")
    print(f"Cascade C* (tau=0.40) Escalation Rate: Pooled={esc_rate_pooled*100:.2f}%")
    
    # ---------------------------------------------------------
    # D.1: Control (iv) Rerun with 20 Independent Draws
    # ---------------------------------------------------------
    print("\n--- D.1: Control (iv) Random Escalation (20 Independent Draws) ---")
    draw_records = []
    
    for draw_seed in range(1, 21):
        rng = np.random.RandomState(draw_seed)
        rnd_stop = (rng.rand(n_pairs) < stop_rate_pooled)
        
        # When escalated: highest proxy of (a500, s4000)
        use_a500 = (p_a500 >= p_s4000)
        err_esc = np.where(use_a500, err_a500, err_s4000)
        
        sim_err = np.where(rnd_stop, err_a500, err_esc)
        sim_kp = np.where(rnd_stop, act_a500, act_a500 + act_s4000)
        
        succ10 = (sim_err < 10.0).mean() * 100.0
        auc5, auc10, auc20 = [v * 100.0 for v in md.pose_auc(sim_err)]
        cum_kp = sim_kp.mean()
        utility = (succ10 / 100.0) - (0.05 / 2000.0) * cum_kp
        
        draw_records.append({
            "draw": draw_seed,
            "succ10": succ10,
            "auc10": auc10,
            "cum_actual_kp": cum_kp,
            "utility": utility
        })
        
    df_draws = pd.DataFrame(draw_records)
    df_draws.to_csv(OUTPUT_DIR / "d1_control_iv_20draws.csv", index=False)
    
    mean_succ10_iv = df_draws["succ10"].mean()
    std_succ10_iv = df_draws["succ10"].std()
    mean_auc10_iv = df_draws["auc10"].mean()
    std_auc10_iv = df_draws["auc10"].std()
    mean_kp_iv = df_draws["cum_actual_kp"].mean()
    std_kp_iv = df_draws["cum_actual_kp"].std()
    mean_ut_iv = df_draws["utility"].mean()
    
    print(f"Control (iv) 20 Draws (Pooled): Succ@10 = {mean_succ10_iv:.2f}% +- {std_succ10_iv:.2f}%, AUC@10 = {mean_auc10_iv:.2f} +- {std_auc10_iv:.2f}, KP = {mean_kp_iv:.1f} +- {std_kp_iv:.1f}, Utility = {mean_ut_iv:.4f}")
    
    # ---------------------------------------------------------
    # D.2: Additional Controls (vi) Anti-Cascade and (vii) Components
    # ---------------------------------------------------------
    print("\n--- D.2: (vi) Anti-Cascade and (vii) Proxy Component Controls ---")
    
    # (vi) Anti-Cascade: Escalate when proxy is HIGH!
    # Choose threshold tau_anti such that exactly esc_rate_pooled fraction escalates on high proxy
    # i.e., proxy >= tau_anti -> escalate (40.53%), proxy < tau_anti -> stop (59.47%)
    tau_anti = np.percentile(p_a500, stop_rate_pooled * 100.0)
    stop_anti = (p_a500 < tau_anti)
    use_a500_anti = (p_a500 >= p_s4000)
    err_esc_anti = np.where(use_a500_anti, err_a500, err_s4000)
    err_anti = np.where(stop_anti, err_a500, err_esc_anti)
    kp_anti = np.where(stop_anti, act_a500, act_a500 + act_s4000)
    
    # (vii) Component-alone escalations (Cross-Scene Thresholding)
    # Target escalation rates: 1 - stop_rate_0015 on 0015; 1 - stop_rate_0022 on 0022
    # Threshold for 0022 chosen on 0015 to match 0015 stop rate; and vice-versa
    
    # (vii.a) Inlier count alone
    th_inl_from_0015 = np.percentile(inl_count_a500[mask_0015], (1.0 - stop_rate_0015) * 100.0) # lower percentile to stop on high count?
    # Stop on high count: count >= th -> stop
    # So stop rate = fraction >= th. Thus th = (1 - stop_rate) percentile!
    th_inl_from_0015 = np.percentile(inl_count_a500[mask_0015], (1.0 - stop_rate_0015) * 100.0)
    th_inl_from_0022 = np.percentile(inl_count_a500[mask_0022], (1.0 - stop_rate_0022) * 100.0)
    
    stop_inl = np.zeros(n_pairs, dtype=bool)
    stop_inl[mask_0015] = (inl_count_a500[mask_0015] >= th_inl_from_0022)
    stop_inl[mask_0022] = (inl_count_a500[mask_0022] >= th_inl_from_0015)
    
    err_inl = np.where(stop_inl, err_a500, np.where(p_a500 >= p_s4000, err_a500, err_s4000))
    kp_inl = np.where(stop_inl, act_a500, act_a500 + act_s4000)
    
    # (vii.b) Inlier ratio alone
    th_ratio_from_0015 = np.percentile(inl_ratio_a500[mask_0015], (1.0 - stop_rate_0015) * 100.0)
    th_ratio_from_0022 = np.percentile(inl_ratio_a500[mask_0022], (1.0 - stop_rate_0022) * 100.0)
    
    stop_ratio = np.zeros(n_pairs, dtype=bool)
    stop_ratio[mask_0015] = (inl_ratio_a500[mask_0015] >= th_ratio_from_0022)
    stop_ratio[mask_0022] = (inl_ratio_a500[mask_0022] >= th_ratio_from_0015)
    
    err_ratio = np.where(stop_ratio, err_a500, np.where(p_a500 >= p_s4000, err_a500, err_s4000))
    kp_ratio = np.where(stop_ratio, act_a500, act_a500 + act_s4000)
    
    # (vii.c) Sampson error term alone (reprojection factor)
    th_reproj_from_0015 = np.percentile(reproj_a500[mask_0015], (1.0 - stop_rate_0015) * 100.0)
    th_reproj_from_0022 = np.percentile(reproj_a500[mask_0022], (1.0 - stop_rate_0022) * 100.0)
    
    stop_reproj = np.zeros(n_pairs, dtype=bool)
    stop_reproj[mask_0015] = (reproj_a500[mask_0015] >= th_reproj_from_0022)
    stop_reproj[mask_0022] = (reproj_a500[mask_0022] >= th_reproj_from_0015)
    
    err_reproj = np.where(stop_reproj, err_a500, np.where(p_a500 >= p_s4000, err_a500, err_s4000))
    kp_reproj = np.where(stop_reproj, act_a500, act_a500 + act_s4000)
    
    # Evaluate and summarize all controls
    controls_dict = {
        "Cascade C* (tau=0.40, Full Proxy)": (
            np.where(stop_c_star, err_a500, np.where(p_a500 >= p_s4000, err_a500, err_s4000)),
            np.where(stop_c_star, act_a500, act_a500 + act_s4000),
            stop_c_star
        ),
        "(iv) Random-Escalation Control (20 Draws Mean)": (
            None, None, None # populated from df_draws
        ),
        "(vi) Anti-Cascade (Escalate on HIGH proxy)": (err_anti, kp_anti, stop_anti),
        "(vii.a) Inlier Count Alone (Cross-Scene)": (err_inl, kp_inl, stop_inl),
        "(vii.b) Inlier Ratio Alone (Cross-Scene)": (err_ratio, kp_ratio, stop_ratio),
        "(vii.c) Sampson Reproj Factor Alone (Cross-Scene)": (err_reproj, kp_reproj, stop_reproj),
    }
    
    results = []
    for name, item in controls_dict.items():
        if name == "(iv) Random-Escalation Control (20 Draws Mean)":
            results.append({
                "control_name": name,
                "succ10": mean_succ10_iv,
                "succ10_std": std_succ10_iv,
                "auc10": mean_auc10_iv,
                "auc10_std": std_auc10_iv,
                "cum_actual_kp": mean_kp_iv,
                "stop_rate_pct": stop_rate_pooled * 100.0,
                "utility": mean_ut_iv
            })
            continue
            
        e_arr, k_arr, st_arr = item
        s10 = (e_arr < 10.0).mean() * 100.0
        auc10 = md.pose_auc(e_arr)[1] * 100.0
        kp_m = k_arr.mean()
        ut = (s10 / 100.0) - (0.05 / 2000.0) * kp_m
        
        results.append({
            "control_name": name,
            "succ10": s10,
            "succ10_std": 0.0,
            "auc10": auc10,
            "auc10_std": 0.0,
            "cum_actual_kp": kp_m,
            "stop_rate_pct": st_arr.mean() * 100.0,
            "utility": ut
        })
        
    df_controls = pd.DataFrame(results)
    df_controls.to_csv(OUTPUT_DIR / "d_controls_summary.csv", index=False)
    print("\nSaved results/phase5b/d_controls_summary.csv:")
    print(df_controls.to_string(index=False))

if __name__ == "__main__":
    run_section_d()
