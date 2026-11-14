import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')
import hpatches
import pipeline_utils as pu
from level2_env import Level2Env, load_split
# metrics import removed

OUTPUT_DIR = Path("results/phase5b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("FINISHING SECTION A.4 DIAGNOSTICS FROM FIXED STAGE 3 DATASET")
    print("=" * 70)
    
    csv_path = OUTPUT_DIR / "megadepth_stage3_afm_rl_v1_fixed.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # 1. Summary by fold and pooled
    summary_rows = []
    
    def summarize_subset(sub, name):
        succ10 = np.mean(sub["pose_err"] < 10.0) * 100.0
        auc5 = np.mean(np.maximum(0.0, 1.0 - sub["pose_err"] / 5.0)) * 100.0
        auc10 = np.mean(np.maximum(0.0, 1.0 - sub["pose_err"] / 10.0)) * 100.0
        auc20 = np.mean(np.maximum(0.0, 1.0 - sub["pose_err"] / 20.0)) * 100.0
        
        cum_kp = np.mean(sub["cumulative_actual_kp"])
        final_kp = np.mean(sub["final_actual_N"])
        mean_steps = np.mean(sub["steps"])
        
        # Pct reaching MAX_STEPS (15) and pct stopping at step 1
        pct_max_steps = np.mean(sub["steps"] >= 15) * 100.0
        pct_step1 = np.mean(sub["steps"] == 1) * 100.0
        
        # Utility
        norm_kp = sub["cumulative_actual_kp"] / 4000.0
        acc = np.maximum(0.0, 1.0 - sub["pose_err"] / 10.0)
        utility = np.mean(acc - 0.1 * norm_kp)
        
        # Actions
        # In fixed env: Act 0: +500, Act 1: +1000, Act 2: -500, Act 3: STOP
        first_a = sub["first_action"]
        pct_fa0 = np.mean(first_a == 0) * 100.0
        pct_fa1 = np.mean(first_a == 1) * 100.0
        pct_fa2 = np.mean(first_a == 2) * 100.0
        pct_fa3 = np.mean(first_a == 3) * 100.0
        
        presets = sub["chosen_preset"].value_counts().to_dict()
        
        return {
            "scope": name,
            "n_pairs": len(sub),
            "succ10_deg": succ10,
            "auc5_deg": auc5,
            "auc10_deg": auc10,
            "auc20_deg": auc20,
            "mean_cum_kp": cum_kp,
            "mean_final_kp": final_kp,
            "mean_steps": mean_steps,
            "pct_max_steps_15": pct_max_steps,
            "pct_stopped_step_1": pct_step1,
            "mean_utility": utility,
            "first_act0_plus500": pct_fa0,
            "first_act1_plus1000": pct_fa1,
            "first_act2_minus500": pct_fa2,
            "first_act3_STOP": pct_fa3,
            "preset_dist": str(presets)
        }
        
    for f in range(5):
        sub_f = df[df["fold"] == f]
        summary_rows.append(summarize_subset(sub_f, f"Fold {f}"))
        
    summary_rows.append(summarize_subset(df, "Pooled (All 5 Folds)"))
    
    df_sum = pd.DataFrame(summary_rows)
    sum_csv = OUTPUT_DIR / "a4_stage3_fixed_summary.csv"
    df_sum.to_csv(sum_csv, index=False)
    print(f"Saved {sum_csv}")
    print(df_sum[["scope", "succ10_deg", "auc10_deg", "mean_cum_kp", "mean_steps", "pct_max_steps_15", "pct_stopped_step_1", "mean_utility"]])
    
    # 2. Step 0 p_epi vs HPatches proxy quantiles
    print("\nEvaluating HPatches Step 0 proxy quantiles...")
    q_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    md_p_epi_step0 = df["p_epi_step0"]
    md_quantiles = md_p_epi_step0.quantile(q_levels).values
    
    # Evaluate HPatches Step 0 proxy directly using Level2Env
    split_data = load_split(None)
    hp_pairs = []
    for seq in split_data["test"]:
        for k in range(2, 7):
            hp_pairs.append((seq, k))
            
    hp_env = Level2Env(split="val", obs_mode="proxy", eval_mode=True)
    hp_proxies = []
    for seq, k in hp_pairs:
        # Step 0 proxy for default preset ALIKED
        obs, _ = hp_env.reset_to(seq, k, "ALIKED")
        hp_proxies.append(obs[26])  # dim 26 is proxy_acc
        
    hp_proxies = np.array(hp_proxies)
    hp_quantiles = np.percentile(hp_proxies, [q * 100 for q in q_levels])
    
    df_q = pd.DataFrame({
        "quantile": ["5%", "25%", "50%", "75%", "95%"],
        "p_epi_megadepth_step0": md_quantiles,
        "proxy_hpatches_step0": hp_quantiles,
        "difference": md_quantiles - hp_quantiles
    })
    q_csv = OUTPUT_DIR / "a4_step0_proxy_quantiles.csv"
    df_q.to_csv(q_csv, index=False)
    print(f"Saved {q_csv}")
    print(df_q)
    
    # 3. Level 2 action probabilities conditioned on step 0 p_epi
    print("\nComputing Level 2 action probabilities conditioned on step 0 p_epi...")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]"]
    df["proxy_bucket"] = pd.cut(df["p_epi_step0"], bins=bins, labels=bin_labels, right=False)
    
    bucket_records = []
    for bl in bin_labels:
        b_df = df[df["proxy_bucket"] == bl]
        n_b = len(b_df)
        if n_b > 0:
            p0 = b_df["prob_act0"].mean()
            p1 = b_df["prob_act1"].mean()
            p2 = b_df["prob_act2"].mean()
            p3 = b_df["prob_act3"].mean()
            # actual chosen action
            act_counts = b_df["first_action"].value_counts(normalize=True).to_dict()
            pct0 = act_counts.get(0, 0.0) * 100.0
            pct1 = act_counts.get(1, 0.0) * 100.0
            pct2 = act_counts.get(2, 0.0) * 100.0
            pct3 = act_counts.get(3, 0.0) * 100.0
        else:
            p0 = p1 = p2 = p3 = pct0 = pct1 = pct2 = pct3 = 0.0
            
        bucket_records.append({
            "proxy_bucket": bl,
            "n_samples": n_b,
            "mean_prob_act0_plus500": p0,
            "mean_prob_act1_plus1000": p1,
            "mean_prob_act2_minus500": p2,
            "mean_prob_act3_STOP": p3,
            "chosen_act0_plus500_pct": pct0,
            "chosen_act1_plus1000_pct": pct1,
            "chosen_act2_minus500_pct": pct2,
            "chosen_act3_STOP_pct": pct3,
        })
        
    df_buckets = pd.DataFrame(bucket_records)
    b_csv = OUTPUT_DIR / "a4_l2_action_probs_by_proxy_bucket.csv"
    df_buckets.to_csv(b_csv, index=False)
    print(f"Saved {b_csv}")
    print(df_buckets[["proxy_bucket", "n_samples", "mean_prob_act0_plus500", "mean_prob_act1_plus1000", "mean_prob_act2_minus500", "mean_prob_act3_STOP"]])
    
    # 4. Bootstrap comparisons against baselines
    print("\nRunning 10,000-resample bootstrap comparison against pre-registered baselines...")
    df_stage2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
    
    # Group stage2a by config
    baselines = {
        "ALIKED-500": df_stage2a[(df_stage2a["preset"] == "ALIKED") & (df_stage2a["N"] == 500)].sort_values(["fname", "i", "j"]),
        "ALIKED-1000": df_stage2a[(df_stage2a["preset"] == "ALIKED") & (df_stage2a["N"] == 1000)].sort_values(["fname", "i", "j"]),
        "SIFT-2000": df_stage2a[(df_stage2a["preset"] == "SIFT") & (df_stage2a["N"] == 2000)].sort_values(["fname", "i", "j"]),
        "SIFT-4000": df_stage2a[(df_stage2a["preset"] == "SIFT") & (df_stage2a["N"] == 4000)].sort_values(["fname", "i", "j"]),
    }
    
    # AFM-RL v1 pooled average per pair across the 5 folds
    df_afm_pair = df.groupby(["fname", "i", "j"]).agg({
        "pose_err": "mean",
        "cumulative_actual_kp": "mean",
        "steps": "mean"
    }).reset_index().sort_values(["fname", "i", "j"])
    
    n_pairs = len(df_afm_pair)
    assert n_pairs == 1500, f"Expected 1500 pairs, got {n_pairs}"
    
    rng = np.random.RandomState(42)
    B = 10000
    boot_indices = rng.randint(0, n_pairs, size=(B, n_pairs))
    
    afm_err = df_afm_pair["pose_err"].values
    afm_kp = df_afm_pair["cumulative_actual_kp"].values
    afm_succ = (afm_err < 10.0).astype(float)
    afm_auc10 = np.maximum(0.0, 1.0 - afm_err / 10.0)
    afm_util = afm_auc10 - 0.1 * (afm_kp / 4000.0)
    
    comp_rows = []
    
    for b_name, b_df in baselines.items():
        assert len(b_df) == 1500, f"Expected 1500 rows for {b_name}, got {len(b_df)}"
        b_err = b_df["pose_err"].values
        b_kp = b_df["actual_kp"].values
        b_succ = (b_err < 10.0).astype(float)
        b_auc10 = np.maximum(0.0, 1.0 - b_err / 10.0)
        b_util = b_auc10 - 0.1 * (b_kp / 4000.0)
        
        diff_succ = afm_succ - b_succ
        diff_auc10 = (afm_auc10 - b_auc10) * 100.0
        diff_kp = afm_kp - b_kp
        diff_util = afm_util - b_util
        
        # Bootstrap
        b_diff_succ = np.mean(diff_succ[boot_indices], axis=1) * 100.0
        b_diff_auc10 = np.mean(diff_auc10[boot_indices], axis=1)
        b_diff_kp = np.mean(diff_kp[boot_indices], axis=1)
        b_diff_util = np.mean(diff_util[boot_indices], axis=1)
        
        comp_rows.append({
            "comparison": f"AFM-RL v1 (fixed) vs {b_name}",
            "delta_succ10_pct": float(np.mean(diff_succ) * 100.0),
            "delta_succ10_ci95_low": float(np.percentile(b_diff_succ, 2.5)),
            "delta_succ10_ci95_high": float(np.percentile(b_diff_succ, 97.5)),
            "delta_auc10": float(np.mean(diff_auc10)),
            "delta_auc10_ci95_low": float(np.percentile(b_diff_auc10, 2.5)),
            "delta_auc10_ci95_high": float(np.percentile(b_diff_auc10, 97.5)),
            "delta_kp": float(np.mean(diff_kp)),
            "delta_kp_ci95_low": float(np.percentile(b_diff_kp, 2.5)),
            "delta_kp_ci95_high": float(np.percentile(b_diff_kp, 97.5)),
            "delta_utility": float(np.mean(diff_util)),
            "delta_utility_ci95_low": float(np.percentile(b_diff_util, 2.5)),
            "delta_utility_ci95_high": float(np.percentile(b_diff_util, 97.5)),
        })
        
    df_comp = pd.DataFrame(comp_rows)
    comp_csv = OUTPUT_DIR / "a4_bootstrap_comparisons.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"Saved {comp_csv}")
    print(df_comp[["comparison", "delta_succ10_pct", "delta_succ10_ci95_low", "delta_succ10_ci95_high", "delta_utility", "delta_utility_ci95_low", "delta_utility_ci95_high"]])

if __name__ == "__main__":
    main()
