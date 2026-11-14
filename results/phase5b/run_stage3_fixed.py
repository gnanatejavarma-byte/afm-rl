import os
import sys
import time
import concurrent.futures
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, 'src')

import megadepth as md
import pipeline_utils as pu
from megadepth_eval_env import MegaDepthEvalEnv

OUTPUT_DIR = Path("results/phase5b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_STAGE3_CSV = OUTPUT_DIR / "megadepth_stage3_afm_rl_v1_fixed.csv"

def evaluate_fold(fold):
    out_csv = OUTPUT_DIR / f"megadepth_stage3_afm_rl_v1_fixed_fold{fold}.csv"
    existing_keys = set()
    if out_csv.exists():
        df_ex = pd.read_csv(out_csv)
        for _, row in df_ex.iterrows():
            existing_keys.add((row["fname"], int(row["i"]), int(row["j"])))
        print(f"[Fold {fold}] Resuming: found {len(existing_keys)} existing records.")
    else:
        pd.DataFrame(columns=[
            "fold", "fname", "i", "j", "chosen_preset", "steps", "actions", "first_action",
            "final_requested_N", "final_actual_N", "cumulative_actual_kp", "is_plateau",
            "p_epi_step0", "p_epi_final", "rot_err", "trans_err", "pose_err", "pose_failure",
            "prob_act0", "prob_act1", "prob_act2", "prob_act3"
        ]).to_csv(out_csv, index=False)
        
    pairs = md.list_pairs()
    l1_path = f"checkpoints_v2/fold{fold}/level1_best/best_model.zip"
    l2_path = f"checkpoints_v2/fold{fold}/level2_14preset_best/best_model.zip"
    l1_model = PPO.load(l1_path)
    l2_model = PPO.load(l2_path)
    env = MegaDepthEvalEnv(eval_mode=True)
    
    batch = []
    processed = 0
    t0 = time.perf_counter()
    
    for pair_idx, (fname, i, j) in enumerate(pairs):
        if (fname, i, j) in existing_keys:
            continue
            
        img0, img1, K0, K1, T_gt = md.load_pair(fname, i, j)
        st = env._stats(fname, i, j, img0, img1)
        action_l1, _ = l1_model.predict(st, deterministic=True)
        preset = pu.PRESET_NAMES[int(action_l1)]
        
        obs, info = env.reset_to(fname, i, j, preset)
        p_epi_step0 = float(env.last_proxy)
        
        obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32)
        with torch.no_grad():
            dist = l2_model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            
        actions = []
        done = False
        first_action = None
        
        while not done:
            action_l2, _ = l2_model.predict(obs, deterministic=True)
            act_int = int(action_l2)
            if first_action is None:
                first_action = act_int
            actions.append(act_int)
            obs, reward, terminated, truncated, info = env.step(act_int)
            done = terminated or truncated
            
        final_pts0 = env.last_r["pts1"]
        final_pts1 = env.last_r["pts2"]
        p_epi_final = float(env.last_proxy)
        
        pose_ret = md.estimate_pose(final_pts0, final_pts1, K0, K1)
        if pose_ret is None:
            pose_failure = True
            rot_err = trans_err = pose_err = 180.0
        else:
            pose_failure = False
            R_est, t_est = pose_ret
            c = np.clip((np.trace(R_est.T @ T_gt[:3, :3]) - 1) / 2, -1, 1)
            rot_err = float(np.rad2deg(np.abs(np.arccos(c))))
            v = np.clip(np.dot(t_est, T_gt[:3, 3]) / (np.linalg.norm(t_est) * np.linalg.norm(T_gt[:3, 3]) + 1e-12), -1, 1)
            t_err = float(np.rad2deg(np.arccos(v)))
            trans_err = min(t_err, 180.0 - t_err)
            pose_err = max(rot_err, trans_err)
            
        batch.append({
            "fold": fold,
            "fname": fname,
            "i": i,
            "j": j,
            "chosen_preset": preset,
            "steps": info["steps"],
            "actions": str(actions),
            "first_action": first_action,
            "final_requested_N": env.n_kp,
            "final_actual_N": env.effective_kp,
            "cumulative_actual_kp": env.cum_kp,
            "is_plateau": int(env.effective_kp >= env.pool_size),
            "p_epi_step0": p_epi_step0,
            "p_epi_final": p_epi_final,
            "rot_err": rot_err,
            "trans_err": trans_err,
            "pose_err": pose_err,
            "pose_failure": pose_failure,
            "prob_act0": float(probs[0]),
            "prob_act1": float(probs[1]),
            "prob_act2": float(probs[2]),
            "prob_act3": float(probs[3])
        })
        processed += 1
        
        if len(batch) >= 20:
            pd.DataFrame(batch).to_csv(out_csv, mode="a", header=False, index=False)
            batch = []
            if processed % 200 == 0:
                print(f"[Fold {fold}] Processed {processed}/{len(pairs)} pairs in {time.perf_counter() - t0:.1f}s...")
                
    if batch:
        pd.DataFrame(batch).to_csv(out_csv, mode="a", header=False, index=False)
        batch = []
        
    print(f"[Fold {fold}] COMPLETED all {len(pairs)} pairs in {time.perf_counter() - t0:.1f}s!")
    return out_csv

def compute_all_summaries():
    print("\nComputing all Section A.4 diagnostic summaries...")
    df = pd.read_csv(FIXED_STAGE3_CSV)
    
    # 1. Per fold and pooled summary table
    summary_rows = []
    
    def calc_metrics(sub_df, name):
        succ10 = (sub_df["pose_err"] < 10.0).mean() * 100.0
        auc5, auc10, auc20 = [v * 100.0 for v in md.pose_auc(sub_df["pose_err"])]
        mean_steps = sub_df["steps"].mean()
        mean_final_n = sub_df["final_requested_N"].mean()
        mean_final_actual_n = sub_df["final_actual_N"].mean()
        mean_cum_kp = sub_df["cumulative_actual_kp"].mean()
        plateau_frac = sub_df["is_plateau"].mean() * 100.0
        
        # Action histograms
        all_actions = []
        for acts_str in sub_df["actions"]:
            try:
                acts = eval(acts_str)
                all_actions.extend(acts)
            except Exception:
                pass
        act_counts = pd.Series(all_actions).value_counts(normalize=True).to_dict()
        pct_act0 = act_counts.get(0, 0.0) * 100.0
        pct_act1 = act_counts.get(1, 0.0) * 100.0
        pct_act2 = act_counts.get(2, 0.0) * 100.0
        pct_act3 = act_counts.get(3, 0.0) * 100.0
        
        first_act_counts = sub_df["first_action"].value_counts(normalize=True).to_dict()
        first_act0 = first_act_counts.get(0, 0.0) * 100.0
        first_act1 = first_act_counts.get(1, 0.0) * 100.0
        first_act2 = first_act_counts.get(2, 0.0) * 100.0
        first_act3 = first_act_counts.get(3, 0.0) * 100.0
        
        preset_dist = sub_df["chosen_preset"].value_counts().to_dict()
        
        return {
            "scope": name,
            "n_rollouts": len(sub_df),
            "succ10_deg": succ10,
            "auc5_deg": auc5,
            "auc10_deg": auc10,
            "auc20_deg": auc20,
            "mean_steps": mean_steps,
            "mean_final_requested_N": mean_final_n,
            "mean_final_actual_N": mean_final_actual_n,
            "mean_cumulative_actual_kp": mean_cum_kp,
            "plateau_pct": plateau_frac,
            "pct_act0_plus500": pct_act0,
            "pct_act1_plus1000": pct_act1,
            "pct_act2_minus500": pct_act2,
            "pct_act3_STOP": pct_act3,
            "first_act0_plus500": first_act0,
            "first_act1_plus1000": first_act1,
            "first_act2_minus500": first_act2,
            "first_act3_STOP": first_act3,
            "preset_distribution": str(preset_dist)
        }
        
    for f in range(5):
        sub = df[df["fold"] == f]
        summary_rows.append(calc_metrics(sub, f"Fold {f}"))
        
    summary_rows.append(calc_metrics(df, "Pooled (All 5 Folds)"))
    
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUTPUT_DIR / "a4_stage3_fixed_summary.csv", index=False)
    print("Saved results/phase5b/a4_stage3_fixed_summary.csv")
    
    # 2. Step 0 p_epi vs HPatches proxy quantiles
    q_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    md_p_epi_step0 = df["p_epi_step0"]
    md_quantiles = md_p_epi_step0.quantile(q_levels).values
    
    # Load HPatches Step 0 proxy from Phase 2c val pairs or Phase 3 folds pairs
    df_hp = pd.read_csv("results/phase3/folds_pairs.csv")
    hp_proxy_step0 = df_hp["proxy"]
    hp_quantiles = hp_proxy_step0.quantile(q_levels).values
    
    df_q = pd.DataFrame({
        "quantile": ["5%", "25%", "50%", "75%", "95%"],
        "p_epi_megadepth_step0": md_quantiles,
        "proxy_hpatches_step0": hp_quantiles,
        "difference": md_quantiles - hp_quantiles
    })
    df_q.to_csv(OUTPUT_DIR / "a4_step0_proxy_quantiles.csv", index=False)
    print("Saved results/phase5b/a4_step0_proxy_quantiles.csv")
    
    # 3. Level 2 action probabilities by proxy bucket
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]"]
    df["proxy_bucket"] = pd.cut(df["p_epi_step0"], bins=bins, labels=bin_labels, right=False)
    
    bucket_records = []
    for bl in bin_labels:
        b_df = df[df["proxy_bucket"] == bl]
        if len(b_df) > 0:
            bucket_records.append({
                "proxy_bucket": bl,
                "n_episodes": len(b_df),
                "mean_prob_act0_plus500": b_df["prob_act0"].mean(),
                "mean_prob_act1_plus1000": b_df["prob_act1"].mean(),
                "mean_prob_act2_minus500": b_df["prob_act2"].mean(),
                "mean_prob_act3_STOP": b_df["prob_act3"].mean(),
                "actual_pct_first_act3_STOP": (b_df["first_action"] == 3).mean() * 100.0
            })
    df_buckets = pd.DataFrame(bucket_records)
    df_buckets.to_csv(OUTPUT_DIR / "a4_l2_action_probs_by_proxy_bucket.csv", index=False)
    print("Saved results/phase5b/a4_l2_action_probs_by_proxy_bucket.csv")
    
    # 4. Bootstrap comparisons vs baselines (reference-image-cluster bootstrap, 10,000 resamples)
    print("Running reference-image-cluster bootstrap (10,000 resamples)...")
    df_stage2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
    df_aliked500 = df_stage2a[(df_stage2a["preset"] == "ALIKED") & (df_stage2a["N"] == 500)].set_index(["fname", "i", "j"])
    df_aliked1000 = df_stage2a[(df_stage2a["preset"] == "ALIKED") & (df_stage2a["N"] == 1000)].set_index(["fname", "i", "j"])
    df_sift4000 = df_stage2a[(df_stage2a["preset"] == "SIFT") & (df_stage2a["N"] == 4000)].set_index(["fname", "i", "j"])
    
    # Group pairs by anchor image (fname, i)
    pairs = md.list_pairs()
    cluster_to_pairs = {}
    for fname, i, j in pairs:
        c_key = (fname, i)
        if c_key not in cluster_to_pairs:
            cluster_to_pairs[c_key] = []
        cluster_to_pairs[c_key].append((fname, i, j))
        
    clusters = list(cluster_to_pairs.keys())
    n_clusters = len(clusters)
    
    # Average across the 5 folds for each pair for AFM-RL v1
    v1_per_pair = df.groupby(["fname", "i", "j"]).agg({
        "pose_err": "mean",
        "cumulative_actual_kp": "mean"
    })
    
    pair_data = []
    for fname, i, j in pairs:
        row_v1 = v1_per_pair.loc[(fname, i, j)]
        row_a500 = df_aliked500.loc[(fname, i, j)]
        row_a1000 = df_aliked1000.loc[(fname, i, j)]
        row_s4000 = df_sift4000.loc[(fname, i, j)]
        
        pair_data.append({
            "cluster": (fname, i),
            "v1_succ": float(row_v1["pose_err"] < 10.0),
            "v1_err": float(row_v1["pose_err"]),
            "v1_kp": float(row_v1["cumulative_actual_kp"]),
            
            "a500_succ": float(row_a500["pose_err"] < 10.0),
            "a500_err": float(row_a500["pose_err"]),
            "a500_kp": float(row_a500["actual_kp"]),
            
            "a1000_succ": float(row_a1000["pose_err"] < 10.0),
            "a1000_err": float(row_a1000["pose_err"]),
            "a1000_kp": float(row_a1000["actual_kp"]),
            
            "s4000_succ": float(row_s4000["pose_err"] < 10.0),
            "s4000_err": float(row_s4000["pose_err"]),
            "s4000_kp": float(row_s4000["actual_kp"]),
        })
    df_p = pd.DataFrame(pair_data)
    
    # Cluster bootstrap 10,000 resamples
    rng = np.random.RandomState(42)
    B = 10000
    
    # Pre-aggregate by cluster
    clust_agg = df_p.groupby("cluster").agg({
        "v1_succ": ["sum", "count"],
        "v1_kp": "sum",
        "a500_succ": "sum",
        "a500_kp": "sum",
        "a1000_succ": "sum",
        "a1000_kp": "sum",
        "s4000_succ": "sum",
        "s4000_kp": "sum"
    })
    
    # Fast array arrays
    c_counts = clust_agg[("v1_succ", "count")].values
    c_v1_succ = clust_agg[("v1_succ", "sum")].values
    c_v1_kp = clust_agg[("v1_kp", "sum")].values
    
    c_a500_succ = clust_agg[("a500_succ", "sum")].values
    c_a500_kp = clust_agg[("a500_kp", "sum")].values
    
    c_a1000_succ = clust_agg[("a1000_succ", "sum")].values
    c_a1000_kp = clust_agg[("a1000_kp", "sum")].values
    
    c_s4000_succ = clust_agg[("s4000_succ", "sum")].values
    c_s4000_kp = clust_agg[("s4000_kp", "sum")].values
    
    boot_indices = rng.randint(0, n_clusters, size=(B, n_clusters))
    
    def get_ci(num_arr, den_arr, base_delta):
        b_num = np.take(num_arr, boot_indices).sum(axis=1)
        b_den = np.take(den_arr, boot_indices).sum(axis=1)
        samples = b_num / b_den
        return np.percentile(samples, 2.5), np.percentile(samples, 97.5)
        
    diff_a500_succ = c_v1_succ - c_a500_succ
    diff_a500_kp = c_v1_kp - c_a500_kp
    
    diff_a1000_succ = c_v1_succ - c_a1000_succ
    diff_a1000_kp = c_v1_kp - c_a1000_kp
    
    diff_s4000_succ = c_v1_succ - c_s4000_succ
    diff_s4000_kp = c_v1_kp - c_s4000_kp
    
    comp_records = []
    
    # 1. vs ALIKED-500
    d_succ = (df_p["v1_succ"].mean() - df_p["a500_succ"].mean()) * 100.0
    ci_succ = [v * 100.0 for v in get_ci(diff_a500_succ, c_counts, d_succ)]
    d_auc10 = (md.pose_auc(df_p["v1_err"])[1] - md.pose_auc(df_p["a500_err"])[1]) * 100.0
    d_kp = df_p["v1_kp"].mean() - df_p["a500_kp"].mean()
    ci_kp = get_ci(diff_a500_kp, c_counts, d_kp)
    comp_records.append({
        "comparison": "AFM-RL v1 (Fixed) vs ALIKED-500",
        "delta_succ10": d_succ,
        "cluster_95_ci_succ10": f"[{ci_succ[0]:.2f}, {ci_succ[1]:.2f}]",
        "delta_auc10": d_auc10,
        "delta_actual_kp": d_kp,
        "cluster_95_ci_actual_kp": f"[{ci_kp[0]:.1f}, {ci_kp[1]:.1f}]"
    })
    
    # 2. vs ALIKED-1000
    d_succ = (df_p["v1_succ"].mean() - df_p["a1000_succ"].mean()) * 100.0
    ci_succ = [v * 100.0 for v in get_ci(diff_a1000_succ, c_counts, d_succ)]
    d_auc10 = (md.pose_auc(df_p["v1_err"])[1] - md.pose_auc(df_p["a1000_err"])[1]) * 100.0
    d_kp = df_p["v1_kp"].mean() - df_p["a1000_kp"].mean()
    ci_kp = get_ci(diff_a1000_kp, c_counts, d_kp)
    comp_records.append({
        "comparison": "AFM-RL v1 (Fixed) vs ALIKED-1000",
        "delta_succ10": d_succ,
        "cluster_95_ci_succ10": f"[{ci_succ[0]:.2f}, {ci_succ[1]:.2f}]",
        "delta_auc10": d_auc10,
        "delta_actual_kp": d_kp,
        "cluster_95_ci_actual_kp": f"[{ci_kp[0]:.1f}, {ci_kp[1]:.1f}]"
    })
    
    # 3. vs SIFT-4000
    d_succ = (df_p["v1_succ"].mean() - df_p["s4000_succ"].mean()) * 100.0
    ci_succ = [v * 100.0 for v in get_ci(diff_s4000_succ, c_counts, d_succ)]
    d_auc10 = (md.pose_auc(df_p["v1_err"])[1] - md.pose_auc(df_p["s4000_err"])[1]) * 100.0
    d_kp = df_p["v1_kp"].mean() - df_p["s4000_kp"].mean()
    ci_kp = get_ci(diff_s4000_kp, c_counts, d_kp)
    comp_records.append({
        "comparison": "AFM-RL v1 (Fixed) vs SIFT-4000",
        "delta_succ10": d_succ,
        "cluster_95_ci_succ10": f"[{ci_succ[0]:.2f}, {ci_succ[1]:.2f}]",
        "delta_auc10": d_auc10,
        "delta_actual_kp": d_kp,
        "cluster_95_ci_actual_kp": f"[{ci_kp[0]:.1f}, {ci_kp[1]:.1f}]"
    })
    
    df_boot = pd.DataFrame(comp_records)
    df_boot.to_csv(OUTPUT_DIR / "a4_bootstrap_comparisons.csv", index=False)
    print("Saved results/phase5b/a4_bootstrap_comparisons.csv")
    print(df_boot.to_string(index=False))

def main():
    print("=" * 70)
    print("RUNNING 5-FOLD PARALLEL EVALUATION OF FIXED AFM-RL v1 (5 WORKERS)")
    print("=" * 70)
    
    t0_all = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(evaluate_fold, f) for f in range(5)]
        results = [f.result() for f in futures]
        
    print("\nAll 5 fold workers finished. Merging into single CSV...")
    df_list = [pd.read_csv(f) for f in results]
    df_merged = pd.concat(df_list, ignore_index=True)
    df_merged.to_csv(FIXED_STAGE3_CSV, index=False)
    print(f"Saved merged dataset with {len(df_merged)} rows to {FIXED_STAGE3_CSV}")
    print(f"Total Parallel Execution Time: {time.perf_counter() - t0_all:.2f} s.")
    
    compute_all_summaries()

if __name__ == "__main__":
    main()
