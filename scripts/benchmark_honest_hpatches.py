import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO

import pipeline_utils as pu
from metrics import evaluate_matches, compute_label_free_proxy
from hpatches import load_split, load_pair, list_pairs
from level1_env import Level1Env
from level2_env import Level2Env
from image_stats import compute_pair_stats, normalize_stats

L1_MODEL_PATH = "checkpoints/level1_best/best_model.zip"
L2_MODEL_PATH = "checkpoints/level2_14preset_best/best_model.zip"
SPLIT = "test"

# Baseline budgets
FIXED_N = {
    "ORB": 1000, "SIFT": 2000, "AKAZE": 1000, "BRISK": 2000,
    "FAST+BRIEF": 1000, "KAZE": 4000, "GFTT+BRIEF": 1000,
    "GFTT+SIFT": 1000, "STAR+BRIEF": 100, "ORB+FREAK": 2000,
    "FAST+FREAK": 1000, "XFeat": 1000, "ALIKED": 500, "ORB+BEBLID": 4000,
}
CLASSICAL_11 = [
    "ORB", "SIFT", "AKAZE", "BRISK", "FAST+BRIEF", "KAZE",
    "GFTT+BRIEF", "GFTT+SIFT", "STAR+BRIEF", "ORB+FREAK", "FAST+FREAK"
]

def auc(errors, tau, n=400):
    e = np.clip(np.nan_to_num(np.asarray(errors, dtype=float), nan=1e6, posinf=1e6), 0, tau)
    ts = np.linspace(0, tau, n)
    recall = np.array([(e < t).mean() for t in ts])
    return float((np.trapezoid(recall, ts) if hasattr(np, 'trapezoid') else np.trapz(recall, ts)) / tau)

def run_benchmark(output_dir="logs", split_file=None):
    os.makedirs(output_dir, exist_ok=True)
    l1_model = PPO.load(L1_MODEL_PATH)
    l2_model = PPO.load(L2_MODEL_PATH)
    pairs = list_pairs(load_split(split_file)[SPLIT])
    print(f"Loaded {len(pairs)} test pairs from {SPLIT} split (17 sequences x 5 pairs).")

    env = Level2Env(split=SPLIT, seed=123, eval_mode=True, split_file=split_file)

    records = []

    for seq, k in tqdm(pairs, desc="Evaluating test pairs"):
        img1, img2, H = load_pair(seq, k)
        kind = "illumination" if seq.startswith("i_") else "viewpoint"
        stats = normalize_stats(compute_pair_stats(img1, img2))

        # 1. Level 1 selection for AFM-RL
        t_l1_start = time.perf_counter()
        l1_act, _ = l1_model.predict(stats, deterministic=True)
        t_l1 = (time.perf_counter() - t_l1_start) * 1000.0
        afm_rl_preset = pu.PRESET_NAMES[int(l1_act)]

        # Run all 14 presets (for individual preset analysis, averages, and oracle)
        pair_preset_results = {}
        for preset in pu.PRESET_NAMES:
            # A. Fixed-N Baseline
            t0 = time.perf_counter()
            r_fix = pu.run_pipeline(img1, img2, preset, FIXED_N[preset], key1=(seq, 1), key2=(seq, k))
            m_fix = evaluate_matches(r_fix["pts1"], r_fix["pts2"], H, img1.shape)
            t_fix_ms = (time.perf_counter() - t0) * 1000.0

            # B. Fixed-4000 Baseline
            t0_4k = time.perf_counter()
            r_4k = pu.run_pipeline(img1, img2, preset, 4000, key1=(seq, 1), key2=(seq, k))
            m_4k = evaluate_matches(r_4k["pts1"], r_4k["pts2"], H, img1.shape)
            t_4k_ms = (time.perf_counter() - t0_4k) * 1000.0

            # C. Level 2 Adaptive (100% Label-Free Proxy Mode)
            t0_rl = time.perf_counter()
            obs, info = env.reset_to(seq, k, preset)
            steps = 0
            cum_kp = env.effective_kp
            while True:
                act, _ = l2_model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(act))
                steps += 1
                cum_kp += env.effective_kp
                if term or trunc:
                    break
            t_rl_ms = (time.perf_counter() - t0_rl) * 1000.0

            pair_preset_results[preset] = {
                "fixed_N": r_fix["n_kp"], "fixed_err": m_fix["corner_err"], "fixed_ok": m_fix["h_ok"], "fixed_ms": t_fix_ms,
                "4k_N": r_4k["n_kp"], "4k_err": m_4k["corner_err"], "4k_ok": m_4k["h_ok"], "4k_ms": t_4k_ms,
                "l2_best_N": info["best_n_kp"], "l2_best_err": info["best_corner_err"], "l2_best_ok": info["best_h_ok"],
                "l2_final_N": info["final_n_kp"], "l2_final_err": info["final_corner_err"], "l2_final_ok": info["final_h_ok"],
                "l2_steps": steps, "l2_cum_kp": cum_kp, "l2_ms": t_rl_ms,
            }

            records.append({
                "seq": seq, "k": k, "kind": kind, "preset": preset,
                "fixed_N": r_fix["n_kp"], "fixed_err": m_fix["corner_err"], "fixed_ok": m_fix["h_ok"], "fixed_ms": t_fix_ms,
                "4k_N": r_4k["n_kp"], "4k_err": m_4k["corner_err"], "4k_ok": m_4k["h_ok"], "4k_ms": t_4k_ms,
                "l2_best_N": info["best_n_kp"], "l2_best_err": info["best_corner_err"], "l2_best_ok": info["best_h_ok"],
                "l2_final_N": info["final_n_kp"], "l2_final_err": info["final_corner_err"], "l2_final_ok": info["final_h_ok"],
                "l2_steps": steps, "l2_cum_kp": cum_kp, "l2_ms": t_rl_ms,
            })

    df_all = pd.DataFrame(records)
    out_all_path = os.path.join(output_dir, "honest_all_presets_test.csv")
    df_all.to_csv(out_all_path, index=False)
    print(f"Saved {out_all_path}")

    # Now assemble the comprehensive method comparisons for Table 2 & baselines
    methods_data = []

    # 1. Fixed-4000 (11 Classical Average)
    df_c11 = df_all[df_all["preset"].isin(CLASSICAL_11)]
    methods_data.append({
        "Method": "Fixed-4000 (11 Classical Avg)",
        "Keypoints": df_c11["4k_N"].mean(),
        "Cum_Keypoints": df_c11["4k_N"].mean(),
        "Time_ms": df_c11["4k_ms"].mean(),
        "Success@3px": df_c11["4k_ok"].mean() * 100.0,
        "AUC@1px": auc(df_c11["4k_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_c11["4k_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_c11["4k_err"], 5.0) * 100.0,
    })

    # 2. Fixed-4000 (14 Preset Full Avg)
    methods_data.append({
        "Method": "Fixed-4000 (14 Preset Avg)",
        "Keypoints": df_all["4k_N"].mean(),
        "Cum_Keypoints": df_all["4k_N"].mean(),
        "Time_ms": df_all["4k_ms"].mean(),
        "Success@3px": df_all["4k_ok"].mean() * 100.0,
        "AUC@1px": auc(df_all["4k_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_all["4k_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_all["4k_err"], 5.0) * 100.0,
    })

    # 3. SIFT + Fixed-2000
    df_sift = df_all[df_all["preset"] == "SIFT"]
    methods_data.append({
        "Method": "SIFT + Fixed-2000",
        "Keypoints": df_sift["fixed_N"].mean(),
        "Cum_Keypoints": df_sift["fixed_N"].mean(),
        "Time_ms": df_sift["fixed_ms"].mean(),
        "Success@3px": df_sift["fixed_ok"].mean() * 100.0,
        "AUC@1px": auc(df_sift["fixed_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_sift["fixed_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_sift["fixed_err"], 5.0) * 100.0,
    })

    # 4. SIFT + Fixed-4000
    methods_data.append({
        "Method": "SIFT + Fixed-4000",
        "Keypoints": df_sift["4k_N"].mean(),
        "Cum_Keypoints": df_sift["4k_N"].mean(),
        "Time_ms": df_sift["4k_ms"].mean(),
        "Success@3px": df_sift["4k_ok"].mean() * 100.0,
        "AUC@1px": auc(df_sift["4k_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_sift["4k_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_sift["4k_err"], 5.0) * 100.0,
    })

    # 5. SIFT + Level 2 (Proxy Scoreboard)
    methods_data.append({
        "Method": "SIFT + Level 2 (Proxy Scoreboard)",
        "Keypoints": df_sift["l2_best_N"].mean(),
        "Cum_Keypoints": df_sift["l2_cum_kp"].mean(),
        "Time_ms": df_sift["l2_ms"].mean(),
        "Success@3px": df_sift["l2_best_ok"].mean() * 100.0,
        "AUC@1px": auc(df_sift["l2_best_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_sift["l2_best_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_sift["l2_best_err"], 5.0) * 100.0,
    })

    # 6. SIFT + Level 2 (Strict STOP Action)
    methods_data.append({
        "Method": "SIFT + Level 2 (Strict STOP)",
        "Keypoints": df_sift["l2_final_N"].mean(),
        "Cum_Keypoints": df_sift["l2_cum_kp"].mean(),
        "Time_ms": df_sift["l2_ms"].mean(),
        "Success@3px": df_sift["l2_final_ok"].mean() * 100.0,
        "AUC@1px": auc(df_sift["l2_final_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_sift["l2_final_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_sift["l2_final_err"], 5.0) * 100.0,
    })

    # 7. Level 2 Alone (Average across 11 classical presets)
    methods_data.append({
        "Method": "Level 2 Alone (11 Classical Avg, Proxy)",
        "Keypoints": df_c11["l2_best_N"].mean(),
        "Cum_Keypoints": df_c11["l2_cum_kp"].mean(),
        "Time_ms": df_c11["l2_ms"].mean(),
        "Success@3px": df_c11["l2_best_ok"].mean() * 100.0,
        "AUC@1px": auc(df_c11["l2_best_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_c11["l2_best_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_c11["l2_best_err"], 5.0) * 100.0,
    })

    # 8. Full AFM-RL (Level 1 + Level 2, 100% Label-Free Proxy)
    # Extract the rows where preset == chosen Level 1 preset for each pair
    l1_env = Level1Env(split=SPLIT, seed=123, eval_mode=True, split_file=split_file)
    afm_rl_rows = []
    oracle_rows = []
    for seq, k in pairs:
        obs, _ = l1_env.reset_to(seq, k)
        act, _ = l1_model.predict(obs, deterministic=True)
        chosen_p = pu.PRESET_NAMES[int(act)]
        
        pair_rows = df_all[(df_all["seq"] == seq) & (df_all["k"] == k)]
        chosen_row = pair_rows[pair_rows["preset"] == chosen_p].iloc[0]
        afm_rl_rows.append(chosen_row)

        # Oracle preset: the preset with lowest corner error under Level 2
        oracle_row = pair_rows.sort_values("l2_best_err").iloc[0]
        oracle_rows.append(oracle_row)

    df_afm = pd.DataFrame(afm_rl_rows)
    methods_data.append({
        "Method": "Full AFM-RL (L1 + L2, Proxy Scoreboard)",
        "Keypoints": df_afm["l2_best_N"].mean(),
        "Cum_Keypoints": df_afm["l2_cum_kp"].mean(),
        "Time_ms": df_afm["l2_ms"].mean(),
        "Success@3px": df_afm["l2_best_ok"].mean() * 100.0,
        "AUC@1px": auc(df_afm["l2_best_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_afm["l2_best_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_afm["l2_best_err"], 5.0) * 100.0,
    })

    methods_data.append({
        "Method": "Full AFM-RL (L1 + L2, Strict STOP)",
        "Keypoints": df_afm["l2_final_N"].mean(),
        "Cum_Keypoints": df_afm["l2_cum_kp"].mean(),
        "Time_ms": df_afm["l2_ms"].mean(),
        "Success@3px": df_afm["l2_final_ok"].mean() * 100.0,
        "AUC@1px": auc(df_afm["l2_final_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_afm["l2_final_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_afm["l2_final_err"], 5.0) * 100.0,
    })

    # 9. Learned Preset Baselines: ALIKED and XFeat
    df_aliked = df_all[df_all["preset"] == "ALIKED"]
    methods_data.append({
        "Method": "ALIKED + Fixed-500 (Learned)",
        "Keypoints": df_aliked["fixed_N"].mean(),
        "Cum_Keypoints": df_aliked["fixed_N"].mean(),
        "Time_ms": df_aliked["fixed_ms"].mean(),
        "Success@3px": df_aliked["fixed_ok"].mean() * 100.0,
        "AUC@1px": auc(df_aliked["fixed_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_aliked["fixed_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_aliked["fixed_err"], 5.0) * 100.0,
    })

    df_xfeat = df_all[df_all["preset"] == "XFeat"]
    methods_data.append({
        "Method": "XFeat + Fixed-1000 (Learned)",
        "Keypoints": df_xfeat["fixed_N"].mean(),
        "Cum_Keypoints": df_xfeat["fixed_N"].mean(),
        "Time_ms": df_xfeat["fixed_ms"].mean(),
        "Success@3px": df_xfeat["fixed_ok"].mean() * 100.0,
        "AUC@1px": auc(df_xfeat["fixed_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_xfeat["fixed_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_xfeat["fixed_err"], 5.0) * 100.0,
    })

    # 10. Oracle Preset Selector (Upper Bound)
    df_oracle = pd.DataFrame(oracle_rows)
    methods_data.append({
        "Method": "Oracle Preset Selection (Upper Bound)",
        "Keypoints": df_oracle["l2_best_N"].mean(),
        "Cum_Keypoints": df_oracle["l2_cum_kp"].mean(),
        "Time_ms": df_oracle["l2_ms"].mean(),
        "Success@3px": df_oracle["l2_best_ok"].mean() * 100.0,
        "AUC@1px": auc(df_oracle["l2_best_err"], 1.0) * 100.0,
        "AUC@3px": auc(df_oracle["l2_best_err"], 3.0) * 100.0,
        "AUC@5px": auc(df_oracle["l2_best_err"], 5.0) * 100.0,
    })

    df_summary = pd.DataFrame(methods_data)
    out_summary_path = os.path.join(output_dir, "honest_summary_table.csv")
    df_summary.to_csv(out_summary_path, index=False)
    print(f"Saved {out_summary_path}")
    print("\n" + "=" * 90)
    print("HONEST HPATCHES BENCHMARK RESULTS (100% LABEL-FREE INFERENCE)")
    print("=" * 90)
    print(df_summary.round(2).to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="logs", help="Directory for output CSVs")
    parser.add_argument("--split-file", default=None, help="Path to custom split JSON file")
    args = parser.parse_args()
    run_benchmark(output_dir=args.output_dir, split_file=args.split_file)

