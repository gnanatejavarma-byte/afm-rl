import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Process

sys.path.insert(0, 'src')
import megadepth as md
from megadepth_eval_env import compute_epipolar_proxy

OUTPUT_DIR = Path("results/phase5b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERM_RECORDS_CSV = OUTPUT_DIR / "b2_permutations_eval_records.csv"
MATCH_DIR = Path("results/phase4b/megadepth_matches")
STAGE2A_CSV = Path("results/phase4b/megadepth_stage2a_priority_evaluations.csv")
SEEDS_OFFSET = [42, 123, 456, 789, 999]

def process_shard_fast(worker_id, pair_sublist, offset_idx):
    shard_csv = OUTPUT_DIR / f"b2_perm_shard_{worker_id}.csv"
    if shard_csv.exists():
        df_old = pd.read_csv(shard_csv)
        if len(df_old) == len(pair_sublist) * 3:
            print(f"[Worker {worker_id}] Shard already complete ({len(df_old)} rows).", flush=True)
            return
            
    df_stage2a = pd.read_csv(STAGE2A_CSV)
    s2a_indexed = df_stage2a.set_index(["fname", "i", "j", "preset", "N"])
    
    img_cache = {}
    def get_image_K(fname, k):
        key = (fname, k)
        if key not in img_cache:
            _, K = md.load_image(fname, k)
            img_cache[key] = K
        return img_cache[key]
        
    scene_cache = {}
    def get_scene_data(fname, i, j):
        if fname not in scene_cache:
            scene_cache[fname] = md._scene(fname)
        sc = scene_cache[fname]
        K0 = get_image_K(fname, i)
        K1 = get_image_K(fname, j)
        T_gt = sc["poses"][j] @ np.linalg.inv(sc["poses"][i])
        return K0, K1, T_gt
        
    configs = [
        ("ALIKED", 500),
        ("ALIKED", 1000),
        ("SIFT", 4000),
    ]
    
    records = []
    t0 = time.perf_counter()
    n_pairs = len(pair_sublist)
    print(f"[Worker {worker_id}] Starting {n_pairs} pairs (index {offset_idx} to {offset_idx + n_pairs - 1})...", flush=True)
    
    for local_idx, (fname, i, j) in enumerate(pair_sublist):
        pair_idx = offset_idx + local_idx
        if (local_idx + 1) % 25 == 0 or (local_idx + 1) == n_pairs:
            elapsed = time.perf_counter() - t0
            rate = (local_idx + 1) / elapsed
            eta = (n_pairs - (local_idx + 1)) / rate
            print(f"[Worker {worker_id}] Processed {local_idx + 1}/{n_pairs} pairs ({rate:.2f} pairs/s, ETA {eta:.1f}s)...", flush=True)
            
        K0, K1, T_gt = get_scene_data(fname, i, j)
        seeds = [pair_idx * 1000 + off for off in SEEDS_OFFSET]
        
        for preset, N in configs:
            clean_preset = preset.replace("+", "_")
            match_f = MATCH_DIR / f"{fname}_{i}_{j}_{clean_preset}_N{N}.npz"
            if match_f.exists():
                d = np.load(match_f)
                pts0, pts1 = d["pts0"], d["pts1"]
            else:
                pts0, pts1 = np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
                
            n_matches = len(pts0)
            
            s2a_row = s2a_indexed.loc[(fname, i, j, preset, N)]
            orig_p_epi = float(s2a_row["p_epi"])
            orig_pose_err = float(s2a_row["pose_err"])
            orig_inlier_ratio = float(s2a_row["inlier_ratio"])
            orig_n_inliers = int(s2a_row["n_inliers"])
            
            rec = {
                "fname": fname, "i": i, "j": j, "pair_idx": pair_idx,
                "preset": preset, "N": N, "n_matches": n_matches,
                "orig_p_epi": orig_p_epi,
                "orig_pose_err": orig_pose_err,
                "orig_inlier_ratio": orig_inlier_ratio,
                "orig_n_inliers": orig_n_inliers
            }
            
            # 5 Permutations
            for s_idx, s_val in enumerate(seeds):
                if n_matches < 5:
                    p_perm_val = 0.0
                    err_perm_val = 180.0
                    inl_perm_val = 0
                else:
                    rng = np.random.RandomState(s_val)
                    perm = rng.permutation(n_matches)
                    p_perm = compute_epipolar_proxy(pts0[perm], pts1[perm], K0, K1)
                    r_perm = md.estimate_pose(pts0[perm], pts1[perm], K0, K1)
                    err_perm_val = md.pose_error(T_gt, r_perm[0], r_perm[1]) if r_perm is not None else 180.0
                    p_perm_val = p_perm["proxy"]
                    inl_perm_val = p_perm["n_inliers"]
                    
                rec[f"perm_{s_idx}_p_epi"] = p_perm_val
                rec[f"perm_{s_idx}_pose_err"] = err_perm_val
                rec[f"perm_{s_idx}_n_inliers"] = inl_perm_val
                
            records.append(rec)
            
    df_shard = pd.DataFrame(records)
    df_shard.to_csv(shard_csv, index=False)
    print(f"[Worker {worker_id}] Successfully saved {len(df_shard)} records to {shard_csv} in {time.perf_counter() - t0:.1f}s.", flush=True)

def run_simulation_and_analysis():
    print("=" * 70)
    print("RUNNING SECTION B.2 & B.3 SIMULATION & BOOTSTRAP ANALYSIS")
    print("=" * 70)
    
    assert PERM_RECORDS_CSV.exists(), f"Missing {PERM_RECORDS_CSV}"
    df_records = pd.read_csv(PERM_RECORDS_CSV)
    print(f"Loaded {len(df_records)} permutation evaluation records from {PERM_RECORDS_CSV}")
    
    pairs = md.list_pairs()
    idx_records = df_records.set_index(["fname", "i", "j", "preset", "N"])
    
    df_stage2a = pd.read_csv(STAGE2A_CSV)
    kp_lookup = df_stage2a.set_index(["fname", "i", "j", "preset", "N"])["actual_kp"].to_dict()
    
    methods = ["ALIKED-500", "ALIKED-1000", "SIFT-4000", "Cascade C* (tau=0.40)", "Always-Both", "Oracle-of-Two"]
    order_names = ["orig"] + [f"perm_{s}" for s in range(5)]
    sim_data = {m: {o: {"errs": [], "kps": []} for o in order_names} for m in methods}
    pair_scenes = []
    
    for fname, i, j in pairs:
        scene = "0015" if fname.startswith("0015") else "0022"
        pair_scenes.append(scene)
        
        row_a500 = idx_records.loc[(fname, i, j, "ALIKED", 500)]
        row_a1000 = idx_records.loc[(fname, i, j, "ALIKED", 1000)]
        row_s4000 = idx_records.loc[(fname, i, j, "SIFT", 4000)]
        
        kp_a500 = float(kp_lookup.get((fname, i, j, "ALIKED", 500), 500))
        kp_a1000 = float(kp_lookup.get((fname, i, j, "ALIKED", 1000), 1000))
        kp_s4000 = float(kp_lookup.get((fname, i, j, "SIFT", 4000), 4000))
        
        for o in order_names:
            p_prefix = f"{o}_p_epi"
            e_prefix = f"{o}_pose_err"
            
            p_a500 = float(row_a500[p_prefix])
            e_a500 = float(row_a500[e_prefix])
            
            p_a1000 = float(row_a1000[p_prefix])
            e_a1000 = float(row_a1000[e_prefix])
            
            p_s4000 = float(row_s4000[p_prefix])
            e_s4000 = float(row_s4000[e_prefix])
            
            sim_data["ALIKED-500"][o]["errs"].append(e_a500)
            sim_data["ALIKED-500"][o]["kps"].append(kp_a500)
            
            sim_data["ALIKED-1000"][o]["errs"].append(e_a1000)
            sim_data["ALIKED-1000"][o]["kps"].append(kp_a1000)
            
            sim_data["SIFT-4000"][o]["errs"].append(e_s4000)
            sim_data["SIFT-4000"][o]["kps"].append(kp_s4000)
            
            # Cascade C* (tau=0.40)
            if p_a500 >= 0.40:
                e_casc = e_a500
                kp_casc = kp_a500
            else:
                kp_casc = kp_a500 + kp_s4000
                e_casc = e_a500 if p_a500 >= p_s4000 else e_s4000
            sim_data["Cascade C* (tau=0.40)"][o]["errs"].append(e_casc)
            sim_data["Cascade C* (tau=0.40)"][o]["kps"].append(kp_casc)
            
            # Always-Both
            kp_both = kp_a500 + kp_s4000
            e_both = e_a500 if p_a500 >= p_s4000 else e_s4000
            sim_data["Always-Both"][o]["errs"].append(e_both)
            sim_data["Always-Both"][o]["kps"].append(kp_both)
            
            # Oracle-of-Two
            e_oracle = min(e_a500, e_s4000)
            sim_data["Oracle-of-Two"][o]["errs"].append(e_oracle)
            sim_data["Oracle-of-Two"][o]["kps"].append(kp_both)
            
    summary_rows = []
    scopes = [
        ("Scene 0015", [s == "0015" for s in pair_scenes]),
        ("Scene 0022", [s == "0022" for s in pair_scenes]),
        ("Pooled (All 1500)", [True] * len(pair_scenes)),
    ]
    
    for scope_name, mask in scopes:
        for m in methods:
            orig_errs = np.array(sim_data[m]["orig"]["errs"])[mask]
            orig_kps = np.array(sim_data[m]["orig"]["kps"])[mask]
            orig_succ = float(np.mean(orig_errs < 10.0)) * 100.0
            orig_auc10 = float(md.pose_auc(orig_errs)[1]) * 100.0
            orig_kp = float(np.mean(orig_kps))
            
            orig_norm_kp = orig_kps / 4000.0
            orig_acc = np.maximum(0.0, 1.0 - orig_errs / 10.0)
            orig_util = float(np.mean(orig_acc - 0.1 * orig_norm_kp))
            
            perm_succs = []
            perm_auc10s = []
            perm_kps = []
            perm_utils = []
            
            for s in range(5):
                p_errs = np.array(sim_data[m][f"perm_{s}"]["errs"])[mask]
                p_kps = np.array(sim_data[m][f"perm_{s}"]["kps"])[mask]
                perm_succs.append(float(np.mean(p_errs < 10.0)) * 100.0)
                perm_auc10s.append(float(md.pose_auc(p_errs)[1]) * 100.0)
                perm_kps.append(float(np.mean(p_kps)))
                
                p_norm_kp = p_kps / 4000.0
                p_acc = np.maximum(0.0, 1.0 - p_errs / 10.0)
                perm_utils.append(float(np.mean(p_acc - 0.1 * p_norm_kp)))
                
            summary_rows.append({
                "scope": scope_name,
                "method": m,
                "n_pairs": int(np.sum(mask)),
                "orig_succ10": orig_succ,
                "perm_mean_succ10": float(np.mean(perm_succs)),
                "perm_std_succ10": float(np.std(perm_succs)),
                "orig_auc10": orig_auc10,
                "perm_mean_auc10": float(np.mean(perm_auc10s)),
                "perm_std_auc10": float(np.std(perm_auc10s)),
                "orig_actual_kp": orig_kp,
                "perm_mean_actual_kp": float(np.mean(perm_kps)),
                "perm_std_actual_kp": float(np.std(perm_kps)),
                "orig_utility": orig_util,
                "perm_mean_utility": float(np.mean(perm_utils)),
                "perm_std_utility": float(np.std(perm_utils)),
            })
            
    df_perm_summary = pd.DataFrame(summary_rows)
    df_perm_summary.to_csv(OUTPUT_DIR / "b2_permutation_robustness_all_pairs.csv", index=False)
    print("\nSaved results/phase5b/b2_permutation_robustness_all_pairs.csv:")
    print(df_perm_summary[df_perm_summary["scope"] == "Pooled (All 1500)"][["method", "orig_succ10", "perm_mean_succ10", "perm_std_succ10", "orig_auc10", "perm_mean_auc10", "orig_utility", "perm_mean_utility"]].to_string(index=False))
    
    # Reference-image-cluster bootstrap
    cluster_keys = [(p[0], p[1]) for p in pairs]
    unique_clusters = list(set(cluster_keys))
    n_clusters = len(unique_clusters)
    cluster_idx_map = {c: idx for idx, c in enumerate(unique_clusters)}
    pair_cluster_indices = np.array([cluster_idx_map[c] for c in cluster_keys])
    
    B = 10000
    rng = np.random.RandomState(42)
    boot_indices = rng.randint(0, n_clusters, size=(B, n_clusters))
    
    def run_clust_bootstrap(arr_succ_a, arr_succ_b, arr_auc_a, arr_auc_b, arr_kp_a, arr_kp_b):
        diff_succ = np.asarray(arr_succ_a, dtype=float) - np.asarray(arr_succ_b, dtype=float)
        diff_kp = np.asarray(arr_kp_a, dtype=float) - np.asarray(arr_kp_b, dtype=float)
        
        c_diff_succ = np.bincount(pair_cluster_indices, weights=diff_succ, minlength=n_clusters)
        c_counts = np.bincount(pair_cluster_indices, minlength=n_clusters)
        c_diff_kp = np.bincount(pair_cluster_indices, weights=diff_kp, minlength=n_clusters)
        
        b_num_succ = np.take(c_diff_succ, boot_indices).sum(axis=1)
        b_den = np.take(c_counts, boot_indices).sum(axis=1)
        samples_succ = (b_num_succ / b_den) * 100.0
        
        b_num_kp = np.take(c_diff_kp, boot_indices).sum(axis=1)
        samples_kp = b_num_kp / b_den
        
        ci_succ = np.percentile(samples_succ, [2.5, 97.5])
        ci_kp = np.percentile(samples_kp, [2.5, 97.5])
        
        d_succ = np.mean(diff_succ) * 100.0
        d_auc10 = (md.pose_auc(arr_auc_a)[1] - md.pose_auc(arr_auc_b)[1]) * 100.0
        d_kp = np.mean(diff_kp)
        
        return {
            "delta_succ10": d_succ,
            "ci_succ10": f"[{ci_succ[0]:.2f}, {ci_succ[1]:.2f}]",
            "delta_auc10": d_auc10,
            "delta_kp": d_kp,
            "ci_kp": f"[{ci_kp[0]:.1f}, {ci_kp[1]:.1f}]"
        }
        
    boot_records = []
    comparisons = [
        ("Cascade C* vs ALIKED-1000", "Cascade C* (tau=0.40)", "ALIKED-1000"),
        ("Cascade C* vs SIFT-4000", "Cascade C* (tau=0.40)", "SIFT-4000"),
        ("Cascade C* vs Always-Both", "Cascade C* (tau=0.40)", "Always-Both"),
        ("Cascade C* vs Oracle-of-Two", "Cascade C* (tau=0.40)", "Oracle-of-Two"),
    ]
    
    for comp_name, m_a, m_b in comparisons:
        err_a = np.array(sim_data[m_a]["orig"]["errs"])
        err_b = np.array(sim_data[m_b]["orig"]["errs"])
        kp_a = np.array(sim_data[m_a]["orig"]["kps"])
        kp_b = np.array(sim_data[m_b]["orig"]["kps"])
        
        res = run_clust_bootstrap(err_a < 10.0, err_b < 10.0, err_a, err_b, kp_a, kp_b)
        boot_records.append({
            "order_mode": "original_order",
            "comparison": comp_name,
            **res
        })
        
    for comp_name, m_a, m_b in comparisons:
        err_a_perm = np.mean([np.array(sim_data[m_a][f"perm_{s}"]["errs"]) for s in range(5)], axis=0)
        err_b_perm = np.mean([np.array(sim_data[m_b][f"perm_{s}"]["errs"]) for s in range(5)], axis=0)
        succ_a_perm = np.mean([np.array(sim_data[m_a][f"perm_{s}"]["errs"]) < 10.0 for s in range(5)], axis=0)
        succ_b_perm = np.mean([np.array(sim_data[m_b][f"perm_{s}"]["errs"]) < 10.0 for s in range(5)], axis=0)
        kp_a_perm = np.mean([np.array(sim_data[m_a][f"perm_{s}"]["kps"]) for s in range(5)], axis=0)
        kp_b_perm = np.mean([np.array(sim_data[m_b][f"perm_{s}"]["kps"]) for s in range(5)], axis=0)
        
        res = run_clust_bootstrap(succ_a_perm, succ_b_perm, err_a_perm, err_b_perm, kp_a_perm, kp_b_perm)
        boot_records.append({
            "order_mode": "permutation_mean",
            "comparison": comp_name,
            **res
        })
        
    df_boot_perm = pd.DataFrame(boot_records)
    df_boot_perm.to_csv(OUTPUT_DIR / "b2_bootstrap_permutation_comparisons.csv", index=False)
    print("\nSaved results/phase5b/b2_bootstrap_permutation_comparisons.csv:")
    print(df_boot_perm[["order_mode", "comparison", "delta_succ10", "ci_succ10", "delta_auc10", "delta_kp", "ci_kp"]].to_string(index=False))
    
    # B.3 Match Order Structure
    print("\n--- SECTION B.3: Investigating Match Order Structure ---")
    struct_records = []
    
    img_cache = {}
    def get_image_K(fname, k):
        key = (fname, k)
        if key not in img_cache:
            _, K = md.load_image(fname, k)
            img_cache[key] = K
        return img_cache[key]
        
    scene_cache = {}
    def get_scene_data(fname, i, j):
        if fname not in scene_cache:
            scene_cache[fname] = md._scene(fname)
        sc = scene_cache[fname]
        K0 = get_image_K(fname, i)
        K1 = get_image_K(fname, j)
        return K0, K1
        
    for fname, i, j in pairs[:200]:
        K0, K1 = get_scene_data(fname, i, j)
        for preset, N in [("ALIKED", 500), ("SIFT", 4000)]:
            clean_preset = preset.replace("+", "_")
            match_f = MATCH_DIR / f"{fname}_{i}_{j}_{clean_preset}_N{N}.npz"
            if match_f.exists():
                d = np.load(match_f)
                pts0, pts1 = d["pts0"], d["pts1"]
                n = len(pts0)
                if n >= 20:
                    p = compute_epipolar_proxy(pts0, pts1, K0, K1)
                    if p["mask"] is not None:
                        mask = p["mask"].ravel()
                        k20 = max(1, int(0.20 * n))
                        inl_first20 = mask[:k20].mean()
                        inl_last20 = mask[-k20:].mean()
                        struct_records.append({
                            "fname": fname, "preset": preset, "N": N, "n_matches": n,
                            "inl_first20": inl_first20, "inl_last20": inl_last20,
                            "inlier_ratio_gap": inl_first20 - inl_last20
                        })
                        
    df_struct = pd.DataFrame(struct_records)
    struct_summary = []
    for preset in ["ALIKED", "SIFT"]:
        sub = df_struct[df_struct["preset"] == preset]
        struct_summary.append({
            "preset": preset,
            "mean_inlier_ratio_first_20pct": sub["inl_first20"].mean(),
            "mean_inlier_ratio_last_20pct": sub["inl_last20"].mean(),
            "mean_gap_front_minus_back": sub["inlier_ratio_gap"].mean(),
            "pct_pairs_front_higher": (sub["inlier_ratio_gap"] > 0).mean() * 100.0
        })
    df_struct_sum = pd.DataFrame(struct_summary)
    df_struct_sum.to_csv(OUTPUT_DIR / "b3_match_order_structure.csv", index=False)
    print("\nSaved results/phase5b/b3_match_order_structure.csv:")
    print(df_struct_sum.to_string(index=False))

def main():
    if not PERM_RECORDS_CSV.exists():
        pairs = md.list_pairs()
        n_pairs = len(pairs)
        n_workers = 5
        chunk_size = int(np.ceil(n_pairs / n_workers))
        
        processes = []
        print(f"Launching {n_workers} parallel workers to evaluate 1500 pairs across 5 permutations...", flush=True)
        
        for w in range(n_workers):
            start = w * chunk_size
            end = min(n_pairs, (w + 1) * chunk_size)
            p_sub = pairs[start:end]
            p = Process(target=process_shard_fast, args=(w, p_sub, start))
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        print("\nAll workers completed. Merging shards...", flush=True)
        all_shards = []
        for w in range(n_workers):
            shard_csv = OUTPUT_DIR / f"b2_perm_shard_{w}.csv"
            assert shard_csv.exists(), f"Missing shard {shard_csv}"
            all_shards.append(pd.read_csv(shard_csv))
            
        df_records = pd.concat(all_shards, ignore_index=True)
        df_records.to_csv(PERM_RECORDS_CSV, index=False)
        print(f"Merged {len(df_records)} records into {PERM_RECORDS_CSV}!", flush=True)
        
    run_simulation_and_analysis()

if __name__ == "__main__":
    main()
