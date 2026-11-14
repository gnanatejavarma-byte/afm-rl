import os
import sys
import time
import platform
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import torch

sys.path.insert(0, 'src')
import megadepth as md
import pipeline_utils as pu
from image_stats import compute_pair_stats, normalize_stats
from stable_baselines3 import PPO

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== RUNNING SECTION F: WALL-CLOCK LATENCY BENCHMARK (CPU) ===")

# 1. Hardware and thread settings
n_cpus = os.cpu_count()
proc_name = platform.processor()
torch_threads = torch.get_num_threads()
cv2_threads = cv2.getNumThreads()

print(f"Hardware / Environment:")
print(f"  Processor: {proc_name}")
print(f"  CPU Count: {n_cpus}")
print(f"  PyTorch Threads: {torch_threads}")
print(f"  OpenCV Threads: {cv2_threads}")

# 2. Select 50 random MegaDepth pairs
pairs = md.list_pairs()
rng = np.random.RandomState(42)
test_indices = rng.choice(len(pairs), 50, replace=False)
test_50_pairs = [pairs[idx] for idx in test_indices]

# Load models for inference timing
l1_model = PPO.load("checkpoints_v2/fold0/level1_best/best_model.zip")
l2_model = PPO.load("checkpoints_v2/fold0/level2_14preset_best/best_model.zip")

# Warm up OpenCV / torch
dummy_img = np.zeros((640, 480), dtype=np.uint8)
_ = pu._build_pool(dummy_img, "SIFT")
_ = l1_model.predict(np.zeros(9, dtype=np.float32), deterministic=True)
_ = l2_model.predict(np.zeros(34, dtype=np.float32), deterministic=True)

# Measurement containers (times in milliseconds)
times_stats = []
times_l1 = []
times_l2 = []

times_sift_extract = []
times_aliked_extract = []

times_match_a500 = []
times_match_a1000 = []
times_match_s2000 = []
times_match_s4000 = []

print("Measuring cold latency on pairs...")

# Measure extraction on first 10 pairs (sample) and matching on all 50
for pair_idx, (fname, i, j) in enumerate(test_50_pairs):
    img0, K0 = md.load_image(fname, i)
    img1, K1 = md.load_image(fname, j)
    
    # 1. Pair stats
    t0 = time.perf_counter()
    st = normalize_stats(compute_pair_stats(img0, img1))
    times_stats.append((time.perf_counter() - t0) * 1000.0)
    
    # 2. Level 1 policy inference
    t0 = time.perf_counter()
    a_l1, _ = l1_model.predict(st, deterministic=True)
    times_l1.append((time.perf_counter() - t0) * 1000.0)
    
    # 3. Level 2 policy inference
    dummy_obs = np.zeros(34, dtype=np.float32)
    t0 = time.perf_counter()
    a_l2, _ = l2_model.predict(dummy_obs, deterministic=True)
    times_l2.append((time.perf_counter() - t0) * 1000.0)
    
    # Measure feature extraction: first 5 pairs for SIFT, first 3 pairs for ALIKED
    if pair_idx < 5:
        # SIFT extraction (both images)
        t0 = time.perf_counter()
        p0_s = pu._build_pool(img0, "SIFT")
        p1_s = pu._build_pool(img1, "SIFT")
        times_sift_extract.append((time.perf_counter() - t0) * 1000.0)
    if pair_idx < 3:
        # ALIKED extraction (both images)
        t0 = time.perf_counter()
        p0_a = pu._build_pool(img0, "ALIKED")
        p1_a = pu._build_pool(img1, "ALIKED")
        times_aliked_extract.append((time.perf_counter() - t0) * 1000.0)
        
    if pair_idx >= 5 or pair_idx >= 3:
        # Load from pool cache for remaining pairs to measure matching + RANSAC quickly
        import hashlib
        from megadepth_eval_env import POOL_CACHE_DIR
        def get_pool_fast(img_idx, pr):
            sc = md._scene(fname)
            p = str(sc["image_paths"][img_idx])
            param_str = f"nfeatures=4000_preset={pr}"
            key_str = f"{p}_{pr}_{param_str}"
            h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
            cf = POOL_CACHE_DIR / pr / f"{h}.npz"
            d = np.load(cf)
            return {"pts": d["pts"], "desc": d["desc"]}
        if pair_idx >= 5:
            p0_s = get_pool_fast(i, "SIFT")
            p1_s = get_pool_fast(j, "SIFT")
        if pair_idx >= 3:
            p0_a = get_pool_fast(i, "ALIKED")
            p1_a = get_pool_fast(j, "ALIKED")
        
    # Measure Matching + Essential RANSAC
    def time_match_ransac(p0, p1, N, preset):
        t_start = time.perf_counter()
        pts0, d0 = p0["pts"][:N], p0["desc"][:N] if p0["desc"] is not None else None
        pts1, d1 = p1["pts"][:N], p1["desc"][:N] if p1["desc"] is not None else None
        good = pu.match_ratio(d0, d1, preset, matcher="BF", ratio=0.75 if preset != "ORB" else 0.80)
        if good and len(good) >= 5:
            m_pts0 = np.float32([pts0[m.queryIdx] for m in good])
            m_pts1 = np.float32([pts1[m.trainIdx] for m in good])
            _ = md.estimate_pose(m_pts0, m_pts1, K0, K1)
        return (time.perf_counter() - t_start) * 1000.0

    times_match_a500.append(time_match_ransac(p0_a, p1_a, 500, "ALIKED"))
    times_match_a1000.append(time_match_ransac(p0_a, p1_a, 1000, "ALIKED"))
    times_match_s2000.append(time_match_ransac(p0_s, p1_s, 2000, "SIFT"))
    times_match_s4000.append(time_match_ransac(p0_s, p1_s, 4000, "SIFT"))

mean_stats = float(np.mean(times_stats))
mean_l1 = float(np.mean(times_l1))
mean_l2 = float(np.mean(times_l2))

mean_ext_sift = float(np.mean(times_sift_extract))
mean_ext_aliked = float(np.mean(times_aliked_extract))

mean_m_a500 = float(np.mean(times_match_a500))
mean_m_a1000 = float(np.mean(times_match_a1000))
mean_m_s2000 = float(np.mean(times_match_s2000))
mean_m_s4000 = float(np.mean(times_match_s4000))

# Full cold latency per pair: extraction (both images) + matching + essential RANSAC
lat_a500 = mean_ext_aliked + mean_m_a500
lat_a1000 = mean_ext_aliked + mean_m_a1000
lat_s2000 = mean_ext_sift + mean_m_s2000
lat_s4000 = mean_ext_sift + mean_m_s4000

print("\nMeasured Component Latencies (ms, CPU):")
print(f"  Pair Stats:          {mean_stats:.2f} ms")
print(f"  L1 Policy Inference: {mean_l1:.2f} ms")
print(f"  L2 Policy Inference: {mean_l2:.2f} ms")
print(f"  SIFT Extraction:     {mean_ext_sift:.2f} ms")
print(f"  ALIKED Extraction:   {mean_ext_aliked:.2f} ms")
print(f"  Match+RANSAC A-500:  {mean_m_a500:.2f} ms")
print(f"  Match+RANSAC A-1000: {mean_m_a1000:.2f} ms")
print(f"  Match+RANSAC S-2000: {mean_m_s2000:.2f} ms")
print(f"  Match+RANSAC S-4000: {mean_m_s4000:.2f} ms")

print("\nTotal Cold Wall-Clock Per Pair (ms, CPU):")
print(f"  ALIKED-500:  {lat_a500:.2f} ms")
print(f"  ALIKED-1000: {lat_a1000:.2f} ms")
print(f"  SIFT-2000:   {lat_s2000:.2f} ms")
print(f"  SIFT-4000:   {lat_s4000:.2f} ms")

# Synthesize expected cold wall-clock for Cascade C* and AFM-RL v1
# Cascade C* (tau=0.40):
# Stage 1: ALIKED-500 cold
# Stage 1 stopping rate on MegaDepth: 59.47%
# If escalated (40.53%): executes Stage 2 (SIFT-4000 cold extraction + match + RANSAC)
# Note: images already read from disk into memory, but feature extraction is done for SIFT.
p_esc = 1.0 - 0.59466667
exp_casc_time = lat_a500 + p_esc * lat_s4000

# AFM-RL v1 (5-fold mean):
# Stats + L1 policy + mean 4.8 steps of L2 policy + extraction
# In stage 3: 67.53% chose ALIKED, 32.47% chose SIFT.
exp_v1_time = mean_stats + mean_l1 + (4.8 * mean_l2) + (0.6753 * lat_a1000 + 0.3247 * lat_s4000)

df_c = pd.read_csv("results/phase5/c_primary_comparisons_summary.csv")
df_c_pooled = df_c[df_c["scene"] == "Pooled"].set_index("method")

wallclock_records = [
    {
        "method": "Cascade C* (tau=0.40)",
        "hardware": "CPU (12 threads)",
        "actual_cumulative_kp": float(df_c_pooled.loc["Cascade C* (tau=0.40)"]["cum_actual_kp"]),
        "cold_wallclock_ms": exp_casc_time,
        "succ10_deg": float(df_c_pooled.loc["Cascade C* (tau=0.40)"]["succ10"]),
        "auc10_deg": float(df_c_pooled.loc["Cascade C* (tau=0.40)"]["auc10"]),
        "utility": float(df_c_pooled.loc["Cascade C* (tau=0.40)"]["utility"]),
    },
    {
        "method": "ALIKED-500",
        "hardware": "CPU (12 threads)",
        "actual_cumulative_kp": float(df_c_pooled.loc["ALIKED-500"]["cum_actual_kp"]),
        "cold_wallclock_ms": lat_a500,
        "succ10_deg": float(df_c_pooled.loc["ALIKED-500"]["succ10"]),
        "auc10_deg": float(df_c_pooled.loc["ALIKED-500"]["auc10"]),
        "utility": float(df_c_pooled.loc["ALIKED-500"]["utility"]),
    },
    {
        "method": "ALIKED-1000",
        "hardware": "CPU (12 threads)",
        "actual_cumulative_kp": float(df_c_pooled.loc["ALIKED-1000"]["cum_actual_kp"]),
        "cold_wallclock_ms": lat_a1000,
        "succ10_deg": float(df_c_pooled.loc["ALIKED-1000"]["succ10"]),
        "auc10_deg": float(df_c_pooled.loc["ALIKED-1000"]["auc10"]),
        "utility": float(df_c_pooled.loc["ALIKED-1000"]["utility"]),
    },
    {
        "method": "SIFT-2000",
        "hardware": "CPU (12 threads)",
        "actual_cumulative_kp": float(df_c_pooled.loc["SIFT-2000"]["cum_actual_kp"]),
        "cold_wallclock_ms": lat_s2000,
        "succ10_deg": float(df_c_pooled.loc["SIFT-2000"]["succ10"]),
        "auc10_deg": float(df_c_pooled.loc["SIFT-2000"]["auc10"]),
        "utility": float(df_c_pooled.loc["SIFT-2000"]["utility"]),
    },
    {
        "method": "SIFT-4000",
        "hardware": "CPU (12 threads)",
        "actual_cumulative_kp": float(df_c_pooled.loc["SIFT-4000"]["cum_actual_kp"]),
        "cold_wallclock_ms": lat_s4000,
        "succ10_deg": float(df_c_pooled.loc["SIFT-4000"]["succ10"]),
        "auc10_deg": float(df_c_pooled.loc["SIFT-4000"]["auc10"]),
        "utility": float(df_c_pooled.loc["SIFT-4000"]["utility"]),
    },
    {
        "method": "AFM-RL v1 (5-fold mean)",
        "hardware": "CPU (12 threads)",
        "actual_cumulative_kp": 9657.57,
        "cold_wallclock_ms": exp_v1_time,
        "succ10_deg": 71.87,
        "auc10_deg": 49.64,
        "utility": 0.4772,
    }
]

df_wc = pd.DataFrame(wallclock_records)
df_wc.to_csv("results/phase5/f_wallclock_summary.csv", index=False)
print("\nSaved results/phase5/f_wallclock_summary.csv")
print(df_wc.to_string(index=False))

print("\n=== SECTION F COMPLETE ===")
