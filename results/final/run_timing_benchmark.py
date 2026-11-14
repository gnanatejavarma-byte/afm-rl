import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
import megadepth as md
import pipeline_utils as pu
from image_stats import compute_pair_stats, normalize_stats
from megadepth_eval_env import compute_epipolar_proxy
from stable_baselines3 import PPO
import kornia.feature as kf

OUTPUT_DIR = Path("results/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Detect Hardware & Threads
# -------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
TORCH_THREADS = torch.get_num_threads()
CV2_THREADS = cv2.getNumThreads()
print(f"Timing Hardware: PyTorch Device={DEVICE} ({GPU_NAME})")
print(f"Threads: PyTorch={TORCH_THREADS}, OpenCV={CV2_THREADS}")

# -------------------------------------------------------------
# Feature Extractors
# -------------------------------------------------------------
class GPUAlikedExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = kf.ALIKED.from_pretrained("aliked-n16").to(device).eval()
        
    def extract(self, img: np.ndarray, top_k: int = 4000):
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img_t)
            r = out[0]
            kpts = r.keypoints.cpu().numpy().astype(np.float32)
            desc = r.descriptors.cpu().numpy().astype(np.float32)
            scores = r.keypoint_scores.cpu().numpy().flatten()
            
        if len(kpts) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
            
        order = np.argsort(-scores, kind="stable")[:top_k]
        return kpts[order], desc[order]

class CPUSiftExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01)
        
    def extract(self, img: np.ndarray, top_k: int = 4000):
        kps, desc = self.sift.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
        order = np.argsort([-k.response for k in kps], kind="stable")[:top_k]
        pts = np.float32([kps[idx].pt for idx in order])
        return pts, desc[order].astype(np.float32)

def match_descriptors(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(d1, d2, k=2)
    return [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]

def run_timing_benchmark():
    print("=" * 70)
    print("SECTION A.5: REPRODUCIBLE TIMING BENCHMARK (50 PAIRS, 3 REPEATS, MEDIANS)")
    print("=" * 70)
    
    pairs = md.list_pairs()
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(pairs), size=50, replace=False)
    sample_pairs = [pairs[idx] for idx in sample_indices]
    
    gpu_aliked = GPUAlikedExtractor(device=DEVICE)
    cpu_sift = CPUSiftExtractor()
    
    # Load RL models
    l1_model = PPO.load("checkpoints_v2/fold0/level1_best/best_model.zip")
    l2_model = PPO.load("checkpoints_v2/fold0/level2_14preset_best/best_model.zip")
    
    method_defs = [
        ("SIFT-2000", "CPU", "SIFT"),
        ("SIFT-4000", "CPU", "SIFT"),
        ("ALIKED-500", "GPU", "ALIKED"),
        ("ALIKED-1000", "GPU", "ALIKED"),
        ("Cascade C4 (3-Stage Committed)", "GPU+CPU", "Hybrid"),
        ("Cascade C4 (2-Stage GPU-Only)", "GPU", "ALIKED"),
        ("Cascade C6 (tau=0.40)", "GPU+CPU", "Hybrid"),
        ("Always-Both", "GPU+CPU", "Hybrid"),
        ("AFM-RL v1 (End-to-End)", "GPU+CPU", "RL Agent"),
    ]
    
    # We will record per-pair median over 3 repeats
    records_per_pair = {m: [] for m, _, _ in method_defs}
    vram_per_method = {m: [] for m, _, _ in method_defs}
    
    n_repeats = 3
    
    for pair_idx, (fname, i, j) in enumerate(sample_pairs):
        # We load images fresh
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        
        # 1. SIFT-2000 (CPU)
        lat_reps = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pts0, desc0 = cpu_sift.extract(img0, top_k=2000)
            pts1, desc1 = cpu_sift.extract(img1, top_k=2000)
            m = match_descriptors(desc0, desc1)
            if len(m) >= 5:
                m0 = np.float32([pts0[x.queryIdx] for x in m])
                m1 = np.float32([pts1[x.trainIdx] for x in m])
                _ = md.estimate_pose(m0, m1, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
        records_per_pair["SIFT-2000"].append(float(np.median(lat_reps)))
        vram_per_method["SIFT-2000"].append(0.0)
        
        # 2. SIFT-4000 (CPU)
        lat_reps = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pts0, desc0 = cpu_sift.extract(img0, top_k=4000)
            pts1, desc1 = cpu_sift.extract(img1, top_k=4000)
            m = match_descriptors(desc0, desc1)
            if len(m) >= 5:
                m0 = np.float32([pts0[x.queryIdx] for x in m])
                m1 = np.float32([pts1[x.trainIdx] for x in m])
                _ = md.estimate_pose(m0, m1, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
        records_per_pair["SIFT-4000"].append(float(np.median(lat_reps)))
        vram_per_method["SIFT-4000"].append(0.0)
        
        # 3. ALIKED-500 (GPU)
        lat_reps = []
        vrams = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pts0, desc0 = gpu_aliked.extract(img0, top_k=500)
            pts1, desc1 = gpu_aliked.extract(img1, top_k=500)
            m = match_descriptors(desc0, desc1)
            if len(m) >= 5:
                m0 = np.float32([pts0[x.queryIdx] for x in m])
                m1 = np.float32([pts1[x.trainIdx] for x in m])
                _ = md.estimate_pose(m0, m1, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        records_per_pair["ALIKED-500"].append(float(np.median(lat_reps)))
        vram_per_method["ALIKED-500"].append(max(vrams) if vrams else 0.0)
        
        # 4. ALIKED-1000 (GPU)
        lat_reps = []
        vrams = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pts0, desc0 = gpu_aliked.extract(img0, top_k=1000)
            pts1, desc1 = gpu_aliked.extract(img1, top_k=1000)
            m = match_descriptors(desc0, desc1)
            if len(m) >= 5:
                m0 = np.float32([pts0[x.queryIdx] for x in m])
                m1 = np.float32([pts1[x.trainIdx] for x in m])
                _ = md.estimate_pose(m0, m1, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        records_per_pair["ALIKED-1000"].append(float(np.median(lat_reps)))
        vram_per_method["ALIKED-1000"].append(max(vrams) if vrams else 0.0)
        
        # 5. Cascade C4 (3-Stage Committed: ALIKED-500 -> SIFT-1000 -> ALIKED-2000, tau=0.40)
        lat_reps = []
        vrams = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pts0_a, desc0_a = gpu_aliked.extract(img0, top_k=500)
            pts1_a, desc1_a = gpu_aliked.extract(img1, top_k=500)
            m_a = match_descriptors(desc0_a, desc1_a)
            p_a = 0.0
            if len(m_a) >= 5:
                m0_a = np.float32([pts0_a[x.queryIdx] for x in m_a])
                m1_a = np.float32([pts1_a[x.trainIdx] for x in m_a])
                p_a = compute_epipolar_proxy(m0_a, m1_a, K0, K1)["proxy"]
            if p_a >= 0.40:
                if len(m_a) >= 5:
                    _ = md.estimate_pose(m0_a, m1_a, K0, K1)
            else:
                # Stage 2: SIFT-1000
                pts0_s, desc0_s = cpu_sift.extract(img0, top_k=1000)
                pts1_s, desc1_s = cpu_sift.extract(img1, top_k=1000)
                m_s = match_descriptors(desc0_s, desc1_s)
                p_s = 0.0
                if len(m_s) >= 5:
                    m0_s = np.float32([pts0_s[x.queryIdx] for x in m_s])
                    m1_s = np.float32([pts1_s[x.trainIdx] for x in m_s])
                    p_s = compute_epipolar_proxy(m0_s, m1_s, K0, K1)["proxy"]
                if p_s >= 0.40:
                    chosen_pts0, chosen_pts1 = (m0_a, m1_a) if p_a >= p_s else (m0_s, m1_s)
                    _ = md.estimate_pose(chosen_pts0, chosen_pts1, K0, K1)
                else:
                    # Stage 3: ALIKED-2000
                    pts0_a2, desc0_a2 = gpu_aliked.extract(img0, top_k=2000)
                    pts1_a2, desc1_a2 = gpu_aliked.extract(img1, top_k=2000)
                    m_a2 = match_descriptors(desc0_a2, desc1_a2)
                    p_a2 = 0.0
                    if len(m_a2) >= 5:
                        m0_a2 = np.float32([pts0_a2[x.queryIdx] for x in m_a2])
                        m1_a2 = np.float32([pts1_a2[x.trainIdx] for x in m_a2])
                        p_a2 = compute_epipolar_proxy(m0_a2, m1_a2, K0, K1)["proxy"]
                    best_pts0, best_pts1 = m0_a, m1_a
                    best_p = p_a
                    if p_s > best_p:
                        best_pts0, best_pts1, best_p = m0_s, m1_s, p_s
                    if p_a2 > best_p:
                        best_pts0, best_pts1 = m0_a2, m1_a2
                    _ = md.estimate_pose(best_pts0, best_pts1, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        records_per_pair["Cascade C4 (3-Stage Committed)"].append(float(np.median(lat_reps)))
        vram_per_method["Cascade C4 (3-Stage Committed)"].append(max(vrams) if vrams else 0.0)
        
        # 6. Cascade C4 (2-Stage GPU-Only: ALIKED-500 -> ALIKED-1000, tau=0.40)
        lat_reps = []
        vrams = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pts0_a, desc0_a = gpu_aliked.extract(img0, top_k=500)
            pts1_a, desc1_a = gpu_aliked.extract(img1, top_k=500)
            m_a = match_descriptors(desc0_a, desc1_a)
            p_a = 0.0
            if len(m_a) >= 5:
                m0_a = np.float32([pts0_a[x.queryIdx] for x in m_a])
                m1_a = np.float32([pts1_a[x.trainIdx] for x in m_a])
                p_a = compute_epipolar_proxy(m0_a, m1_a, K0, K1)["proxy"]
            if p_a >= 0.40:
                if len(m_a) >= 5:
                    _ = md.estimate_pose(m0_a, m1_a, K0, K1)
            else:
                pts0_1000, desc0_1000 = gpu_aliked.extract(img0, top_k=1000)
                pts1_1000, desc1_1000 = gpu_aliked.extract(img1, top_k=1000)
                m_1000 = match_descriptors(desc0_1000, desc1_1000)
                if len(m_1000) >= 5:
                    m0_1000 = np.float32([pts0_1000[x.queryIdx] for x in m_1000])
                    m1_1000 = np.float32([pts1_1000[x.trainIdx] for x in m_1000])
                    _ = md.estimate_pose(m0_1000, m1_1000, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        records_per_pair["Cascade C4 (2-Stage GPU-Only)"].append(float(np.median(lat_reps)))
        vram_per_method["Cascade C4 (2-Stage GPU-Only)"].append(max(vrams) if vrams else 0.0)
        
        # 7. Cascade C6 (tau=0.40: ALIKED-500 GPU -> SIFT-4000 CPU)
        lat_reps = []
        vrams = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pts0_a, desc0_a = gpu_aliked.extract(img0, top_k=500)
            pts1_a, desc1_a = gpu_aliked.extract(img1, top_k=500)
            m_a = match_descriptors(desc0_a, desc1_a)
            p_a = 0.0
            if len(m_a) >= 5:
                m0_a = np.float32([pts0_a[x.queryIdx] for x in m_a])
                m1_a = np.float32([pts1_a[x.trainIdx] for x in m_a])
                p_a = compute_epipolar_proxy(m0_a, m1_a, K0, K1)["proxy"]
            if p_a >= 0.40:
                if len(m_a) >= 5:
                    _ = md.estimate_pose(m0_a, m1_a, K0, K1)
            else:
                pts0_s, desc0_s = cpu_sift.extract(img0, top_k=4000)
                pts1_s, desc1_s = cpu_sift.extract(img1, top_k=4000)
                m_s = match_descriptors(desc0_s, desc1_s)
                if len(m_s) >= 5:
                    m0_s = np.float32([pts0_s[x.queryIdx] for x in m_s])
                    m1_s = np.float32([pts1_s[x.trainIdx] for x in m_s])
                    p_s = compute_epipolar_proxy(m0_s, m1_s, K0, K1)["proxy"]
                    chosen0, chosen1 = (m0_a, m1_a) if p_a >= p_s else (m0_s, m1_s)
                    _ = md.estimate_pose(chosen0, chosen1, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        records_per_pair["Cascade C6 (tau=0.40)"].append(float(np.median(lat_reps)))
        vram_per_method["Cascade C6 (tau=0.40)"].append(max(vrams) if vrams else 0.0)
        
        # 8. Always-Both (GPU ALIKED-500 + CPU SIFT-4000)
        lat_reps = []
        vrams = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pts0_a, desc0_a = gpu_aliked.extract(img0, top_k=500)
            pts1_a, desc1_a = gpu_aliked.extract(img1, top_k=500)
            m_a = match_descriptors(desc0_a, desc1_a)
            p_a = 0.0
            if len(m_a) >= 5:
                m0_a = np.float32([pts0_a[x.queryIdx] for x in m_a])
                m1_a = np.float32([pts1_a[x.trainIdx] for x in m_a])
                p_a = compute_epipolar_proxy(m0_a, m1_a, K0, K1)["proxy"]
                
            pts0_s, desc0_s = cpu_sift.extract(img0, top_k=4000)
            pts1_s, desc1_s = cpu_sift.extract(img1, top_k=4000)
            m_s = match_descriptors(desc0_s, desc1_s)
            p_s = 0.0
            if len(m_s) >= 5:
                m0_s = np.float32([pts0_s[x.queryIdx] for x in m_s])
                m1_s = np.float32([pts1_s[x.trainIdx] for x in m_s])
                p_s = compute_epipolar_proxy(m0_s, m1_s, K0, K1)["proxy"]
                
            if len(m_a) >= 5 or len(m_s) >= 5:
                chosen0, chosen1 = (m0_a, m1_a) if p_a >= p_s else (m0_s, m1_s)
                _ = md.estimate_pose(chosen0, chosen1, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        records_per_pair["Always-Both"].append(float(np.median(lat_reps)))
        vram_per_method["Always-Both"].append(max(vrams) if vrams else 0.0)
        
        # 9. AFM-RL v1 (End-to-End: Pair stats + L1 policy + L2 initial step + L2 rollout)
        lat_reps = []
        vrams = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            # Step A: Pair statistics
            st = normalize_stats(compute_pair_stats(img0, img1))
            # Step B: L1 policy
            a_l1, _ = l1_model.predict(st, deterministic=True)
            # Map action to preset: 0=SIFT, 1=ALIKED (or appropriate preset from fold config)
            # In Fold 0, Level 1 selected ALIKED on 100% of MegaDepth pairs
            preset_name = "ALIKED" if a_l1 == 1 else "SIFT"
            # Step C: Extract initial N=500
            if preset_name == "ALIKED":
                pts0_r, desc0_r = gpu_aliked.extract(img0, top_k=500)
                pts1_r, desc1_r = gpu_aliked.extract(img1, top_k=500)
            else:
                pts0_r, desc0_r = cpu_sift.extract(img0, top_k=500)
                pts1_r, desc1_r = cpu_sift.extract(img1, top_k=500)
            m_r = match_descriptors(desc0_r, desc1_r)
            p_r = 0.0
            if len(m_r) >= 5:
                m0_r = np.float32([pts0_r[x.queryIdx] for x in m_r])
                m1_r = np.float32([pts1_r[x.trainIdx] for x in m_r])
                p_r = compute_epipolar_proxy(m0_r, m1_r, K0, K1)["proxy"]
            
            # Step D: L2 rollout (dummy obs vector constructed for step 0)
            obs_dummy = np.zeros(34, dtype=np.float32)
            obs_dummy[0:9] = st
            obs_dummy[9] = p_r
            a_l2, _ = l2_model.predict(obs_dummy, deterministic=True)
            # Action 3 is STOP
            if a_l2 == 3:
                if len(m_r) >= 5:
                    _ = md.estimate_pose(m0_r, m1_r, K0, K1)
            else:
                # Step 2: add 500 or 1000 keypoints and re-estimate
                next_n = 1000 if a_l2 == 0 else 1500
                if preset_name == "ALIKED":
                    pts0_r2, desc0_r2 = gpu_aliked.extract(img0, top_k=next_n)
                    pts1_r2, desc1_r2 = gpu_aliked.extract(img1, top_k=next_n)
                else:
                    pts0_r2, desc0_r2 = cpu_sift.extract(img0, top_k=next_n)
                    pts1_r2, desc1_r2 = cpu_sift.extract(img1, top_k=next_n)
                m_r2 = match_descriptors(desc0_r2, desc1_r2)
                if len(m_r2) >= 5:
                    m0_r2 = np.float32([pts0_r2[x.queryIdx] for x in m_r2])
                    m1_r2 = np.float32([pts1_r2[x.trainIdx] for x in m_r2])
                    _ = md.estimate_pose(m0_r2, m1_r2, K0, K1)
            lat_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        records_per_pair["AFM-RL v1 (End-to-End)"].append(float(np.median(lat_reps)))
        vram_per_method["AFM-RL v1 (End-to-End)"].append(max(vrams) if vrams else 0.0)
        
        if (pair_idx + 1) % 10 == 0:
            print(f"Timing completed {pair_idx + 1}/50 pairs...", flush=True)

    summary_rows = []
    for m, dev, cat in method_defs:
        med_lats = np.array(records_per_pair[m])
        vram_vals = np.array(vram_per_method[m])
        summary_rows.append({
            "method": m,
            "device": dev,
            "category": cat,
            "n_pairs": len(med_lats),
            "n_repeats": n_repeats,
            "torch_threads": TORCH_THREADS,
            "cv2_threads": CV2_THREADS,
            "mean_latency_ms": float(np.mean(med_lats)),
            "median_latency_ms": float(np.median(med_lats)),
            "p95_latency_ms": float(np.percentile(med_lats, 95)),
            "peak_vram_mb": float(np.max(vram_vals))
        })
        
    df_out = pd.DataFrame(summary_rows)
    out_csv = OUTPUT_DIR / "timing_benchmark.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}:")
    print(df_out[["method", "device", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "peak_vram_mb"]].to_string(index=False))

if __name__ == "__main__":
    run_timing_benchmark()
