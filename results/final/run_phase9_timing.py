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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
TORCH_THREADS = torch.get_num_threads()
CV2_THREADS = cv2.getNumThreads()

print(f"Timing Hardware: PyTorch Device={DEVICE} ({GPU_NAME})")
print(f"Threads: PyTorch={TORCH_THREADS}, OpenCV={CV2_THREADS}")

# Feature Extractors
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
        
    def extract_full(self, img: np.ndarray):
        kps, desc = self.sift.detectAndCompute(img, None)
        if desc is None or len(kps) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
        order = np.argsort([-k.response for k in kps], kind="stable")
        pts = np.float32([kps[idx].pt for idx in order])
        return pts, desc[order].astype(np.float32)

def match_descriptors(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(d1, d2, k=2)
    return [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]

def run_phase9_timing():
    print("=" * 70)
    print("PHASE 9: REPRODUCIBLE TIMING BENCHMARK (50 PAIRS, 3 REPEATS, MEDIANS)")
    print("=" * 70)
    
    pairs = md.list_pairs()
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(pairs), size=50, replace=False)
    sample_pairs = [pairs[idx] for idx in sample_indices]
    
    gpu_aliked = GPUAlikedExtractor(device=DEVICE)
    cpu_sift = CPUSiftExtractor()
    
    # Warmup GPU
    dummy = np.zeros((480, 640), dtype=np.uint8)
    _ = gpu_aliked.extract(dummy, top_k=500)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    l1_model = PPO.load("checkpoints_v2/fold0/level1_best/best_model.zip")
    l2_model = PPO.load("checkpoints_v2/fold0/level2_14preset_best/best_model.zip")
    
    methods = [
        "SIFT-500",
        "SIFT-1000",
        "SIFT-2000",
        "SIFT-4000",
        "ALIKED-500",
        "ALIKED-1000",
        "Cascade C4 (tau=0.40)",
        "Cascade C4 (tau=0.45)",
        "Cascade C2 (tau=0.40)",
        "Cascade C6 (tau=0.40)",
        "Always-Both",
        "AFM-RL v1 (End-to-End)",
    ]
    
    components = [
        "afm_pair_stats",
        "afm_level1_eval",
        "afm_pool_extraction_4000",
        "afm_level2_policy_step",
        "afm_matching_bf",
        "afm_ransac_pose",
    ]
    
    records_per_pair = {m: [] for m in methods}
    vram_per_method = {m: [] for m in methods}
    comp_records = {c: [] for c in components}
    
    n_repeats = 3
    
    for pair_idx, (fname, i, j) in enumerate(sample_pairs):
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        
        # -------------------------------------------------------------
        # Measure SIFT feature extraction once per image (repeated 3 times for median)
        # -------------------------------------------------------------
        sift_ext_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pts0_sift_all, desc0_sift_all = cpu_sift.extract_full(img0)
            pts1_sift_all, desc1_sift_all = cpu_sift.extract_full(img1)
            sift_ext_times.append((time.perf_counter() - t0) * 1000.0)
        t_sift_ext = float(np.median(sift_ext_times))
        
        # SIFT baselines (500, 1000, 2000, 4000)
        for n_kps, m_name in [(500, "SIFT-500"), (1000, "SIFT-1000"), (2000, "SIFT-2000"), (4000, "SIFT-4000")]:
            lat_reps = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                d0 = desc0_sift_all[:n_kps]
                d1 = desc1_sift_all[:n_kps]
                p0 = pts0_sift_all[:n_kps]
                p1 = pts1_sift_all[:n_kps]
                m = match_descriptors(d0, d1)
                if len(m) >= 5:
                    m0 = np.float32([p0[x.queryIdx] for x in m])
                    m1 = np.float32([p1[x.trainIdx] for x in m])
                    _ = md.estimate_pose(m0, m1, K0, K1)
                lat_reps.append(t_sift_ext + (time.perf_counter() - t0) * 1000.0)
            records_per_pair[m_name].append(float(np.median(lat_reps)))
            vram_per_method[m_name].append(0.0)
            
        # -------------------------------------------------------------
        # Measure ALIKED baselines on GPU
        # -------------------------------------------------------------
        for n_kps, m_name in [(500, "ALIKED-500"), (1000, "ALIKED-1000")]:
            lat_reps = []
            vrams = []
            for _ in range(n_repeats):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pts0, desc0 = gpu_aliked.extract(img0, top_k=n_kps)
                pts1, desc1 = gpu_aliked.extract(img1, top_k=n_kps)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                m = match_descriptors(desc0, desc1)
                if len(m) >= 5:
                    m0 = np.float32([pts0[x.queryIdx] for x in m])
                    m1 = np.float32([pts1[x.trainIdx] for x in m])
                    _ = md.estimate_pose(m0, m1, K0, K1)
                lat_reps.append((time.perf_counter() - t0) * 1000.0)
                if torch.cuda.is_available():
                    vrams.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
            records_per_pair[m_name].append(float(np.median(lat_reps)))
            vram_per_method[m_name].append(max(vrams) if vrams else 0.0)
            
        # -------------------------------------------------------------
        # Stage 1: ALIKED-500 for Cascades
        # -------------------------------------------------------------
        # Extract ALIKED-500 and compute proxy
        al500_reps = []
        vram_c_reps = []
        for _ in range(n_repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pts0_al500, desc0_al500 = gpu_aliked.extract(img0, top_k=500)
            pts1_al500, desc1_al500 = gpu_aliked.extract(img1, top_k=500)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            m_al500 = match_descriptors(desc0_al500, desc1_al500)
            p_al500 = 0.0
            if len(m_al500) >= 5:
                m0_al500 = np.float32([pts0_al500[x.queryIdx] for x in m_al500])
                m1_al500 = np.float32([pts1_al500[x.trainIdx] for x in m_al500])
                p_al500 = compute_epipolar_proxy(m0_al500, m1_al500, K0, K1)["proxy"]
            else:
                m0_al500, m1_al500 = None, None
            al500_reps.append((time.perf_counter() - t0) * 1000.0)
            if torch.cuda.is_available():
                vram_c_reps.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        t_stage1 = float(np.median(al500_reps))
        vram_stage1 = max(vram_c_reps) if vram_c_reps else 0.0
        
        # Helper: Time pose recovery
        t_pose_start = time.perf_counter()
        if m0_al500 is not None and len(m0_al500) >= 5:
            _ = md.estimate_pose(m0_al500, m1_al500, K0, K1)
        t_pose_only = (time.perf_counter() - t_pose_start) * 1000.0

        # Helper: Time SIFT-1000 match + proxy
        t_s1k_match_start = time.perf_counter()
        m_s1k = match_descriptors(desc0_sift_all[:1000], desc1_sift_all[:1000])
        p_s1k = 0.0
        if len(m_s1k) >= 5:
            m0_s1k = np.float32([pts0_sift_all[x.queryIdx] for x in m_s1k])
            m1_s1k = np.float32([pts1_sift_all[x.trainIdx] for x in m_s1k])
            p_s1k = compute_epipolar_proxy(m0_s1k, m1_s1k, K0, K1)["proxy"]
        else:
            m0_s1k, m1_s1k = None, None
        t_s1k_total = t_sift_ext + (time.perf_counter() - t_s1k_match_start) * 1000.0

        # Helper: Time SIFT-4000 match + proxy
        t_s4k_match_start = time.perf_counter()
        m_s4k = match_descriptors(desc0_sift_all[:4000], desc1_sift_all[:4000])
        p_s4k = 0.0
        if len(m_s4k) >= 5:
            m0_s4k = np.float32([pts0_sift_all[x.queryIdx] for x in m_s4k])
            m1_s4k = np.float32([pts1_sift_all[x.trainIdx] for x in m_s4k])
            p_s4k = compute_epipolar_proxy(m0_s4k, m1_s4k, K0, K1)["proxy"]
        else:
            m0_s4k, m1_s4k = None, None
        t_s4k_total = t_sift_ext + (time.perf_counter() - t_s4k_match_start) * 1000.0

        # Cascade C4 at tau=0.40 (ALIKED-500 -> SIFT-1000 -> ALIKED-2000)
        if p_al500 >= 0.40:
            lat_c4_40 = t_stage1 + t_pose_only
        else:
            if p_s1k >= 0.40:
                lat_c4_40 = t_stage1 + t_s1k_total + t_pose_only
            else:
                # Stage 3: ALIKED-2000
                t_al2k_start = time.perf_counter()
                pts0_al2k, desc0_al2k = gpu_aliked.extract(img0, top_k=2000)
                pts1_al2k, desc1_al2k = gpu_aliked.extract(img1, top_k=2000)
                m_al2k = match_descriptors(desc0_al2k, desc1_al2k)
                lat_c4_40 = t_stage1 + t_s1k_total + (time.perf_counter() - t_al2k_start) * 1000.0 + t_pose_only
        records_per_pair["Cascade C4 (tau=0.40)"].append(lat_c4_40)
        vram_per_method["Cascade C4 (tau=0.40)"].append(vram_stage1)

        # Cascade C4 at tau=0.45
        if p_al500 >= 0.45:
            lat_c4_45 = t_stage1 + t_pose_only
        else:
            if p_s1k >= 0.45:
                lat_c4_45 = t_stage1 + t_s1k_total + t_pose_only
            else:
                t_al2k_start = time.perf_counter()
                pts0_al2k, desc0_al2k = gpu_aliked.extract(img0, top_k=2000)
                pts1_al2k, desc1_al2k = gpu_aliked.extract(img1, top_k=2000)
                lat_c4_45 = t_stage1 + t_s1k_total + (time.perf_counter() - t_al2k_start) * 1000.0 + t_pose_only
        records_per_pair["Cascade C4 (tau=0.45)"].append(lat_c4_45)
        vram_per_method["Cascade C4 (tau=0.45)"].append(vram_stage1)

        # Cascade C2 at tau=0.40 (ALIKED-500 -> ALIKED-1000 -> SIFT-4000)
        if p_al500 >= 0.40:
            lat_c2_40 = t_stage1 + t_pose_only
        else:
            t_al1k_start = time.perf_counter()
            pts0_al1k, desc0_al1k = gpu_aliked.extract(img0, top_k=1000)
            pts1_al1k, desc1_al1k = gpu_aliked.extract(img1, top_k=1000)
            m_al1k = match_descriptors(desc0_al1k, desc1_al1k)
            p_al1k = 0.0
            if len(m_al1k) >= 5:
                m0_al1k = np.float32([pts0_al1k[x.queryIdx] for x in m_al1k])
                m1_al1k = np.float32([pts1_al1k[x.trainIdx] for x in m_al1k])
                p_al1k = compute_epipolar_proxy(m0_al1k, m1_al1k, K0, K1)["proxy"]
            t_al1k_total = (time.perf_counter() - t_al1k_start) * 1000.0
            if p_al1k >= 0.40:
                lat_c2_40 = t_stage1 + t_al1k_total + t_pose_only
            else:
                lat_c2_40 = t_stage1 + t_al1k_total + t_s4k_total + t_pose_only
        records_per_pair["Cascade C2 (tau=0.40)"].append(lat_c2_40)
        vram_per_method["Cascade C2 (tau=0.40)"].append(vram_stage1)

        # Cascade C6 at tau=0.40 (ALIKED-500 -> SIFT-4000)
        if p_al500 >= 0.40:
            lat_c6_40 = t_stage1 + t_pose_only
        else:
            lat_c6_40 = t_stage1 + t_s4k_total + t_pose_only
        records_per_pair["Cascade C6 (tau=0.40)"].append(lat_c6_40)
        vram_per_method["Cascade C6 (tau=0.40)"].append(vram_stage1)

        # Always-Both (ALIKED-500 + SIFT-4000)
        lat_always_both = t_stage1 + t_s4k_total + t_pose_only
        records_per_pair["Always-Both"].append(lat_always_both)
        vram_per_method["Always-Both"].append(vram_stage1)

        # -------------------------------------------------------------
        # AFM-RL v1 and Component Breakdown
        # -------------------------------------------------------------
        # Component 1: Pair stats (9 dims)
        t_comp = time.perf_counter()
        raw_st = compute_pair_stats(img0, img1)
        norm_st = normalize_stats(raw_st)
        t_stats = (time.perf_counter() - t_comp) * 1000.0

        # Component 2: Level 1 policy evaluation
        t_comp = time.perf_counter()
        a_l1, _ = l1_model.predict(norm_st, deterministic=True)
        t_l1 = (time.perf_counter() - t_comp) * 1000.0
        preset_name = "ALIKED" if a_l1 == 1 else "SIFT"

        # Component 3: Pool extraction (N=4000)
        t_comp = time.perf_counter()
        if preset_name == "ALIKED":
            pts0_pool, desc0_pool = gpu_aliked.extract(img0, top_k=4000)
            pts1_pool, desc1_pool = gpu_aliked.extract(img1, top_k=4000)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            pts0_pool, desc0_pool = pts0_sift_all, desc0_sift_all
            pts1_pool, desc1_pool = pts1_sift_all, desc1_sift_all
        t_pool = (time.perf_counter() - t_comp) * 1000.0

        # Component 5: Matching initial 500
        t_comp = time.perf_counter()
        m_r = match_descriptors(desc0_pool[:500], desc1_pool[:500])
        t_match = (time.perf_counter() - t_comp) * 1000.0

        # Component 4: Level 2 policy evaluation
        m0_r = np.float32([pts0_pool[:500][x.queryIdx] for x in m_r]) if len(m_r) >= 5 else np.zeros((0, 2), np.float32)
        m1_r = np.float32([pts1_pool[:500][x.trainIdx] for x in m_r]) if len(m_r) >= 5 else np.zeros((0, 2), np.float32)
        proxy_r = compute_epipolar_proxy(m0_r, m1_r, K0, K1)

        t_comp = time.perf_counter()
        obs = np.zeros(34, dtype=np.float32)
        obs[0:9] = norm_st
        obs[9] = proxy_r["proxy"]
        obs[10] = 500.0 / 4000.0
        obs[11] = 500.0 / 4000.0
        obs[12] = proxy_r["inlier_ratio"]
        obs[13] = len(m_r) / 1000.0
        obs[14] = 0.0
        a_l2, _ = l2_model.predict(obs, deterministic=True)
        t_l2 = (time.perf_counter() - t_comp) * 1000.0

        # If action != STOP (action 3), re-match
        t_extra_match = 0.0
        if a_l2 != 3:
            t_comp = time.perf_counter()
            delta_n = 500 if a_l2 == 0 else (1000 if a_l2 == 1 else 2000)
            cur_n = min(4000, 500 + delta_n)
            m_r2 = match_descriptors(desc0_pool[:cur_n], desc1_pool[:cur_n])
            if len(m_r2) >= 5:
                m0_r = np.float32([pts0_pool[:cur_n][x.queryIdx] for x in m_r2])
                m1_r = np.float32([pts1_pool[:cur_n][x.trainIdx] for x in m_r2])
            t_extra_match = (time.perf_counter() - t_comp) * 1000.0

        # Component 6: RANSAC pose
        t_comp = time.perf_counter()
        if len(m0_r) >= 5:
            _ = md.estimate_pose(m0_r, m1_r, K0, K1)
        t_ransac = (time.perf_counter() - t_comp) * 1000.0

        lat_afm = t_stats + t_l1 + t_pool + t_match + t_l2 + t_extra_match + t_ransac
        records_per_pair["AFM-RL v1 (End-to-End)"].append(lat_afm)
        vram_per_method["AFM-RL v1 (End-to-End)"].append(vram_stage1)

        comp_records["afm_pair_stats"].append(t_stats)
        comp_records["afm_level1_eval"].append(t_l1)
        comp_records["afm_pool_extraction_4000"].append(t_pool)
        comp_records["afm_level2_policy_step"].append(t_l2)
        comp_records["afm_matching_bf"].append(t_match + t_extra_match)
        comp_records["afm_ransac_pose"].append(t_ransac)

        print(f"Timing completed {pair_idx + 1}/50 pairs...", flush=True)

    # -------------------------------------------------------------
    # Fold-Selected Cascade Latency
    # -------------------------------------------------------------
    # Weighted mean over the 5 folds: Folds 0,2,3 (C4 0.40), Fold 1 (C4 0.45), Fold 4 (C2 0.40)
    c4_40_lats = np.array(records_per_pair["Cascade C4 (tau=0.40)"])
    c4_45_lats = np.array(records_per_pair["Cascade C4 (tau=0.45)"])
    c2_40_lats = np.array(records_per_pair["Cascade C2 (tau=0.40)"])
    fold_sel_lats = (3.0 * c4_40_lats + 1.0 * c4_45_lats + 1.0 * c2_40_lats) / 5.0
    
    records_per_pair["Fold-Selected Cascade (Mean over 5 folds)"] = list(fold_sel_lats)
    vram_per_method["Fold-Selected Cascade (Mean over 5 folds)"] = list(vram_per_method["Cascade C4 (tau=0.40)"])
    methods.append("Fold-Selected Cascade (Mean over 5 folds)")

    # Build Summary Dataframe
    summary_rows = []
    for m in methods:
        lats = np.array(records_per_pair[m])
        vrams = np.array(vram_per_method[m])
        device_str = "CPU" if "SIFT" in m else ("GPU" if "ALIKED" in m else "GPU+CPU")
        summary_rows.append({
            "method": m,
            "device": device_str,
            "n_pairs": len(lats),
            "n_repeats": n_repeats,
            "mean_latency_ms": float(np.mean(lats)),
            "median_latency_ms": float(np.median(lats)),
            "std_latency_ms": float(np.std(lats)),
            "p95_latency_ms": float(np.percentile(lats, 95)),
            "peak_vram_mb": float(np.max(vrams))
        })
    df_methods = pd.DataFrame(summary_rows)
    methods_csv = OUTPUT_DIR / "timing_benchmark_phase9.csv"
    df_methods.to_csv(methods_csv, index=False)
    print(f"\nSaved {methods_csv}:")
    print(df_methods[["method", "device", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "peak_vram_mb"]].to_string(index=False))

    # Build Breakdown Dataframe
    comp_rows = []
    for c in components:
        c_lats = np.array(comp_records[c])
        comp_rows.append({
            "component": c,
            "mean_latency_ms": float(np.mean(c_lats)),
            "median_latency_ms": float(np.median(c_lats)),
            "std_latency_ms": float(np.std(c_lats)),
            "p95_latency_ms": float(np.percentile(c_lats, 95)),
            "pct_of_e2e": float(np.mean(c_lats) / np.mean(records_per_pair["AFM-RL v1 (End-to-End)"])) * 100.0
        })
    df_comps = pd.DataFrame(comp_rows)
    comps_csv = OUTPUT_DIR / "timing_breakdown_phase9.csv"
    df_comps.to_csv(comps_csv, index=False)
    print(f"\nSaved {comps_csv}:")
    print(df_comps.to_string(index=False))

if __name__ == "__main__":
    run_phase9_timing()
