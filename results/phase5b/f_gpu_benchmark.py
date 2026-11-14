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
from megadepth_eval_env import (
    compute_epipolar_proxy,
    get_megadepth_pool,
    POOL_CACHE_DIR,
)

OUTPUT_DIR = Path("results/phase5b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# GPU ALIKED Extractor
# -------------------------------------------------------------
import kornia.feature as kf

class GPUAlikedExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = kf.ALIKED.from_pretrained("aliked-n16").to(device).eval()
        
    def extract(self, img: np.ndarray, top_k: int = 4000):
        # img: (H, W) uint8 grayscale
        h, w = img.shape
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

# -------------------------------------------------------------
# CPU SIFT Extractor
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Section F.3: Cold Per-Pair Latency Benchmarking (50 pairs)
# -------------------------------------------------------------
def benchmark_f3_latency():
    print("=" * 70)
    print("SECTION F.3: COLD PER-PAIR LATENCY ON 50 RANDOM MEGADEPTH PAIRS")
    print("=" * 70)
    
    pairs = md.list_pairs()
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(pairs), size=50, replace=False)
    sample_pairs = [pairs[idx] for idx in sample_indices]
    
    gpu_aliked = GPUAlikedExtractor(device="cuda")
    cpu_sift = CPUSiftExtractor()
    
    methods = [
        "ALIKED-500 (GPU)",
        "ALIKED-1000 (GPU)",
        "SIFT-2000 (CPU)",
        "SIFT-4000 (CPU)",
        "Cascade C6 (tau=0.40)",
        "Fold-Selected Cascade C4 (tau=0.40)",
        "Always-Both (ALIKED-500 GPU + SIFT-4000 CPU)",
    ]
    
    records = {m: {"latencies": [], "peak_vram_mb": []} for m in methods}
    
    for pair_cnt, (fname, i, j) in enumerate(sample_pairs):
        sc = md._scene(fname)
        T_gt = sc["poses"][j] @ np.linalg.inv(sc["poses"][i])
        
        # 1. ALIKED-500 (GPU)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        pts0_a500, desc0_a500 = gpu_aliked.extract(img0, top_k=500)
        pts1_a500, desc1_a500 = gpu_aliked.extract(img1, top_k=500)
        matches_a500 = match_descriptors(desc0_a500, desc1_a500)
        if len(matches_a500) >= 5:
            m_pts0 = np.float32([pts0_a500[m.queryIdx] for m in matches_a500])
            m_pts1 = np.float32([pts1_a500[m.trainIdx] for m in matches_a500])
            r_a500 = md.estimate_pose(m_pts0, m_pts1, K0, K1)
        else:
            r_a500 = None
        t1 = time.perf_counter()
        vram_a500 = torch.cuda.max_memory_allocated() / (1024 * 1024)
        records["ALIKED-500 (GPU)"]["latencies"].append((t1 - t0) * 1000.0)
        records["ALIKED-500 (GPU)"]["peak_vram_mb"].append(vram_a500)
        
        # 2. ALIKED-1000 (GPU)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        pts0_a1000, desc0_a1000 = gpu_aliked.extract(img0, top_k=1000)
        pts1_a1000, desc1_a1000 = gpu_aliked.extract(img1, top_k=1000)
        matches_a1000 = match_descriptors(desc0_a1000, desc1_a1000)
        if len(matches_a1000) >= 5:
            m_pts0 = np.float32([pts0_a1000[m.queryIdx] for m in matches_a1000])
            m_pts1 = np.float32([pts1_a1000[m.trainIdx] for m in matches_a1000])
            r_a1000 = md.estimate_pose(m_pts0, m_pts1, K0, K1)
        else:
            r_a1000 = None
        t1 = time.perf_counter()
        vram_a1000 = torch.cuda.max_memory_allocated() / (1024 * 1024)
        records["ALIKED-1000 (GPU)"]["latencies"].append((t1 - t0) * 1000.0)
        records["ALIKED-1000 (GPU)"]["peak_vram_mb"].append(vram_a1000)
        
        # 3. SIFT-2000 (CPU)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        pts0_s2000, desc0_s2000 = cpu_sift.extract(img0, top_k=2000)
        pts1_s2000, desc1_s2000 = cpu_sift.extract(img1, top_k=2000)
        matches_s2000 = match_descriptors(desc0_s2000, desc1_s2000)
        if len(matches_s2000) >= 5:
            m_pts0 = np.float32([pts0_s2000[m.queryIdx] for m in matches_s2000])
            m_pts1 = np.float32([pts1_s2000[m.trainIdx] for m in matches_s2000])
            r_s2000 = md.estimate_pose(m_pts0, m_pts1, K0, K1)
        else:
            r_s2000 = None
        t1 = time.perf_counter()
        records["SIFT-2000 (CPU)"]["latencies"].append((t1 - t0) * 1000.0)
        records["SIFT-2000 (CPU)"]["peak_vram_mb"].append(0.0)
        
        # 4. SIFT-4000 (CPU)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        pts0_s4000, desc0_s4000 = cpu_sift.extract(img0, top_k=4000)
        pts1_s4000, desc1_s4000 = cpu_sift.extract(img1, top_k=4000)
        matches_s4000 = match_descriptors(desc0_s4000, desc1_s4000)
        if len(matches_s4000) >= 5:
            m_pts0 = np.float32([pts0_s4000[m.queryIdx] for m in matches_s4000])
            m_pts1 = np.float32([pts1_s4000[m.trainIdx] for m in matches_s4000])
            r_s4000 = md.estimate_pose(m_pts0, m_pts1, K0, K1)
        else:
            r_s4000 = None
        t1 = time.perf_counter()
        records["SIFT-4000 (CPU)"]["latencies"].append((t1 - t0) * 1000.0)
        records["SIFT-4000 (CPU)"]["peak_vram_mb"].append(0.0)
        
        # 5. Cascade C6 (tau=0.40): ALIKED-500 GPU -> SIFT-4000 CPU if proxy < 0.40
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        pts0_a, desc0_a = gpu_aliked.extract(img0, top_k=500)
        pts1_a, desc1_a = gpu_aliked.extract(img1, top_k=500)
        m_a = match_descriptors(desc0_a, desc1_a)
        if len(m_a) >= 5:
            m0_a = np.float32([pts0_a[m.queryIdx] for m in m_a])
            m1_a = np.float32([pts1_a[m.trainIdx] for m in m_a])
            p_a = compute_epipolar_proxy(m0_a, m1_a, K0, K1)["proxy"]
        else:
            p_a = 0.0
            
        if p_a >= 0.40:
            if len(m_a) >= 5:
                _ = md.estimate_pose(m0_a, m1_a, K0, K1)
        else:
            # Escalate to SIFT-4000
            pts0_s, desc0_s = cpu_sift.extract(img0, top_k=4000)
            pts1_s, desc1_s = cpu_sift.extract(img1, top_k=4000)
            m_s = match_descriptors(desc0_s, desc1_s)
            if len(m_s) >= 5:
                m0_s = np.float32([pts0_s[m.queryIdx] for m in m_s])
                m1_s = np.float32([pts1_s[m.trainIdx] for m in m_s])
                p_s = compute_epipolar_proxy(m0_s, m1_s, K0, K1)["proxy"]
                chosen_pts0, chosen_pts1 = (m0_a, m1_a) if p_a >= p_s else (m0_s, m1_s)
                _ = md.estimate_pose(chosen_pts0, chosen_pts1, K0, K1)
        t1 = time.perf_counter()
        vram_c6 = torch.cuda.max_memory_allocated() / (1024 * 1024)
        records["Cascade C6 (tau=0.40)"]["latencies"].append((t1 - t0) * 1000.0)
        records["Cascade C6 (tau=0.40)"]["peak_vram_mb"].append(vram_c6)
        
        # 6. Fold-Selected Cascade C4 (tau=0.40): ALIKED-500 GPU -> ALIKED-1000 GPU if proxy < 0.40
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        pts0_a, desc0_a = gpu_aliked.extract(img0, top_k=500)
        pts1_a, desc1_a = gpu_aliked.extract(img1, top_k=500)
        m_a = match_descriptors(desc0_a, desc1_a)
        if len(m_a) >= 5:
            m0_a = np.float32([pts0_a[m.queryIdx] for m in m_a])
            m1_a = np.float32([pts1_a[m.trainIdx] for m in m_a])
            p_a = compute_epipolar_proxy(m0_a, m1_a, K0, K1)["proxy"]
        else:
            p_a = 0.0
            
        if p_a >= 0.40:
            if len(m_a) >= 5:
                _ = md.estimate_pose(m0_a, m1_a, K0, K1)
        else:
            # Escalate to ALIKED-1000
            pts0_1000, desc0_1000 = gpu_aliked.extract(img0, top_k=1000)
            pts1_1000, desc1_1000 = gpu_aliked.extract(img1, top_k=1000)
            m_1000 = match_descriptors(desc0_1000, desc1_1000)
            if len(m_1000) >= 5:
                m0_1000 = np.float32([pts0_1000[m.queryIdx] for m in m_1000])
                m1_1000 = np.float32([pts1_1000[m.trainIdx] for m in m_1000])
                _ = md.estimate_pose(m0_1000, m1_1000, K0, K1)
        t1 = time.perf_counter()
        vram_c4 = torch.cuda.max_memory_allocated() / (1024 * 1024)
        records["Fold-Selected Cascade C4 (tau=0.40)"]["latencies"].append((t1 - t0) * 1000.0)
        records["Fold-Selected Cascade C4 (tau=0.40)"]["peak_vram_mb"].append(vram_c4)
        
        # 7. Always-Both: ALIKED-500 GPU + SIFT-4000 CPU in sequence
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        pts0_a, desc0_a = gpu_aliked.extract(img0, top_k=500)
        pts1_a, desc1_a = gpu_aliked.extract(img1, top_k=500)
        m_a = match_descriptors(desc0_a, desc1_a)
        p_a = 0.0
        if len(m_a) >= 5:
            m0_a = np.float32([pts0_a[m.queryIdx] for m in m_a])
            m1_a = np.float32([pts1_a[m.trainIdx] for m in m_a])
            p_a = compute_epipolar_proxy(m0_a, m1_a, K0, K1)["proxy"]
            
        pts0_s, desc0_s = cpu_sift.extract(img0, top_k=4000)
        pts1_s, desc1_s = cpu_sift.extract(img1, top_k=4000)
        m_s = match_descriptors(desc0_s, desc1_s)
        p_s = 0.0
        if len(m_s) >= 5:
            m0_s = np.float32([pts0_s[m.queryIdx] for m in m_s])
            m1_s = np.float32([pts1_s[m.trainIdx] for m in m_s])
            p_s = compute_epipolar_proxy(m0_s, m1_s, K0, K1)["proxy"]
            
        if len(m_a) >= 5 or len(m_s) >= 5:
            chosen_pts0, chosen_pts1 = (m0_a, m1_a) if p_a >= p_s else (m0_s, m1_s)
            _ = md.estimate_pose(chosen_pts0, chosen_pts1, K0, K1)
        t1 = time.perf_counter()
        vram_both = torch.cuda.max_memory_allocated() / (1024 * 1024)
        records["Always-Both (ALIKED-500 GPU + SIFT-4000 CPU)"]["latencies"].append((t1 - t0) * 1000.0)
        records["Always-Both (ALIKED-500 GPU + SIFT-4000 CPU)"]["peak_vram_mb"].append(vram_both)
        
        if (pair_cnt + 1) % 10 == 0:
            print(f"Benchmarked {pair_cnt + 1}/50 pairs...", flush=True)
            
    summary_rows = []
    for m in methods:
        lats = np.array(records[m]["latencies"])
        vrams = np.array(records[m]["peak_vram_mb"])
        summary_rows.append({
            "method": m,
            "n_pairs": len(lats),
            "mean_latency_ms": float(np.mean(lats)),
            "median_latency_ms": float(np.median(lats)),
            "p95_latency_ms": float(np.percentile(lats, 95)),
            "peak_vram_mb": float(np.max(vrams))
        })
        
    df_lat = pd.DataFrame(summary_rows)
    lat_csv = OUTPUT_DIR / "f3_gpu_latency_summary.csv"
    df_lat.to_csv(lat_csv, index=False)
    print(f"\nSaved {lat_csv}:")
    print(df_lat.to_string(index=False))

# -------------------------------------------------------------
# Section F.4: GPU Feature Pool Compatibility (20 pairs)
# -------------------------------------------------------------
def benchmark_f4_compatibility():
    print("=" * 70)
    print("SECTION F.4: GPU FEATURE POOL COMPATIBILITY (20 PAIRS)")
    print("=" * 70)
    
    pairs = md.list_pairs()
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(pairs), size=20, replace=False)
    sample_pairs = [pairs[idx] for idx in sample_indices]
    
    gpu_aliked = GPUAlikedExtractor(device="cuda")
    
    comp_rows = []
    
    for pair_idx, (fname, i, j) in enumerate(sample_pairs):
        sc = md._scene(fname)
        T_gt = sc["poses"][j] @ np.linalg.inv(sc["poses"][i])
        
        img0, K0 = md.load_image(fname, i)
        img1, K1 = md.load_image(fname, j)
        path0 = sc["image_paths"][i]
        path1 = sc["image_paths"][j]
        
        # 1. CPU-cached pools
        cpu_p0 = get_megadepth_pool(img0, "ALIKED", path0)
        cpu_p1 = get_megadepth_pool(img1, "ALIKED", path1)
        
        cpu_pts0 = cpu_p0["pts"][:500]
        cpu_desc0 = cpu_p0["desc"][:500]
        cpu_pts1 = cpu_p1["pts"][:500]
        cpu_desc1 = cpu_p1["desc"][:500]
        
        # 2. GPU-extracted fresh pools
        gpu_pts0, gpu_desc0 = gpu_aliked.extract(img0, top_k=500)
        gpu_pts1, gpu_desc1 = gpu_aliked.extract(img1, top_k=500)
        
        # Keypoint count check
        cnt_match = (len(cpu_pts0) == len(gpu_pts0)) and (len(cpu_pts1) == len(gpu_pts1))
        
        # Max coordinate diff
        min_k0 = min(len(cpu_pts0), len(gpu_pts0))
        max_coord_diff0 = float(np.max(np.abs(cpu_pts0[:min_k0] - gpu_pts0[:min_k0]))) if min_k0 > 0 else 0.0
        min_k1 = min(len(cpu_pts1), len(gpu_pts1))
        max_coord_diff1 = float(np.max(np.abs(cpu_pts1[:min_k1] - gpu_pts1[:min_k1]))) if min_k1 > 0 else 0.0
        max_coord_diff = max(max_coord_diff0, max_coord_diff1)
        
        # CPU match & eval
        m_cpu = match_descriptors(cpu_desc0, cpu_desc1)
        if len(m_cpu) >= 5:
            m0_c = np.float32([cpu_pts0[m.queryIdx] for m in m_cpu])
            m1_c = np.float32([cpu_pts1[m.trainIdx] for m in m_cpu])
            p_cpu = compute_epipolar_proxy(m0_c, m1_c, K0, K1)["proxy"]
            r_cpu = md.estimate_pose(m0_c, m1_c, K0, K1)
            err_cpu = md.pose_error(T_gt, r_cpu[0], r_cpu[1]) if r_cpu is not None else 180.0
        else:
            p_cpu = 0.0
            err_cpu = 180.0
            
        # GPU match & eval
        m_gpu = match_descriptors(gpu_desc0, gpu_desc1)
        if len(m_gpu) >= 5:
            m0_g = np.float32([gpu_pts0[m.queryIdx] for m in m_gpu])
            m1_g = np.float32([gpu_pts1[m.trainIdx] for m in m_gpu])
            p_gpu = compute_epipolar_proxy(m0_g, m1_g, K0, K1)["proxy"]
            r_gpu = md.estimate_pose(m0_g, m1_g, K0, K1)
            err_gpu = md.pose_error(T_gt, r_gpu[0], r_gpu[1]) if r_gpu is not None else 180.0
        else:
            p_gpu = 0.0
            err_gpu = 180.0
            
        # Cascade decision (tau = 0.40)
        dec_cpu = "STOP" if p_cpu >= 0.40 else "ESCALATE"
        dec_gpu = "STOP" if p_gpu >= 0.40 else "ESCALATE"
        same_dec = (dec_cpu == dec_gpu)
        
        comp_rows.append({
            "pair_idx": pair_idx,
            "fname": fname, "i": i, "j": j,
            "cpu_n_kpts0": len(cpu_pts0),
            "gpu_n_kpts0": len(gpu_pts0),
            "kpts_count_match": cnt_match,
            "max_coord_diff_px": max_coord_diff,
            "cpu_p_epi": p_cpu,
            "gpu_p_epi": p_gpu,
            "abs_diff_p_epi": abs(p_cpu - p_gpu),
            "cpu_pose_err": err_cpu,
            "gpu_pose_err": err_gpu,
            "abs_diff_pose_err": abs(err_cpu - err_gpu),
            "cpu_cascade_decision": dec_cpu,
            "gpu_cascade_decision": dec_gpu,
            "decisions_match": same_dec
        })
        
    df_comp = pd.DataFrame(comp_rows)
    comp_csv = OUTPUT_DIR / "f4_gpu_pool_compatibility.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"\nSaved {comp_csv}")
    
    summary = [{
        "n_pairs_evaluated": len(df_comp),
        "pct_identical_kp_count": float(df_comp["kpts_count_match"].mean() * 100.0),
        "mean_max_coord_diff_px": float(df_comp["max_coord_diff_px"].mean()),
        "max_max_coord_diff_px": float(df_comp["max_coord_diff_px"].max()),
        "mean_abs_diff_p_epi": float(df_comp["abs_diff_p_epi"].mean()),
        "max_abs_diff_p_epi": float(df_comp["abs_diff_p_epi"].max()),
        "mean_abs_diff_pose_err": float(df_comp["abs_diff_pose_err"].mean()),
        "pct_identical_cascade_decisions": float(df_comp["decisions_match"].mean() * 100.0)
    }]
    df_sum = pd.DataFrame(summary)
    sum_csv = OUTPUT_DIR / "f4_pool_compatibility_summary.csv"
    df_sum.to_csv(sum_csv, index=False)
    print(f"\nSaved {sum_csv}:")
    print(df_sum.to_string(index=False))

def main():
    benchmark_f3_latency()
    benchmark_f4_compatibility()

if __name__ == "__main__":
    main()
