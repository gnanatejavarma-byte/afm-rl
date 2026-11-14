import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== RUNNING SECTION D: ABLATIONS ===")

# Load evaluations
df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")

def get_scene(fname):
    return "0015" if "0015" in fname else "0022"

df2a["scene"] = df2a["fname"].apply(get_scene)

pairs = md.list_pairs()
scenes = np.array([get_scene(f) for f, i, j in pairs])
mask_0015 = (scenes == "0015")
mask_0022 = (scenes == "0022")

# Helper to get pair-indexed arrays
def get_arrays(preset, N):
    sub = df2a[(df2a["preset"] == preset) & (df2a["N"] == N)].set_index(["fname", "i", "j"])
    p_epi = np.array([float(sub.loc[(f, i, j)]["p_epi"]) for f, i, j in pairs])
    act_kp = np.array([float(sub.loc[(f, i, j)]["actual_kp"]) for f, i, j in pairs])
    pose_err = np.array([float(sub.loc[(f, i, j)]["pose_err"]) for f, i, j in pairs])
    return p_epi, act_kp, pose_err

p_a500, act_a500, err_a500 = get_arrays("ALIKED", 500)
p_a1000, act_a1000, err_a1000 = get_arrays("ALIKED", 1000)
p_a2000, act_a2000, err_a2000 = get_arrays("ALIKED", 2000)

p_s2000, act_s2000, err_s2000 = get_arrays("SIFT", 2000)
p_s4000, act_s4000, err_s4000 = get_arrays("SIFT", 4000)

# Pre-registered Cascade C* (tau=0.40) stopping mask
s1_stop_c_star = (p_a500 >= 0.40)
stop_rate_pooled = float(np.mean(s1_stop_c_star))

def summarize_result(name, err_arr, kp_arr, mask=None):
    if mask is None:
        mask = np.ones(len(pairs), dtype=bool)
    e = err_arr[mask]
    k = kp_arr[mask]
    
    succ10 = float(np.mean(e < 10.0)) * 100.0
    auc = md.pose_auc(e)
    mean_kp = float(np.mean(k))
    utility = (succ10 / 100.0) - 0.1 * (mean_kp / 4000.0)
    
    return {
        "method": name,
        "n_pairs": int(np.sum(mask)),
        "succ10": succ10,
        "auc5": auc[0] * 100.0,
        "auc10": auc[1] * 100.0,
        "auc20": auc[2] * 100.0,
        "cum_actual_kp": mean_kp,
        "utility": utility
    }

records = []

# --- (i) Always run both ALIKED-500 + SIFT-4000, take higher proxy ---
use_a500_i = (p_a500 >= p_s4000)
err_i = np.where(use_a500_i, err_a500, err_s4000)
kp_i = act_a500 + act_s4000

# --- (ii) Oracle-of-two (GT picks better of ALIKED-500 and SIFT-4000) ---
use_a500_ii = (err_a500 <= err_s4000)
err_ii = np.where(use_a500_ii, err_a500, err_s4000)
kp_ii = act_a500 + act_s4000  # Both extracted to know

# --- (iii) Proxy-selected among {ALIKED-500, 1000, 2000} ---
stack_p = np.stack([p_a500, p_a1000, p_a2000], axis=1) # (1500, 3)
stack_err = np.stack([err_a500, err_a1000, err_a2000], axis=1)
best_proxy_idx = np.argmax(stack_p, axis=1)
err_iii = np.array([stack_err[idx, best_proxy_idx[idx]] for idx in range(len(pairs))])
kp_iii = act_a500 + act_a1000 + act_a2000

# --- (iv) Random-escalation control with same stage-1 stopping rate ---
np.random.seed(42)
n_sims = 100
err_iv_sims = []
kp_iv_sims = []

for _ in range(n_sims):
    rnd_stop = (np.random.rand(len(pairs)) < stop_rate_pooled)
    # If stopped -> a500. If escalated -> highest proxy of (a500, s4000)
    use_a500_esc = (p_a500 >= p_s4000)
    err_esc = np.where(use_a500_esc, err_a500, err_s4000)
    
    sim_err = np.where(rnd_stop, err_a500, err_esc)
    sim_kp = np.where(rnd_stop, act_a500, act_a500 + act_s4000)
    err_iv_sims.append(sim_err)
    kp_iv_sims.append(sim_kp)

err_iv = np.mean(err_iv_sims, axis=0)
kp_iv = np.mean(kp_iv_sims, axis=0)

# --- (v) Nearest equal-cost fixed baselines ---
# SIFT-2000 and ALIKED-2000

# --- Pre-registered Cascade C* for direct reference ---
err_c_star = np.where(s1_stop_c_star, err_a500, np.where(p_a500 >= p_s4000, err_a500, err_s4000))
kp_c_star = np.where(s1_stop_c_star, act_a500, act_a500 + act_s4000)

ablation_dict = {
    "Cascade C* (tau=0.40)": (err_c_star, kp_c_star),
    "(i) Always Both (ALIKED-500 + SIFT-4000, proxy pick)": (err_i, kp_i),
    "(ii) Oracle-of-Two (GT pick ALIKED-500 vs SIFT-4000)": (err_ii, kp_ii),
    "(iii) Proxy Pick {ALIKED-500, 1000, 2000}": (err_iii, kp_iii),
    "(iv) Random-Escalation Control (matching 59.47% stop rate)": (err_iv, kp_iv),
    "(v.a) Nearest Fixed Baseline: SIFT-2000": (err_s2000, act_s2000),
    "(v.b) Nearest Fixed Baseline: ALIKED-2000": (err_a2000, act_a2000),
}

for name, (e_arr, k_arr) in ablation_dict.items():
    for sc_name, msk in [("0015", mask_0015), ("0022", mask_0022), ("Pooled", None)]:
        res = summarize_result(name, e_arr, k_arr, msk)
        res["scene"] = sc_name
        records.append(res)

df_abl = pd.DataFrame(records)
# Reorder columns
df_abl = df_abl[["method", "scene", "n_pairs", "succ10", "auc5", "auc10", "auc20", "cum_actual_kp", "utility"]]
df_abl.to_csv("results/phase5/d_ablations_summary.csv", index=False)
print("Saved results/phase5/d_ablations_summary.csv")

print("\n=== ABLATIONS SUMMARY (POOLED) ===")
print(df_abl[df_abl["scene"] == "Pooled"].to_string(index=False))

print("\n=== SECTION D COMPLETE ===")

