import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')
import megadepth as md

OUTPUT_DIR = Path("results/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== RUNNING SECTION C: CASCADE UNDER AMENDMENT 1 ===")

# 1. Load priority evaluations
df2a = pd.read_csv("results/phase4b/megadepth_stage2a_priority_evaluations.csv")

def get_scene(fname):
    return "0015" if "0015" in fname else "0022"

df2a["scene"] = df2a["fname"].apply(get_scene)

# Extract Stage 1 (ALIKED-500) and Stage 2 (SIFT-4000)
df_s1 = df2a[(df2a["preset"] == "ALIKED") & (df2a["N"] == 500)].set_index(["fname", "i", "j"])
df_s2 = df2a[(df2a["preset"] == "SIFT") & (df2a["N"] == 4000)].set_index(["fname", "i", "j"])

# Align pairs
pairs = md.list_pairs()

# Compute pair-level arrays for S1 and S2
scenes = np.array([get_scene(f) for f, i, j in pairs])
p_epi_s1 = np.array([float(df_s1.loc[(f, i, j)]["p_epi"]) for f, i, j in pairs])
act_kp_s1 = np.array([float(df_s1.loc[(f, i, j)]["actual_kp"]) for f, i, j in pairs])
pose_err_s1 = np.array([float(df_s1.loc[(f, i, j)]["pose_err"]) for f, i, j in pairs])

p_epi_s2 = np.array([float(df_s2.loc[(f, i, j)]["p_epi"]) for f, i, j in pairs])
act_kp_s2 = np.array([float(df_s2.loc[(f, i, j)]["actual_kp"]) for f, i, j in pairs])
pose_err_s2 = np.array([float(df_s2.loc[(f, i, j)]["pose_err"]) for f, i, j in pairs])

def eval_cascade(tau, mask=None):
    if mask is None:
        mask = np.ones(len(pairs), dtype=bool)
        
    s1_stop = p_epi_s1 >= tau
    
    # Cumulative actual keypoints
    cum_act_kp = np.where(s1_stop, act_kp_s1, act_kp_s1 + act_kp_s2)[mask]
    
    # Final state selection: Highest-Proxy Attempted Config (Protocol v2 Final Section 2)
    # If stopped at s1 -> use s1. If escalated -> argmax(p_epi_s1, p_epi_s2)
    use_s1 = np.where(s1_stop, True, p_epi_s1 >= p_epi_s2)
    final_err = np.where(use_s1, pose_err_s1, pose_err_s2)[mask]
    
    succ10 = np.mean(final_err < 10.0) * 100.0
    auc = md.pose_auc(final_err)
    mean_kp = float(np.mean(cum_act_kp))
    stop_rate = float(np.mean(s1_stop[mask])) * 100.0
    
    # Utility = Succ@10° - 0.1 * (mean_cum_act_kp / 4000)
    # Note: Succ@10° is in [0, 1] or %? Protocol v2 Final: U = Succ@3px - 0.1*(Cum_KP / 4000) where Succ is in [0, 1].
    # Succ in [0, 1]:
    succ10_frac = float(np.mean(final_err < 10.0))
    utility = succ10_frac - 0.1 * (mean_kp / 4000.0)
    
    return {
        "tau": tau,
        "n_pairs": int(np.sum(mask)),
        "succ10": succ10,
        "succ10_frac": succ10_frac,
        "auc5": auc[0] * 100.0,
        "auc10": auc[1] * 100.0,
        "auc20": auc[2] * 100.0,
        "cum_actual_kp": mean_kp,
        "stop_rate_s1": stop_rate,
        "utility": utility
    }

# -------------------------------------------------------------
# 1. CROSS-SCENE FITTING ON EMPIRICAL TAU GRID
# -------------------------------------------------------------
mask_0015 = (scenes == "0015")
mask_0022 = (scenes == "0022")

# Empirical tau grid on Scene 0015 (quantiles 0.05 to 0.95 in 0.05 steps)
p_0015 = p_epi_s1[mask_0015]
q_grid_0015 = np.percentile(p_0015, np.arange(5, 100, 5))

# Fit on 0015
best_u_0015 = -1e9
best_tau_0015 = None
fit_0015_records = []

for tau in q_grid_0015:
    res = eval_cascade(tau, mask_0015)
    fit_0015_records.append(res)
    if res["utility"] > best_u_0015:
        best_u_0015 = res["utility"]
        best_tau_0015 = tau

# Evaluate best_tau_0015 on Scene 0022
eval_0022_with_tau_0015 = eval_cascade(best_tau_0015, mask_0022)

# Empirical tau grid on Scene 0022
p_0022 = p_epi_s1[mask_0022]
q_grid_0022 = np.percentile(p_0022, np.arange(5, 100, 5))

# Fit on 0022
best_u_0022 = -1e9
best_tau_0022 = None
fit_0022_records = []

for tau in q_grid_0022:
    res = eval_cascade(tau, mask_0022)
    fit_0022_records.append(res)
    if res["utility"] > best_u_0022:
        best_u_0022 = res["utility"]
        best_tau_0022 = tau

# Evaluate best_tau_0022 on Scene 0015
eval_0015_with_tau_0022 = eval_cascade(best_tau_0022, mask_0015)

# Cross-scene mean
cross_scene_summary = [
    {
        "direction": "Train 0015 -> Test 0022",
        "optimal_tau": best_tau_0015,
        "test_scene": "0022",
        "n_pairs": eval_0022_with_tau_0015["n_pairs"],
        "succ10": eval_0022_with_tau_0015["succ10"],
        "auc5": eval_0022_with_tau_0015["auc5"],
        "auc10": eval_0022_with_tau_0015["auc10"],
        "auc20": eval_0022_with_tau_0015["auc20"],
        "cum_actual_kp": eval_0022_with_tau_0015["cum_actual_kp"],
        "stop_rate_s1": eval_0022_with_tau_0015["stop_rate_s1"],
        "utility": eval_0022_with_tau_0015["utility"]
    },
    {
        "direction": "Train 0022 -> Test 0015",
        "optimal_tau": best_tau_0022,
        "test_scene": "0015",
        "n_pairs": eval_0015_with_tau_0022["n_pairs"],
        "succ10": eval_0015_with_tau_0022["succ10"],
        "auc5": eval_0015_with_tau_0022["auc5"],
        "auc10": eval_0015_with_tau_0022["auc10"],
        "auc20": eval_0015_with_tau_0022["auc20"],
        "cum_actual_kp": eval_0015_with_tau_0022["cum_actual_kp"],
        "stop_rate_s1": eval_0015_with_tau_0022["stop_rate_s1"],
        "utility": eval_0015_with_tau_0022["utility"]
    }
]

df_cross = pd.DataFrame(cross_scene_summary)
# Add mean row
mean_row = {
    "direction": "Cross-Scene Mean",
    "optimal_tau": (best_tau_0015 + best_tau_0022) / 2.0,
    "test_scene": "Mean(0015, 0022)",
    "n_pairs": 1500,
    "succ10": (cross_scene_summary[0]["succ10"] + cross_scene_summary[1]["succ10"]) / 2.0,
    "auc5": (cross_scene_summary[0]["auc5"] + cross_scene_summary[1]["auc5"]) / 2.0,
    "auc10": (cross_scene_summary[0]["auc10"] + cross_scene_summary[1]["auc10"]) / 2.0,
    "auc20": (cross_scene_summary[0]["auc20"] + cross_scene_summary[1]["auc20"]) / 2.0,
    "cum_actual_kp": (cross_scene_summary[0]["cum_actual_kp"] + cross_scene_summary[1]["cum_actual_kp"]) / 2.0,
    "stop_rate_s1": (cross_scene_summary[0]["stop_rate_s1"] + cross_scene_summary[1]["stop_rate_s1"]) / 2.0,
    "utility": (cross_scene_summary[0]["utility"] + cross_scene_summary[1]["utility"]) / 2.0,
}
df_cross = pd.concat([df_cross, pd.DataFrame([mean_row])], ignore_index=True)
df_cross.to_csv("results/phase5/c_cross_scene_fitting_summary.csv", index=False)
print("Saved results/phase5/c_cross_scene_fitting_summary.csv")
print(df_cross.to_string(index=False))

# -------------------------------------------------------------
# 2. HPATCHES-FIXED CHOICE (C*, tau=0.40) ON ALL 1500 PAIRS
# -------------------------------------------------------------
res_c_star_0015 = eval_cascade(0.40, mask_0015)
res_c_star_0022 = eval_cascade(0.40, mask_0022)
res_c_star_pooled = eval_cascade(0.40, None)

hpatches_choice_records = [
    {"split": "Scene 0015", **res_c_star_0015},
    {"split": "Scene 0022", **res_c_star_0022},
    {"split": "Pooled (All 1500)", **res_c_star_pooled},
]
df_hpatches_choice = pd.DataFrame(hpatches_choice_records)
df_hpatches_choice.to_csv("results/phase5/c_hpatches_preregistered_c_star.csv", index=False)
print("\nSaved results/phase5/c_hpatches_preregistered_c_star.csv")
print(df_hpatches_choice.to_string(index=False))

# -------------------------------------------------------------
# 3. PRIMARY COMPARISON TABLE: CASCADE vs KEY BASELINES
# -------------------------------------------------------------
def eval_fixed_baseline(preset, N, mask=None):
    if mask is None:
        mask = np.ones(len(pairs), dtype=bool)
    sub = df2a[(df2a["preset"] == preset) & (df2a["N"] == N)].set_index(["fname", "i", "j"])
    
    errs = np.array([float(sub.loc[(f, i, j)]["pose_err"]) for f, i, j in pairs])[mask]
    kps = np.array([float(sub.loc[(f, i, j)]["actual_kp"]) for f, i, j in pairs])[mask]
    
    auc = md.pose_auc(errs)
    succ10_frac = float(np.mean(errs < 10.0))
    mean_kp = float(np.mean(kps))
    utility = succ10_frac - 0.1 * (mean_kp / 4000.0)
    
    return {
        "method": f"{preset}-{N}",
        "n_pairs": int(np.sum(mask)),
        "succ10": succ10_frac * 100.0,
        "auc5": auc[0] * 100.0,
        "auc10": auc[1] * 100.0,
        "auc20": auc[2] * 100.0,
        "cum_actual_kp": mean_kp,
        "utility": utility
    }

comp_records = []
targets = [
    ("ALIKED", 500),
    ("ALIKED", 1000),
    ("SIFT", 2000),
    ("SIFT", 4000)
]

for sc_name, msk in [("Scene 0015", mask_0015), ("Scene 0022", mask_0022), ("Pooled", None)]:
    # Cascade C*
    casc_res = eval_cascade(0.40, msk)
    comp_records.append({
        "method": "Cascade C* (tau=0.40)",
        "scene": sc_name,
        "succ10": casc_res["succ10"],
        "auc5": casc_res["auc5"],
        "auc10": casc_res["auc10"],
        "auc20": casc_res["auc20"],
        "cum_actual_kp": casc_res["cum_actual_kp"],
        "utility": casc_res["utility"]
    })
    for preset, N in targets:
        fb_res = eval_fixed_baseline(preset, N, msk)
        comp_records.append({
            "method": fb_res["method"],
            "scene": sc_name,
            "succ10": fb_res["succ10"],
            "auc5": fb_res["auc5"],
            "auc10": fb_res["auc10"],
            "auc20": fb_res["auc20"],
            "cum_actual_kp": fb_res["cum_actual_kp"],
            "utility": fb_res["utility"]
        })

df_comp = pd.DataFrame(comp_records)
df_comp.to_csv("results/phase5/c_primary_comparisons_summary.csv", index=False)
print("\nSaved results/phase5/c_primary_comparisons_summary.csv")
print(df_comp.to_string(index=False))

print("\n=== SECTION C COMPLETE ===")
