import os
import sys
from pathlib import Path

# Ensure writable matplotlib cache
mpl_cache = Path("results/final/mpl_cache")
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_IMAGES_DIR = Path("paper/images")
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DIR = Path("results/final")

# Styling
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 11,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.autolayout': True
})

def make_all_figures():
    print("Loading results/final/all_results_v2.csv...")
    df_res = pd.read_csv("results/final/all_results_v2.csv")

    def get_val(table, row, col):
        sub = df_res[(df_res["table"] == table) & (df_res["row"] == row) & (df_res["column"] == col)]
        if len(sub) == 0:
            raise KeyError(f"Missing key in all_results_v2: table={table}, row={row}, col={col}")
        return sub.iloc[0]["value"]

    # =========================================================================
    # FIG 5: HPatches Pareto Frontier
    # =========================================================================
    print("Generating Figure 5: Pareto frontier on HPatches...")
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    sift_data = [
        {"Method": "SIFT-500", "Cum_KP": 500.0, "Succ@3px": float(get_val("T1_HPatches_Main", "SIFT-500", "succ_3px_pct"))},
        {"Method": "SIFT-1000", "Cum_KP": 1000.0, "Succ@3px": float(get_val("T1_HPatches_Main", "SIFT-1000", "succ_3px_pct"))},
        {"Method": "SIFT-2000", "Cum_KP": 1975.23, "Succ@3px": float(get_val("T1_HPatches_Main", "SIFT-2000", "succ_3px_pct"))},
        {"Method": "SIFT-4000", "Cum_KP": 3590.86, "Succ@3px": float(get_val("T1_HPatches_Main", "SIFT-4000", "succ_3px_pct"))},
    ]
    df_sift = pd.DataFrame(sift_data)
    ax.plot(df_sift["Cum_KP"], df_sift["Succ@3px"], 'o--', color='#1f77b4', label="SIFT (500-4000)", alpha=0.85)

    al500_kp = float(get_val("T1_HPatches_Main", "ALIKED-500", "cum_kp"))
    al500_succ = float(get_val("T1_HPatches_Main", "ALIKED-500", "succ_3px_pct"))
    ax.plot([al500_kp], [al500_succ], 's', color='#2ca02c', label="ALIKED-500", markersize=7)

    xf1k_kp = float(get_val("T1_HPatches_Main", "XFeat-1000", "cum_kp"))
    xf1k_succ = float(get_val("T1_HPatches_Main", "XFeat-1000", "succ_3px_pct"))
    ax.plot([xf1k_kp], [xf1k_succ], 'd', color='#9467bd', label="XFeat-1000", markersize=7)

    # Full AFM-RL v1 with 95% CI
    afm_kp = float(get_val("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "cum_kp"))
    afm_succ = float(get_val("T1_HPatches_Main", "Full AFM-RL v1 (Strict STOP)", "succ_3px_pct"))
    # CI from P1 comparison relative to Best Fixed (73.97%): [-5.34%, +1.90%] -> [68.63%, 75.87%]
    ax.errorbar([afm_kp], [afm_succ], yerr=[[afm_succ - 68.63], [75.87 - afm_succ]], fmt='*', color='#d62728',
                markersize=12, label="Full AFM-RL v1 (72.41%)", capsize=4, elinewidth=1.5)

    # Best Fixed VAL
    bf_kp = float(get_val("T1_HPatches_Main", "Best Fixed (Fold VAL)", "cum_kp"))
    bf_succ = float(get_val("T1_HPatches_Main", "Best Fixed (Fold VAL)", "succ_3px_pct"))
    ax.plot([bf_kp], [bf_succ], 'P', color='#ff7f0e', markersize=9, label="Best Fixed (VAL: SIFT-4k, 73.97%)")

    # Oracle Preset Selection Bound
    orc_kp = float(get_val("T1_HPatches_Main", "Oracle Preset (N=4000 Fixed Ceiling)*", "cum_kp"))
    orc_succ = float(get_val("T1_HPatches_Main", "Oracle Preset (N=4000 Fixed Ceiling)*", "succ_3px_pct"))
    ax.plot([orc_kp], [orc_succ], 'X', color='#8c564b', markersize=9, label="Oracle Preset Bound* (80.69%)")

    # Proxy-guided cascade (Seed 0, corrected 920.69 KP)
    casc_kp = float(get_val("T1_HPatches_Main", "Proxy-Guided Cascade (Original Seed 0)", "cum_kp"))
    casc_succ = float(get_val("T1_HPatches_Main", "Proxy-Guided Cascade (Original Seed 0)", "succ_3px_pct"))
    ax.plot([casc_kp], [casc_succ], 'D', color='#e377c2', markersize=8, label="Proxy-Guided Cascade (76.21%)")

    ax.set_xlabel("Cumulative Keypoints Evaluated ($K_{cum}$)")
    ax.set_ylabel("Success Rate (Succ@3px, %)")
    ax.set_title("HPatches Pareto Frontier: Success vs. Cumulative Keypoints")
    ax.set_xlim(200, 4300)
    ax.set_ylim(40, 85)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_pareto_hpatches.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save figure data
    df_f5 = pd.DataFrame([
        {"Method": "SIFT-500", "Cum_KP": 500.0, "Succ@3px": 59.66},
        {"Method": "SIFT-1000", "Cum_KP": 1000.0, "Succ@3px": 67.59},
        {"Method": "SIFT-2000", "Cum_KP": 1975.23, "Succ@3px": 72.76},
        {"Method": "SIFT-4000", "Cum_KP": 3590.86, "Succ@3px": 72.93},
        {"Method": "ALIKED-500", "Cum_KP": al500_kp, "Succ@3px": al500_succ},
        {"Method": "XFeat-1000", "Cum_KP": xf1k_kp, "Succ@3px": xf1k_succ},
        {"Method": "Full AFM-RL v1", "Cum_KP": afm_kp, "Succ@3px": afm_succ, "CI_low": 68.63, "CI_high": 75.87},
        {"Method": "Best Fixed (VAL)", "Cum_KP": bf_kp, "Succ@3px": bf_succ},
        {"Method": "Oracle Preset Bound*", "Cum_KP": orc_kp, "Succ@3px": orc_succ},
        {"Method": "Proxy-Guided Cascade", "Cum_KP": casc_kp, "Succ@3px": casc_succ},
    ])
    df_f5.to_csv(FINAL_DIR / "figure_data_pareto_hpatches.csv", index=False)

    # =========================================================================
    # FIG 6: MegaDepth-1500 Pareto Frontier
    # =========================================================================
    print("Generating Figure 6: Pareto frontier on MegaDepth...")
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    md_sift_kp = [
        float(get_val("T2_MegaDepth_Main", "SIFT-500", "cum_actual_kp")),
        float(get_val("T2_MegaDepth_Main", "SIFT-1000", "cum_actual_kp")),
        float(get_val("T2_MegaDepth_Main", "SIFT-2000", "cum_actual_kp")),
        float(get_val("T2_MegaDepth_Main", "SIFT-4000", "cum_actual_kp")),
    ]
    md_sift_auc = [
        float(get_val("T2_MegaDepth_Main", "SIFT-500", "auc_10_deg")),
        float(get_val("T2_MegaDepth_Main", "SIFT-1000", "auc_10_deg")),
        float(get_val("T2_MegaDepth_Main", "SIFT-2000", "auc_10_deg")),
        float(get_val("T2_MegaDepth_Main", "SIFT-4000", "auc_10_deg")),
    ]
    ax.plot(md_sift_kp, md_sift_auc, 'o--', color='#1f77b4', label="SIFT (500-4000)", alpha=0.85)

    md_aliked_kp = [
        float(get_val("T2_MegaDepth_Main", "ALIKED-500", "cum_actual_kp")),
        float(get_val("T2_MegaDepth_Main", "ALIKED-1000", "cum_actual_kp")),
    ]
    md_aliked_auc = [
        float(get_val("T2_MegaDepth_Main", "ALIKED-500", "auc_10_deg")),
        float(get_val("T2_MegaDepth_Main", "ALIKED-1000", "auc_10_deg")),
    ]
    ax.plot(md_aliked_kp, md_aliked_auc, 's--', color='#2ca02c', label="ALIKED (500-1000)", alpha=0.85)

    # AFM-RL v1 (Fixed Rerun) - use 55.17 consistently
    afm_md_kp = float(get_val("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "cum_actual_kp"))
    afm_md_auc = float(get_val("T2_MegaDepth_Main", "AFM-RL v1 (Fixed Rerun)", "auc_10_deg")) # 55.17
    # 95% CI vs ALIKED-500: [-1.15, +0.94] relative to 55.24 -> [54.09, 56.18]
    ax.errorbar([afm_md_kp], [afm_md_auc], yerr=[[afm_md_auc - 54.09], [56.18 - afm_md_auc]], fmt='*', color='#d62728',
                markersize=12, label="AFM-RL v1 (Fixed, 55.17)", capsize=4, elinewidth=1.5)

    # Cascade C4 (Committed)
    c4_kp = float(get_val("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "cum_actual_kp"))
    c4_auc = float(get_val("T2_MegaDepth_Main", "Cascade C4 (3-Stage Committed)", "auc_10_deg"))
    ax.plot([c4_kp], [c4_auc], '^', color='#ff7f0e', markersize=9, label="Cascade C4 (Committed, 57.58)")

    # Cascade C6 (tau=0.40)
    c6_kp = float(get_val("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "cum_actual_kp"))
    c6_auc = float(get_val("T2_MegaDepth_Main", "Cascade C6 (tau=0.40, Post-hoc)", "auc_10_deg"))
    # Permutation std is 0.21
    ax.errorbar([c6_kp], [c6_auc], yerr=[0.42], fmt='D', color='#9467bd',
                markersize=9, label="Cascade C6 (tau=0.40, 59.94)", capsize=4, elinewidth=1.5)

    ax.set_xlabel("Cumulative Keypoints Evaluated ($K_{cum}$)")
    ax.set_ylabel("Pose Accuracy (AUC@10°)")
    ax.set_title("MegaDepth-1500 Pareto Frontier: AUC@10° vs. Keypoints")
    ax.set_xlim(200, 4200)
    ax.set_ylim(20, 65)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_pareto_megadepth.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save figure data
    df_f6 = pd.DataFrame([
        {"Method": "SIFT-500", "Cum_KP": md_sift_kp[0], "AUC@10": md_sift_auc[0]},
        {"Method": "SIFT-1000", "Cum_KP": md_sift_kp[1], "AUC@10": md_sift_auc[1]},
        {"Method": "SIFT-2000", "Cum_KP": md_sift_kp[2], "AUC@10": md_sift_auc[2]},
        {"Method": "SIFT-4000", "Cum_KP": md_sift_kp[3], "AUC@10": md_sift_auc[3]},
        {"Method": "ALIKED-500", "Cum_KP": md_aliked_kp[0], "AUC@10": md_aliked_auc[0]},
        {"Method": "ALIKED-1000", "Cum_KP": md_aliked_kp[1], "AUC@10": md_aliked_auc[1]},
        {"Method": "AFM-RL v1 (Fixed)", "Cum_KP": afm_md_kp, "AUC@10": afm_md_auc, "CI_low": 54.09, "CI_high": 56.18},
        {"Method": "Cascade C4 (Committed)", "Cum_KP": c4_kp, "AUC@10": c4_auc},
        {"Method": "Cascade C6 (tau=0.40)", "Cum_KP": c6_kp, "AUC@10": c6_auc, "Perm_std": 0.21},
    ])
    df_f6.to_csv(FINAL_DIR / "figure_data_pareto_megadepth.csv", index=False)

    # =========================================================================
    # FIG 3: Cascade Stages (Stop vs Escalation Rates)
    # =========================================================================
    print("Generating Figure 3: Cascade stage stop percentages...")
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=300)

    # Remove (vii.a), format clean multiline labels to avoid overlapping ticks
    cascades = [
        "Cascade C6\n(AL500->SI4k)",
        "Cascade C4\n(Folds 0,2,3)",
        "Cascade C4\n(Fold 1, $\\tau$=0.45)",
        "Cascade C2\n(Fold 4)",
        "Inlier Count\nAlone Control"
    ]
    step1_stops = [
        float(get_val("T3_Cascade_Results", "Cascade C6 (tau=0.40) Original", "step1_stop_rate")), # 59.47
        float(get_val("T3_Cascade_Results", "MegaDepth Fold 0 Selected", "step1_stop_rate")),       # 59.47
        float(get_val("T3_Cascade_Results", "MegaDepth Fold 1 Selected", "step1_stop_rate")),       # 55.47
        float(get_val("T3_Cascade_Results", "MegaDepth Fold 4 Selected", "step1_stop_rate")),       # 59.47
        float(get_val("T3_Cascade_Results", "Control: (vii.a) Inlier Count Alone (Cross-Scene)", "step1_stop_rate")) # 59.93
    ]
    escalations = [100.0 - s for s in step1_stops]

    x = np.arange(len(cascades))
    width = 0.52

    p1 = ax.bar(x, step1_stops, width, label="Stage 1 Stop Rate (Early Exit)", color='#2ca02c', alpha=0.85)
    p2 = ax.bar(x, escalations, width, bottom=step1_stops, label="Escalation Rate (Stage 2+)", color='#d62728', alpha=0.85)

    for i in range(len(cascades)):
        ax.text(x[i], step1_stops[i] / 2, f"{step1_stops[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        ax.text(x[i], step1_stops[i] + escalations[i] / 2, f"{escalations[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)

    ax.set_ylabel("Proportion of Test Pairs (%)")
    ax.set_title("Early-Exit vs. Escalation Rates Across Cascades")
    ax.set_xticks(x)
    ax.set_xticklabels(cascades, fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(axis='y', linestyle=":", alpha=0.6)
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_cascade_stages.pdf", bbox_inches="tight")
    plt.close(fig)

    df_f3 = pd.DataFrame({
        "Cascade": cascades,
        "Stage1_Stop_Rate_%": step1_stops,
        "Escalation_Rate_%": escalations
    })
    df_f3.to_csv(FINAL_DIR / "figure_data_cascade_stages.csv", index=False)

    # =========================================================================
    # FIG 2: Policy Collapse (Standardize to Step 1)
    # =========================================================================
    print("Generating Figure 2: Policy collapse bar charts...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=300)

    folds = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Pooled"]
    # HPatches ALIKED %: Fold 0 had 100% ALIKED in final policy evaluation
    hp_aliked = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    md_aliked = [67.53, 100.0, 100.0, 100.0, 99.67, 93.44]

    x = np.arange(len(folds))
    w = 0.35
    ax1.bar(x - w/2, hp_aliked, w, label="HPatches", color='#1f77b4', alpha=0.85)
    ax1.bar(x + w/2, md_aliked, w, label="MegaDepth", color='#ff7f0e', alpha=0.85)
    ax1.set_ylabel("ALIKED Selection Share (%)")
    ax1.set_title("(a) Level 1 Policy Collapse to ALIKED")
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds, rotation=25, fontsize=8)
    ax1.set_ylim(50, 105)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis='y', linestyle=":", alpha=0.6)

    # Subplot 2: Step 1 Action Frequency on MegaDepth
    actions = ["+500 KP", "+1000 KP", "-500 KP", "STOP"]
    shares = [8.48, 11.05, 0.0, 80.47]
    colors = ['#1f77b4', '#2ca02c', '#7f7f7f', '#d62728']

    bars = ax2.bar(actions, shares, color=colors, width=0.5, alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5)

    ax2.set_ylabel("Action Frequency at Step 1 (%)")
    ax2.set_title("(b) Level 2 Step 1 Action Distribution")
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle=":", alpha=0.6)

    fig.savefig(OUTPUT_IMAGES_DIR / "fig_policy_collapse.pdf", bbox_inches="tight")
    plt.close(fig)

    df_f2 = pd.DataFrame({
        "Fold": folds,
        "HPatches_ALIKED_%": hp_aliked,
        "MegaDepth_ALIKED_%": md_aliked
    })
    df_f2.to_csv(FINAL_DIR / "figure_data_policy_collapse.csv", index=False)

    # =========================================================================
    # FIG 4: Cascade Architecture Diagram
    # =========================================================================
    print("Generating Figure 4: Cascade Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)
    ax.axis('off')

    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e6f2ff', edgecolor='#1f77b4', linewidth=1.5)
    box_props_stage2 = dict(boxstyle='round,pad=0.5', facecolor='#fff2e6', edgecolor='#ff7f0e', linewidth=1.5)
    box_props_exit = dict(boxstyle='round,pad=0.5', facecolor='#e6ffe6', edgecolor='#2ca02c', linewidth=1.5)

    ax.text(0.15, 0.75, "Stage 1:\nALIKED-500\n(Fast GPU Neural)", ha='center', va='center', bbox=box_props, fontsize=9)
    ax.text(0.48, 0.75, "Proxy Test:\n$\\hat{P}_{\\text{epi}} \\geq \\tau$ ?\n($\\tau = 0.40$)", ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', edgecolor='#cccc00', linewidth=1.5), fontsize=8.5)

    ax.text(0.85, 0.75, "Early Exit:\nEstimate Pose\n(59.5% Pairs)", ha='center', va='center', bbox=box_props_exit, fontsize=9)
    ax.text(0.48, 0.25, "Stage 2 Escalation:\nSIFT-4000 (CPU)\n(40.5% Hard Pairs)", ha='center', va='center', bbox=box_props_stage2, fontsize=9)
    ax.text(0.85, 0.25, "Select Best Proxy:\nEstimate Pose", ha='center', va='center', bbox=box_props_exit, fontsize=9)

    ax.annotate("", xy=(0.34, 0.75), xytext=(0.28, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.72, 0.75), xytext=(0.62, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.67, 0.79, "Yes", ha='center', fontsize=8.5, color='green', fontweight='bold')

    ax.annotate("", xy=(0.48, 0.42), xytext=(0.48, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.52, 0.50, "No", ha='left', fontsize=8.5, color='red', fontweight='bold')

    ax.annotate("", xy=(0.72, 0.25), xytext=(0.63, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("Proxy-Guided Cascade Decision Pipeline (Two-Stage Asymmetric)", fontsize=10, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_cascade_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print("All figures successfully regenerated into paper/images/ and results/final/figure_data_*.csv!")

if __name__ == "__main__":
    make_all_figures()
