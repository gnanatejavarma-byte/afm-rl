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

# Paths
OUTPUT_TABLES_DIR = Path("results/final/tables")
OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGES_DIR = Path("paper/images_new")
OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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

def make_figures():
    print("Generating Figure 1: Pareto frontier on HPatches...")
    # F1.a: HPatches Pareto
    df_hp = pd.read_csv("results/phase3/pareto_plot_data.csv")
    
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    
    # Filter fixed configs
    sift_sub = df_hp[df_hp["Method"].str.startswith("SIFT Fixed-")].copy()
    sift_sub["n"] = sift_sub["Method"].apply(lambda x: int(x.split("-")[-1]))
    sift_sub = sift_sub.sort_values("n")
    
    ax.plot(sift_sub["Cum_KP"], sift_sub["Succ@3px"], 'o--', color='#1f77b4', label="SIFT (500-4000)", alpha=0.8)
    
    # ALIKED-500
    al500 = df_hp[df_hp["Method"] == "ALIKED-500"]
    if len(al500) > 0:
        ax.plot(al500["Cum_KP"], al500["Succ@3px"], 's', color='#2ca02c', label="ALIKED-500", markersize=7)
        
    # XFeat-1000
    xf1k = df_hp[df_hp["Method"] == "XFeat-1000"]
    if len(xf1k) > 0:
        ax.plot(xf1k["Cum_KP"], xf1k["Succ@3px"], 'd', color='#9467bd', label="XFeat-1000", markersize=7)
        
    # Mark AFM-RL v1 with CI
    # Succ@3px = 72.41% [-1.55% vs best fixed], Cum KP = 1215.38
    # From b3_bootstrap_comparisons.csv: DeltaSucc 95% CI is [-5.34%, +1.90%] relative to Best Fixed (73.97%) -> [68.63%, 75.87%]
    ax.errorbar([1215.38], [72.41], yerr=[[72.41 - 68.63], [75.87 - 72.41]], fmt='*', color='#d62728', 
                markersize=12, label="Full AFM-RL v1 (72.41%)", capsize=4, elinewidth=1.5)
                
    # Mark Best Fixed VAL
    ax.plot([4000.0], [73.97], 'P', color='#ff7f0e', markersize=9, label="Best Fixed (VAL: SIFT-4k, 73.97%)")
    
    # Mark Oracle Preset Upper Bound
    ax.plot([4000.0], [80.69], 'X', color='#8c564b', markersize=9, label="Oracle Preset Bound* (80.69%)")
    
    # Mark Exploratory Cascade
    ax.plot([1548.30], [76.21], 'D', color='#e377c2', markersize=8, label="AFM-RL v2 Cascade (76.21%)")

    ax.set_xlabel("Cumulative Keypoints Evaluated ($K_{cum}$)")
    ax.set_ylabel("Success Rate (Succ@3px, %)")
    ax.set_title("HPatches Pareto Frontier: Success vs. Cumulative Keypoints")
    ax.set_xlim(200, 4300)
    ax.set_ylim(40, 85)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_pareto_hpatches.pdf", bbox_inches="tight")
    plt.close(fig)

    # F1.b: MegaDepth Pareto
    print("Generating Figure 2: Pareto frontier on MegaDepth...")
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    
    # Fixed baselines on MegaDepth
    md_sift_kp = [496.26, 993.43, 1989.67, 3821.14]
    md_sift_auc = [23.32, 38.83, 46.14, 54.78]
    md_aliked_kp = [497.47, 976.97]
    md_aliked_auc = [55.24, 55.46]
    
    ax.plot(md_sift_kp, md_sift_auc, 'o--', color='#1f77b4', label="SIFT (500-4000)", alpha=0.8)
    ax.plot(md_aliked_kp, md_aliked_auc, 's--', color='#2ca02c', label="ALIKED (500-1000)", alpha=0.8)
    
    # AFM-RL v1 (Fixed Rerun)
    # AUC@10 = 55.16, CI from a1_bootstrap_comparisons_fixed: delta vs ALIKED-500 is -0.07 [-0.53, +0.37] -> AUC in [54.71, 55.61]
    ax.errorbar([870.55], [55.16], yerr=[[55.16 - 54.71], [55.61 - 55.16]], fmt='*', color='#d62728', 
                markersize=12, label="AFM-RL v1 (Fixed, 55.16)", capsize=4, elinewidth=1.5)
                
    # Cascade C4 (3-Stage Committed)
    ax.plot([1529.96], [57.58], '^', color='#ff7f0e', markersize=9, label="Cascade C4 (Committed, 57.58)")
    
    # Cascade C6 (tau=0.40)
    # From b2_permutation_robustness_all_pairs: AUC@10 = 59.94, perm std = 0.24
    ax.errorbar([2013.49], [59.94], yerr=[0.48], fmt='D', color='#9467bd', 
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

    # F2: Cascade Stage-Stop Percentages
    print("Generating Figure 3: Cascade stage stop percentages...")
    fig, ax = plt.subplots(figsize=(6.0, 3.5), dpi=300)
    
    cascades = [
        "Cascade C6\n(AL500->SI4k)",
        "Cascade C4 (Fold 0-3)\n(AL500->SI1k->AL2k)",
        "Cascade C4 (Fold 1)\n(tau=0.45)",
        "Cascade C2 (Fold 4)\n(AL500->AL1k->SI4k)",
        "Control (vii.a)\nInlier Count Alone"
    ]
    step1_stops = [59.47, 59.47, 55.47, 59.47, 59.93]
    escalations = [100.0 - s for s in step1_stops]
    
    x = np.arange(len(cascades))
    width = 0.55
    
    p1 = ax.bar(x, step1_stops, width, label="Stage 1 Stop (Early Exit)", color='#2ca02c', alpha=0.85)
    p2 = ax.bar(x, escalations, width, bottom=step1_stops, label="Escalated to Stage 2+", color='#d62728', alpha=0.85)
    
    for i in range(len(cascades)):
        ax.text(x[i], step1_stops[i] / 2, f"{step1_stops[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        ax.text(x[i], step1_stops[i] + escalations[i] / 2, f"{escalations[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        
    ax.set_ylabel("Percentage of Test Pairs (%)")
    ax.set_title("Early-Exit vs. Escalation Proportion Across Cascades")
    ax.set_xticks(x)
    ax.set_xticklabels(cascades, fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis='y', linestyle=":", alpha=0.6)
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_cascade_stages.pdf", bbox_inches="tight")
    plt.close(fig)

    # F3: Policy Collapse Bar Charts
    print("Generating Figure 4: Policy collapse bar charts...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=300)
    
    # Subplot 1: Level 1 Preset Distribution on HPatches vs MegaDepth
    folds = ["Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Pooled"]
    # HPatches ALIKED %
    hp_aliked = [73.91, 100.0, 100.0, 100.0, 100.0, 94.83]
    # MegaDepth ALIKED %
    # F0: 1013/1500=67.53%, F1: 100%, F2: 100%, F3: 100%, F4: 1495/1500=99.67%, Pooled: 7008/7500=93.44%
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
    ax1.legend(loc="lower right")
    ax1.grid(axis='y', linestyle=":", alpha=0.6)
    
    # Subplot 2: Level 2 Action Histogram on MegaDepth
    actions = ["+500 KP", "+1000 KP", "-500 KP", "STOP (Step 1)"]
    # Pooled action shares: +500: 8.48%, +1000: 11.05%, -500: 0.0%, STOP: 80.47%
    shares = [8.48, 11.05, 0.0, 80.47]
    colors = ['#1f77b4', '#2ca02c', '#7f7f7f', '#d62728']
    
    bars = ax2.bar(actions, shares, color=colors, width=0.5, alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=8.5)
        
    ax2.set_ylabel("Initial Action Frequency (%)")
    ax2.set_title("(b) Level 2 Step 0 Action Distribution")
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle=":", alpha=0.6)
    
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_policy_collapse.pdf", bbox_inches="tight")
    plt.close(fig)

    # F4: Cascade Architecture Diagram
    print("Generating Figure 5: Cascade Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)
    ax.axis('off')
    
    # Draw simple conceptual schematic boxes
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e6f2ff', edgecolor='#1f77b4', linewidth=1.5)
    box_props_stage2 = dict(boxstyle='round,pad=0.5', facecolor='#fff2e6', edgecolor='#ff7f0e', linewidth=1.5)
    box_props_exit = dict(boxstyle='round,pad=0.5', facecolor='#e6ffe6', edgecolor='#2ca02c', linewidth=1.5)
    
    ax.text(0.15, 0.75, "Stage 1:\nALIKED-500\n(Fast GPU Neural)", ha='center', va='center', bbox=box_props, fontsize=9)
    ax.text(0.48, 0.75, "Proxy Test:\n$\\hat{P}_{epi} \\geq \\tau$ ?\n($\\tau = 0.40$)", ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', edgecolor='#cccc00', linewidth=1.5), fontsize=8.5)
            
    ax.text(0.85, 0.75, "Early Exit:\nEstimate Pose\n(59.5% Pairs)", ha='center', va='center', bbox=box_props_exit, fontsize=9)
    
    ax.text(0.48, 0.25, "Stage 2 Escalation:\nSIFT-4000 (CPU)\n(40.5% Hard Pairs)", ha='center', va='center', bbox=box_props_stage2, fontsize=9)
    ax.text(0.85, 0.25, "Select Best Proxy:\nEstimate Pose", ha='center', va='center', bbox=box_props_exit, fontsize=9)
    
    # Arrows
    ax.annotate("", xy=(0.34, 0.75), xytext=(0.28, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.72, 0.75), xytext=(0.62, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.67, 0.79, "Yes", ha='center', fontsize=8.5, color='green', fontweight='bold')
    
    ax.annotate("", xy=(0.48, 0.42), xytext=(0.48, 0.58), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.52, 0.50, "No", ha='left', fontsize=8.5, color='red', fontweight='bold')
    
    ax.annotate("", xy=(0.72, 0.25), xytext=(0.63, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.set_title("AFM Cascade Decision Pipeline (2-Stage Asymmetric)", fontsize=10, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    fig.savefig(OUTPUT_IMAGES_DIR / "fig_cascade_diagram.pdf", bbox_inches="tight")
    plt.close(fig)

def make_tables():
    print("Generating LaTeX Tables...")
    
    # -------------------------------------------------------------
    # T1: HPatches Main Table
    # -------------------------------------------------------------
    t1_tex = r"""% Data Table T1: HPatches Main (116 Sequences, 5 Folds)
% Source: results/final/all_results.csv (T1_HPatches_Main)
\begin{table*}[t]
\centering
\caption{HPatches 5-fold frozen evaluation across all 116 sequences (580 test pairs). All metrics evaluated under honest label-free protocol without test-time ground truth. Primary comparison deltas (P1--P5) include sequence-clustered bootstrap 95\% confidence intervals (10,000 resamples).}
\label{tab:hpatches_main}
\small
\begin{tabular}{lcccccccc}
\hline
\textbf{Method} & \textbf{Eval Type} & \textbf{Succ@3px (\%)} & \textbf{AUC@1px} & \textbf{AUC@3px} & \textbf{AUC@5px} & \textbf{Final KP} & \textbf{Cum. KP} & \textbf{Steps} \\
\hline
\multicolumn{9}{l}{\textit{Fixed Baselines (Frozen Grid)}} \\
SIFT-500 & Pooled (580) & 67.59 & 18.32 & 44.68 & 53.41 & 500 & 500 & 1.00 \\
SIFT-1000 & Pooled (580) & 71.21 & 19.88 & 47.50 & 56.45 & 1000 & 1000 & 1.00 \\
SIFT-2000 & Pooled (580) & 72.76 & 20.91 & 49.01 & 57.98 & 2000 & 2000 & 1.00 \\
SIFT-4000 & Pooled (580) & 72.93 & 22.39 & 51.13 & 59.98 & 4000 & 4000 & 1.00 \\
ALIKED-500 & Pooled (580) & 72.59 & 20.25 & 46.88 & 55.89 & 500 & 500 & 1.00 \\
ALIKED-1000 & Pooled (580) & 73.62 & 20.73 & 47.56 & 56.54 & 1000 & 1000 & 1.00 \\
XFeat-1000 & Pooled (580) & 71.90 & 20.15 & 47.21 & 56.02 & 1000 & 1000 & 1.00 \\
Best Fixed (Fold VAL) & Pooled (580) & 73.97 & 21.64 & 49.64 & 58.62 & 4000 & 4000 & 1.00 \\
\hline
\multicolumn{9}{l}{\textit{Adaptive \& Upper Bound Methods}} \\
Full AFM-RL v1 (Strict STOP) & Pooled (580) & 72.41 & 20.12 & 46.81 & 55.83 & 500 & 1215.38 & 1.00 \\
AFM-RL v2 Cascade (Exploratory) & Pooled (580) & 76.21 & 21.68 & 49.61 & 58.74 & 1165.50 & 1548.30 & 1.41 \\
Oracle Preset Selection* & Pooled (580) & 80.69 & 24.97 & 56.33 & 65.31 & 4000 & 4000 & 1.00 \\
\hline
\multicolumn{9}{l}{\textit{Primary Pre-Registered Comparisons (AFM-RL v1 vs. Baseline, 95\% Clustered Bootstrap CIs)}} \\
P1 vs. Best Fixed VAL & Delta & \multicolumn{2}{l}{$-1.55\%\ [-5.34\%, +1.90\%]$} & \multicolumn{2}{l}{$-2.83\ [-5.67\%, -0.29\%]$} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P2 vs. ALIKED-500 & Delta & \multicolumn{2}{l}{$-0.17\%\ [-1.03\%, +0.52\%]$} & \multicolumn{2}{l}{$-0.08\ [-0.56\%, +0.38\%]$} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P3 vs. SIFT-2000 & Delta & \multicolumn{2}{l}{$-0.34\%\ [-4.48\%, +3.45\%]$} & \multicolumn{2}{l}{$-2.20\ [-5.25\%, +0.74\%]$} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P4 vs. SIFT-4000 & Delta & \multicolumn{2}{l}{$-0.52\%\ [-4.66\%, +3.28\%]$} & \multicolumn{2}{l}{$-4.32\ [-7.31\%, -1.54\%]$} & \multicolumn{3}{l}{Indistinguishable (CI crosses 0)} \\
P5 vs. Oracle Preset* & Delta & \multicolumn{2}{l}{$-8.28\%\ [-11.72\%, -4.83\%]$} & \multicolumn{2}{l}{$-9.52\ [-12.21, -6.84]$} & \multicolumn{3}{l}{Significant Deficit ($p < 0.001$)} \\
\hline
\end{tabular}
\end{table*}
"""
    with open(OUTPUT_TABLES_DIR / "tab_hpatches_main.tex", "w", encoding="utf-8") as f:
        f.write(t1_tex)

    # -------------------------------------------------------------
    # T2: MegaDepth Main Table
    # -------------------------------------------------------------
    t2_tex = r"""% Data Table T2: MegaDepth Main (1,500 Pairs)
% Source: results/final/all_results.csv (T2_MegaDepth_Main)
\begin{table*}[t]
\centering
\caption{MegaDepth-1500 benchmark performance across 1,500 pairs (Scenes 0015 and 0022). Metrics report pose recall (Succ@10°) and pose AUC curves up to 5°, 10°, and 20°. Computational overhead accounts for cumulative keypoints evaluated and wall-clock execution time on identical hardware (PyTorch CUDA on NVIDIA RTX 3050 Laptop GPU, Intel Core CPU).}
\label{tab:megadepth_main}
\small
\begin{tabular}{lcccccccc}
\hline
\textbf{Method} & \textbf{Device} & \textbf{Succ@10° (\%)} & \textbf{AUC@5°} & \textbf{AUC@10°} & \textbf{AUC@20°} & \textbf{Req. N} & \textbf{Cum. KP} & \textbf{Latency (ms)} \\
\hline
\multicolumn{9}{l}{\textit{Fixed Baselines}} \\
SIFT-500 & CPU & 35.80 & 14.19 & 23.32 & 33.37 & 500 & 496.26 & 948.3 \\
SIFT-1000 & CPU & 55.40 & 26.85 & 38.83 & 49.92 & 1000 & 993.43 & 1182.4 \\
SIFT-2000 & CPU & 66.80 & 33.91 & 46.14 & 57.51 & 2000 & 1989.67 & 1286.3 \\
SIFT-4000 & CPU & 77.20 & 39.73 & 54.78 & 67.43 & 4000 & 3821.14 & 1456.1 \\
ALIKED-500 & GPU & 76.13 & 39.46 & 55.24 & 68.42 & 500 & 497.47 & 264.8 \\
ALIKED-1000 & GPU & 77.67 & 39.52 & 55.46 & 68.61 & 1000 & 976.97 & 276.8 \\
\hline
\multicolumn{9}{l}{\textit{RL Policy \& Pre-Registered Cascades}} \\
AFM-RL v1 (Fixed Rerun) & GPU+CPU & 76.05 & 39.55 & 55.16 & 68.23 & Adaptive & 870.55 & 2087.5 \\
Cascade C4 (3-Stage Committed) & GPU+CPU & 78.87 & 41.37 & 57.58 & 70.90 & 500--2000 & 1529.96 & 435.9 \\
Cascade C4 (2-Stage GPU-Only) & GPU & 77.67 & 39.52 & 55.46 & 68.61 & 500--1000 & 823.90 & 402.1 \\
\hline
\multicolumn{9}{l}{\textit{Exploratory Extension \& Upper Bound}} \\
Cascade C6 ($\tau=0.40$, Post-hoc) & GPU+CPU & 82.13 & 43.32 & 59.94 & 73.52 & 500 / 4000 & 2013.49 & 433.7 \\
Always-Both (AL500 + SI4k) & GPU+CPU & 82.53 & 44.12 & 60.47 & 74.02 & 4500 & 4318.61 & 1908.9 \\
Oracle-of-Two (AL500 / SI4k)* & Bound & 87.53 & 52.14 & 68.38 & 80.12 & 4000 & 4318.61 & -- \\
\hline
\end{tabular}
\end{table*}
"""
    with open(OUTPUT_TABLES_DIR / "tab_megadepth_main.tex", "w", encoding="utf-8") as f:
        f.write(t2_tex)

    # -------------------------------------------------------------
    # T3: Cascade Results Table
    # -------------------------------------------------------------
    t3_tex = r"""% Data Table T3: Cascade Results, Robustness & Component Studies
% Source: results/final/all_results.csv (T3_Cascade_Results)
\begin{table*}[t]
\centering
\caption{Detailed evaluation of feature cascades on MegaDepth-1500: cross-validation fold selections, correspondence permutation sensitivity, and ablation controls. All cascades use unsupervised epipolar proxy $\hat{P}_{epi}$ for escalation decisions.}
\label{tab:cascade_results}
\small
\begin{tabular}{lcccccc}
\hline
\textbf{Cascade / Control Setup} & \textbf{Protocol Status} & \textbf{Succ@10° (\%)} & \textbf{AUC@10°} & \textbf{Cum. KP} & \textbf{Stop Rate (\%)} & \textbf{Escalation (\%)} \\
\hline
\multicolumn{7}{l}{\textit{Fold-Selected Cascades (Protocol v2 Amendment 1)}} \\
Fold 0: C4 ($\tau=0.40$, AL500 $\to$ SI1k $\to$ AL2k) & Pre-registered & 78.87 & 57.58 & 1529.96 & 59.47 & 40.53 \\
Fold 1: C4 ($\tau=0.45$, AL500 $\to$ SI1k $\to$ AL2k) & Pre-registered & 78.87 & 57.49 & 1642.44 & 55.47 & 44.53 \\
Fold 2: C4 ($\tau=0.40$, AL500 $\to$ SI1k $\to$ AL2k) & Pre-registered & 78.87 & 57.58 & 1529.96 & 59.47 & 40.53 \\
Fold 3: C4 ($\tau=0.40$, AL500 $\to$ SI1k $\to$ AL2k) & Pre-registered & 78.87 & 57.58 & 1529.96 & 59.47 & 40.53 \\
Fold 4: C2 ($\tau=0.40$, AL500 $\to$ AL1k $\to$ SI4k) & Pre-registered & 82.40 & 60.04 & 1909.65 & 59.47 & 40.53 \\
Pooled Fold-Selected Ensemble & Pre-registered & 79.57 & 58.07 & 1608.40 & 58.67 & 41.33 \\
\hline
\multicolumn{7}{l}{\textit{Post-Hoc Cascade C6 ($\tau=0.40$, ALIKED-500 $\to$ SIFT-4000) Permutation Robustness}} \\
Original Match Ordering & Post-hoc & 82.13 & 59.94 & 2013.49 & 59.47 & 40.53 \\
Permutation 1 & Post-hoc & 82.00 & 59.88 & 2018.12 & 59.27 & 40.73 \\
Permutation 2 & Post-hoc & 82.40 & 60.05 & 2009.84 & 59.60 & 40.40 \\
Permutation 3 & Post-hoc & 82.20 & 59.98 & 2012.30 & 59.40 & 40.60 \\
Permutation Mean $\pm$ Std & Post-hoc & $82.18 \pm 0.16$ & $59.96 \pm 0.08$ & $2013.44 \pm 3.48$ & $59.43 \pm 0.14$ & $40.57 \pm 0.14$ \\
\hline
\multicolumn{7}{l}{\textit{Component Ablations \& Calibration Controls}} \\
Random-Escalation Control (20 Draws Mean) & Control & $78.62 \pm 0.44$ & $57.30 \pm 0.34$ & 2040.64 & 59.47 & 40.53 \\
Anti-Cascade (Escalate on HIGH proxy) & Control & 76.33 & 55.29 & 2082.01 & 59.47 & 40.53 \\
Inlier Count Alone (Threshold 125, Cross-Scene) & Control & 82.07 & 59.86 & 2000.57 & 59.93 & 40.07 \\
Inlier Ratio Alone (Cross-Scene) & Control & 81.20 & 59.24 & 2337.58 & 49.87 & 50.13 \\
Reprojection Factor Alone (Cross-Scene) & Control & 79.67 & 57.90 & 2217.64 & 54.00 & 46.00 \\
\hline
\end{tabular}
\end{table*}
"""
    with open(OUTPUT_TABLES_DIR / "tab_cascade_results.tex", "w", encoding="utf-8") as f:
        f.write(t3_tex)

    # -------------------------------------------------------------
    # T4: Policy Behaviour Table
    # -------------------------------------------------------------
    t4_tex = r"""% Data Table T4: Policy Behaviour, Preset Entropy & Action Telemetry
% Source: results/final/all_results.csv (T4_Policy_Behaviour)
\begin{table*}[t]
\centering
\caption{Policy behavioral telemetry across HPatches (5 folds) and MegaDepth (zero-shot transfer). Level 1 measures preset selection entropy (in bits) over the 14-preset action space. Level 2 measures initial action distribution and termination step counts.}
\label{tab:policy_behaviour}
\small
\begin{tabular}{lcccccc}
\hline
\textbf{Split / Benchmark} & \textbf{Total Pairs} & \textbf{ALIKED Share (\%)} & \textbf{SIFT Share (\%)} & \textbf{L1 Entropy (bits)} & \textbf{Step 1 STOP (\%)} & \textbf{Mean Steps} \\
\hline
\multicolumn{7}{l}{\textit{HPatches 5-Fold Test Evaluation (In-Distribution)}} \\
Fold 0 & 115 & 73.91 (85/115) & 25.22 (29/115) & 0.883 & 100.00 & 1.00 \\
Fold 1 & 115 & 100.00 (115/115) & 0.00 (0/115) & 0.000 & 100.00 & 1.00 \\
Fold 2 & 115 & 100.00 (115/115) & 0.00 (0/115) & 0.000 & 100.00 & 1.00 \\
Fold 3 & 115 & 100.00 (115/115) & 0.00 (0/115) & 0.000 & 100.00 & 1.00 \\
Fold 4 & 120 & 100.00 (120/120) & 0.00 (0/120) & 0.000 & 100.00 & 1.00 \\
Pooled HPatches & 580 & 94.83 (550/580) & 5.00 (29/580) & 0.305 & 100.00 & 1.00 \\
\hline
\multicolumn{7}{l}{\textit{MegaDepth-1500 Evaluation (Zero-Shot Transfer)}} \\
Fold 0 Policy & 1500 & 67.53 (1013/1500) & 32.47 (487/1500) & 0.909 & 67.80 & 1.57 \\
Fold 1 Policy & 1500 & 100.00 (1500/1500) & 0.00 (0/1500) & 0.000 & 87.27 & 1.13 \\
Fold 2 Policy & 1500 & 100.00 (1500/1500) & 0.00 (0/1500) & 0.000 & 87.00 & 1.13 \\
Fold 3 Policy & 1500 & 100.00 (1500/1500) & 0.00 (0/1500) & 0.000 & 65.53 & 1.38 \\
Fold 4 Policy & 1500 & 99.67 (1495/1500) & 0.13 (2/1500) & 0.024 & 94.73 & 1.05 \\
Pooled MegaDepth & 7500 & 93.44 (7008/7500) & 6.52 (489/7500) & 0.354 & 80.47 & 1.25 \\
\hline
\end{tabular}
\end{table*}
"""
    with open(OUTPUT_TABLES_DIR / "tab_policy_behaviour.tex", "w", encoding="utf-8") as f:
        f.write(t4_tex)

    # -------------------------------------------------------------
    # T5: Audit Table
    # -------------------------------------------------------------
    t5_tex = r"""% Data Table T5: Methodological Audit & Discrepancy Reconciliation
% Source: results/final/all_results.csv (T5_Audit_Table)
\begin{table*}[t]
\centering
\caption{Systematic audit of discrepancies between earlier drafts and frozen results. All values recomputed from raw per-pair match records.}
\label{tab:audit_table}
\small
\begin{tabular}{p{3.5cm}p{3.5cm}p{3.5cm}p{5.5cm}}
\hline
\textbf{Issue / Claim} & \textbf{Old Value (Leaky / Bugged)} & \textbf{Honest Value (Frozen)} & \textbf{Root Cause / Mechanism} \\
\hline
HPatches SIFT Performance & Succ@3px = 84.8\% & Succ@3px = 72.93\% & Phase 1 evaluated on test set with ground-truth corner error inside reward loop and non-disjoint train/test splits. \\
\hline
MegaDepth AFM-RL v1 Action Mapping & Succ@10° = 74.20\%, Cum KP = 100.0, 100\% STOP & Succ@10° = 76.05\%, Cum KP = 870.55, 80.47\% STOP & Environment bug: \texttt{src/megadepth\_eval\_env.py} mapped actions 0..3 to $[-500, -250, +250, +500]$ instead of $[+500, +1000, -500, \text{STOP}]$, forcing depletion. Corrected in Phase 5b. \\
\hline
Cascade C4 Definition Nomenclature & C4: ALIKED-500 $\to$ ALIKED-1000 (KP = 823.90) & C4: ALIKED-500 $\to$ SIFT-1000 $\to$ ALIKED-2000 (KP = 1529.96) & Protocol v2 Amendment 1 pre-registered 3-stage C4. A 2-stage GPU abbreviation was tested in text/benchmark. Both are reported side-by-side. \\
\hline
MegaDepth Clustered Bootstrap & $\Delta\text{Succ@10°} = -1.93\%$, $\Delta\text{AUC@10°} = -2.89$ & $\Delta\text{Succ@10°} = -0.08\%$ $[-0.66\%, +0.50\%]$, $\Delta\text{AUC@10°} = -0.07$ $[-0.53, +0.37]$ & Earlier bootstrap averaged pose error across folds before thresholding $< 10.0$ and used piecewise linear approximation. Corrected via reference-image cluster bootstrap. \\
\hline
RANSAC Seed Reproducibility & \texttt{cv2.setRNGSeed} hook caused variation & Hook had zero effect; correspondence order is sole noise source & OpenCV's \texttt{cv2.findHomography} uses internal deterministic RANSAC seed; correspondence order variation produces $\pm 0.3\%$ to $\pm 0.8\%$ selection noise. \\
\hline
\end{tabular}
\end{table*}
"""
    with open(OUTPUT_TABLES_DIR / "tab_audit_table.tex", "w", encoding="utf-8") as f:
        f.write(t5_tex)

    # Concatenate all tables
    print("Concatenating all tables into results/final/all_tables_for_paste.tex...")
    with open("results/final/all_tables_for_paste.tex", "w", encoding="utf-8") as f_out:
        f_out.write("% ==========================================================================\n")
        f_out.write("% ALL TABLES CONCATENATED FOR EASY COPY/PASTE INTO LATEX MANUSCRIPT\n")
        f_out.write("% Generated automatically by results/final/make_assets.py\n")
        f_out.write("% ==========================================================================\n\n")
        for t_file in ["tab_hpatches_main.tex", "tab_megadepth_main.tex", "tab_cascade_results.tex", "tab_policy_behaviour.tex", "tab_audit_table.tex"]:
            p = OUTPUT_TABLES_DIR / t_file
            with open(p, "r", encoding="utf-8") as f_in:
                f_out.write(f_in.read())
                f_out.write("\n\n")

if __name__ == "__main__":
    make_figures()
    make_tables()
    print("Assets generation finished successfully.")
