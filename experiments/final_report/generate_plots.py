"""Generates all verified comparison plots for the AFM-RL Final Report & Presentation.

Plots Generated:
1. hpatches_auc3_overall.png: Overall HPatches AUC@3 comparison across all 5 methods
2. hpatches_auc5_overall.png: Overall HPatches AUC@5 comparison across all 5 methods
3. hpatches_keypoints_overall.png: Average keypoints allocated (Adaptive vs Fixed baselines)
4. hpatches_auc3_by_preset.png: AUC@3 across all 11 presets (5 bars per preset)
5. hpatches_accuracy_vs_keypoints.png: Scatter plot of AUC@3 vs Keypoints for each preset
6. hpatches_adaptive_vs_fixed4000.png: Direct comparison of Adaptive vs Fixed-4000 per preset
7. hpatches_auc_all_thresholds.png: Grouped bars for AUC@1px, AUC@3px, AUC@5px across methods
8. megadepth_auc20_comparison.png: MegaDepth AUC@20 comparison (MD-trained, Zero-Shot, Fixed 500-4000)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set global style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial", "DejaVu Sans", "Helvetica"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0

# -----------------------------------------------------------------------------
# Color Palette
# -----------------------------------------------------------------------------
COLOR_ADAPTIVE = "#1f77b4"   # Solid Blue
COLOR_F500     = "#aec7e8"   # Light Blue
COLOR_F1000    = "#ffbb78"   # Light Orange
COLOR_F2000    = "#ff7f0e"   # Orange
COLOR_F4000    = "#d62728"   # Red / Dark Orange
COLOR_MD_TRAIN = "#9467bd"   # Purple
COLOR_HP_ZS    = "#2ca02c"   # Green

# -----------------------------------------------------------------------------
# 1. HPatches Verified Overall Data
# -----------------------------------------------------------------------------
methods_order = ["Adaptive Level-2", "Fixed 500", "Fixed 1000", "Fixed 2000", "Fixed 4000"]
hpatches_overall = {
    "method": methods_order,
    "avg_kp": [1191.7, 495.6, 979.5, 1897.0, 3424.4],
    "auc_1": [14.24, 10.54, 13.54, 14.73, 15.76],
    "auc_3": [38.30, 26.89, 31.08, 35.07, 36.34],
    "auc_5": [51.97, 35.51, 40.64, 45.74, 47.09],
    "succ_1": [31.55, 22.57, 26.74, 29.73, 31.55],
    "succ_3": [64.39, 42.67, 48.88, 56.26, 56.79],
    "succ_5": [78.18, 54.01, 60.11, 66.20, 68.66],
}
df_overall = pd.DataFrame(hpatches_overall)

# -----------------------------------------------------------------------------
# 2. HPatches Verified Per-Preset Data (AUC@3)
# -----------------------------------------------------------------------------
presets = [
    "SIFT", "KAZE", "GFTT+SIFT", "AKAZE", "BRISK",
    "FAST+FREAK", "GFTT+BRIEF", "FAST+BRIEF", "ORB",
    "STAR+BRIEF", "ORB+FREAK"
]

preset_data = {
    "preset": presets,
    "adaptive_auc3": [50.03, 45.49, 44.61, 42.71, 42.43, 38.64, 36.09, 36.01, 33.72, 30.21, 21.37],
    "adaptive_kp":   [942.4, 1198.4, 1117.6, 1310.9, 1401.2, 1165.9, 1324.2, 1000.0, 1686.7, 725.8, 1235.2],
    "fixed500_auc3": [35.96, 31.12, 35.25, 28.64, 25.84, 34.28, 26.47, 29.25, 15.76, 24.09, 9.15],
    "fixed1000_auc3":[46.95, 36.20, 40.87, 33.00, 34.81, 32.97, 27.72, 32.88, 18.77, 28.25, 9.48],
    "fixed2000_auc3":[52.31, 42.91, 41.15, 37.38, 36.60, 37.16, 31.75, 29.16, 28.51, 30.88, 17.94],
    "fixed4000_auc3":[52.91, 44.30, 41.49, 36.69, 38.96, 37.99, 32.98, 33.60, 33.34, 29.83, 17.69],
}
df_presets = pd.DataFrame(preset_data)


# -----------------------------------------------------------------------------
# PLOT 1: Overall HPatches AUC@3 Comparison
# -----------------------------------------------------------------------------
def plot_hpatches_auc3_overall():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(
        df_overall["method"],
        df_overall["auc_3"],
        color=[COLOR_ADAPTIVE, COLOR_F500, COLOR_F1000, COLOR_F2000, COLOR_F4000],
        width=0.55,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.set_ylabel("Mean AUC@3px (%)", fontsize=12, fontweight="bold")
    ax.set_title("HPatches Homography Estimation — Overall AUC@3px Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.8,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    # Highlight adaptive gain
    ax.axhline(df_overall.loc[0, "auc_3"], color=COLOR_ADAPTIVE, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = PLOTS_DIR / "hpatches_auc3_overall.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# PLOT 2: Overall HPatches AUC@5 Comparison
# -----------------------------------------------------------------------------
def plot_hpatches_auc5_overall():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(
        df_overall["method"],
        df_overall["auc_5"],
        color=[COLOR_ADAPTIVE, COLOR_F500, COLOR_F1000, COLOR_F2000, COLOR_F4000],
        width=0.55,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.set_ylabel("Mean AUC@5px (%)", fontsize=12, fontweight="bold")
    ax.set_title("HPatches Homography Estimation — Overall AUC@5px Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 60)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    ax.axhline(df_overall.loc[0, "auc_5"], color=COLOR_ADAPTIVE, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = PLOTS_DIR / "hpatches_auc5_overall.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# PLOT 3: Average Keypoints Overall Comparison
# -----------------------------------------------------------------------------
def plot_hpatches_keypoints_overall():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(
        df_overall["method"],
        df_overall["avg_kp"],
        color=[COLOR_ADAPTIVE, COLOR_F500, COLOR_F1000, COLOR_F2000, COLOR_F4000],
        width=0.55,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.set_ylabel("Average Keypoint Budget (Points)", fontsize=12, fontweight="bold")
    ax.set_title("Keypoint Budget Comparison — Adaptive RL vs Fixed Baselines", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 4000)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 70,
            f"{int(round(height)):,} KP",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
    
    # Annotate saving vs Fixed 4000
    ax.annotate(
        "65% Budget Reduction\nvs Fixed 4000",
        xy=(0, 1191.7), xytext=(0.8, 2600),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10.5, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba")
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / "hpatches_keypoints_overall.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# PLOT 4: HPatches AUC@3 by Preset (All 11 Presets)
# -----------------------------------------------------------------------------
def plot_hpatches_auc3_by_preset():
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    
    x = np.arange(len(presets))
    width = 0.15
    
    ax.bar(x - 2*width, df_presets["adaptive_auc3"], width, label="Adaptive Level-2", color=COLOR_ADAPTIVE, edgecolor="#222")
    ax.bar(x - 1*width, df_presets["fixed500_auc3"], width, label="Fixed 500", color=COLOR_F500, edgecolor="#555")
    ax.bar(x,           df_presets["fixed1000_auc3"], width, label="Fixed 1000", color=COLOR_F1000, edgecolor="#555")
    ax.bar(x + 1*width, df_presets["fixed2000_auc3"], width, label="Fixed 2000", color=COLOR_F2000, edgecolor="#555")
    ax.bar(x + 2*width, df_presets["fixed4000_auc3"], width, label="Fixed 4000", color=COLOR_F4000, edgecolor="#555")
    
    ax.set_ylabel("AUC@3px (%)", fontsize=12, fontweight="bold")
    ax.set_title("HPatches Homography Estimation — AUC@3px by Feature Preset", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(presets, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    
    plt.tight_layout()
    out_path = PLOTS_DIR / "hpatches_auc3_by_preset.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# PLOT 5: Adaptive AUC@3 vs Keypoints Scatter Plot
# -----------------------------------------------------------------------------
def plot_hpatches_accuracy_vs_keypoints():
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    x = df_presets["adaptive_kp"]
    y = df_presets["adaptive_auc3"]
    
    scatter = ax.scatter(x, y, color=COLOR_ADAPTIVE, s=160, edgecolor="black", linewidth=1.5, zorder=5)
    
    # Annotate each point with preset name
    for i, txt in enumerate(presets):
        dx, dy = 18, 0.4
        if txt == "SIFT":
            dy = 1.0
        elif txt == "KAZE":
            dy = -1.5
        elif txt == "GFTT+SIFT":
            dy = 1.0
        elif txt == "FAST+BRIEF":
            dy = -1.5
        elif txt == "AKAZE":
            dy = 1.0
        elif txt == "BRISK":
            dy = -1.5
        ax.annotate(
            txt,
            (x[i], y[i]),
            xytext=(x[i] + dx, y[i] + dy),
            fontsize=10.5,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ddd", alpha=0.8)
        )
    
    ax.set_xlabel("Average Keypoints Allocated by Agent", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUC@3px (%)", fontsize=12, fontweight="bold")
    ax.set_title("Trade-off Analysis: Adaptive AUC@3px vs Average Keypoints per Preset", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(600, 1850)
    ax.set_ylim(15, 55)
    
    plt.tight_layout()
    out_path = PLOTS_DIR / "hpatches_accuracy_vs_keypoints.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# PLOT 6: Adaptive vs Fixed-4000 Direct Comparison
# -----------------------------------------------------------------------------
def plot_hpatches_adaptive_vs_fixed4000():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    x = np.arange(len(presets))
    width = 0.35
    
    ax.bar(x - width/2, df_presets["adaptive_auc3"], width, label="Adaptive Level-2 (avg ~1,192 KP)", color=COLOR_ADAPTIVE, edgecolor="#222")
    ax.bar(x + width/2, df_presets["fixed4000_auc3"], width, label="Fixed 4000 (avg ~3,424 KP)", color=COLOR_F4000, edgecolor="#222")
    
    ax.set_ylabel("AUC@3px (%)", fontsize=12, fontweight="bold")
    ax.set_title("Adaptive Level-2 vs Fixed-4000 across all 11 Presets", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(presets, rotation=25, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    
    # Annotate that 10 of 11 presets match or beat Fixed-4000
    ax.text(
        0.03, 0.92,
        "Adaptive matches or outperforms Fixed-4000 on 10/11 presets\nwhile using ~65% fewer keypoints overall.",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#bce8f1")
    )
    
    plt.tight_layout()
    out_path = PLOTS_DIR / "hpatches_adaptive_vs_fixed4000.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# PLOT 7: HPatches AUC Across All Thresholds (AUC@1, AUC@3, AUC@5)
# -----------------------------------------------------------------------------
def plot_hpatches_auc_all_thresholds():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    x = np.arange(len(methods_order))
    width = 0.25
    
    ax.bar(x - width, df_overall["auc_1"], width, label="AUC@1px", color="#4daf4a", edgecolor="#222")
    ax.bar(x,         df_overall["auc_3"], width, label="AUC@3px", color="#377eb8", edgecolor="#222")
    ax.bar(x + width, df_overall["auc_5"], width, label="AUC@5px", color="#984ea3", edgecolor="#222")
    
    ax.set_ylabel("Mean Normalized AUC (%)", fontsize=12, fontweight="bold")
    ax.set_title("HPatches Multi-Threshold Comparison (AUC@1px, AUC@3px, AUC@5px)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_order, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, fontsize=11, loc="upper left")
    
    plt.tight_layout()
    out_path = PLOTS_DIR / "hpatches_auc_all_thresholds.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# PLOT 8: MegaDepth AUC@20 Comparison
# -----------------------------------------------------------------------------
def plot_megadepth_auc20_comparison():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    md_methods = [
        "MegaDepth-Trained\nAdaptive",
        "HPatches Zero-Shot\nAdaptive",
        "Fixed 500",
        "Fixed 1000",
        "Fixed 2000",
        "Fixed 4000"
    ]
    md_auc20 = [26.02, 35.39, 34.13, 44.61, 53.71, 62.08]
    colors = [COLOR_MD_TRAIN, COLOR_HP_ZS, COLOR_F500, COLOR_F1000, COLOR_F2000, COLOR_F4000]
    
    bars = ax.bar(
        md_methods,
        md_auc20,
        color=colors,
        width=0.55,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.set_ylabel("Pose AUC@20° (%)", fontsize=12, fontweight="bold")
    ax.set_title("MegaDepth-1500 Relative Pose Estimation — AUC@20° Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 72)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
    
    ax.text(
        0.04, 0.85,
        "Honest Finding: MegaDepth-trained agent trails HPatches zero-shot policy\ndue to the simple Sampson-distance surrogate reward mismatch.",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="#856404",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba")
    )
    
    plt.tight_layout()
    out_path = PLOTS_DIR / "megadepth_auc20_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    print("Generating final comparison plots...")
    plot_hpatches_auc3_overall()
    plot_hpatches_auc5_overall()
    plot_hpatches_keypoints_overall()
    plot_hpatches_auc3_by_preset()
    plot_hpatches_accuracy_vs_keypoints()
    plot_hpatches_adaptive_vs_fixed4000()
    plot_hpatches_auc_all_thresholds()
    plot_megadepth_auc20_comparison()
    print("All plots generated successfully in experiments/final_report/plots/.")
