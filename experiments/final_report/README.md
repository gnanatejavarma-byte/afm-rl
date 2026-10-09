# AFM-RL Final Documentation and Presentation Package

This directory contains the final documentation, presentation deck, verified comparison plots, and reproduction scripts for the **Adaptive Feature Matching using Hierarchical Reinforcement Learning (AFM-RL)** project.

---

## Directory Contents

```
experiments/final_report/
│
├── FINAL_AFM_RL_REPORT.md             # Complete 17-section final report in simple English
├── AFM_RL_Final_Presentation.pptx     # 16-slide widescreen PowerPoint presentation
├── README.md                          # This documentation file
│
├── generate_plots.py                  # Script to regenerate all 8 comparison plots
├── create_presentation.py             # Script to regenerate the PowerPoint presentation
│
└── plots/
    ├── hpatches_auc3_overall.png          # Overall HPatches AUC@3px comparison (5 methods)
    ├── hpatches_auc5_overall.png          # Overall HPatches AUC@5px comparison (5 methods)
    ├── hpatches_keypoints_overall.png     # Keypoint budget comparison (showing 65% reduction)
    ├── hpatches_auc3_by_preset.png        # AUC@3px across all 11 presets (grouped bars)
    ├── hpatches_accuracy_vs_keypoints.png # Scatter plot of AUC@3 vs keypoints per preset
    ├── hpatches_adaptive_vs_fixed4000.png # Direct comparison of Adaptive vs Fixed-4000
    ├── hpatches_auc_all_thresholds.png    # Multi-threshold comparison (AUC@1, @3, @5px)
    └── megadepth_auc20_comparison.png     # MegaDepth-1500 AUC@20° comparison
```

---

## Key Experimental Results Summary

### 1. HPatches Homography Benchmark (85 Held-Out Test Pairs)
- **Adaptive Level-2 RL:** **38.30% AUC@3px**, **51.97% AUC@5px**, **1,191.7 Average Keypoints**.
- **Fixed 4000 Baseline:** **36.34% AUC@3px**, **47.09% AUC@5px**, **3,424.4 Average Keypoints**.
- **Key Takeaway:** The Adaptive Level-2 RL agent **beats the brute-force 4,000-keypoint baseline (+1.96% AUC@3px)** while using **over 65% fewer keypoints** on average (1,192 vs 3,424 points).
- **Beats Fixed 500 on all 11 presets** (+11.41 percentage points on average).
- **Beats or matches Fixed 4000 on 10 of 11 presets**.
- **Best Preset:** SIFT (50.03% AUC@3px, 942 keypoints avg).
- **Weakest Preset:** ORB+FREAK (21.37% AUC@3px, 1,235 keypoints avg).

### 2. MegaDepth-1500 3D Pose Benchmark (225 Held-Out Test Pairs)
- **MegaDepth-Trained Adaptive:** **26.02% AUC@20°** (529 Average Keypoints).
- **HPatches Zero-Shot Adaptive:** **35.39% AUC@20°** (619 Average Keypoints).
- **Fixed 4000 Baseline:** **62.08% AUC@20°**.
- **Honest Finding:** The MegaDepth-specific agent became overly conservative due to the uncalibrated Sampson-distance surrogate reward, demonstrating that future 3D RL models need direct Essential matrix pose rewards.

---

## How to Reproduce

### Regenerate All Comparison Plots:
```bash
python experiments/final_report/generate_plots.py
```

### Regenerate the PowerPoint Presentation:
```bash
python experiments/final_report/create_presentation.py
```

### Re-run HPatches Benchmark Evaluation:
```bash
python experiments/hpatches_ripe_eval/evaluate_hpatches_auc.py --split test
```
