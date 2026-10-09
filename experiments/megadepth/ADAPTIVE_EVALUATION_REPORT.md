# MegaDepth-1500 Full Adaptive Level-2 RL Evaluation Report

- **Evaluated Pairs:** 1500 calibrated MegaDepth pairs
- **Presets Evaluated:** 11 detector/descriptor pipelines
- **Total Adaptive Episodes:** 16,500
- **Total Elapsed Time:** 1311.3s (21.9 minutes)
- **RL Checkpoint:** `checkpoints/level2_deltaobs_best/best_model.zip` (Trained on HPatches, evaluated zero-shot)
- **Observation Dimensions:** 31 (9 pair visual stats, 11 preset one-hot, 11 runtime state variables)

## 1. Zero-Shot Transfer & Surrogate Observation Methodology

The Level-2 RL agent was trained entirely on the HPatches planar homography dataset.
When deployed zero-shot to MegaDepth non-planar, multi-view camera scenes:
1. **Visual Statistics (9 dims)** & **Preset One-Hot (11 dims)**: Transferred 100% identically without modification.
2. **Runtime Accuracy Signal (`last_acc`)**: HPatches uses corner error with ground-truth homography: $\exp(-\text{corner\_err} / 5.0)$.
   In MegaDepth, relative pose has no ground-truth homography. A geometrically matched surrogate using Fundamental Matrix RANSAC mean Sampson distance was used:
   $$\text{surrogate\_acc} = \exp(-\text{mean\_sampson\_err} / 5.0)$$
   This preserves the identical $[0, 1]$ numerical range and smooth falloff expected by the policy's value and actor heads.

## 2. Adaptive Policy Performance Summary

| Preset | Avg KP | Med KP | Avg Steps | Valid Pose | Mean Err (°) | Med Err (°) | @5° (%) | @10° (%) | @20° (%) | AUC@5 | AUC@10 | AUC@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ORB | 901 | 500 | 1.41 | 1500/1500 | 69.33 | 50.88 | 2.47 | 6.27 | 18.00 | 1.00 | 2.78 | 7.52 |
| SIFT | 542 | 500 | 1.08 | 1500/1500 | 44.31 | 21.87 | 18.80 | 31.40 | 47.93 | 9.92 | 17.63 | 28.93 |
| AKAZE | 536 | 500 | 1.07 | 1500/1500 | 42.97 | 23.32 | 13.40 | 26.53 | 45.13 | 6.78 | 13.53 | 24.94 |
| BRISK | 752 | 500 | 1.35 | 1496/1500 | 52.38 | 36.75 | 8.07 | 16.53 | 32.07 | 3.74 | 7.92 | 16.00 |
| FAST+BRIEF | 514 | 500 | 1.03 | 1500/1500 | 63.13 | 39.42 | 13.67 | 22.40 | 35.07 | 7.18 | 12.74 | 20.94 |
| KAZE | 647 | 500 | 1.30 | 1487/1500 | 44.91 | 25.45 | 12.73 | 25.27 | 43.27 | 6.40 | 12.87 | 23.85 |
| GFTT+BRIEF | 522 | 500 | 1.03 | 1500/1500 | 64.56 | 39.32 | 12.87 | 22.73 | 35.13 | 6.25 | 12.14 | 20.82 |
| GFTT+SIFT | 568 | 500 | 1.14 | 1499/1500 | 51.75 | 24.40 | 21.27 | 32.73 | 46.13 | 11.03 | 19.29 | 29.84 |
| STAR+BRIEF | 494 | 500 | 1.01 | 1500/1500 | 67.57 | 40.58 | 12.33 | 21.60 | 33.00 | 5.76 | 11.54 | 19.39 |
| ORB+FREAK | 818 | 500 | 2.12 | 1330/1500 | 66.79 | 48.54 | 2.53 | 6.87 | 17.13 | 0.93 | 2.68 | 7.31 |
| FAST+FREAK | 516 | 500 | 1.03 | 1500/1500 | 70.10 | 47.79 | 12.20 | 20.40 | 30.87 | 6.81 | 11.78 | 18.99 |

## 3. Comparison with Fixed-Budget Baseline (500, 1000, 2000, 4000 KP)

| Preset | Adaptive KP | Adaptive @20° | Best Fixed Budget | Best Fixed @20° | Gap vs Best @20° | Fixed-4000 @20° | Gap vs Fixed-4000 | Fixed-500 @20° | Gap vs Fixed-500 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ORB | 901 | 18.00% | 4000 | 47.40% | -29.40% | 47.40% | -29.40% | 14.20% | +3.80% |
| SIFT | 542 | 47.93% | 4000 | 82.40% | -34.47% | 82.40% | -34.47% | 47.00% | +0.93% |
| AKAZE | 536 | 45.13% | 4000 | 76.93% | -31.80% | 76.93% | -31.80% | 44.53% | +0.60% |
| BRISK | 752 | 32.07% | 4000 | 70.73% | -38.66% | 70.73% | -38.66% | 27.07% | +5.00% |
| FAST+BRIEF | 514 | 35.07% | 4000 | 56.53% | -21.46% | 56.53% | -21.46% | 34.93% | +0.14% |
| KAZE | 647 | 43.27% | 4000 | 70.67% | -27.40% | 70.67% | -27.40% | 40.33% | +2.94% |
| GFTT+BRIEF | 522 | 35.13% | 4000 | 56.53% | -21.40% | 56.53% | -21.40% | 35.07% | +0.06% |
| GFTT+SIFT | 568 | 46.13% | 4000 | 65.20% | -19.07% | 65.20% | -19.07% | 45.07% | +1.06% |
| STAR+BRIEF | 494 | 33.00% | 4000 | 49.20% | -16.20% | 49.20% | -16.20% | 32.93% | +0.07% |
| ORB+FREAK | 818 | 17.13% | 4000 | 37.00% | -19.87% | 37.00% | -19.87% | 14.07% | +3.06% |
| FAST+FREAK | 516 | 30.87% | 4000 | 51.47% | -20.60% | 51.47% | -20.60% | 30.60% | +0.27% |

## 4. Key Diagnostic Observations & Systematic Early Stopping Analysis

1. **Policy Stopping Behavior:** The agent took an average of **1.23 steps** per episode across all presets and stayed at an average budget of **619 keypoints** (starting from $N_{start}=500$).
2. **Early Stopping Mechanism:** On HPatches (planar scenes with uniform illumination and high overlap), 500–1000 keypoints was typically sufficient to achieve near-perfect corner accuracy. Because the HPatches cost penalty $\lambda=0.1$ penalizes larger keypoint counts ($-\lambda \cdot N / 4000$), the agent learned a conservative stopping policy on HPatches.
3. **Domain Shift to MegaDepth (3D Extreme Viewpoint & Scale Changes):** In 3D wide-baseline outdoor environments like MegaDepth, matching is significantly harder and benefits monotonically from maximal keypoints (4000 KP achieves the highest accuracy for every single preset).
4. **Adaptive vs Fixed-500 Baseline:** When comparing the adaptive agent against the equivalent fixed budget (500 KP), the adaptive agent achieves comparable or slightly improved accuracy due to selective budget increments on pairs where initial inliers were sparse.
5. **Implication for Level 1 / Fine-Tuning:** The Level-2 policy's zero-shot behavior faithfully reflects its HPatches training objective. For optimal performance on outdoor 3D datasets, either multi-dataset training or domain-adapted utility functions are required.
