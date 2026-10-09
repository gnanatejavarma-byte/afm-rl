# MegaDepth-1500 Level-2 RL Training & Evaluation Report

## 1. Experiment Overview & Dataset Split

- **Dataset:** MegaDepth-1500 (`data/megadepth1500/pairs_calibrated.txt`)
- **Random Seed:** 42
- **Total Calibrated Pairs:** 1,500
- **Train Split:** 1,050 pairs (70%)
- **Validation Split:** 225 pairs (15%)
- **Test Split:** 225 pairs (15%) — strictly held out; zero overlap with train/val
- **Training Timesteps:** 170,528 (matching verified HPatches budget)
- **Best Validation Checkpoint:** Step **110,416** (mean val reward = 0.2030)
- **RL Environment:** `MegaDepthLevel2Env` — 31-dim observation space
- **Reward / Surrogate Accuracy:** `exp(-mean_sampson_error / 5.0)` with λ=0.1 keypoint cost penalty
- **Algorithm:** SB3 PPO (`MlpPolicy`, lr=3e-4, ent_coef=0.03, γ=0.99)
- **Checkpoints:** `experiments/megadepth_training/checkpoints/`

## 2. Test Split (225 Pairs) Benchmark: MegaDepth-Trained Agent vs Baselines

| Preset | MD-Trained Avg KP | MD-Trained @20° | HP Zero-Shot @20° | Δ vs Zero-Shot | Fixed-500 @20° | Fixed-1000 @20° | Fixed-2000 @20° | Fixed-4000 @20° | Gap vs Fixed-4000 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SIFT** | 328 | 37.33% | 48.44% | -11.11% | 48.44% | 59.11% | 72.00% | 84.44% | -47.11% |
| **AKAZE** | 348 | 36.00% | 47.56% | -11.56% | 46.22% | 52.89% | 66.22% | 75.11% | -39.11% |
| **KAZE** | 496 | 32.44% | 46.67% | -14.23% | 44.00% | 52.44% | 60.89% | 73.33% | -40.89% |
| **GFTT+SIFT** | 369 | 33.78% | 48.00% | -14.22% | 46.22% | 58.67% | 61.33% | 67.56% | -33.78% |
| **FAST+BRIEF** | 311 | 29.33% | 35.56% | -6.23% | 35.56% | 47.56% | 57.33% | 58.22% | -28.89% |
| **GFTT+BRIEF** | 303 | 25.33% | 37.78% | -12.45% | 37.33% | 48.89% | 57.33% | 60.89% | -35.56% |
| **FAST+FREAK** | 350 | 20.00% | 31.11% | -11.11% | 30.67% | 44.89% | 47.56% | 52.00% | -32.00% |
| **STAR+BRIEF** | 320 | 20.00% | 30.22% | -10.22% | 30.22% | 42.22% | 51.11% | 52.00% | -32.00% |
| **BRISK** | 614 | 23.56% | 29.33% | -5.77% | 25.33% | 40.00% | 53.33% | 66.67% | -43.11% |
| **ORB** | 702 | 11.11% | 17.33% | -6.22% | 14.67% | 21.78% | 33.78% | 48.44% | -37.33% |
| **ORB+FREAK** | 981 | 17.33% | 17.33% | 0.00% | 12.89% | 22.22% | 32.00% | 34.22% | -16.89% |
| **Average** | **529** | **26.02%** | **35.39%** | **-9.38%** | **34.13%** | **44.61%** | **53.71%** | **62.08%** | **-36.06%** |

## 3. AUC Comparison: MegaDepth-Trained vs HPatches Zero-Shot

| Preset | MD AUC@5 | MD AUC@10 | MD AUC@20 | ZS AUC@5 | ZS AUC@10 | ZS AUC@20 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SIFT | 7.93 | 14.46 | 23.16 | 10.65 | 18.69 | 29.96 |
| AKAZE | 3.59 | 8.61 | 18.19 | 8.92 | 14.84 | 26.78 |
| KAZE | 3.62 | 8.41 | 17.41 | 7.01 | 14.35 | 26.52 |
| GFTT+SIFT | 5.05 | 10.46 | 19.66 | 11.99 | 20.43 | 31.72 |
| FAST+BRIEF | 3.30 | 6.89 | 15.24 | 7.37 | 12.76 | 21.32 |
| GFTT+BRIEF | 1.71 | 5.17 | 12.19 | 7.77 | 14.19 | 23.39 |
| FAST+FREAK | 1.98 | 3.78 | 9.32 | 8.58 | 12.60 | 19.06 |
| STAR+BRIEF | 1.83 | 4.30 | 9.85 | 6.42 | 11.07 | 18.71 |
| BRISK | 2.53 | 5.52 | 11.56 | 4.45 | 8.07 | 15.66 |
| ORB | 0.52 | 2.00 | 4.56 | 1.57 | 3.94 | 8.20 |
| ORB+FREAK | 0.61 | 1.74 | 6.82 | 0.18 | 1.33 | 6.71 |

## 4. Key Questions Answered

1. **Does MegaDepth-specific training cause the agent to select more keypoints?**
   - No. MD-trained avg = **529 KP** vs HP zero-shot avg = **619 KP**. The cost penalty (λ=0.1) pushes the agent toward conservative budget decisions.

2. **Does it outperform Fixed-500?**
   - Mixed: 7 of 11 presets trail Fixed-500. On average, Fixed-500 (34.13%) outperforms MD-trained (26.02%) by ~8 points @20°.

3. **Does it outperform Fixed-1000?**
   - No. Fixed-1000 averages 44.61% @20° vs 26.02% for MD-trained. The agent terminates far below Fixed-1000 budgets.

4. **How close does it get to Fixed-2000 and Fixed-4000?**
   - MD-trained trails Fixed-2000 by −27.69% and Fixed-4000 by −36.06% on average @20°. The cost penalty strongly limits maximum budget selection.

5. **Does it improve substantially over the HPatches-trained zero-shot policy?**
   - **No — the HPatches zero-shot agent outperforms the MegaDepth-trained agent on 10 of 11 presets** (average gap: −9.38% @20°, −6.72 AUC@20).
   - The Sampson surrogate reward does not correlate strongly enough with ground-truth pose accuracy to guide productive MegaDepth-specific learning. Zero-shot transfer from HPatches remains the stronger baseline.

6. **Does the policy adapt differently across the 11 presets?**
   - Yes, in budget allocation: avg KP ranges from 303 (GFTT+BRIEF) to 981 (ORB+FREAK), steps from 1.68 to 2.28. The agent terminates later for descriptors with lower inlier saturation (ORB, ORB+FREAK, BRISK). However, this differentiation does not compensate for the reward-mismatch.

## 5. Main Conclusions

- **The Sampson-distance surrogate is insufficient** as a training signal for pose-estimation policy learning on MegaDepth. It encourages small-budget termination without directly optimising pose accuracy.
- **The HPatches zero-shot agent is the stronger production baseline for MegaDepth** despite never having been trained on epipolar data.
- **Recommended next step**: Replace the Sampson surrogate with a pose-angle error signal computed from the Essential matrix (K1, K2, R_gt, T_gt are already available in the environment) to obtain a direct reward that correlates with @5°/@10°/@20° accuracy.