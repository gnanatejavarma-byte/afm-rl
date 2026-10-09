# HPatches Homography Estimation AUC Evaluation (RIPE Protocol)

This experiment evaluates the **original HPatches-trained Level-2 RL Adaptive Agent** against **fixed keypoint budgets** (500, 1000, 2000, 4000) using the RIPE-style homography metric (AUC@1px, AUC@3px, AUC@5px) across all 11 feature matching presets.

---

## Evaluation Protocol

### 1. Dataset & Split
- **Dataset:** HPatches (116 sequences, 580 image pairs: reference image 1 vs images 2..6)
- **Split Configuration (`configs/split.json`):**
  - **Train:** 81 sequences (40 illumination, 41 viewpoint) = 405 pairs
  - **Validation:** 18 sequences (9 illumination, 9 viewpoint) = 90 pairs
  - **Test (Held-Out):** 17 sequences (8 illumination, 9 viewpoint) = 85 pairs

### 2. Evaluated Methods
1. **Adaptive Level-2 RL:** Original frozen checkpoint (`checkpoints/level2_deltaobs_best/best_model.zip`, 31-dim observation space, adaptive budget control).
2. **Fixed 500:** Constant budget of 500 keypoints.
3. **Fixed 1000:** Constant budget of 1,000 keypoints.
4. **Fixed 2000:** Constant budget of 2,000 keypoints.
5. **Fixed 4000:** Constant budget of 4,000 keypoints.

### 3. Presets (All 11 Pipelines)
- `ORB`, `SIFT`, `AKAZE`, `BRISK`, `FAST+BRIEF`, `KAZE`, `GFTT+BRIEF`, `GFTT+SIFT`, `STAR+BRIEF`, `ORB+FREAK`, `FAST+FREAK`

### 4. Metrics (RIPE Benchmark Style)
- **Corner Reprojection Error:** Mean Euclidean distance (pixels) between 4 image corners projected with ground-truth $H_{gt}$ vs estimated $\hat{H}$ (via OpenCV RANSAC on matched keypoints).
- **AUC@1px, AUC@3px, AUC@5px:** Area Under the Cumulative Success Curve normalized over thresholds of 1, 3, and 5 pixels via trapezoidal integration.
- **Success@1px, Success@3px, Success@5px:** Percentage of image pairs with mean corner error below 1px, 3px, and 5px.
- **Average Keypoints & Steps:** Mean effective keypoints utilized and episode steps taken.

---

## Reproducing the Evaluation

To run the evaluation on the held-out test split:
```bash
python experiments/hpatches_ripe_eval/evaluate_hpatches_auc.py --split test
```

To run on the validation split:
```bash
python experiments/hpatches_ripe_eval/evaluate_hpatches_auc.py --split val
```

---

## Output Files

- `results/hpatches_auc_results.csv`: Complete aggregated metrics per preset and method.
- `results/hpatches_auc_summary.md`: Detailed formatted markdown report with tables and comparison analysis.
- `results/raw_evaluations_test.csv`: Per-pair raw matching and corner error logs.
