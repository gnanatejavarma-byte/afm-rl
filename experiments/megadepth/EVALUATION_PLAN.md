# MegaDepth-1500 Evaluation Plan for AFM-RL Level 2

---

## 1. Overview & Objective

This document defines the evaluation methodology for assessing the **AFM-RL Level-2 Keypoint Agent** on the **MegaDepth-1500** benchmark.

The goal is to test whether the frozen Level-2 RL policy (trained only on HPatches homography pairs) generalizes **zero-shot** to complex 3D outdoor scenes by dynamically tuning keypoint budgets to optimize **relative camera pose estimation accuracy** versus computational cost.

---

## 2. Dataset Structure & Assets

* **Location:** `data/megadepth1500/`
* **Sub-directories & Files:**
  * `pairs.txt`: Plain image pair list (1,500 pairs).
  * `pairs_calibrated.txt`: 1,500 calibrated pairs with 32 fields per line (`img1`, `img2`, `K1` [9], `K2` [9], `R` [9], `T` [3]).
  * `views.txt`: Per-view intrinsic and extrinsics table.
  * `images/`: High-resolution scene images organized by scene ID:
    * `0015/` (St. Peter's / Colosseum / Urban monuments)
    * `0022/` (Piazza Navona / Historic architectural facades)
  * `depths/`: Dense depth maps for 0015 and 0022.

### Are Depth Maps Required for Relative Pose Evaluation?
* **NO.** Relative camera pose evaluation requires only:
  1. Image pairs $(I_1, I_2)$,
  2. Camera intrinsics $(K_1, K_2)$,
  3. Ground-truth relative rotation $R \in SO(3)$ and relative translation direction $T \in \mathbb{R}^3 / \|T\|$.
* All these geometric ground-truth parameters are fully available in `pairs_calibrated.txt`. Depth maps are only needed for dense 3D re-projection or depth-check tasks, which are unnecessary for 2-view relative pose estimation.

---

## 3. Ground-Truth Information & Geometry Formulation

Each line of `pairs_calibrated.txt` provides:
* **Camera Calibration Matrices:**
  $$K_1 = \begin{bmatrix} f_{x1} & 0 & c_{x1} \\ 0 & f_{y1} & c_{y1} \\ 0 & 0 & 1 \end{bmatrix}, \quad K_2 = \begin{bmatrix} f_{x2} & 0 & c_{x2} \\ 0 & f_{y2} & c_{y2} \\ 0 & 0 & 1 \end{bmatrix}$$
* **Ground-Truth Relative Pose:**
  * Rotation matrix $R \in \mathbb{R}^{3\times 3}$ ($I_1 \to I_2$)
  * Translation vector $T \in \mathbb{R}^{3}$ ($I_1 \to I_2$)
* **Ground-Truth Essential Matrix:**
  $$E_{\text{gt}} = [T]_\times R$$

---

## 4. Evaluation Task & Metric Formulation

### 4.1 Evaluation Task: Two-View Relative Pose Estimation
For an image pair $(I_1, I_2)$ and active preset $p$:
1. Run feature extraction and matching to obtain 2D correspondence pairs $(x_1, x_2)$.
2. Normalize points: $\tilde{x}_1 = K_1^{-1} [x_1, 1]^T, \; \tilde{x}_2 = K_2^{-1} [x_2, 1]^T$.
3. Estimate Essential Matrix $E_{\text{est}}$ via 5-point / RANSAC algorithm (`cv2.findEssentialMat`).
4. Recover relative rotation $R_{\text{est}}$ and translation direction $t_{\text{est}}$ via `cv2.recoverPose`.
5. Compute angular rotation error $e_R$ and translation angular error $e_t$:
   $$e_R = \arccos\left(\frac{\operatorname{Tr}(R_{\text{est}}^T R_{\text{gt}}) - 1}{2}\right) \times \frac{180^\circ}{\pi}$$
   $$e_t = \arccos\left(\frac{t_{\text{est}} \cdot T_{\text{gt}}}{\|t_{\text{est}}\| \|T_{\text{gt}}\|}\right) \times \frac{180^\circ}{\pi}$$
6. Combined pose error: $e_{\text{pose}} = \max(e_R, e_t)$.

### 4.2 Benchmark Metrics
* **Pose Error AUC at Thresholds:** Standard MegaDepth AUC at $5^\circ$, $10^\circ$, and $20^\circ$ error thresholds ($\text{AUC@5}^\circ$, $\text{AUC@10}^\circ$, $\text{AUC@20}^\circ$).
* **Pose Accuracy / Success Rate:** Percentage of pairs with $e_{\text{pose}} < 5^\circ$, $< 10^\circ$, and $< 20^\circ$.
* **Keypoint Efficiency:** Mean keypoints used ($\bar{N}$) by Level-2 vs. fixed baselines ($N=1000, 2000, 4000$).
* **Epipolar Sampson Distance & Inlier Ratio:** Mean inlier percentage under RANSAC threshold $\tau = 3.0\text{ px}$.

---

## 5. Zero-Shot Evaluation Feasibility of Level-2

### Feasibility: **HIGH (100% Zero-Shot Compatible)**
The frozen Level-2 policy in `checkpoints/level2_deltaobs_best/best_model.zip` expects a **31-dimensional observation vector**:
1. **9 Visual Pair Statistics:** Computed directly from raw image pairs $(I_1, I_2)$ using `src/image_stats.py` (entropy, blur, corner density, gradients, intensity shift).
2. **11 Preset One-Hot Indicator:** Identifies the active pipeline.
3. **11 Runtime Indicators:** In MegaDepth, during an evaluation rollout, runtime accuracy is evaluated using epipolar/pose reward or RANSAC inlier ratio, and Level-2 selects actions (`+500`, `+1000`, `-500`, `STOP`) until termination.

---

## 6. What Can Be Reused from AFM-RL

| Module | Reusable Elements | Notes |
| :--- | :--- | :--- |
| `src/pipeline_utils.py` | All 11 Presets, `run_pipeline()`, caching, detectors/descriptors | 100% Reusable |
| `src/image_stats.py` | `compute_pair_stats()`, `normalize_stats()`, `N_STATS=9` | 100% Reusable |
| `src/cv_compat.py` | OpenCV factory wrappers | 100% Reusable |
| `checkpoints/level2_deltaobs_best/best_model.zip` | Frozen PPO policy weights | Used directly for zero-shot inference |

---

## 7. New Evaluation Code to Create (Under `experiments/megadepth/`)

1. `experiments/megadepth/dataset.py`: Lightweight parser for `pairs_calibrated.txt` that yields $(I_1, I_2, K_1, K_2, R, T)$.
2. `experiments/megadepth/pose_metrics.py`: Computes Essential Matrix, recovers relative pose $(R, t)$, calculates angular errors $(e_R, e_t)$, and computes AUC@$5^\circ, 10^\circ, 20^\circ$.
3. `experiments/megadepth/evaluate_megadepth.py`: Runs benchmark comparison:
   * Fixed keypoint baselines ($N=500, 1000, 2000, 4000$) across all 11 presets.
   * Frozen AFM-RL Level-2 adaptive policy across all 11 presets.
   * Outputs summary table, CSV records under `experiments/megadepth/results/`, and keypoint savings breakdown.

---

## 8. Protected Files (DO NOT MODIFY)

* `src/level2_env.py`
* `src/metrics.py`
* `src/hpatches.py`
* `scripts/train_level2.py`
* `checkpoints/` and `logs/`
* `README.md`
