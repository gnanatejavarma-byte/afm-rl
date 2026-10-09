# Aachen Day-Night v1.1 Evaluation Plan for AFM-RL Level 2

---

## 1. Overview & Objective

This document defines the evaluation methodology for assessing the **AFM-RL Level-2 Keypoint Agent** on the **Aachen Day-Night v1.1** benchmark.

The goal is to evaluate whether the frozen Level-2 RL policy dynamically adapts its keypoint budget to overcome severe **day-to-night illumination changes, glare, and dynamic shadows** in long-term visual localization tasks without needing fine-tuning or retraining.

---

## 2. Dataset Structure & Assets

* **Location:** `data/aachen_v1_1/`
* **Sub-directories & Files:**
  * `3D-models/aachen_v_1_1/`:
    * `cameras.bin`: COLMAP camera calibration models for database images.
    * `images.bin`: COLMAP registered database poses and 2D keypoint projections.
    * `points3D.bin`: 3D triangulated point cloud of Aachen old town.
    * `database_intrinsics_v1_1.txt`: Plaintext list of reference image camera models and focal parameters.
    * `aachen_v_1_1.nvm`: VisualSFM export of the 3D model.
  * `images_upright/`:
    * `query/night/`: High-resolution nighttime query photos captured across multiple mobile phone sensors (`nexus5x/`, `nexus4/`).
    * `sequences/`: Daytime reference video frames (`gopro3_undistorted/`, `nexus4_sequences/`).
  * `queries/`:
    * `night_time_queries_with_intrinsics.txt`: Query image filenames, camera model (`SIMPLE_RADIAL`), dimensions, and intrinsic parameters ($f, c_x, c_y, k$).

---

## 3. Ground-Truth Information & Geometry Formulation

* **Database Images:** Full 6-DoF poses $(R_{\text{db}}, t_{\text{db}})$ and calibrated intrinsics $K_{\text{db}}$ from the COLMAP reconstruction model.
* **Query Images:** Calibrated camera intrinsics $K_q$ with radial distortion coefficients provided in `night_time_queries_with_intrinsics.txt`.
* **Ground-Truth 3D Structure:** Over 240 MB of triangulated 3D scene points (`points3D.bin`) linking daytime database keypoints to global $(X, Y, Z)$ coordinates.

---

## 4. Evaluation Tasks & Benchmark Metrics

### 4.1 Evaluation Task: Visual Localization & Feature Matching Under Day-Night Shifts
1. **2-View Pairwise Matching:** Pair nighttime queries with corresponding daytime database images covering the same spatial views.
2. **Keypoint Budget Allocation:**
   * Run fixed-budget baselines ($N = 500, 1000, 2000, 4000$) across all 11 presets.
   * Run AFM-RL Level-2 agent to dynamically allocate keypoints.
3. **6-DoF Pose Estimation (PnP + RANSAC):**
   * Triangulate 2D query keypoints to 3D model points.
   * Solve Perspective-n-Point (PnP + RANSAC) to recover query camera pose $(R_q, t_q)$.

### 4.2 Standard Visual Localization Metrics
* **Pose Recall at Standard Thresholds:** Percentage of successfully localized night queries within:
  * **High precision:** $(0.25\text{m}, 2^\circ)$
  * **Medium precision:** $(0.50\text{m}, 5^\circ)$
  * **Coarse precision:** $(5.00\text{m}, 10^\circ)$
* **Matching Quality Under Darkness:**
  * Mean RANSAC inlier count and inlier ratio $\rho_{\text{inlier}} = \frac{N_{\text{inliers}}}{N_{\text{matches}}}$.
  * Epipolar Sampson distance across day-night pairs.
* **Keypoint Budget Efficiency:**
  * Mean keypoint count allocated on difficult night scenes vs daytime references.
  * Demonstrating whether AFM-RL automatically scales up keypoint density for night scenes while remaining frugal on well-lit scenes.

---

## 5. Zero-Shot Evaluation Feasibility of Level-2

### Feasibility: **HIGH (100% Zero-Shot Compatible)**
* The 31-dimensional observation vector uses:
  * **9 Visual Statistics:** Computed directly from the night query vs daytime reference image pair (e.g. large mean intensity difference $|\bar{I}_1 - \bar{I}_2|$ and Laplacian blur differences).
  * **11 One-Hot Presets:** Represents the active detector/descriptor.
  * **11 Runtime Indicators:** Dynamic step count, RANSAC inlier ratio, and match metrics.
* The frozen agent will receive strong visual darkness signals (from the 9 pair statistics) and can immediately execute sequential budgeting (`+500`, `+1000`, `-500`, `STOP`) without modifying model weights.

---

## 6. What Can Be Reused from AFM-RL

| Module | Reusable Elements | Notes |
| :--- | :--- | :--- |
| `src/pipeline_utils.py` | All 11 Presets, `run_pipeline()`, caching, detectors/descriptors | 100% Reusable |
| `src/image_stats.py` | `compute_pair_stats()`, `normalize_stats()`, `N_STATS=9` | 100% Reusable |
| `src/cv_compat.py` | OpenCV factory wrappers | 100% Reusable |
| `checkpoints/level2_deltaobs_best/best_model.zip` | Frozen PPO policy weights | Direct zero-shot execution |

---

## 7. New Evaluation Code to Create (Under `experiments/aachen/`)

1. `experiments/aachen/dataset.py`: Indexer that reads query intrinsics (`night_time_queries_with_intrinsics.txt`), loads night queries, and pairs them with spatially aligned daytime sequence frames.
2. `experiments/aachen/localization_metrics.py`: Computes 2D-2D epipolar metrics, RANSAC inlier ratios, and PnP pose errors.
3. `experiments/aachen/evaluate_aachen.py`: Benchmark runner executing:
   * Fixed keypoint baselines across all 11 presets.
   * Frozen AFM-RL Level-2 adaptive agent.
   * Generates summary tables and saves results to `experiments/aachen/results/`.

---

## 8. Protected Files (DO NOT MODIFY)

* `src/level2_env.py`
* `src/metrics.py`
* `src/hpatches.py`
* `scripts/train_level2.py`
* `checkpoints/` and `logs/`
* `README.md`
