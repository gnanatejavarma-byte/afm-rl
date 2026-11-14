# PROTOCOL v2 AMENDMENT 1: EXECUTION STAGING, PRIORITY SCHEDULING, AND MEGADEPTH BENCHMARK SPECIFICATIONS

**Pre-registration Status:** LOCKED & BINDING  
**Date:** November 14, 2026  
**Supercedes / Amends:** `results/phase4/PROTOCOL_v2_FINAL.md`  

---

## 1. Context and Objective

This amendment establishes the operational staging, priority execution order, empirical threshold grid calibration, and hardware constraints for the zero-shot confirmation evaluation on the **MegaDepth-1500** benchmark.

All analyses on HPatches (Phases 1–4b) are designated as **exploratory / diagnostic**. The MegaDepth-1500 evaluation represents the **primary confirmatory benchmark** of label-free adaptive keypoint budgeting.

---

## 2. Staged Execution Architecture

To guarantee strict completion within the 12-hour compute envelope while ensuring critical comparisons are evaluated first, execution is partitioned into four sequential stages:

### Stage 1: Complete Feature Pool Pre-Extraction (Mandatory Pre-requisite)
* **Scope:** All 14 presets across all 806 unique images in the MegaDepth-1500 test set ($806 \times 14 = 11,284$ image pools).
* **Storage:** Cached to disk under `data/pool_cache_megadepth/` in compressed `.npz` format.
* **Guarantee:** Keypoints and descriptors are extracted once per image at maximum budget ($N=4000$) and reused across all pair evaluations. For presets where budget-specific extraction differs from slicing (e.g., ORB octree clustering and border-trimmed descriptors), separate per-$N$ pools are maintained.

### Stage 2a: Priority Fixed Baselines (14 Core Configurations)
* **Scope:** Evaluated on all 1,500 pairs of MegaDepth-1500 ($1,500 \times 14 = 21,000$ evaluations).
* **Configurations:**
  1. `SIFT` @ $N \in \{500, 1000, 2000, 4000\}$ (4 configs)
  2. `ALIKED` @ $N \in \{500, 1000, 2000, 4000\}$ (4 configs)
  3. `XFeat` @ $N \in \{500, 1000, 2000, 4000\}$ (4 configs)
  4. `ORB` @ $N \in \{1000, 4000\}$ (2 configs)
* **Priority:** Highest. Directly covers primary pre-registered comparisons P1, P2, P3, and P4.

### Stage 3: AFM-RL Adaptive Rollouts (5 Fold Models)
* **Scope:** 5 fold-specific models evaluated on all 1,500 pairs ($1,500 \times 5 = 7,500$ rollouts).
* **Protocol:** Strictly label-free execution in `obs_mode="proxy"` with runtime guard forbidding access to ground-truth relative pose $\mathbf{T}_{\text{gt}}$.
* **Outputs:** Rollout trajectory, cumulative keypoints, proxy stopping decisions, and post-hoc pose estimation error ($R_{\text{err}}, t_{\text{err}}$).

### Stage 2b: Remaining Secondary Baselines (42 Configurations)
* **Scope:** The remaining 42 preset-budget combinations ($1,500 \times 42 = 63,000$ evaluations).
* **Execution Status:** Resumable, append-safe, background execution. Can be gracefully terminated once Stages 1, 2a, and 3 are complete without compromising core protocol findings.

---

## 3. Empirical Escalation Threshold Grid ($\tau$)

In HPatches, the homography inlier proxy saturated differently than the epipolar geometry proxy on MegaDepth. On MegaDepth:
* The label-free epipolar proxy $p_{\text{epi}}$ is defined by Eq. 7:
  $$p_{\text{epi}} = \min\left(1.0, \frac{n_{\text{inliers}}}{120}\right) \times \left( 0.6 \cdot \text{ratio}_{\text{inliers}} + 0.4 \cdot \exp\left(-\frac{\text{sampson}_{\text{px}}}{2.0}\right) \right)$$
* The escalation grid $\tau$ for cascade execution is calibrated to the 5th through 95th percentiles (in 5% increments) of the empirical distribution of $p_{\text{epi}}$ measured across the fitting scene:
  $$\tau \in \{ Q_{0.05}, Q_{0.10}, Q_{0.15}, \dots, Q_{0.95} \}$$
* All grid decisions and model selection criteria remain identical to Protocol v2 Final.

---

## 4. Benchmark Specifications & Out-of-Scope Methods

1. **Image Resolution:**
   * Images are loaded with maximum dimension clamped to $1200$ pixels:
     $$s = \min\left(1.0, \frac{1200}{\max(H, W)}\right)$$
   * Intrinsics $\mathbf{K}$ are scaled proportionally ($K_{00}, K_{11}, K_{02}, K_{12} \leftarrow s \cdot K$).
   * *Notice:* Benchmark results are internally consistent across all methods evaluated herein, but raw AUC metrics are not directly comparable to published literature using $840$ px (LoFTR) or $1600$ px (SuperGlue).

2. **Feature Matching:**
   * Keypoints matched using standard mutual nearest-neighbor or OpenCV `BFMatcher` with Lowe's ratio test ($0.75$ for float descriptors, $0.80$ for binary).
   * RANSAC pose estimation via `cv2.findEssentialMat` (prob=0.99999, threshold=0.5 px) followed by `cv2.recoverPose`.

3. **Out-of-Scope Methods:**
   * **LightGlue** is not installed in the local environment and is strictly out of scope.
   * No learned matchers or proprietary backends are injected.

4. **Match Archival & Reproducibility:**
   * All matched keypoint coordinate pairs $(p_0, p_1)$ in their original matcher order are compressed and archived to `results/phase4b/megadepth_matches/` as `.npz` records for complete post-run verification.

