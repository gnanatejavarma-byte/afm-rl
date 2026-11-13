# PROTOCOL v2 FINAL & MEGADEPTH BENCHMARK ADDENDUM

**Status:** LOCKED & PRE-REGISTERED  
**Date:** November 14, 2026  
**Scope:** Label-Free Cascaded Architecture (v2), Zero-Shot Transfer to MegaDepth-1500, and Rigorous Statistical Harness  

---

## 1. Provenance, Methodological Separation, and Scientific Status

* **AFM-RL v1 (Hierarchical Bandit/PPO):**  
  AFM-RL v1 uses a 9-feature contextual bandit at Level 1 to choose among 14 feature presets, and a 4-action PPO agent at Level 2 to iteratively budget keypoints. Models are trained exclusively on HPatches 5-fold cross-validation (`checkpoints_v2/fold0..4`).
* **Exploratory Status of HPatches v2 Analysis:**  
  The candidate cascade architectures ($C_1$--$C_6$) and escalation thresholds $\tau$ of AFM-RL v2 were explicitly formulated and refined using diagnostic headroom and noise-floor analyses conducted on the 116 sequences of HPatches. Therefore, **all evaluations of AFM-RL v2 on HPatches are classified as post-hoc / exploratory**.
* **Confirmatory Status of MegaDepth-1500:**  
  The official 1,500-pair MegaDepth wide-baseline benchmark serves as the **true, unbiased confirmation benchmark**. No policy parameters, cascade orderings, or thresholds are fit on MegaDepth data, and no agent is trained on MegaDepth.
* **Discarded Legacy Artifacts:**  
  The legacy "225-pair MegaDepth validation split" (generated via random index shuffle without scene disjointness) and any "MegaDepth-trained" RL agent are completely discarded and must not be cited or used.

---

## 2. AFM-RL v2 Cascade Definitions & Execution Rule

### Fixed Candidate Set ($C_1$--$C_6$)
To strictly constrain model selection, the candidate space is restricted to exactly six pre-registered cascades:
1. **$C_1$ (Modern-to-Classical Standard):** `[ALIKED-500 -> SIFT-2000 -> SIFT-4000]`  
   *Attempts efficient modern learned keypoints first; escalates to robust dense classical keypoints if confidence is low.*
2. **$C_2$ (Learned Escalation then Classical Fallback):** `[ALIKED-500 -> ALIKED-1000 -> SIFT-4000]`  
   *Doubles modern keypoint capacity before falling back to maximum classical budget.*
3. **$C_3$ (Pure Classical Multi-Scale):** `[SIFT-1000 -> SIFT-2000 -> SIFT-4000]`  
   *Progressive classical budget expansion.*
4. **$C_4$ (Modern Exploration):** `[ALIKED-500 -> SIFT-1000 -> ALIKED-2000]`  
   *Interleaves modern and classical feature extractors.*
5. **$C_5$ (Conservative Classical):** `[SIFT-500 -> SIFT-2000 -> SIFT-4000]`  
   *Low-floor classical matcher with progressive escalation.*
6. **$C_6$ (Fast Two-Stage Cascade):** `[ALIKED-500 -> SIFT-4000]`  
   *Bimodal binary decision: rapid pass or maximal rescue.*

### Escalation & Stopping Criterion
For a chosen cascade $C = ((P_1, N_1), \dots, (P_M, N_M))$ and escalation threshold $\tau \in [0, 1]$:
1. Execute stage $m=1$: extract $(P_1, N_1)$, match, and compute the label-free proxy $p^{(1)}$.
2. If $p^{(1)} \ge \tau$ or $m = M$ (final stage), terminate rollout.
3. Otherwise, escalate to stage $m+1$ and execute $(P_{m+1}, N_{m+1})$.

### Final State Selection: Highest-Proxy Attempted Config
* The final returned match configuration is the **Highest-Proxy Attempted Config**:
  $$m^* = \arg\max_{1 \le m \le m_{\text{stopped}}} p^{(m)}$$
* *Rationale:* All attempted configurations are already computed during the episode. Selecting the highest-proxy attempt ensures monotonicity: if an escalated configuration $(P_{m+1}, N_{m+1})$ fails due to extreme feature noise or out-of-distribution visual distortion, the system safely falls back to the best earlier attempt without degradation.

### Hyperparameter Selection on HPatches ($C^*, \tau^*$)
* **Search Space:** $C \in \{C_1, \dots, C_6\}$, $\tau \in \{0.05, 0.10, 0.15, \dots, 0.95\}$ (19 grid points).
* **Cross-Validation:** Selected per fold strictly on that fold's `train + val` sequences via sequence-grouped inner CV (`GroupKFold(n_splits=5, groups=sequence)`). Test sequences are never seen during selection.
* **Optimization Objective:**
  $$U = \text{Succ@3px} - 0.1 \times \left( \frac{\text{Cum\_KP}}{4000} \right)$$
  Ties broken by smaller cumulative keypoints.

---

## 3. Cost Accounting Protocol

All methods report two standardized cost metrics:
1. **Cumulative Keypoints ($\text{Cum\_KP}$):** Sum of all keypoint allocations across all attempted stages in the episode:
   $$\text{Cum\_KP} = \sum_{m=1}^{m_{\text{stopped}}} N_m$$
   For fixed baselines, $\text{Cum\_KP} = N$. For AFM-RL v1, $\text{Cum\_KP}$ counts all keypoints allocated across all PPO rollout steps.
2. **Cold Wall-Clock Time (ms / pair):** Measured end-to-end execution time per pair, including image loading, feature extraction (without cache hits), descriptor matching, and RANSAC geometric verification.

---

## 4. MegaDepth-1500 Evaluation Protocol (LoFTR-Style)

### Dataset & Split
* All **1,500 official test pairs** across scenes `0015` (600 pairs, 328 unique images) and `0022` (900 pairs, 478 unique images).
* Data directory: `data/megadepth/megadepth_test_1500/`.

### Image Preprocessing & Resolution
* Grayscale loading.
* Following standard benchmark conventions (LoFTR, SuperPoint), images are resized such that the maximum side length $\le 1200$ pixels:
  $$s = \min\left(1.0, \frac{1200}{\max(H, W)}\right)$$
  If $s < 1.0$, images are downsampled using `cv2.INTER_AREA`, and the camera intrinsics matrix is scaled by $s$: $K_{0,1}[:2] \leftarrow K_{0,1}[:2] \times s$.

### Relative Pose Estimation Protocol
1. **Normalized Coordinates:** Given matched 2D points $p_0, p_1$ and intrinsics $K_0, K_1$, points are converted to normalized camera rays:
   $$n_0 = K_0^{-1} [p_0, 1]^T, \quad n_1 = K_1^{-1} [p_1, 1]^T$$
2. **Essential Matrix RANSAC:**
   `cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC, prob=0.99999, threshold=0.5 / f_mean)`
   where $f_{\text{mean}} = \frac{1}{2} (f_{0x} + f_{1y})$.
3. **Cheirality & Pose Recovery:**
   `cv2.recoverPose(E, n0, n1, np.eye(3), distance=1e9, mask=mask)`
   Returns relative rotation $R$ and translation unit vector $t$.
4. **Angular Pose Error:**
   $$\Delta R = \arccos\left( \text{clip}\left(\frac{\text{Tr}(R^T R_{\text{gt}}) - 1}{2}, -1, 1\right) \right) \times \frac{180^\circ}{\pi}$$
   $$\Delta t = \arccos\left( \text{clip}\left(\frac{t^T t_{\text{gt}}}{\|t\| \|t_{\text{gt}}\|}, -1, 1\right) \right) \times \frac{180^\circ}{\pi}$$
   $$\text{Error}_{\text{pose}} = \max\left( \Delta R, \min(\Delta t, 180^\circ - \Delta t) \right)$$
   *(Translation direction compared up to sign).*
5. **Accuracy Metrics:**
   * Relative Pose AUC at thresholds $5^\circ$, $10^\circ$, and $20^\circ$ (cumulative trapezoidal integration of recall curve up to threshold).
   * Registration Recall at $5^\circ$, $10^\circ$, and $20^\circ$ ($\% \text{ pairs with } \text{Error}_{\text{pose}} < \theta$).

---

## 5. Label-Free Epipolar Verification Proxy for MegaDepth

During evaluation and agent rollouts, ground-truth pose $T_{\text{gt}}$ is **strictly inaccessible**. The agent observes only the label-free epipolar proxy $p_{\text{epi}}$, which mirrors Eq. 7 of the AFM-RL paper:
* **Inputs:** Correspondences $p_0, p_1$, intrinsics $K_0, K_1$, image dimensions.
* **Essential Fit:** Compute essential matrix $E$ and inlier mask via `cv2.findEssentialMat` with normalized coordinates and threshold $0.5 / f_{\text{mean}}$.
* **Metrics:**
  * Inlier count: $n_{\text{inliers}} = \sum \text{mask}$
  * Inlier ratio: $\text{inlier\_ratio} = n_{\text{inliers}} / n_{\text{matches}}$
  * Mean Sampson error $\bar{e}_{\text{sampson}}$ in pixels computed across essential-matrix inliers:
    $$e_{\text{sampson}} = \frac{|x_1^T E x_0|}{\sqrt{(E x_0)_1^2 + (E x_0)_2^2 + (E^T x_1)_1^2 + (E^T x_1)_2^2}} \times f_{\text{mean}}$$
* **Proxy Formula (Eq. 7 functional form, frozen constants):**
  $$\text{count\_sat} = \min\left(1.0, \frac{n_{\text{inliers}}}{120.0}\right)$$
  $$\text{reproj\_factor} = \exp\left(-\frac{\bar{e}_{\text{sampson}}}{2.0}\right)$$
  $$p_{\text{epi}} = \text{clip}\left( \text{count\_sat} \times \left(0.6 \times \text{inlier\_ratio} + 0.4 \times \text{reproj\_factor}\right), 0.0, 1.0 \right)$$
* *Constraint:* May utilize camera intrinsics $K$; must **never** access ground-truth pose $T_{\text{gt}}$ or depth maps.

---

## 6. Models, Baselines, and Primary Comparisons

### Evaluated Models
1. **AFM-RL v1 (HPatches 5-Fold Models):**
   * The 5 fold models (`checkpoints_v2/fold0..4`, each Level 1 + Level 2) are transferred **zero-shot** to all 1,500 MegaDepth pairs.
   * Reported as mean $\pm$ std across the 5 models, and pooled across pairs.
2. **AFM-RL v2 Cascade:**
   * Candidate cascade structures $C_1$--$C_6$ defined from HPatches.
   * Cross-scene transfer: $\tau$ fitted on Scene 0015 and evaluated on Scene 0022; $\tau$ fitted on Scene 0022 and evaluated on Scene 0015 (bidirectional cross-validation).

### Baseline Methods & Learned Matchers
* **Classical Baselines:** SIFT-1000, SIFT-2000, SIFT-4000, ORB-1000, ORB-4000.
* **Learned Extractors:** ALIKED-500, ALIKED-1000, ALIKED-2000, ALIKED-4000; XFeat-1000, XFeat-4000.
* **Best Fixed on Complementary Scene:** Best fixed configuration chosen on Scene 0015 evaluated on Scene 0022, and vice versa.
* **Third-Party Learned Matchers Note:** `LightGlue` is **not installed** in the environment and is out of scope. Baselines are evaluated with standard nearest-neighbor mutual or ratio matching as defined in `pipeline_utils.py`.

### Statistical Harness & Clustered Bootstrap
* **Cluster Unit:** In MegaDepth, pairs within a scene share images (e.g. 600 pairs share 328 images in 0015; co-visibility graph forms a single dominant giant component). Pairs are clustered by reference image $I_0$ (anchor view), yielding 328 clusters in 0015 and 478 clusters in 0022.
* **Bootstrap:** $B = 10,000$ cluster-level resamples for 95% percentile confidence intervals.
* **Permutation Robustness:** Primary results report original match ordering; 3 correspondence permutations evaluated as a sensitivity check.
