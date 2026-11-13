# Phase 3 Pre-Registration Protocol: Frozen Test Evaluation of AFM-RL (v1)

**Date:** 2026-11-13  
**Status:** FROZEN PRE-REGISTRATION (Committed prior to inspecting any Phase 3 test split outputs)  
**Repository:** `gnanatejavarma-byte/afm-rl` (`main` branch)  

---

## 1. Objective and Anti-Leakage Principle
This protocol pre-registers all evaluation methods, metrics, statistical comparison procedures, and hypothesis tests for the frozen test evaluation of AFM-RL under the honest observation regime (`obs_mode="proxy"`, label-free proxy observation, strict STOP execution).

**Binding Commitment:**
All models, checkpoints (`checkpoints_v2/fold0` through `checkpoints_v2/fold4`), thresholds (e.g., $CORNER\_PX = 3.0$, $REPROJ\_PX = 3.0$), reward functions, and hyperparameters are strictly frozen. No hyperparameter, architecture, or policy may be altered after observing test set performance. Any analysis conducted after observing test results must be explicitly designated as **post-hoc**.

---

## 2. Evaluation Datasets & Cross-Validation Structure
Evaluation is conducted across the 5 standard disjoint folds (`configs/folds/fold0.json` through `configs/folds/fold4.json`):
* Each fold partitions the 116 HPatches sequences into Train ($\sim 81$), Validation ($11\text{--}12$), and Test ($23\text{--}24$ sequences).
* Each sequence has 5 evaluation pairs ($k=2, 3, 4, 5, 6$), yielding:
  * Fold 0 Test: 23 sequences = 115 pairs
  * Fold 1 Test: 23 sequences = 115 pairs
  * Fold 2 Test: 23 sequences = 115 pairs
  * Fold 3 Test: 23 sequences = 115 pairs
  * Fold 4 Test: 24 sequences = 120 pairs
* **Total Pooled Test Pairs:** Exactly 580 pairs across the 5 folds (partitioning the entire dataset with zero sequence leakage).
* **Reporting:** Both:
  1. **Pooled Evaluation:** All 580 test pairs concatenated across the 5 folds.
  2. **Per-Fold Cross-Validation:** Mean $\pm$ sample standard deviation across the 5 folds ($k=0, 1, 2, 3, 4$).

---

## 3. Registered Metrics
For every method on every pair $(s, k)$, the evaluation harness computes:
1. **Succ@3px (%):** Indicator $\mathbb{I}(\text{corner\_err} < 3.0\text{ px})$, where $\text{corner\_err}$ is the mean Euclidean reprojection error of the 4 image corners between ground-truth $H$ and RANSAC-estimated $\hat{H}$.
2. **AUC@1px, AUC@3px, AUC@5px (%):** Area under the cumulative corner error curve normalized to $[0, 1]$, computed up to thresholds 1, 3, and 5 pixels.
3. **Final KP:** Number of keypoints in the final returned feature set passed to matching and homography estimation.
4. **Cumulative KP:** Total keypoints extracted across all Level 2 steps in the episode ($\sum_{t} N_t$).
5. **Mean Steps:** Average number of Level 2 decision steps per pair before the strict `STOP` action.

---

## 4. Evaluated Methods

### Primary Policy
* **Full AFM-RL (v1, Strict STOP):** Hierarchical policy. Level 1 selects the preset $P \in \{1, \dots, 14\}$ from the 9 pair statistics. Level 2 iteratively adapts budget $\Delta N$ until it emits `STOP` under label-free proxy observation. Evaluated using the respective fold checkpoint (`checkpoints_v2/fold{k}`).

### Primary Comparisons (P1 – P5)
* **P1: Best Fixed $(P^*, N^*)$ on Fold Validation:**
  * For each fold $k$, the optimal fixed configuration $(P^*, N^*)$ is selected strictly on fold $k$'s VAL split from the $14 \times 4 = 56$ grid ($P \in 14\text{ presets}$, $N \in \{500, 1000, 2000, 4000\}$).
  * Selection criterion: Highest $\text{Succ@3px}$ on VAL. Ties are broken by lowest keypoint budget $N$.
  * That chosen fixed config is then evaluated on fold $k$'s TEST split.
* **P2: ALIKED Fixed-500:** Single modern learned baseline with low keypoint budget.
* **P3: SIFT Fixed-2000:** Standard classical baseline with medium keypoint budget.
* **P4: SIFT Fixed-4000:** High-budget classical baseline.
* **P5: Oracle Preset Selection (GT Upper Bound):** For each test pair, the preset $P \in 14\text{ presets}$ that maximizes ground-truth utility under Level 2 adaptation is selected in hindsight. Represents the theoretical upper bound of preset selection.

### Ablation Rows
1. **L1 Preset + Fixed-500 (No L2):** Level 1 chooses the preset, fixed budget $N=500$ without Level 2 adaptation.
2. **L1 Preset + Fixed-1000 (No L2):** Level 1 chooses the preset, fixed budget $N=1000$ without Level 2 adaptation.
3. **ALIKED + L2:** Fixed preset = ALIKED, Level 2 adapts budget dynamically.
4. **SIFT + L2:** Fixed preset = SIFT, Level 2 adapts budget dynamically.
5. **14-Preset Average + L2:** Average performance of Level 2 across all 14 individual presets.
6. **Full AFM-RL:** Level 1 + Level 2 hierarchical policy.

---

## 5. Statistical Inference & Hypothesis Testing
To account for sequence clustering (5 pairs per sequence share the same underlying scene structure):
* **Sequence-Level Paired Bootstrap:**
  * Number of resamples: $B = 10,000$.
  * Resampling unit: Sequences (clusters), not individual image pairs. All 5 pairs for a sampled sequence are taken together.
  * Computed quantities: 95% two-sided percentile confidence intervals for:
    * $\Delta\text{Succ@3px} = \text{Succ}_{\text{Full}} - \text{Succ}_{\text{Baseline}}$
    * $\Delta\text{AUC@3px} = \text{AUC3}_{\text{Full}} - \text{AUC3}_{\text{Baseline}}$
* **Exact McNemar Test:**
  * Evaluated on paired binary success indicators ($b = \text{Full succeeds, Baseline fails}$; $c = \text{Full fails, Baseline succeeds}$).
  * Exact two-sided binomial $p$-value: $p = 2 \sum_{i=0}^{\min(b, c)} \binom{b+c}{i} 0.5^{b+c}$.
